import fractions
import logging
from typing import Optional, cast

import av
from av.frame import Frame
from av.packet import Packet
from av.video.codeccontext import VideoCodecContext
from av.video.frame import VideoFrame

from ..jitterbuffer import JitterFrame
from ..mediastreams import VIDEO_TIME_BASE, convert_timebase
from .base import Decoder, Encoder

logger = logging.getLogger(__name__)

DEFAULT_BITRATE = 1000000  # 1 Mbps
MIN_BITRATE = 500000  # 500 kbps
MAX_BITRATE = 3000000  # 3 Mbps

MAX_FRAME_RATE = 30
# Largest RTP payload, the same budget as H264 (see h264.PACKET_MAX).
PACKET_MAX = 1100

# RTP payload format for AV1: https://aomediacodec.github.io/av1-rtp-spec/
AGGREGATION_Z = 0x80  # the first OBU element continues the previous packet's last
AGGREGATION_Y = 0x40  # the last OBU element continues in the next packet
AGGREGATION_N = 0x08  # the packet starts a new coded video sequence

OBU_SEQUENCE_HEADER = 1
OBU_TEMPORAL_DELIMITER = 2
OBU_TILE_LIST = 8
OBU_HAS_EXTENSION = 0x04
OBU_HAS_SIZE = 0x02

TEMPORAL_DELIMITER = bytes([OBU_TEMPORAL_DELIMITER << 3 | OBU_HAS_SIZE, 0])


def leb128_encode(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def leb128_decode(data: bytes, pos: int) -> tuple[int, int]:
    """
    Read a LEB128 value at `pos`; return it and the position after it.
    """
    value = 0
    for i in range(8):
        if pos + i >= len(data):
            raise ValueError("LEB128 value is truncated")
        byte = data[pos + i]
        value |= (byte & 0x7F) << (7 * i)
        if not byte & 0x80:
            return value, pos + i + 1
    raise ValueError("LEB128 value is too long")


def obu_type(obu: bytes) -> int:
    return (obu[0] >> 3) & 0x0F


def _header_size(obu: bytes) -> int:
    return 2 if obu[0] & OBU_HAS_EXTENSION else 1


def split_obus(data: bytes) -> list[bytes]:
    """
    Split a low overhead bitstream, each OBU carrying its size, into OBUs.
    """
    obus = []
    pos = 0
    while pos < len(data):
        if not data[pos] & OBU_HAS_SIZE:
            # an OBU without a size field runs to the end
            obus.append(data[pos:])
            break
        size, start = leb128_decode(data, pos + _header_size(data[pos:]))
        if start + size > len(data):
            raise ValueError("OBU is truncated")
        obus.append(data[pos : start + size])
        pos = start + size
    return obus


def without_size(obu: bytes) -> bytes:
    """
    Remove an OBU's size field, as the RTP payload format recommends.
    """
    if not obu[0] & OBU_HAS_SIZE:
        return obu
    header_size = _header_size(obu)
    _, start = leb128_decode(obu, header_size)
    return bytes([obu[0] & ~OBU_HAS_SIZE]) + obu[1:header_size] + obu[start:]


def with_size(obu: bytes) -> bytes:
    """
    Give an OBU a size field, as decoders read a low overhead bitstream.
    """
    if obu[0] & OBU_HAS_SIZE:
        return obu
    header_size = _header_size(obu)
    payload = obu[header_size:]
    return (
        bytes([obu[0] | OBU_HAS_SIZE])
        + obu[1:header_size]
        + leb128_encode(len(payload))
        + payload
    )


def _obu_elements(payload: bytes) -> list[bytes]:
    """
    The OBU elements of one RTP payload (after its aggregation header).
    """
    count = (payload[0] >> 4) & 0x03  # W: 0 means each element has a length
    elements: list[bytes] = []
    pos = 1
    while pos < len(payload):
        if count == 0 or len(elements) < count - 1:
            size, pos = leb128_decode(payload, pos)
        else:
            size = len(payload) - pos
        if pos + size > len(payload):
            raise ValueError("OBU element is truncated")
        elements.append(payload[pos : pos + size])
        pos += size
    return elements


def depacketize(data: bytes) -> bytes:
    """
    Rebuild a temporal unit from a frame's packets, as av1_depayload left them.

    OBUs may span packets (the Z and Y aggregation flags), so they are
    reassembled across the frame. A continuation whose start was lost is
    dropped.
    """
    obus: list[bytes] = []
    fragment: Optional[bytes] = None
    pos = 0
    while pos < len(data):
        length, pos = leb128_decode(data, pos)
        payload = data[pos : pos + length]
        pos += length
        if not payload:
            raise ValueError("AV1 payload is empty")
        elements = _obu_elements(payload)
        if payload[0] & AGGREGATION_Z and elements:
            continued = elements.pop(0)
            if fragment is not None:
                elements.insert(0, fragment + continued)
        fragment = None
        if payload[0] & AGGREGATION_Y and elements:
            fragment = elements.pop()
        obus.extend(elements)
    kept = [
        with_size(obu)
        for obu in obus
        if obu and obu_type(obu) not in (OBU_TEMPORAL_DELIMITER, OBU_TILE_LIST)
    ]
    return TEMPORAL_DELIMITER + b"".join(kept)


def av1_depayload(payload: bytes) -> bytes:
    """
    Keep a packet's payload, prefixed with its length.

    OBUs span packets, so the decoder reassembles a frame's OBUs from its
    packets (see :func:`depacketize`); the jitter buffer joins the frame's
    packets, and the length prefixes keep them apart.
    """
    if not payload:
        raise ValueError("AV1 payload is empty")
    return leb128_encode(len(payload)) + payload


def packetize(data: bytes, new_sequence: bool = False) -> list[bytes]:
    """
    Split a temporal unit (a low overhead bitstream) into RTP payloads.

    Every OBU element carries its length (W = 0); an OBU larger than a
    packet continues in the next ones (the Y and Z flags).
    """
    obus = [
        without_size(obu)
        for obu in split_obus(data)
        if obu_type(obu) not in (OBU_TEMPORAL_DELIMITER, OBU_TILE_LIST)
    ]
    payloads: list[bytes] = []
    body = bytearray()
    continues = False  # the next packet continues the last element

    def flush(last_continues: bool) -> None:
        nonlocal body, continues
        header = 0
        if continues:
            header |= AGGREGATION_Z
        if last_continues:
            header |= AGGREGATION_Y
        if new_sequence and not payloads:
            header |= AGGREGATION_N
        payloads.append(bytes([header]) + bytes(body))
        body = bytearray()
        continues = last_continues

    for obu in obus:
        remaining = obu
        while remaining:
            room = PACKET_MAX - 1 - len(body)
            room -= len(leb128_encode(room))
            if room <= 0:
                flush(last_continues=False)
                continue
            chunk = remaining[:room]
            body += leb128_encode(len(chunk)) + chunk
            remaining = remaining[len(chunk) :]
            if remaining:
                flush(last_continues=True)
    if body:
        flush(last_continues=False)
    return payloads


class Av1Decoder(Decoder):
    def __init__(self) -> None:
        self.codec = cast(VideoCodecContext, av.CodecContext.create("libdav1d", "r"))
        # dav1d holds back several frames by default; return each at once
        self.codec.options = {"max_frame_delay": "1"}

    def decode(self, encoded_frame: JitterFrame) -> list[Frame]:
        try:
            packet = av.Packet(depacketize(encoded_frame.data))
            packet.pts = encoded_frame.timestamp
            packet.time_base = VIDEO_TIME_BASE
            return cast(list[Frame], self.codec.decode(packet))
        except (av.FFmpegError, ValueError) as e:
            logger.warning("Av1Decoder() failed to decode, skipping package: " + str(e))
            return []


class Av1Encoder(Encoder):
    def __init__(self) -> None:
        self.codec: Optional[VideoCodecContext] = None
        self.__target_bitrate = DEFAULT_BITRATE

    def encode(
        self, frame: Frame, force_keyframe: bool = False
    ) -> tuple[list[bytes], int]:
        assert isinstance(frame, VideoFrame)
        if frame.format.name != "yuv420p":
            frame = frame.reformat(format="yuv420p")

        if self.codec and (
            frame.width != self.codec.width
            or frame.height != self.codec.height
            # We only adjust bitrate if it changes by over 10%.
            or abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate
            > 0.1
        ):
            self.codec = None

        if force_keyframe:
            frame.pict_type = av.video.frame.PictureType.I
        else:
            frame.pict_type = av.video.frame.PictureType.NONE

        new_sequence = self.codec is None
        if self.codec is None:
            self.codec = av.CodecContext.create("libsvtav1", "w")
            self.codec.width = frame.width
            self.codec.height = frame.height
            self.codec.bit_rate = self.target_bitrate
            self.codec.pix_fmt = "yuv420p"
            self.codec.framerate = fractions.Fraction(MAX_FRAME_RATE, 1)
            self.codec.time_base = fractions.Fraction(1, MAX_FRAME_RATE)
            self.codec.options = {
                # real-time mode: low-delay prediction, no lookahead, CBR
                "preset": "10",
                "svtav1-params": "rtc=1:rc=2",
            }

        data = b""
        keyframe = False
        for package in self.codec.encode(frame):
            data += bytes(package)
            keyframe = keyframe or package.is_keyframe
        payloads = (
            packetize(data, new_sequence=new_sequence or keyframe) if data else []
        )
        timestamp = convert_timebase(frame.pts, frame.time_base, VIDEO_TIME_BASE)
        return payloads, timestamp

    def pack(self, packet: Packet) -> tuple[list[bytes], int]:
        payloads = packetize(bytes(packet), new_sequence=packet.is_keyframe)
        timestamp = convert_timebase(packet.pts, packet.time_base, VIDEO_TIME_BASE)
        return payloads, timestamp

    @property
    def target_bitrate(self) -> int:
        """
        Target bitrate in bits per second.
        """
        return self.__target_bitrate

    @target_bitrate.setter
    def target_bitrate(self, bitrate: int) -> None:
        bitrate = max(MIN_BITRATE, min(bitrate, MAX_BITRATE))
        self.__target_bitrate = bitrate
