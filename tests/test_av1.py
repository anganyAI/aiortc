import io
from contextlib import redirect_stderr
from unittest import TestCase

from aiortc.codecs import depayload, get_decoder, get_encoder
from aiortc.codecs.av1 import (
    AGGREGATION_N,
    AGGREGATION_Y,
    AGGREGATION_Z,
    OBU_SEQUENCE_HEADER,
    OBU_TEMPORAL_DELIMITER,
    PACKET_MAX,
    TEMPORAL_DELIMITER,
    Av1Encoder,
    av1_depayload,
    depacketize,
    leb128_decode,
    leb128_encode,
    packetize,
    split_obus,
    with_size,
    without_size,
)
from aiortc.jitterbuffer import JitterFrame
from aiortc.rtcrtpparameters import RTCRtpCodecParameters

from .codecs import CodecTestCase

AV1_CODEC = RTCRtpCodecParameters(
    mimeType="video/AV1", clockRate=90000, payloadType=116
)

OBU_FRAME = 6


def obu(kind: int, payload: bytes, extension: bool = False) -> bytes:
    """An OBU without its size field, as RTP carries it."""
    header = bytes([kind << 3 | (0x04 if extension else 0)])
    return header + (b"\x10" if extension else b"") + payload


def join(payloads: list[bytes]) -> bytes:
    """What the jitter buffer hands the decoder for one frame."""
    return b"".join(av1_depayload(p) for p in payloads)


class Leb128Test(TestCase):
    def test_roundtrip(self) -> None:
        for value in (0, 1, 127, 128, 300, 16383, 16384, 2**32):
            data = leb128_encode(value)
            self.assertEqual(leb128_decode(data, 0), (value, len(data)))

    def test_truncated(self) -> None:
        with self.assertRaises(ValueError):
            leb128_decode(b"\x80", 0)
        with self.assertRaises(ValueError):
            leb128_decode(b"\x80" * 9, 0)


class ObuTest(TestCase):
    def test_size_field_roundtrip(self) -> None:
        for o in (obu(OBU_FRAME, b"x" * 300), obu(OBU_FRAME, b"abc", extension=True)):
            sized = with_size(o)
            self.assertTrue(sized[0] & 0x02)
            self.assertEqual(without_size(sized), o)
            self.assertEqual(with_size(sized), sized)

    def test_split_low_overhead_bitstream(self) -> None:
        obus = [obu(OBU_SEQUENCE_HEADER, b"seq"), obu(OBU_FRAME, b"f" * 500)]
        stream = b"".join(with_size(o) for o in obus)
        self.assertEqual([without_size(o) for o in split_obus(stream)], obus)

    def test_split_truncated(self) -> None:
        with self.assertRaises(ValueError):
            split_obus(with_size(obu(OBU_FRAME, b"f" * 50))[:-1])


class PacketizeTest(TestCase):
    def test_small_obus_share_a_packet(self) -> None:
        stream = TEMPORAL_DELIMITER + with_size(obu(OBU_SEQUENCE_HEADER, b"seq"))
        stream += with_size(obu(OBU_FRAME, b"f" * 100))
        payloads = packetize(stream, new_sequence=True)
        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0][0], AGGREGATION_N)  # W = 0, no Z/Y
        # the temporal delimiter is not sent
        self.assertEqual(
            depacketize(join(payloads)),
            TEMPORAL_DELIMITER
            + with_size(obu(OBU_SEQUENCE_HEADER, b"seq"))
            + with_size(obu(OBU_FRAME, b"f" * 100)),
        )

    def test_large_obu_spans_packets(self) -> None:
        frame = obu(OBU_FRAME, bytes(range(256)) * 20)  # 5121 bytes
        payloads = packetize(with_size(frame))
        self.assertGreater(len(payloads), 4)
        for payload in payloads:
            self.assertLessEqual(len(payload), PACKET_MAX)
        flags = [p[0] & (AGGREGATION_Z | AGGREGATION_Y) for p in payloads]
        self.assertEqual(flags[0], AGGREGATION_Y)
        self.assertEqual(set(flags[1:-1]), {AGGREGATION_Z | AGGREGATION_Y})
        self.assertEqual(flags[-1], AGGREGATION_Z)
        self.assertEqual(
            depacketize(join(payloads)), TEMPORAL_DELIMITER + with_size(frame)
        )

    def test_counted_elements(self) -> None:
        # W = 2: the first element has a length, the last runs to the end
        first, last = obu(OBU_SEQUENCE_HEADER, b"seq"), obu(OBU_FRAME, b"frame")
        payload = bytes([0x20]) + leb128_encode(len(first)) + first + last
        self.assertEqual(
            depacketize(join([payload])),
            TEMPORAL_DELIMITER + with_size(first) + with_size(last),
        )

    def test_continuation_without_its_start_is_dropped(self) -> None:
        frame = obu(OBU_FRAME, b"f" * 3000)
        payloads = packetize(with_size(frame))
        self.assertEqual(depacketize(join(payloads[1:])), TEMPORAL_DELIMITER)

    def test_depayload_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            av1_depayload(b"")
        self.assertEqual(depayload(AV1_CODEC, b"\x00ab"), b"\x03\x00ab")


class Av1Test(CodecTestCase):
    def test_decoder_rejects_garbage(self) -> None:
        decoder = get_decoder(AV1_CODEC)
        with redirect_stderr(io.StringIO()):
            frames = decoder.decode(JitterFrame(data=b"\x05\x00junk", timestamp=0))
        self.assertEqual(frames, [])

    def test_encoder_keyframe_starts_a_sequence(self) -> None:
        encoder = self.ensureIsInstance(get_encoder(AV1_CODEC), Av1Encoder)
        frames = self.create_video_frames(width=320, height=240, count=3)
        payloads, _ = encoder.encode(frames[0])
        self.assertTrue(payloads[0][0] & AGGREGATION_N)
        payloads, _ = encoder.encode(frames[1])
        self.assertFalse(payloads[0][0] & AGGREGATION_N)
        payloads, _ = encoder.encode(frames[2], force_keyframe=True)
        self.assertTrue(payloads[0][0] & AGGREGATION_N)
        for payload in payloads:
            self.assertLessEqual(len(payload), PACKET_MAX)

    def test_encoder_target_bitrate(self) -> None:
        encoder = self.ensureIsInstance(get_encoder(AV1_CODEC), Av1Encoder)
        self.assertEqual(encoder.target_bitrate, 1000000)
        encoder.target_bitrate = 10
        self.assertEqual(encoder.target_bitrate, 500000)

    def test_roundtrip_640_360(self) -> None:
        self.roundtrip_video(AV1_CODEC, 640, 360)

    def test_roundtrip_temporal_delimiters(self) -> None:
        # every decoded unit starts with a temporal delimiter OBU
        unit = depacketize(join(packetize(with_size(obu(OBU_FRAME, b"f")))))
        self.assertEqual(unit[0] >> 3, OBU_TEMPORAL_DELIMITER)
