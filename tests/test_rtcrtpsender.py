import asyncio
import time
from struct import pack
from unittest import TestCase
from unittest.mock import MagicMock, patch

from aiortc import MediaStreamTrack
from aiortc.codecs import PCMU_CODEC
from aiortc.exceptions import InvalidStateError
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack
from aiortc.rtcrtpparameters import (
    RTCRtpCapabilities,
    RTCRtpCodecCapability,
    RTCRtpCodecParameters,
    RTCRtpHeaderExtensionCapability,
    RTCRtpSendParameters,
)
from aiortc.rtcrtpsender import RTCRtpSender
from aiortc.rtp import (
    RTCP_PSFB_APP,
    RTCP_PSFB_FIR,
    RTCP_PSFB_PLI,
    RTCP_RTPFB_NACK,
    RtcpPsfbPacket,
    RtcpReceiverInfo,
    RtcpRrPacket,
    RtcpRtpfbPacket,
    RtpPacket,
    is_rtcp,
    pack_remb_fci,
)
from aiortc.stats import RTCStatsReport

from tests.test_mediastreams import VideoPacketStreamTrack

from .utils import ClosedDtlsTransport, asynctest, dummy_dtls_transport_pair

VP8_CODEC = RTCRtpCodecParameters(
    mimeType="video/VP8", clockRate=90000, payloadType=100
)

H264_CODEC = RTCRtpCodecParameters(
    mimeType="video/H264", clockRate=90000, payloadType=98
)


class BuggyStreamTrack(MediaStreamTrack):
    kind = "audio"

    async def recv(self) -> None:
        raise Exception("I'm a buggy track!")


class RTCRtpSenderTest(TestCase):
    def test_capabilities(self) -> None:
        # audio
        capabilities = RTCRtpSender.getCapabilities("audio")
        self.assertIsInstance(capabilities, RTCRtpCapabilities)
        self.assertEqual(
            capabilities.codecs,
            [
                RTCRtpCodecCapability(
                    mimeType="audio/opus", clockRate=48000, channels=2
                ),
                RTCRtpCodecCapability(
                    mimeType="audio/G722", clockRate=8000, channels=1
                ),
                RTCRtpCodecCapability(
                    mimeType="audio/PCMU", clockRate=8000, channels=1
                ),
                RTCRtpCodecCapability(
                    mimeType="audio/PCMA", clockRate=8000, channels=1
                ),
            ],
        )
        self.assertEqual(
            capabilities.headerExtensions,
            [
                RTCRtpHeaderExtensionCapability(
                    uri="urn:ietf:params:rtp-hdrext:sdes:mid"
                ),
                RTCRtpHeaderExtensionCapability(
                    uri="urn:ietf:params:rtp-hdrext:ssrc-audio-level"
                ),
            ],
        )

        # video
        capabilities = RTCRtpSender.getCapabilities("video")
        self.assertIsInstance(capabilities, RTCRtpCapabilities)
        self.assertEqual(
            capabilities.codecs,
            [
                RTCRtpCodecCapability(mimeType="video/VP8", clockRate=90000),
                RTCRtpCodecCapability(mimeType="video/rtx", clockRate=90000),
                RTCRtpCodecCapability(
                    mimeType="video/H264",
                    clockRate=90000,
                    parameters={
                        "level-asymmetry-allowed": "1",
                        "packetization-mode": "1",
                        "profile-level-id": "42001f",
                    },
                ),
                RTCRtpCodecCapability(
                    mimeType="video/H264",
                    clockRate=90000,
                    parameters={
                        "level-asymmetry-allowed": "1",
                        "packetization-mode": "1",
                        "profile-level-id": "42e01f",
                    },
                ),
                RTCRtpCodecCapability(mimeType="video/AV1", clockRate=90000),
            ],
        )
        self.assertEqual(
            capabilities.headerExtensions,
            [
                RTCRtpHeaderExtensionCapability(
                    uri="urn:ietf:params:rtp-hdrext:sdes:mid"
                ),
                RTCRtpHeaderExtensionCapability(
                    uri="http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time"
                ),
            ],
        )

        # bogus
        with self.assertRaises(ValueError):
            RTCRtpSender.getCapabilities("bogus")

    @asynctest
    async def test_construct(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender("audio", local_transport)
            self.assertEqual(sender.kind, "audio")
            self.assertEqual(sender.transport, local_transport)

    def test_construct_invalid_dtls_transport_state(self) -> None:
        dtlsTransport = ClosedDtlsTransport()
        with self.assertRaises(InvalidStateError):
            RTCRtpSender("audio", dtlsTransport)  # type: ignore

    @asynctest
    async def test_connection_error(self) -> None:
        """
        Close the underlying transport before the sender.
        """
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(AudioStreamTrack(), local_transport)
            self.assertEqual(sender.kind, "audio")

            await sender.send(RTCRtpSendParameters(codecs=[PCMU_CODEC]))

            await local_transport.stop()

    @asynctest
    async def test_handle_rtcp_nack(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(VideoStreamTrack(), local_transport)
            self.assertEqual(sender.kind, "video")

            await sender.send(RTCRtpSendParameters(codecs=[VP8_CODEC]))

            # receive RTCP feedback NACK
            packet = RtcpRtpfbPacket(
                fmt=RTCP_RTPFB_NACK, ssrc=1234, media_ssrc=sender._ssrc
            )
            packet.lost.append(7654)
            await sender._handle_rtcp_packet(packet)

            # clean shutdown
            await sender.stop()

    @asynctest
    async def test_handle_rtcp_pli(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(VideoStreamTrack(), local_transport)
            self.assertEqual(sender.kind, "video")

            await sender.send(RTCRtpSendParameters(codecs=[VP8_CODEC]))

            # receive RTCP feedback PLI
            packet = RtcpPsfbPacket(
                fmt=RTCP_PSFB_PLI, ssrc=1234, media_ssrc=sender._ssrc
            )
            await sender._handle_rtcp_packet(packet)

            # clean shutdown
            await sender.stop()

    @asynctest
    async def test_handle_rtcp_fir(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(VideoStreamTrack(), local_transport)
            self.assertEqual(sender.kind, "video")

            await sender.send(RTCRtpSendParameters(codecs=[VP8_CODEC]))

            # receive RTCP feedback FIR
            packet = RtcpPsfbPacket(
                fmt=RTCP_PSFB_FIR, ssrc=1234, media_ssrc=sender._ssrc
            )
            await sender._handle_rtcp_packet(packet)

            # clean shutdown
            await sender.stop()

    @asynctest
    async def test_handle_rtcp_fir_retransmission(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(VideoStreamTrack(), local_transport)
            await sender.send(RTCRtpSendParameters(codecs=[VP8_CODEC]))

            def fir(seq: int) -> RtcpPsfbPacket:
                # RFC 5104: media SSRC is 0, the FCI names the sender
                return RtcpPsfbPacket(
                    fmt=RTCP_PSFB_FIR,
                    ssrc=1234,
                    media_ssrc=0,
                    fci=pack("!LB3x", sender._ssrc, seq),
                )

            with patch.object(sender, "_send_keyframe") as send_keyframe:
                # a retransmitted FIR asks for the same keyframe
                await sender._handle_rtcp_packet(fir(1))
                await sender._handle_rtcp_packet(fir(1))
                self.assertEqual(send_keyframe.call_count, 1)

                # a new sequence number is a new request
                await sender._handle_rtcp_packet(fir(2))
                self.assertEqual(send_keyframe.call_count, 2)

            # clean shutdown
            await sender.stop()

    @asynctest
    async def test_handle_rtcp_remb(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(VideoStreamTrack(), local_transport)
            self.assertEqual(sender.kind, "video")

            await sender.send(RTCRtpSendParameters(codecs=[VP8_CODEC]))

            # receive RTCP feedback REMB
            packet = RtcpPsfbPacket(
                fmt=RTCP_PSFB_APP,
                ssrc=1234,
                media_ssrc=0,
                fci=pack_remb_fci(4160000, [sender._ssrc]),
            )
            await sender._handle_rtcp_packet(packet)

            # receive RTCP feedback REMB (malformed)
            packet = RtcpPsfbPacket(
                fmt=RTCP_PSFB_APP, ssrc=1234, media_ssrc=0, fci=b"JUNK"
            )
            await sender._handle_rtcp_packet(packet)

            # clean shutdown
            await sender.stop()

    @asynctest
    async def test_handle_rtcp_rr(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(VideoStreamTrack(), local_transport)
            self.assertEqual(sender.kind, "video")

            await sender.send(RTCRtpSendParameters(codecs=[VP8_CODEC]))

            # receive RTCP RR
            packet = RtcpRrPacket(
                ssrc=1234,
                reports=[
                    RtcpReceiverInfo(
                        ssrc=sender._ssrc,
                        fraction_lost=0,
                        packets_lost=0,
                        highest_sequence=630,
                        jitter=1906,
                        lsr=0,
                        dlsr=0,
                    )
                ],
            )
            await sender._handle_rtcp_packet(packet)

            # check stats
            report = await sender.getStats()
            self.assertIsInstance(report, RTCStatsReport)
            self.assertEqual(
                sorted([s.type for s in report.values()]),
                ["outbound-rtp", "remote-inbound-rtp", "transport"],
            )

            # clean shutdown
            await sender.stop()

    @asynctest
    async def test_sender_report_is_stamped_when_sent(self) -> None:
        """RFC 3550 §6.4.1: two reports with no RTP in between are told apart,
        and the RTP timestamp follows the clock from the last packet."""
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(AudioStreamTrack(), local_transport)
            sender._RTCRtpSender__rtp_timestamp = 1000
            sender._RTCRtpSender__rtp_sent_at = time.time() - 0.5
            sender._RTCRtpSender__clock_rate = 48000

            first_ntp, rtp_timestamp = sender._RTCRtpSender__sender_report_time()
            await asyncio.sleep(0.01)
            second_ntp, _ = sender._RTCRtpSender__sender_report_time()

            self.assertNotEqual(first_ntp >> 16, second_ntp >> 16)
            self.assertAlmostEqual(rtp_timestamp, 1000 + 24000, delta=480)

    @asynctest
    async def test_round_trip_time_on_a_paused_stream(self) -> None:
        """The report a receiver names is the one sent last, at its own time:
        no RTP left since the previous one, the round trip stays positive."""
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(AudioStreamTrack(), local_transport)
            ntp, _ = sender._RTCRtpSender__sender_report_time()
            sender._RTCRtpSender__lsr = (ntp >> 16) & 0xFFFFFFFF
            sender._RTCRtpSender__lsr_time = time.time() - 0.2
            await sender._handle_rtcp_packet(
                RtcpRrPacket(
                    ssrc=1234,
                    reports=[
                        RtcpReceiverInfo(
                            ssrc=sender._ssrc,
                            fraction_lost=0,
                            packets_lost=0,
                            highest_sequence=630,
                            jitter=0,
                            lsr=sender._RTCRtpSender__lsr,
                            dlsr=int(0.15 * 65536),
                        )
                    ],
                )
            )
            remote = next(
                s
                for s in (await sender.getStats()).values()
                if s.type == "remote-inbound-rtp"
            )
            self.assertAlmostEqual(remote.roundTripTime, 0.05, delta=0.02)

    @patch("aiortc.rtcrtpsender.logger.isEnabledFor")
    @asynctest
    async def test_log_debug(self, mock_is_enabled_for: MagicMock) -> None:
        mock_is_enabled_for.return_value = True

        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(VideoStreamTrack(), local_transport)
            await sender.send(RTCRtpSendParameters(codecs=[VP8_CODEC]))

            # clean shutdown
            await sender.stop()

    @asynctest
    async def test_send_keyframe(self) -> None:
        """
        Ask for a keyframe.
        """
        queue: asyncio.Queue[RtpPacket] = asyncio.Queue()

        async def mock_send_rtp(data: bytes) -> None:
            if not is_rtcp(data):
                await queue.put(RtpPacket.parse(data))

        async with dummy_dtls_transport_pair() as (local_transport, _):
            local_transport._send_rtp = mock_send_rtp  # type: ignore

            sender = RTCRtpSender(VideoStreamTrack(), local_transport)
            self.assertEqual(sender.kind, "video")

            await sender.send(RTCRtpSendParameters(codecs=[VP8_CODEC]))

            # wait for one packet to be transmitted, and ask for keyframe
            await queue.get()
            sender._send_keyframe()

            # wait for packet to be transmitted, then shutdown
            await asyncio.sleep(0.1)
            await sender.stop()

    @asynctest
    async def test_handle_encoded_packet(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(VideoPacketStreamTrack(), local_transport)
            self.assertEqual(sender.kind, "video")

            await sender.send(RTCRtpSendParameters(codecs=[H264_CODEC]))

            # wait for cleanup
            await asyncio.sleep(0.1)
            await sender.stop()

    @asynctest
    async def test_retransmit(self) -> None:
        """
        Ask for an RTP packet retransmission.
        """
        queue: asyncio.Queue[RtpPacket] = asyncio.Queue()

        async def mock_send_rtp(data: bytes) -> None:
            if not is_rtcp(data):
                await queue.put(RtpPacket.parse(data))

        async with dummy_dtls_transport_pair() as (local_transport, _):
            local_transport._send_rtp = mock_send_rtp  # type: ignore

            sender = RTCRtpSender(VideoStreamTrack(), local_transport)
            sender._ssrc = 1234
            self.assertEqual(sender.kind, "video")

            await sender.send(RTCRtpSendParameters(codecs=[VP8_CODEC]))

            # wait for one packet to be transmitted, and ask to retransmit
            packet = await queue.get()
            await sender._retransmit(packet.sequence_number)

            # wait for packet to be retransmitted, then shutdown
            await asyncio.sleep(0.1)
            await sender.stop()

            # check packet was retransmitted
            found_rtx = None
            while not queue.empty():
                queue_packet = queue.get_nowait()
                if queue_packet.sequence_number == packet.sequence_number:
                    found_rtx = queue_packet
                    break
            self.assertIsNotNone(found_rtx)
            self.assertEqual(found_rtx.payload_type, 100)
            self.assertEqual(found_rtx.ssrc, 1234)

    @asynctest
    async def test_retransmit_with_rtx(self) -> None:
        """
        Ask for an RTP packet retransmission.
        """
        queue: asyncio.Queue[RtpPacket] = asyncio.Queue()

        async def mock_send_rtp(data: bytes) -> None:
            if not is_rtcp(data):
                await queue.put(RtpPacket.parse(data))

        async with dummy_dtls_transport_pair() as (local_transport, _):
            local_transport._send_rtp = mock_send_rtp  # type: ignore

            sender = RTCRtpSender(VideoStreamTrack(), local_transport)
            sender._ssrc = 1234
            sender._rtx_ssrc = 2345
            self.assertEqual(sender.kind, "video")

            await sender.send(
                RTCRtpSendParameters(
                    codecs=[
                        VP8_CODEC,
                        RTCRtpCodecParameters(
                            mimeType="video/rtx",
                            clockRate=90000,
                            payloadType=101,
                            parameters={"apt": 100},
                        ),
                    ]
                )
            )

            # wait for one packet to be transmitted, and ask to retransmit
            packet = await queue.get()
            await sender._retransmit(packet.sequence_number)

            # wait for packet to be retransmitted, then shutdown
            await asyncio.sleep(0.1)
            await sender.stop()

            # check packet was retransmitted
            found_rtx = None
            while not queue.empty():
                queue_packet = queue.get_nowait()
                if queue_packet.payload_type == 101:
                    found_rtx = queue_packet
                    break
            self.assertIsNotNone(found_rtx)
            self.assertEqual(found_rtx.payload_type, 101)
            self.assertEqual(found_rtx.ssrc, 2345)
            self.assertEqual(found_rtx.payload[0:2], pack("!H", packet.sequence_number))

    @asynctest
    async def test_disabled(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(AudioStreamTrack(), local_transport)
            self.assertEqual(sender.kind, "audio")
            self.assertEqual(sender._enabled, True)

            await sender.send(RTCRtpSendParameters(codecs=[PCMU_CODEC]))
            sender._enabled = False
            self.assertEqual(sender._enabled, False)

            # check stats
            report = await sender.getStats()
            self.assertIsInstance(report, RTCStatsReport)
            self.assertEqual(
                sorted([s.type for s in report.values()]),
                ["outbound-rtp", "transport"],
            )

            outbound_rtp = report["outbound-rtp_" + str(id(sender))]
            self.assertEqual(outbound_rtp.packetsSent, 0)

            # clean shutdown
            await sender.stop()

    @asynctest
    async def test_stop(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(AudioStreamTrack(), local_transport)
            self.assertEqual(sender.kind, "audio")

            await sender.send(RTCRtpSendParameters(codecs=[PCMU_CODEC]))

            # clean shutdown
            await sender.stop()

    @asynctest
    async def test_stop_before_send(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(AudioStreamTrack(), local_transport)
            await sender.stop()

    @asynctest
    async def test_stop_on_exception(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            sender = RTCRtpSender(BuggyStreamTrack(), local_transport)
            self.assertEqual(sender.kind, "audio")

            await sender.send(RTCRtpSendParameters(codecs=[PCMU_CODEC]))

            # clean shutdown
            await sender.stop()

    @asynctest
    async def test_track_ended(self) -> None:
        async with dummy_dtls_transport_pair() as (local_transport, _):
            track = AudioStreamTrack()
            sender = RTCRtpSender(track, local_transport)

            await sender.send(RTCRtpSendParameters(codecs=[PCMU_CODEC]))

            # stop track and wait for RTP loop to exit
            track.stop()
            await asyncio.sleep(0.1)
