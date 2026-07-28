"""Temporary skeleton glue for SyLC Cast (Task 7) -- superseded by CastController (Task 12).

Drives the native cast pipeline straight into a transport: for each pushed YUV eye-pair it
uploads the six plane SRVs (NativeRenderer.set_yuv_frame), asks the renderer to pack+encode
one SBS HEVC access unit (cast_encode), and sends every resulting NAL over the transport
(WifiTransport.send_video). No timing, no state machine -- just enough to stand up
the first live PC->PC video stream end to end. The real session lifecycle (handshake,
bitrate feedback, teardown ordering) belongs to CastController and is out of scope here.

Task 9 adds audio_pump(): a one-line rewire from an AudioTap's on_pcm to
transport.send_audio, the audio counterpart of push()'s video path (no timing/state
machine there either -- the receiver side's audio-master-clock A/V sync logic lives in
cast_sender.loopback_receiver, not here).
"""

import logging

logger = logging.getLogger(__name__)


class SkeletonSession:
    """Minimal renderer -> transport pump. The caller owns start()/stop() of both the
    renderer (initialize + cast_start ... cast_stop + shutdown) and the transport
    (start + set_peer ... stop); this object only turns pushed frames into sent NALs."""

    def __init__(self, renderer, transport):
        self._renderer = renderer
        self._transport = transport

    def push(self, left, right, pts_ms, force_idr=False):
        """Encode ONE stereo frame and send it. `left`/`right` are (Y, U, V) per eye
        (per-eye planes, as NativeRenderer.set_yuv_frame expects: Y 1080x1920, U/V
        540x960, uint8). force_idr forces a keyframe (True only for the very first frame
        of a stream); it also flags the wire units as keyframes. Returns the NAL count.

        Raises on a GPU upload/encode failure (surfacing renderer.last_error) rather than
        silently sending nothing -- the skeleton's whole job is to prove the live path,
        so a real GPU failure must be loud, not swallowed."""
        if not self._renderer.set_yuv_frame(*left, *right):
            raise RuntimeError(f"set_yuv_frame failed: {self._renderer.last_error()}")
        nals = self._renderer.cast_encode(pts_ms, force_idr)
        if not nals:
            raise RuntimeError(f"cast_encode returned no packets: {self._renderer.last_error()}")
        for nal in nals:
            self._transport.send_video(pts_ms, nal, force_idr)
        return len(nals)

    def audio_pump(self, audio_tap) -> None:
        """Wire an AudioTap's decoded PCM straight into this session's transport
        (WifiTransport.send_audio) -- the audio counterpart of push()'s video
        encode->send_video path, but audio needs no encode step: AudioTap already emits
        ready-to-send s16le PCM. Rewires audio_tap's on_pcm callback post-construction
        (AudioTap has no public setter/property for it, only the constructor argument) so
        the caller can build the AudioTap however it likes and hand it here right before
        start()ing it; the caller still owns audio_tap.start()/.seek()/.stop() -- this
        method only sets where its output goes, same division of ownership as push()
        leaving renderer/transport lifecycle to the caller."""
        audio_tap._on_pcm = lambda pts_ms, pcm, sample_rate, channels: self._transport.send_audio(pts_ms, pcm)
