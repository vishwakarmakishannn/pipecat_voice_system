"""Compatibility guard for Pipecat's audio-only Small WebRTC transport."""

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType


def install_audio_only_cv2_shim() -> None:
    """Keep OpenCV's incompatible bundled FFmpeg out of the voice process.

    Pipecat imports OpenCV unconditionally even when WebRTC video is disabled.
    PyAV/aiortc and the macOS OpenCV wheel bundle different FFmpeg major
    versions, which makes Objective-C media classes collide at import time.
    This process is explicitly audio-only, so expose only the symbols Pipecat
    reads while defining its dormant video conversion table.
    """
    if "cv2" in sys.modules:
        return

    shim = ModuleType("cv2")
    shim.__spec__ = ModuleSpec("cv2", loader=None)
    shim.__aura_audio_only__ = True
    shim.COLOR_YUV2RGB_I420 = 100
    shim.COLOR_YUV2RGB_NV12 = 90
    shim.COLOR_GRAY2RGB = 8

    def video_disabled(*_args, **_kwargs):
        raise RuntimeError(
            "OpenCV video conversion is disabled in the audio-only voice process"
        )

    shim.cvtColor = video_disabled
    sys.modules["cv2"] = shim
