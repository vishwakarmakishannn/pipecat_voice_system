import subprocess
import sys


def test_audio_only_webrtc_import_does_not_load_native_opencv_or_collide():
    script = """
from core.webrtc_compat import install_audio_only_cv2_shim
install_audio_only_cv2_shim()
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
import cv2
import av
print(bool(getattr(cv2, '__aura_audio_only__', False)))
print(getattr(cv2, '__file__', None))
print(bool(SmallWebRTCTransport), bool(av))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "AVFFrameReceiver" not in completed.stderr
    assert "AVFAudioReceiver" not in completed.stderr
    assert completed.stdout.splitlines()[-3:] == ["True", "None", "True True"]
