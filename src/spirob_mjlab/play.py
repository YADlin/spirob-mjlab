"""Keep mjlab's play CLI, with physics-frame playback for camera knockdown.

Only the interactive viewer changes. Training, policy observations, decisions,
motor waypoints and the 600 physics steps per decision remain in mjlab.
"""
from __future__ import annotations

import sys


def main():
    from mjlab.scripts import play

    camera_tasks = {
        "Mjlab-SpiRob-EggKnockdown-Camera",
        "Mjlab-SpiRob-EggKnockdown-Camera-Fixed",
        "Mjlab-SpiRob-EggKnockdown-Camera-Eval",
    }
    if not any(arg in camera_tasks for arg in sys.argv[1:]):
        return play.main()

    from spirob_mjlab.knockdown.playback import PhysicsFramePlayback

    native, viser = play.NativeMujocoViewer, play.ViserPlayViewer

    class CameraNativeViewer(PhysicsFramePlayback, native):
        pass

    class CameraViserViewer(PhysicsFramePlayback, viser):
        pass

    print("[SpiRob play] Drawing intermediate physics states at 60 Hz; "
          "policy decisions remain 1.2 s apart. Real move/HOLD pauses remain.")
    play.NativeMujocoViewer = CameraNativeViewer
    play.ViserPlayViewer = CameraViserViewer
    try:
        return play.main()
    finally:
        play.NativeMujocoViewer, play.ViserPlayViewer = native, viser


if __name__ == "__main__":
    main()
