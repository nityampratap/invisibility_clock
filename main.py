"""
main.py
-------
Real-time "Invisibility Cloak" application.

Pipeline per frame:
    1. Capture frame from webcam.
    2. Run hand-gesture detection (fist vs. open) with temporal debouncing.
    3. Run person segmentation to get a soft foreground mask.
    4. Maintain / slowly adapt a clean background plate.
    5. Based on the gesture state, cross-fade between:
         - the live frame (visible), and
         - the frame with the person's mask-region replaced by the
           background plate (invisible).
    6. Display the result.

Controls:
    b       - recalibrate background instantly using the current frame
              (make sure you are NOT in frame when you press this)
    q / ESC - quit
"""

import cv2
import numpy as np
import time

from modules.hand_gesture import HandGestureDetector
from modules.segmentation import PersonSegmenter
from modules.background_manager import BackgroundManager


# ---------------------------------------------------------------------------
# Configuration -- tune these to your camera / lighting / taste
# ---------------------------------------------------------------------------
CAM_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
CALIBRATION_FRAMES = 60          # frames averaged to build the initial background
CROSSFADE_SPEED = 0.12           # how fast the effect ramps in/out, per frame (0-1)
SHOW_DEBUG_OVERLAY = True        # draw hand skeleton + status text


def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check CAM_INDEX, and that no other "
            "application is using the camera."
        )

    gesture_detector = HandGestureDetector()
    segmenter = PersonSegmenter()
    bg_manager = BackgroundManager()

    print(">>> Step out of frame. Calibrating background...")
    bg_manager.calibrate(cap, num_frames=CALIBRATION_FRAMES)
    print(">>> Background captured. You may step back in.")

    # cloak_amount ranges 0.0 (fully visible) -> 1.0 (fully invisible).
    # We ramp this smoothly each frame instead of snapping instantly, which
    # both looks more "magical" and absorbs any single-frame gesture/mask
    # jitter so the effect doesn't flicker.
    cloak_amount = 0.0

    prev_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Warning: failed to read frame from camera.")
            continue

        frame = cv2.flip(frame, 1)  # mirror -- feels like looking in a mirror
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- 1. Gesture detection ---
        gesture_state, hand_landmarks = gesture_detector.update(frame_rgb)

        # --- 2. Person segmentation ---
        person_mask = segmenter.get_mask(frame_rgb)  # float32 HxW, 1 = person

        # --- 3. Background maintenance ---
        bg_manager.update(frame, person_mask)
        background = bg_manager.get()

        # --- 4. Smoothly ramp the cloak effect toward its target state ---
        target = 1.0 if gesture_state == "fist" else 0.0
        if cloak_amount < target:
            cloak_amount = min(target, cloak_amount + CROSSFADE_SPEED)
        elif cloak_amount > target:
            cloak_amount = max(target, cloak_amount - CROSSFADE_SPEED)

        # --- 5. Compose the final image ---
        # person_mask is 1.0 where you are. We only want to reveal the
        # background there, scaled by how "activated" the cloak currently is.
        mask_3ch = person_mask[..., None]  # HxW -> HxWx1, broadcasts over BGR
        effective_reveal = mask_3ch * cloak_amount

        composited = (
            background.astype(np.float32) * effective_reveal
            + frame.astype(np.float32) * (1 - effective_reveal)
        ).astype(np.uint8)

        # --- 6. Debug overlay ---
        if SHOW_DEBUG_OVERLAY:
            gesture_detector.draw(composited, hand_landmarks)
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            status_text = (
                f"Gesture: {gesture_state.upper()}  |  "
                f"Cloak: {cloak_amount:.2f}  |  FPS: {fps:.1f}"
            )
            cv2.putText(
                composited, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )
            cv2.putText(
                composited, "Press 'b' to recalibrate background, 'q' to quit",
                (10, composited.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1
            )

        cv2.imshow("Invisibility Cloak", composited)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('b'):
            print(">>> Recalibrating background from current frame "
                  "(step out of frame first)...")
            bg_manager.recalibrate_single(frame)

    cap.release()
    cv2.destroyAllWindows()
    gesture_detector.close()
    segmenter.close()


if __name__ == "__main__":
    main()
