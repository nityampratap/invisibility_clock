"""
hand_gesture.py
----------------
Detects whether a hand is open or closed (fist) using MediaPipe Hands.

Why geometry instead of a trained classifier?
    A trained gesture-classifier model adds latency, a training pipeline,
    and a dependency on labeled data. Since "fist vs. open hand" is a very
    simple, well-defined shape difference, we can detect it directly from
    the 21 hand landmarks MediaPipe already gives us for free -- this is
    essentially instant to compute and very robust.

Why debounce ("smoothing_window" / majority vote)?
    Per-frame hand detection is noisy: motion blur, partial occlusion, or a
    hand mid-transition between open and closed can cause a single bad
    reading. If we acted on every raw frame, the cloak effect would
    flicker on and off rapidly. Instead we keep a short rolling history and
    only flip the "stable" state once a clear majority of recent frames
    agree.
"""

import mediapipe as mp
import numpy as np
from collections import deque


class HandGestureDetector:
    """
    Wraps MediaPipe Hands to detect a simple two-state gesture:
        - "fist"  -> triggers the invisibility effect
        - "open"  -> shows the person normally
    """

    # MediaPipe hand landmark indices we need.
    # See: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
    TIP_IDS = [4, 8, 12, 16, 20]        # thumb, index, middle, ring, pinky tips
    PIP_IDS = [3, 6, 10, 14, 18]        # the joint just below each tip

    def __init__(self, detection_confidence=0.7, tracking_confidence=0.7,
                 smoothing_window=7, majority_threshold=0.6):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self.mp_draw = mp.solutions.drawing_utils

        # Rolling buffer of recent raw gesture readings, used to smooth out
        # single-frame misfires (e.g. motion blur momentarily looking like
        # a fist).
        self.history = deque(maxlen=smoothing_window)
        self.majority_threshold = majority_threshold

        # Stable, debounced state exposed to the rest of the app.
        self.stable_state = "open"

    def _classify_raw(self, hand_landmarks):
        """
        Returns 'fist' or 'open' for a single frame based on landmark
        geometry.

        Logic: a finger is considered "curled" if its tip is closer to the
        wrist than its PIP joint is. We compare *distances to the wrist*
        rather than raw y-coordinates, which makes this robust to the hand
        being rotated or tilted (not just held perfectly upright).
        """
        landmarks = hand_landmarks.landmark
        wrist = landmarks[0]

        def dist(a, b):
            return np.hypot(a.x - b.x, a.y - b.y)

        curled_count = 0

        # Four fingers: index, middle, ring, pinky.
        for tip_id, pip_id in zip(self.TIP_IDS[1:], self.PIP_IDS[1:]):
            tip_dist = dist(landmarks[tip_id], wrist)
            pip_dist = dist(landmarks[pip_id], wrist)
            if tip_dist < pip_dist:
                curled_count += 1

        # Thumb is handled separately: it folds sideways across the palm
        # rather than inward toward the wrist, so we compare it to the
        # index finger's base knuckle instead.
        thumb_tip = landmarks[self.TIP_IDS[0]]
        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        thumb_curled = dist(thumb_tip, index_mcp) < dist(thumb_mcp, index_mcp)
        if thumb_curled:
            curled_count += 1

        # 4 or 5 curled fingers counts as a fist. This threshold tolerates
        # one slightly-misread finger without losing the gesture.
        return "fist" if curled_count >= 4 else "open"

    def update(self, frame_rgb):
        """
        Runs detection on a single RGB frame and returns:
            (stable_state, hand_landmarks_or_None)

        stable_state is the debounced "fist"/"open" string the app should
        act on. hand_landmarks_or_None lets the caller draw the hand
        skeleton for visual feedback.
        """
        results = self.hands.process(frame_rgb)

        landmarks_out = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            raw_state = self._classify_raw(hand_landmarks)
            landmarks_out = hand_landmarks
            self.history.append(raw_state)
        # else: no hand visible this frame. We deliberately do NOT push
        # anything into history here, so a brief occlusion (hand leaves
        # frame for a split second) doesn't immediately flip the state.

        # Majority vote over recent history for a stable decision.
        if self.history:
            fist_ratio = self.history.count("fist") / len(self.history)
            if fist_ratio >= self.majority_threshold:
                self.stable_state = "fist"
            elif (1 - fist_ratio) >= self.majority_threshold:
                self.stable_state = "open"
            # else: ambiguous window (near 50/50) -- keep previous state
            # rather than guessing.

        return self.stable_state, landmarks_out

    def draw(self, frame_bgr, hand_landmarks):
        """Draws the hand skeleton on the frame for visual debugging."""
        if hand_landmarks is not None:
            self.mp_draw.draw_landmarks(
                frame_bgr, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
            )

    def close(self):
        self.hands.close()
