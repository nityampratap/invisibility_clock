"""
segmentation.py
---------------
Wraps MediaPipe Selfie Segmentation to produce a smooth, feathered
foreground (person) mask in real time.

Why this model instead of something heavier (SAM, DeepLabV3, etc.)?
    Selfie Segmentation is purpose-built for exactly this use case --
    real-time, single-person, webcam-style segmentation -- and runs on CPU
    at well over 30 FPS on modest hardware. A general-purpose segmentation
    model would be noticeably slower for no real quality benefit here.

Why smooth the mask over time AND blur its edges?
    Raw per-frame segmentation masks tend to "shimmer" -- the silhouette
    edge jitters a pixel or two frame to frame, which is very noticeable
    once you composite in a background. Blending with the previous mask
    (temporal smoothing) and feathering the edges (spatial blur) together
    turn a jittery cutout into a stable, soft-edged one.
"""

import cv2
import numpy as np
import mediapipe as mp


class PersonSegmenter:
    """
    Produces a float32 mask in [0, 1] where 1.0 = person / foreground and
    0.0 = background.
    """

    def __init__(self, model_selection=1, smoothing_alpha=0.7):
        """
        model_selection:
            0 = "general" model (faster, tuned for close-up faces/shoulders)
            1 = "landscape" model (slightly heavier, better for full body /
                further-from-camera framing -- recommended for this app)
        smoothing_alpha:
            Exponential-moving-average weight for the new frame's mask vs.
            the previous one. Higher = more responsive to fast movement,
            but more flicker. Lower = smoother, but more motion lag/trailing.
            0.7 is a good default for a full-body cloak effect.
        """
        self.mp_selfie = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie.SelfieSegmentation(
            model_selection=model_selection
        )
        self.smoothing_alpha = smoothing_alpha
        self.prev_mask = None

    def get_mask(self, frame_rgb):
        """
        Returns a smoothed, feathered mask (float32, same H x W as the
        frame, values in [0, 1]) representing "person-ness" per pixel.
        """
        result = self.segmenter.process(frame_rgb)
        raw_mask = result.segmentation_mask  # float32, values roughly in [0, 1]

        # --- Temporal smoothing ---
        # Blend with the previous frame's mask so the silhouette edge
        # doesn't jitter frame-to-frame.
        if self.prev_mask is None:
            smoothed = raw_mask
        else:
            smoothed = (
                self.smoothing_alpha * raw_mask
                + (1 - self.smoothing_alpha) * self.prev_mask
            )
        self.prev_mask = smoothed

        # --- Spatial feathering ---
        # Soften the mask edges so the person-to-background transition in
        # the final composite looks like a soft fade, not a hard cutout.
        feathered = cv2.GaussianBlur(smoothed, (15, 15), 0)

        return np.clip(feathered, 0.0, 1.0)

    def close(self):
        self.segmenter.close()
