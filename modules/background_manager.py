"""
background_manager.py
----------------------
Captures and maintains a clean "plate" of the background (the scene with
no person in it), and slowly adapts it over time so lighting changes
(clouds passing outside a window, a light flickering, etc.) don't break
the illusion.
"""

import numpy as np


class BackgroundManager:
    def __init__(self, adapt_rate=0.01):
        """
        adapt_rate:
            Fraction of each background-only pixel that gets nudged toward
            the current frame, per frame. Kept small (1%) so real lighting
            drift is absorbed smoothly, but a person standing still for a
            moment doesn't slowly "burn into" the background plate.
        """
        self.background = None
        self.adapt_rate = adapt_rate

    def calibrate(self, cap, num_frames=60, target_size=None):
        """
        Captures and averages `num_frames` frames to build the initial
        clean background plate. The user must step out of frame while this
        runs. Averaging multiple frames (rather than taking one snapshot)
        reduces ordinary sensor/compression noise in the plate.

        target_size: optional (width, height) tuple. If the camera's native
        resolution doesn't match what main.py requested (some
        webcams/drivers ignore CAP_PROP_FRAME_WIDTH/HEIGHT), we resize here
        too -- otherwise the background plate and the later live frames
        would be different shapes and fail to composite together.
        """
        import cv2  # local import to keep this module's only hard dependency numpy

        frames = []
        collected = 0
        while collected < num_frames:
            ok, frame = cap.read()
            if not ok:
                continue
            if target_size is not None and (frame.shape[1], frame.shape[0]) != target_size:
                frame = cv2.resize(frame, target_size)
            frames.append(frame.astype(np.float32))
            collected += 1

        self.background = np.mean(frames, axis=0).astype(np.uint8)
        return self.background

    def recalibrate_single(self, frame):
        """Manual override: instantly reset the background to one clean frame."""
        self.background = frame.copy()

    def update(self, frame, person_mask):
        """
        Slowly blends newly-seen background pixels (where person_mask is
        near 0) into the stored background plate. This keeps the plate
        fresh if lighting drifts, without letting the person themselves
        leak into the stored background.
        """
        if self.background is None:
            self.background = frame.copy()
            return

        # Only adapt in areas that are confidently background right now --
        # i.e. the segmentation mask is very close to 0 there.
        confidently_bg = (person_mask < 0.05).astype(np.float32)[..., None]
        blend_amount = self.adapt_rate * confidently_bg

        self.background = (
            blend_amount * frame.astype(np.float32)
            + (1 - blend_amount) * self.background.astype(np.float32)
        ).astype(np.uint8)

    def get(self):
        return self.background
