# Invisibility Cloak

A real-time webcam effect: **close your fist and you vanish**, revealing the
background behind you. **Open your hand and you reappear.**

## How it works (short version)

1. **MediaPipe Selfie Segmentation** finds "you" vs. "background" in every
   frame, producing a soft mask.
2. **MediaPipe Hands** reads your hand's landmark positions to tell fist vs.
   open hand, geometrically (no trained gesture model needed).
3. `main.py` cross-fades between the live frame and a stored clean
   background plate, using the mask, gated by your gesture.

See the top-of-file docstrings in each module for the detailed reasoning.

## Project structure

```
invisibility_cloak/
├── requirements.txt          # Python dependencies
├── main.py                   # Entry point - run this
├── README.md                 # This file
└── modules/
    ├── __init__.py
    ├── hand_gesture.py        # Fist / open-hand detection (MediaPipe Hands)
    ├── segmentation.py        # Person mask (MediaPipe Selfie Segmentation)
    └── background_manager.py  # Background capture + adaptive maintenance
```

## Setup

**1. Create a virtual environment** (Python 3.9–3.11 recommended — MediaPipe
does not yet support 3.12+ on all platforms):

```bash
python3.11 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Run it:**

```bash
python main.py
```

When it starts, **step out of frame** — it captures ~2 seconds of empty
background before showing the live window. Step back in, make a fist, and
watch yourself disappear.

## Controls

| Key | Action |
|---|---|
| `b` | Recalibrate the background instantly from the current frame (step out of frame first) |
| `q` / `Esc` | Quit |

## Tuning

All the knobs live at the top of `main.py` and in each module's constructor:

- `CROSSFADE_SPEED` (main.py) — how fast the disappear/reappear fade plays out. Lower = slower, more dramatic fade.
- `smoothing_alpha` (segmentation.py) — mask responsiveness vs. stability.
- `adapt_rate` (background_manager.py) — how fast the background plate adjusts to lighting drift.
- `smoothing_window` / `majority_threshold` (hand_gesture.py) — how many frames of agreement are needed before the gesture state flips.

## Troubleshooting

- **Camera doesn't open**: try `CAM_INDEX = 1` (or 2) in `main.py` if you have multiple cameras, or close other apps using the webcam.
- **Effect looks patchy / holes appear in your silhouette**: improve lighting — segmentation models rely on decent contrast between you and the background. Avoid backlighting.
- **Background "ghosts" appear where you stood for a while**: lower `adapt_rate` in `background_manager.py`, or press `b` to hard-reset the plate.
- **Gesture doesn't register**: make sure your whole hand (not just fingertips) is in frame and reasonably close to the camera; MediaPipe Hands needs a clear view of the palm/back of hand.
- **Low FPS**: lower `FRAME_WIDTH`/`FRAME_HEIGHT` in `main.py`, or set `model_selection=0` in `segmentation.py` for the lighter segmentation model.

## Ideas to extend

- Add a subtle "shimmer" (sinusoidal edge distortion) during the cross-fade for a more magical look.
- Trigger a sound effect on state change.
- Support multiple people/masks at once.
- Swap the static background plate for a virtual scene (green-screen style) instead of your real room.
