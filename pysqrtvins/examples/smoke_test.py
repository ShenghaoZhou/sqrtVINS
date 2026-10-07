"""Smoke test for the pysqrtvins frontend bindings.

Tracks a synthetic cloud of dots drifting across the image:
  1. with the default built-in OpenCV backend, and
  2. with a pure-Python detector (cv2.goodFeaturesToTrack) plugged in through
     pysqrtvins.vision.Backend -- the same extension point an XFeat user
     would implement (see xfeat_frontend_example.py).
"""

import sys

import cv2
import numpy as np

sys.path.insert(0, "build/pysqrtvins")
import pysqrtvins as sv

W, H = 752, 480
NUM_FRAMES = 20


def make_frames():
    rng = np.random.default_rng(42)
    pts = rng.uniform([30, 30], [W - 30, H - 30], size=(400, 2))
    frames = []
    for i in range(NUM_FRAMES):
        img = np.zeros((H, W), np.uint8)
        for x, y in pts + i * 2.0:  # drift 2 px/frame to the lower right
            cv2.circle(img, (int(x), int(y)), 2, 255, -1)
        frames.append(img)
    return frames


def make_camera():
    cam = sv.cam.CamRadtan(W, H)
    cam.set_value(np.array([400.0, 400.0, W / 2, H / 2, 0.0, 0.0, 0.0, 0.0]))
    return cam


def run(backend=None, label="default"):
    tracker = sv.frontend.TrackKLT(
        cameras={0: make_camera()},
        numfeats=150,
        numaruco=1024,
        stereo=False,
        histmethod=sv.frontend.HistogramMethod.NONE,
        fast_threshold=20,
        gridx=5,
        gridy=5,
        minpxdist=8,
        ransacth=1.0,
        backend=backend,
    )
    n_tracks = []
    for i, img in enumerate(make_frames()):
        tracker.feed(float(i) * 0.05, [img], [0])
        ids = tracker.get_last_ids().get(0, np.empty(0, np.uint64))
        n_tracks.append(len(ids))
    db = tracker.get_feature_database()
    feats = db.features_containing((NUM_FRAMES - 1) * 0.05)
    lengths = [f.num_measurements() for f in feats]
    print(
        f"[{label}] tracks/frame min={min(n_tracks)} max={max(n_tracks)} "
        f"last={n_tracks[-1]}, db_size={db.size()}, "
        f"max_track_len={max(lengths) if lengths else 0}"
    )
    assert min(n_tracks[1:]) >= 10, f"[{label}] too few tracks: {n_tracks}"
    assert n_tracks[-1] > 50, f"[{label}] too few tracks: {n_tracks}"
    assert lengths and max(lengths) >= NUM_FRAMES - 2, (
        f"[{label}] tracks not persistent: max len {max(lengths) if lengths else 0}"
    )
    return tracker


class ShiTomasiDetector:
    """Example custom detector implemented in pure Python.

    The frontend calls detect() once per frame on the full-resolution image;
    it must return an (N,3) array of (x, y, response).
    """

    def __init__(self):
        self.num_calls = 0
        self.last_shape = None

    def detect(self, image, mask, threshold, nonmax_suppression):
        self.num_calls += 1
        self.last_shape = image.shape
        # NOTE: the frontend mask marks EXCLUDED regions with values > 127,
        # while goodFeaturesToTrack expects VALID regions to be non-zero
        gftt_mask = None if mask is None else cv2.bitwise_not(mask)
        pts = cv2.goodFeaturesToTrack(
            image, maxCorners=2000, qualityLevel=0.01, minDistance=4,
            mask=gftt_mask
        )
        if pts is None:
            return np.empty((0, 3), np.float32)
        pts = pts.reshape(-1, 2)
        response = np.ones(len(pts), np.float32)
        return np.hstack([pts, response[:, None]]).astype(np.float32)


def main():
    run(None, "builtin-opencv")

    det = ShiTomasiDetector()
    backend = sv.vision.Backend(det, name="python-shitomasi")
    assert backend.name() == "python-shitomasi"
    assert backend.supports(sv.vision.Op.FeatureDetection)
    run(backend, "python-shitomasi")
    print(
        f"[python-shitomasi] detect() called {det.num_calls}x, "
        f"last input shape {det.last_shape} (full image = {(H, W)})"
    )
    assert det.num_calls > 0
    assert det.last_shape == (H, W), "detector should see the full image"

    print("smoke test PASSED")


if __name__ == "__main__":
    main()
