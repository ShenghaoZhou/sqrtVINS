"""XFeat feature detection in the Sqrt-VINS frontend, via pysqrtvins.

This shows how a user extends the visual frontend with a learned detector
from a Python library: XFeat is loaded through the vendored vismatch package
(third_party/vismatch -> vismatch/third_party/accelerated_features) and
plugged into the tracker through pysqrtvins.vision.Backend. Everything else
(pyramids, KLT tracking, two-view RANSAC) stays in the built-in OpenCV
backend.

Usage:
    # Track a directory of images (e.g. EuRoC mav0/cam0/data)
    python xfeat_frontend_example.py --images /path/to/mav0/cam0/data

    # Synthetic demo (homography-warped textured image, no dataset needed)
    python xfeat_frontend_example.py --demo

Requires: torch, and the vismatch accelerated_features submodule:
    git submodule update --init third_party/vismatch/vismatch/third_party/accelerated_features
XFeat weights are taken from the submodule (weights/xfeat.pt) or, if missing,
downloaded from the huggingface hub (pip install huggingface_hub).
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "build" / "pysqrtvins"))
ACCEL_FEATURES = (
    REPO / "third_party" / "vismatch" / "vismatch" / "third_party"
    / "accelerated_features"
)
sys.path.insert(0, str(ACCEL_FEATURES))

import pysqrtvins as sv


class XFeatDetector:
    """Frontend detector callback backed by XFeat.

    Implements the pysqrtvins detect() protocol: called once per frame on the
    full-resolution grayscale image, returns (N,3) = (x, y, response).
    """

    def __init__(self, device="cpu", max_keypoints=1024):
        import torch

        weights = ACCEL_FEATURES / "weights" / "xfeat.pt"
        if not weights.exists() or weights.stat().st_size < 1 << 20:
            # Repo weights not checked out (git-lfs); fall back to the hub
            from huggingface_hub import snapshot_download

            weights = Path(snapshot_download("vismatch/xfeat")) / "xfeat.pt"

        from modules.xfeat import XFeat

        self.torch = torch
        self.device = device
        self.max_keypoints = max_keypoints
        self.model = XFeat(weights=str(weights))
        self.model.net = self.model.net.to(device)
        self.model.dev = torch.device(device)

    def detect(self, image, mask, threshold, nonmax_suppression):
        # image: (H,W) uint8 grayscale
        # mask:  (H,W) uint8 or None; values > 127 mark EXCLUDED regions
        with self.torch.no_grad():
            tensor = self.torch.from_numpy(image).float() / 255.0
            tensor = tensor[None, None].to(self.device)  # [1,1,H,W]
            out = self.model.detectAndCompute(
                tensor, top_k=self.max_keypoints
            )[0]
        kpts = out["keypoints"].detach().cpu().numpy()  # (N,2)
        scores = out["scores"].detach().cpu().numpy()  # (N,)
        if mask is not None and len(kpts):
            keep = mask[kpts[:, 1].astype(int), kpts[:, 0].astype(int)] <= 127
            kpts, scores = kpts[keep], scores[keep]
        return np.hstack([kpts, scores[:, None]]).astype(np.float32)


def make_camera(width, height):
    # EuRoC cam0 radtan intrinsics (scaled to the actual image size)
    cam = sv.cam.CamRadtan(width, height)
    sx, sy = width / 752.0, height / 480.0
    cam.set_value(
        np.array(
            [458.654 * sx, 457.296 * sy, 367.215 * sx, 248.375 * sy,
             -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]
        )
    )
    return cam


def demo_frames(num_frames=30, size=(752, 480)):
    """Synthetic sequence: homography warps of a textured vismatch example
    image (keeps the test free of any dataset download)."""
    base = None
    assets = REPO / "third_party" / "vismatch" / "vismatch" / "assets"
    for name in ["example_pairs/outdoor/montmartre_close.jpg",
                 "example_test/original.jpg"]:
        path = assets / name
        if path.exists():
            base = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            break
    if base is None:
        raise RuntimeError(f"no example image found under {assets}")
    base = cv2.resize(base, (size[0] * 2, size[1] * 2))
    w, h = size
    frames = []
    for i in range(num_frames):
        angle = 0.01 * i
        scale = 1.0 + 0.005 * i
        tx, ty = 3.0 * i, 1.5 * i
        cos, sin = np.cos(angle), np.sin(angle)
        # warp around the image center, then translate
        M = np.array(
            [
                [scale * cos, -scale * sin, tx + (1 - scale * cos) * w / 2 + scale * sin * h / 2],
                [scale * sin, scale * cos, ty + -scale * sin * w / 2 + (1 - scale * cos) * h / 2],
            ]
        )
        M[0, 2] += (base.shape[1] - w) / 2
        M[1, 2] += (base.shape[0] - h) / 2
        frames.append(cv2.warpAffine(base, M, (w, h)))
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=str, default=None,
                        help="directory with an image sequence (sorted by name)")
    parser.add_argument("--demo", action="store_true",
                        help="run on a synthetic warped sequence")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    args = parser.parse_args()

    if args.images:
        paths = sorted(
            p for p in Path(args.images).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )[: args.max_frames]
        frames = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]
        print(f"loaded {len(frames)} images from {args.images}")
    else:
        frames = demo_frames(args.max_frames)
        print(f"generated {len(frames)} synthetic demo frames")
    if not frames:
        raise SystemExit("no frames to process")
    height, width = frames[0].shape

    detector = XFeatDetector(device=args.device,
                             max_keypoints=args.max_keypoints)
    backend = sv.vision.Backend(detector, name="xfeat")
    tracker = sv.frontend.TrackKLT(
        cameras={0: make_camera(width, height)},
        numfeats=200,
        stereo=False,
        histmethod=sv.frontend.HistogramMethod.HISTOGRAM,
        gridx=5,
        gridy=5,
        minpxdist=8,
        backend=backend,
    )

    print(f"tracking {len(frames)} frames with XFeat detection "
          f"({args.device})...")
    t_detect_total = 0.0
    n_tracks = []
    for i, img in enumerate(frames):
        t0 = time.perf_counter()
        tracker.feed(float(i) / 20.0, [img], [0])
        t1 = time.perf_counter()
        ids = tracker.get_last_ids().get(0, np.empty(0, np.uint64))
        n_tracks.append(len(ids))
        t_detect_total += t1 - t0
        if i % 10 == 0 or i == len(frames) - 1:
            print(f"  frame {i:3d}: {len(ids):3d} tracks "
                  f"({1e3 * (t1 - t0):6.1f} ms)")

    db = tracker.get_feature_database()
    feats = db.features_containing((len(frames) - 1) / 20.0)
    lengths = [f.num_measurements() for f in feats]
    print(f"done: {min(n_tracks)}-{max(n_tracks)} tracks/frame, "
          f"db size {db.size()}, longest track "
          f"{max(lengths) if lengths else 0} frames, "
          f"avg {1e3 * t_detect_total / len(frames):.1f} ms/frame")
    assert min(n_tracks[1:]) >= 50, "too few tracks"
    assert lengths and max(lengths) >= len(frames) // 2, "tracks not persistent"
    print("xfeat frontend example PASSED")


if __name__ == "__main__":
    main()
