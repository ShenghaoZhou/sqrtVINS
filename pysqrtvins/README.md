# pysqrtvins — Python API for the Sqrt-VINS visual frontend

`pysqrtvins` exposes the Sqrt-VINS visual frontend (camera models, KLT feature
tracker, feature database) to Python and lets you **replace the low-level
computer vision operations with your own Python implementations** — e.g. using
a learned detector such as XFeat instead of FAST.

The frontend talks to a `vision::CVBackend` abstraction for all vision
operations (detection, KLT tracking, two-view RANSAC, histogram
pre-processing, pyramids). `pysqrtvins.vision.Backend` wraps any duck-typed
Python object into such a backend; operations the object does not implement
fall back to the built-in OpenCV backend.

## Building

The module is part of the normal build (requires pybind11, provided by pixi):

```bash
pixi run configure && pixi run build
# module is at build/pysqrtvins/pysqrtvins.*.so
python -c "import sys; sys.path.insert(0, 'build/pysqrtvins'); import pysqrtvins"
```

## Quick start

```python
import numpy as np
import pysqrtvins as sv

# Camera calibration (EuRoC cam0 intrinsics, radtan)
cam = sv.cam.CamRadtan(752, 480)
cam.set_value(np.array([458.654, 457.296, 367.215, 248.375,
                        -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05]))

tracker = sv.frontend.TrackKLT(
    cameras={0: cam},
    numfeats=200,
    stereo=False,
    histmethod=sv.frontend.HistogramMethod.HISTOGRAM,
    fast_threshold=20, gridx=5, gridy=5, minpxdist=10,
    ransacth=1.0,
    backend=None,  # None -> built-in OpenCV backend
)

for timestamp, image in frames:       # image: uint8 HxW (or HxWx3/4, converted to gray)
    tracker.feed(timestamp, [image], [0])
    obs = tracker.get_last_obs()[0]   # (N,2) float32 pixel positions
    ids = tracker.get_last_ids()[0]   # (N,) uint64 feature ids (persistent across frames)

db = tracker.get_feature_database()
for feat in db.features_containing(timestamp):
    feat.featid, feat.timestamps()[0], feat.uvs()[0], feat.uvs_norm()[0]
```

## Extending the frontend from Python

Pass `backend=sv.vision.Backend(impl)` with a duck-typed `impl` object. Every
method below is **optional**; missing ones are delegated to the built-in
fallback backend (`fallback="opencv"` by default).

```python
class MyDetector:
    def detect(self, image, mask, threshold, nonmax_suppression):
        """Called ONCE PER FRAME on the full-resolution image.

        image:  (H,W) uint8 grayscale
        mask:   (H,W) uint8 or None -- values > 127 mark EXCLUDED regions
                (note this is the opposite convention of cv2.goodFeaturesToTrack)
        threshold / nonmax_suppression: the FAST parameters from the tracker
                config (interpret freely for a learned detector)
        returns: (N,3) or (N,5) float array of (x, y, response[, size[, octave]]),
                 larger response = stronger keypoint
        """
        ...

backend = sv.vision.Backend(MyDetector(), name="my-detector")
tracker = sv.frontend.TrackKLT(..., backend=backend)
```

The full protocol (all methods optional):

| method | signature | returns |
|---|---|---|
| `detect` | `(image, mask, threshold, nonmax_suppression)` | `(N,3)/(N,5)` keypoints `(x, y, response[, size[, octave]])` |
| `refine_subpix` | `(image, pts, win_size, max_iters, eps)` | refined `(N,2)` points, or `None` to skip |
| `track` | `(prev_pyramid, curr_pyramid, pts_prev, pts_next, win_size, max_levels, max_iters, eps)` | `(pts_next (N,2), status (N,))` |
| `reject` | `(pts0_n, pts1_n, focal, threshold, confidence)` | inlier flags `(N,)` |
| `equalize_histogram` | `(image)` | uint8 image |
| `apply_clahe` | `(image, clip_limit, tile_size)` | uint8 image |
| `build_pyramid` | `(image, levels, win_size)` | list of uint8 images (level 0 first) |

Notes:

- Pyramids are lists of grayscale uint8 images, level 0 = full resolution.
  `pts_prev`/`pts_next` are in level-0 coordinates; `pts_next` contains the
  initial guesses (OpenCV `OPTFLOW_USE_INITIAL_FLOW` semantics).
- `pts0_n`/`pts1_n` in `reject` are *normalized* (undistorted) coordinates;
  the error threshold is in pixels scaled by `focal`.
- All numpy arrays handed to Python are **copies** and may be retained.
- If a callback raises, its traceback is printed and the operation is
  permanently delegated to the fallback backend (so a buggy callback cannot
  crash frontend worker threads).
- Detection is invoked once per frame on the full image; the frontend
  distributes the returned keypoints over its grid cells itself, keeping the
  strongest per cell. Learned detectors therefore pay only one forward pass
  per frame.

## Examples

- `examples/smoke_test.py` — synthetic sequence; runs the tracker with the
  built-in backend and with a pure-Python Shi-Tomasi detector.
- `examples/xfeat_frontend_example.py` — XFeat (via the vendored
  `third_party/vismatch`) as the frontend detector on a EuRoC sequence.

## Running the XFeat example

The example needs torch and the XFeat code + weights from the vismatch
`accelerated_features` submodule:

```bash
# XFeat code (needs modules/*.py and weights/xfeat.pt under
# third_party/vismatch/vismatch/third_party/accelerated_features):
git submodule update --init third_party/vismatch/vismatch/third_party/accelerated_features

# ...or, if github is unreachable, fetch the files via the jsdelivr CDN:
#   modules/{__init__,xfeat,model,interpolator}.py and weights/xfeat.pt from
#   https://cdn.jsdelivr.net/gh/verlab/accelerated_features@main/<path>

# torch (CPU build is enough; on python 3.14 use torch >= 2.14):
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
#   China mirror: https://mirror.sjtu.edu.cn/pytorch-wheels/cpu

pixi run xfeat-demo     # synthetic warped sequence, no dataset needed
# or: python pysqrtvins/examples/xfeat_frontend_example.py --images /path/to/mav0/cam0/data
```

Useful mirrors if the default package indexes are slow (measured on the
dev machine): USTC / NJU for conda-forge (`mirrors.ustc.edu.cn/anaconda/cloud/conda-forge`,
`mirror.nju.edu.cn/anaconda/cloud/conda-forge`, ~5 MB/s vs ~75 kB/s on
conda.anaconda.org), USTC pypi (`mirrors.ustc.edu.cn/pypi/simple`), SJTU for
pytorch wheels (`mirror.sjtu.edu.cn/pytorch-wheels`), jsdelivr for GitHub
raw files.
