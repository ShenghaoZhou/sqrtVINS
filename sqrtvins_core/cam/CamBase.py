"""
CamBase — JAX port of ov_core/src/cam/CamBase.h.

Base pinhole camera. Owns an 8-vector of intrinsics:

    camera_values = [f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2]

`set_value` asserts size 8 in the C++ (`CamBase::set_value`), which is what
`VioManagerOptions.cpp:173` builds from the YAML `intrinsics` (4-vector) and
`distortion_coeffs` (4-vector) keys. Width/height are separate members.

The distorted-to-normalized mapping (`undistort`) is an iterative inverse
problem; the C++ delegates it to `cv::undistortPoints`. We keep that
delegation — `cv2.undistortPoints` is the same iteration loop, and
reimplementing it in JAX would only add fragility. See the note at the bottom
of this module about the float32 intermediate.
"""

from __future__ import annotations

import numpy as np


class CamBase:
    """Base pinhole camera model. Holds the 8-vector intrinsic bundle."""

    def __init__(self, width: int, height: int):
        self.width = int(width)
        self.height = int(height)
        self._values = np.zeros(8, dtype=np.float64)
        self._type = "pinhole"

    # -------------------------------------------------------------- values
    def set_value(self, calib: np.ndarray) -> None:
        """Set the 8-vector of intrinsics. Mirrors `CamBase::set_value`.

        The C++ stores an OpenCV `Matx33d` K and `Vec4d` D alongside the raw
        vector; those are reconstruction-only in our port (callers that need
        them use `get_K()` / `get_D()`), so we keep just the vector.
        """
        calib = np.asarray(calib, dtype=np.float64).reshape(-1)
        assert calib.size == 8, f"cam_calib must be size 8, got {calib.size}"
        self._values = calib.copy()

    def get_value(self) -> np.ndarray:
        return self._values.copy()

    def get_K(self) -> np.ndarray:
        """3x3 intrinsic matrix in OpenCV layout (matches `get_K`)."""
        fx, fy, cx, cy = self._values[:4]
        return np.array([[fx, 0.0, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def get_D(self) -> np.ndarray:
        """4-vector [k1, k2, p1, p2] (matches `get_D`)."""
        return self._values[4:].copy()

    # ------------------------------------------------------------- intrinsics
    @property
    def fx(self) -> float:
        return float(self._values[0])

    @property
    def fy(self) -> float:
        return float(self._values[1])

    @property
    def cx(self) -> float:
        return float(self._values[2])

    @property
    def cy(self) -> float:
        return float(self._values[3])

    # -------------------------------------------------------- pure interface
    def undistort(self, uv_dist: np.ndarray) -> np.ndarray:
        """Raw pixel → normalized camera coords. Abstract."""
        raise NotImplementedError

    def distort(self, uv_norm: np.ndarray) -> np.ndarray:
        """Normalized camera coords → raw pixel. Abstract."""
        raise NotImplementedError

    def compute_distort_jacobian(
        self, uv_norm: np.ndarray, do_calib: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Jacobians of the pixel measurement w.r.t. normalized coords (2x2) and
        w.r.t. the 8 intrinsics (2x8). Abstract.
        """
        raise NotImplementedError

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _to_2vec(v: np.ndarray) -> np.ndarray:
        return np.asarray(v, dtype=np.float64).reshape(-1)[:2]


# ---------------------------------------------------------------------------
# float32 caveat (matches the C++ `undistort` behaviour)
# ---------------------------------------------------------------------------
#
# `CamRadtan::undistort` packs the input into a `cv::Mat` of type `CV_32F` and
# calls `cv::undistortPoints`, so the iteration runs in float32 even though the
# rest of the estimator is double. Our JAX port uses float64 throughout, which
# makes `undistort` *more accurate* than the C++ reference. The discrepancy
# shows up in the first ATE comparison against the C++ build — it is expected,
# not a bug (see plan Phase 7, Gate 3).
