"""
CamRadtan — JAX port of ov_core/src/cam/CamRadtan.h.

Brown–Conrady radial-tangential model. Brown, Conrady's model is:

    x1 = x_n * (1 + k1 r^2 + k2 r^4) + 2 p1 x_n y_n + p2 (r^2 + 2 x_n^2)
    y1 = y_n * (1 + k1 r^2 + k2 r^4) +   p1 (r^2 + 2 y_n^2) + 2 p2 x_n y_n
    r^2 = x_n^2 + y_n^2
    z = [f_x x1 + c_x, f_y y1 + c_y]

with `values = [f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2]` (8-vector).

Two layers, mirroring how the C++ is used:
  * the `CamRadtan` class, which the estimator's `State` holds as
    `cam_intrinsics_cameras[cam_id]` and which is the public-facing API;
  * the module-level `radtan_distort` / `radtan_jacobian_dzn` /
    `radtan_jacobian_dzeta` functions, which are pure JAX ops taking
    `(N, 2)` batches. Phase 4's `UpdaterHelper.get_feature_jacobian_full`
    calls these inside `vmap`, so they must stay pure and differentiable —
    do not route them through the class.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from .CamBase import CamBase


# ---------------------------------------------------------------------------
# Pure JAX kernels (batched: uv_norm is (N, 2), values is (8,))
# ---------------------------------------------------------------------------

def _parts(uv_norm: jnp.ndarray, values: jnp.ndarray):
    """Shared intermediates for distort and both Jacobians.

    Returns a flat tuple:
      (x, y, r_2, r_4, alpha, beta, x1, y1, fx, fy, cx, cy, k1, k2, p1, p2)
    """
    x = uv_norm[..., 0]
    y = uv_norm[..., 1]
    fx, fy, cx, cy, k1, k2, p1, p2 = values

    r_2 = x * x + y * y
    r_4 = r_2 * r_2
    alpha = 1.0 + k1 * r_2 + k2 * r_4
    x1 = x * alpha + 2.0 * p1 * x * y + p2 * (r_2 + 2.0 * x * x)
    y1 = y * alpha + p1 * (r_2 + 2.0 * y * y) + 2.0 * p2 * x * y
    # beta = dx1/dy_n = dy1/dx_n (the mixed partial is symmetric)
    beta = (2.0 * k1 * x * y + 4.0 * k2 * x * y * r_2
            + 2.0 * p1 * x + 2.0 * p2 * y)
    return x, y, r_2, r_4, alpha, beta, x1, y1, fx, fy, cx, cy, k1, k2, p1, p2


def radtan_distort(uv_norm: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """
    Normalize coords → distorted pixels, matching `CamRadtan::distort`.

    Args:
        uv_norm: (N, 2) array of normalized coordinates.
        values:  (8,) intrinsics `[fx, fy, cx, cy, k1, k2, p1, p2]`.
    Returns:
        (N, 2) array of distorted pixel coordinates.
    """
    (_x, _y, _r2, _r4, _a, _b, x1, y1, fx, fy, cx, cy, *_rest) = \
        _parts(uv_norm, values)
    return jnp.stack([fx * x1 + cx, fy * y1 + cy], axis=-1)


def radtan_jacobian_dzn(
    uv_norm: jnp.ndarray, values: jnp.ndarray
) -> jnp.ndarray:
    """
    d(z)/d(z_n), a (N, 2, 2) array — `H_dz_dzn` from
    `CamRadtan::compute_distort_jacobian`.

    Transcribed literally from the C++; each entry is verified below
    `test/test_camera.py` against `jax.jacfwd` of `radtan_distort`.
    """
    x, y, r_2, _r4, alpha, beta, _x1, _y1, fx, fy, _cx, _cy, k1, k2, p1, p2 = \
        _parts(uv_norm, values)

    dzn00 = fx * (alpha
                  + (2.0 * k1 * x * x + 4.0 * k2 * x * x * r_2)
                  + 2.0 * p1 * y + 6.0 * p2 * x)
    dzn11 = fy * (alpha
                  + (2.0 * k1 * y * y + 4.0 * k2 * y * y * r_2)
                  + 2.0 * p2 * x + 6.0 * p1 * y)
    dzn01 = fx * beta
    dzn10 = fy * beta

    # The inner stacks are (N, 2) with axis -1 already holding columns, so the
    # outer stack MUST go along -2. `axis=-1` there creates a new last axis and
    # silently transposes the whole matrix: dzn01 would land at [1, 0] and dzn10
    # at [0, 1]. (`radtan_jacobian_dzeta` below does this correctly.)
    return jnp.stack([
        jnp.stack([dzn00, dzn01], axis=-1),
        jnp.stack([dzn10, dzn11], axis=-1),
    ], axis=-2)


def radtan_jacobian_dzeta(
    uv_norm: jnp.ndarray, values: jnp.ndarray
) -> jnp.ndarray:
    """
    d(z)/d(zeta), a (N, 2, 8) array — `H_dz_dzeta` from
    `CamRadtan::compute_distort_jacobian` with `do_calib=True`.

    Columns are `[dfx, dfy, dcx, dcy, dk1, dk2, dp1, dp2]`.
    """
    x, y, r_2, r_4, _alpha, _beta, x1, y1, fx, fy, _cx, _cy, *_rest = \
        _parts(uv_norm, values)

    row_u = jnp.stack([
        x1,                 # d/dfx
        jnp.zeros_like(x1), # d/dfy
        jnp.ones_like(x1),  # d/dcx
        jnp.zeros_like(x1), # d/dcy
        fx * x * r_2,       # d/dk1
        fx * x * r_4,       # d/dk2
        2.0 * fx * x * y,   # d/dp1
        fx * (r_2 + 2.0 * x * x),  # d/dp2
    ], axis=-1)

    row_v = jnp.stack([
        jnp.zeros_like(y1), # d/dfx
        y1,                 # d/dfy
        jnp.zeros_like(y1), # d/dcx
        jnp.ones_like(y1),  # d/dcy
        fy * y * r_2,       # d/dk1
        fy * y * r_4,       # d/dk2
        fy * (r_2 + 2.0 * y * y),  # d/dp1
        2.0 * fy * x * y,   # d/dp2
    ], axis=-1)

    return jnp.stack([row_u, row_v], axis=-2)


# ---------------------------------------------------------------------------
# Undistort — delegates to cv2, as the C++ does
# ---------------------------------------------------------------------------

def _cv2_undistort(uv_dist: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """cv2.undistortPoints wrapper.

    NOTE: `cv2.undistortPoints` requires the input to be a (N, 1, 2) float32
    array. We therefore downcast to float32 exactly like the C++ port, which
    makes this the same numerical operation (same iteration count, same
    float32 intermediates).

    All arguments are positional: the OpenCV Python bindings declare `distCoeffs`
    as a positional-only name, so `dist=...` raises "Overload resolution failed".
    """
    import cv2

    uv = np.asarray(uv_dist, dtype=np.float32)
    if uv.ndim == 1:
        uv = uv.reshape(1, 2)
    if uv.ndim == 2:
        uv = uv.reshape(uv.shape[0], 1, 2)
    out = cv2.undistortPoints(
        uv,
        np.asarray(K, dtype=np.float64),
        np.asarray(D, dtype=np.float64),
        R=None,
        P=None,
    )
    out = out.reshape(out.shape[0], 2).astype(np.float64)
    if uv_dist.shape[0] == 2 and np.asarray(uv_dist).ndim == 1:
        return out[0]
    return out


class CamRadtan(CamBase):
    """Brown–Conrady radial-tangential pinhole camera."""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._type = "pinhole-radtan"

    # --------------------------------------------------------- distort
    def distort(self, uv_norm: np.ndarray) -> np.ndarray:
        """Normalized coords → raw pixel coords.

        Accepts a single (2,) point or a batch (N, 2); returns the same shape.
        """
        uv = np.asarray(uv_norm, dtype=np.float64)
        if uv.ndim == 1:
            uv = uv.reshape(1, 2)
        out = np.asarray(
            radtan_distort(jnp.asarray(uv), jnp.asarray(self._values))
        )
        return out[0] if uv_norm.ndim == 1 else out

    # ------------------------------------------------------- undistort
    def undistort(self, uv_dist: np.ndarray) -> np.ndarray:
        """Raw pixel → normalized coords, via `cv2.undistortPoints`.

        Same float32 intermediate as the C++ `CamRadtan::undistort`.
        """
        uv = np.asarray(uv_dist, dtype=np.float64)
        return _cv2_undistort(uv, self.get_K(), self.get_D())

    # ------------------------------------------------------- jacobians
    def compute_distort_jacobian(
        self, uv_norm: np.ndarray, do_calib: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Returns (H_dz_dzn [2x2], H_dz_dzeta [2x8] or None).
        """
        H_dz_dzn = np.asarray(
            radtan_jacobian_dzn(
                jnp.asarray(uv_norm, dtype=np.float64).reshape(1, 2),
                jnp.asarray(self._values),
            )
        )[0]
        if not do_calib:
            return H_dz_dzn, None
        H_dz_dzeta = np.asarray(
            radtan_jacobian_dzeta(
                jnp.asarray(uv_norm, dtype=np.float64).reshape(1, 2),
                jnp.asarray(self._values),
            )
        )[0]
        return H_dz_dzn, H_dz_dzeta

    # -------------------------------------------------------------- misc
    def clone(self) -> "CamRadtan":
        out = CamRadtan(self.width, self.height)
        out._values = self._values.copy()
        return out

    def __repr__(self) -> str:  # noqa: D105
        return (f"CamRadtan({self.width}x{self.height}, "
                f"K={self.get_K().diagonal()[:2]}, "
                f"principle=({self.cx}, {self.cy}), D={self.get_D()})")
