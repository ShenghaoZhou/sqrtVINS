"""
CamEqui — JAX port of ov_core/src/cam/CamEqui.h.

OpenCV equidistant (fisheye) model:

    theta   = atan(r),   r^2 = x_n^2 + y_n^2
    theta_d = theta + k1 theta^3 + k2 theta^5 + k3 theta^7 + k4 theta^9
    x1      = x_n * theta_d / r
    y1      = y_n * theta_d / r
    z       = [f_x x1 + c_x, f_y y1 + c_y]

`values = [f_x, f_y, c_x, c_y, k_1, k_2, k_3, k_4]` (8-vector; the four
distortion slots hold odd-order fisheye terms rather than k1/k2/p1/p2).

Same two-layer split as `CamRadtan`: the `CamEqui` class for the estimator,
and pure JAX kernels (`equi_distort`, `equi_jacobian_dzn`, `equi_jacobian_dzeta`)
for `vmap` in the update loop.

The C++ guards `r > 1e-8` with a Python-style ternary; under JAX we use
`jnp.where` so the graph stays straight-line. The `r → 0` fallbacks are
numerically irrelevant (they only affect the first ulp at the image center)
but we keep them so the traced graph never divides by zero.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from .CamBase import CamBase

_R_TOL = jnp.float64(1e-8)


# ---------------------------------------------------------------------------
# Pure JAX kernels (uv_norm is (N, 2), values is (8,))
# ---------------------------------------------------------------------------

def _parts(uv_norm: jnp.ndarray, values: jnp.ndarray):
    """Shared intermediates.

    Returns (x, y, r, theta, theta_d, inv_r, cdist, x1, y1, fx, fy, cx, cy,
             k1, k2, k3, k4).
    """
    x = uv_norm[..., 0]
    y = uv_norm[..., 1]
    fx, fy, cx, cy, k1, k2, k3, k4 = values

    r = jnp.sqrt(x * x + y * y)
    theta = jnp.arctan(r)
    theta_d = (theta
               + k1 * theta ** 3
               + k2 * theta ** 5
               + k3 * theta ** 7
               + k4 * theta ** 9)

    # C++: inv_r = (r > 1e-8) ? 1/r : 1.0 ; cdist = (r > 1e-8) ? theta_d/r : 1.0
    inv_r = jnp.where(r > _R_TOL, 1.0 / jnp.where(r > _R_TOL, r, 1.0), 1.0)
    cdist = jnp.where(r > _R_TOL, theta_d * inv_r, 1.0)

    x1 = x * cdist
    y1 = y * cdist
    return x, y, r, theta, theta_d, inv_r, cdist, x1, y1, fx, fy, cx, cy, k1, k2, k3, k4


def equi_distort(uv_norm: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """Normalized coords → distorted pixels, matching `CamEqui::distort`."""
    (x, y, r, theta, theta_d, inv_r, cdist, x1, y1,
     fx, fy, cx, cy, k1, k2, k3, k4) = _parts(uv_norm, values)
    return jnp.stack([fx * x1 + cx, fy * y1 + cy], axis=-1)


def equi_jacobian_dzn(uv_norm: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """d(z)/d(z_n), (N, 2, 2). Literal transcription of the chain-rule stack."""
    (x, y, r, theta, theta_d, inv_r, cdist, x1, y1,
     fx, fy, cx, cy, k1, k2, k3, k4) = _parts(uv_norm, values)

    # duv_dxy = diag(fx, fy) — applied at the end
    # dxy_dxyn = diag(theta_d * inv_r)
    dxy_dxyn_diag = theta_d * inv_r

    # dxy_dr = [-x*theta_d/r^2, -y*theta_d/r^2]
    dxy_dr = jnp.stack([-x * theta_d * inv_r * inv_r,
                        -y * theta_d * inv_r * inv_r], axis=-1)

    # dr_dxyn = [x/r, y/r]  (1x2)
    dr_dxyn = jnp.stack([x * inv_r, y * inv_r], axis=-1)

    # dxy_dthd = [x/r, y/r] — numerically identical to dr_dxyn
    dxy_dthd = dr_dxyn

    dthd_dth = (1.0
                + 3.0 * k1 * theta ** 2
                + 5.0 * k2 * theta ** 4
                + 7.0 * k3 * theta ** 6
                + 9.0 * k4 * theta ** 8)
    dth_dr = 1.0 / (r * r + 1.0)

    # dH/dz_n = dxy_dxyn + (dxy_dr + dxy_dthd * dthd_dth * dth_dr) * dr_dxyn^T
    # `dthd_dth` and `dth_dr` are (N,)-vectors — `[..., None]` is mandatory, else
    # a batch of N > 1 fails to broadcast against the (N, 2) rows.
    outer = dxy_dr + dxy_dthd * dthd_dth[..., None] * dth_dr[..., None]  # (N, 2)
    # (N, 2, 2) outer product of `outer` and `dr_dxyn`. The outer stack MUST use
    # axis=-2: the inner stacks are already (N, 2) with -1 holding columns, so
    # axis=-1 would transpose the result. Numerically harmless *here* -- both
    # off-diagonals collapse to x*y/r * dcdist/dr -- but it is a transpose and would
    # break silently if this decomposition changed.
    rank1 = jnp.stack([
        jnp.stack([outer[..., 0] * dr_dxyn[..., 0], outer[..., 0] * dr_dxyn[..., 1]], axis=-1),
        jnp.stack([outer[..., 1] * dr_dxyn[..., 0], outer[..., 1] * dr_dxyn[..., 1]], axis=-1),
    ], axis=-2)
    # (N, 1, 1) so `eye` gets the batch dim broadcast up to (N, 2, 2). With
    # only one trailing axis -- `dxy_dxyn_diag[..., None]` -- this is
    # (2, 2) * (N, 1) and raises for N > 1, while N == 1 silently passes.
    inner = (jnp.eye(2, dtype=values.dtype)[None, :, :]
             * dxy_dxyn_diag[..., None, None] + rank1)

    # duv/dxy = diag(fx, fy) — a left-multiply scales the ROWS. `jnp.stack([fx, fy])`
    # is shape (2,) (fx/fy are scalars, unpacked from the (8,) `values`), so a plain
    # `* inner` broadcasts against the LAST axis and scales the columns instead. The
    # diagonal entries are fx/fy either way, which is exactly why the symmetric
    # diagonal tests passed while the off-diagonals came out transposed. Build the
    # diagonal and multiply with matmul, not elementwise — `diag_duv * inner` would
    # zero the off-diagonals with the diagonal's own zeros.
    zero = jnp.zeros_like(fx)
    diag_duv = jnp.stack([
        jnp.stack([fx, zero], axis=-1),
        jnp.stack([zero, fy], axis=-1),
    ], axis=-2)
    return jnp.matmul(diag_duv, inner)


def equi_jacobian_dzeta(uv_norm: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """d(z)/d(zeta), (N, 2, 8). Columns `[dfx, dfy, dcx, dcy, k1, k2, k3, k4]`."""
    (x, y, r, theta, theta_d, inv_r, cdist, x1, y1,
     fx, fy, cx, cy, k1, k2, k3, k4) = _parts(uv_norm, values)

    zero = jnp.zeros_like(x1)
    one = jnp.ones_like(x1)

    row_u = jnp.stack([
        x1,                  # d/dfx
        zero,                # d/dfy
        one,                 # d/dcx
        zero,                # d/dcy
        fx * x * inv_r * theta ** 3,
        fx * x * inv_r * theta ** 5,
        fx * x * inv_r * theta ** 7,
        fx * x * inv_r * theta ** 9,
    ], axis=-1)

    row_v = jnp.stack([
        zero,                # d/dfx
        y1,                  # d/dfy
        zero,                # d/dcx
        one,                 # d/dcy
        fy * y * inv_r * theta ** 3,
        fy * y * inv_r * theta ** 5,
        fy * y * inv_r * theta ** 7,
        fy * y * inv_r * theta ** 9,
    ], axis=-1)

    return jnp.stack([row_u, row_v], axis=-2)


def _cv2_equi_undistort(uv_dist: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    """cv2.fisheye.undistortPoints wrapper — mirrors `cv::fisheye::undistortPoints`.

    The C++ builds a `CV_32F` mat of shape `(1, 2)` and calls `.reshape(2)`,
    which makes it **one row × two channels** — `(N, 1, 2)`. Both details are
    load-bearing and are reproduced here:

    * `cv2.fisheye.undistortPoints` asserts
      `distorted.type() == CV_32FC2 || CV_64FC2`, i.e. it needs the two channels
      explicit. A plain `(N, 2)` array is read as `N`-rows-×-2-channels in the
      same shape, but this build rejects it outright, so reshape to
      `(N, 1, 2)` like the C++ does.
    * float32 is the C++ intermediate, so the round trip through
      `distort() → undistort()` is limited to ~1e-6 in normalized coords, not
      ~1e-15. Keeping float32 means the JAX port and the C++ port disagree with
      each other at the same magnitude, which is what an ATE comparison wants.

    Arguments are positional — `D` is a positional-only name in the binding.
    """
    import cv2

    uv = np.asarray(uv_dist, dtype=np.float32)
    was_1d = uv.ndim == 1
    uv = uv.reshape(1, 2) if was_1d else uv.reshape(uv.shape[0], 1, 2)
    out = cv2.fisheye.undistortPoints(
        uv,
        np.asarray(K, dtype=np.float64),
        np.asarray(D, dtype=np.float64),
        R=None,
        P=None,
    )
    out = out.reshape(out.shape[0], 2).astype(np.float64)
    return out[0] if was_1d else out


class CamEqui(CamBase):
    """OpenCV equidistant / fisheye pinhole camera."""

    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self._type = "pinhole-equi"

    def distort(self, uv_norm: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv_norm, dtype=np.float64)
        if uv.ndim == 1:
            uv = uv.reshape(1, 2)
        out = np.asarray(equi_distort(jnp.asarray(uv), jnp.asarray(self._values)))
        return out[0] if uv_norm.ndim == 1 else out

    def undistort(self, uv_dist: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv_dist, dtype=np.float64)
        return _cv2_equi_undistort(uv, self.get_K(), self.get_D())

    def compute_distort_jacobian(
        self, uv_norm: np.ndarray, do_calib: bool = True
    ) -> tuple[np.ndarray, np.ndarray | None]:
        uvn = jnp.asarray(uv_norm, dtype=np.float64).reshape(1, 2)
        H_dz_dzn = np.asarray(
            equi_jacobian_dzn(uvn, jnp.asarray(self._values)))[0]
        if not do_calib:
            return H_dz_dzn, None
        H_dz_dzeta = np.asarray(
            equi_jacobian_dzeta(uvn, jnp.asarray(self._values)))[0]
        return H_dz_dzn, H_dz_dzeta

    def clone(self) -> "CamEqui":
        out = CamEqui(self.width, self.height)
        out._values = self._values.copy()
        return out

    def __repr__(self) -> str:  # noqa: D105
        return (f"CamEqui({self.width}x{self.height}, "
                f"K={self.get_K().diagonal()[:2]}, "
                f"principle=({self.cx}, {self.cy}), D={self.get_D()})")
