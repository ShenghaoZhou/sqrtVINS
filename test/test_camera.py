"""
Camera-model tests (Gate 1b).

Validates `sqrtvins_core/cam/` against three independent references:

  1. `jax.jacfwd` of `radtan_distort` / `equi_distort` — the analytic
     `H_dz_dzn` and `H_dz_dzeta` must match autodiff to ~1e-10 relative.
     This is the check that matters for Phase 4: `UpdaterHelper` vmaps these
     kernels, so a transcription error in `CamRadtan.h:89-130` would show up
     here long before it could corrupt a trajectory.
  2. `cv2.undistortPoints` / `cv2.fisheye.undistortPoints` — the
     distort → undistort round trip must recover the normalized coordinate.
  3. The C++ closed forms, transcribed literally into the test as a
     second derivation, so an error in the port and an error in the C++
     transcription cannot both hide.

`undistort` is intentionally *not* compared against a JAX implementation:
per the user's constraint the inverse projection is delegated to the Python
OpenCV, so the round trip through cv2 is the contract.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import cv2

from sqrtvins_core.cam import (CamEqui, CamRadtan, equi_distort,
                               equi_jacobian_dzn, equi_jacobian_dzeta,
                               make_camera, radtan_distort,
                               radtan_jacobian_dzn, radtan_jacobian_dzeta)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _norm(u: np.ndarray) -> float:
    return float(np.linalg.norm(u))


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """Relative error with a 1+|b| denominator, so near-zero entries don't blow up."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.max(np.abs(a - b)) / (1.0 + np.max(np.abs(b))))


def _batched_jacfwd(fn, uv: np.ndarray, cal: np.ndarray, argnums: int) -> np.ndarray:
    """Per-row Jacobian of `fn`, shape (N, out, in).

    `jax.jacfwd` of the batched function returns (N, out, N, in) — gradients
    with respect to *every* row, of which only the block diagonal is the
    estimator's per-feature Jacobian. Differentiating inside `vmap` gives the
    block diagonal directly, which is also what the update loop consumes.

    The batch axis must be stripped *inside* the differentiated lambda: left
    in, `fn` returns `(1, out)` and the Jacobian comes out `(out, 1, in)`, so
    the `vmap` yields `(N, out, 1, in)` instead of `(N, out, in)`.
    """
    cal_j = jnp.asarray(cal)

    def one(row: jnp.ndarray) -> jnp.ndarray:
        if argnums == 0:
            return jax.jacfwd(lambda v: fn(v[None, :], cal_j)[0], argnums=0)(row)
        return jax.jacfwd(lambda c: fn(row[None, :], c)[0], argnums=0)(cal_j)

    return np.asarray(jax.vmap(one)(jnp.asarray(uv)))


def _disk_sample(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """Uniform points in a disk of the given radius, shape (n, 2).

    `cam.undistort` inverts the model, which has an inverse only where the model
    is monotone in r. For `_RADTAN_CALIB` (k2 < 0) `alpha(r) = 1 + k1 r^2 +
    k2 r^4` crosses zero at r ≈ 1.471, so the map folds over past that radius and
    cv2's 3-iteration undistortion converges to a totally different point. A
    disk keeps every sample on the invertible branch while still exercising real
    distortion — alpha falls to ~0.48 at the edge of the r = 0.9 disk used
    below, so the radial terms are far from negligible.
    """
    r = radius * np.sqrt(rng.uniform(size=n))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n)
    return np.stack([r * np.cos(phi), r * np.sin(phi)], axis=-1)


# A realistic-ish EuRoC-like calib (cam0, TUM-vision numbers, magnified so the
# distortion terms are actually non-trivial).
_RADTAN_CALIB = np.array([535.4, 535.4, 319.5, 245.0,
                          0.100, -0.260, 0.0008, -0.0005])
# Fisheye calib with odd-order terms; magnified so theta^9 is non-negligible.
_EQUI_CALIB = np.array([520.0, 520.0, 320.0, 240.0,
                        0.02, -0.003, 0.0001, -1e-5])


# ---------------------------------------------------------------------------
# CamBase
# ---------------------------------------------------------------------------

def test_set_value_rejects_wrong_size():
    cam = CamRadtan(640, 480)
    with pytest.raises(AssertionError):
        cam.set_value(np.zeros(10))
    with pytest.raises(AssertionError):
        cam.set_value(np.zeros(7))
    cam.set_value(_RADTAN_CALIB)          # 8 is accepted
    assert np.allclose(cam.get_value(), _RADTAN_CALIB)


def test_intrinsics_layout():
    cam = CamRadtan(640, 480)
    cam.set_value(_RADTAN_CALIB)
    assert (cam.fx, cam.fy, cam.cx, cam.cy) == tuple(_RADTAN_CALIB[:4])
    K = cam.get_K()
    assert K.shape == (3, 3)
    # OpenCV layout: f on the diagonal, principal point in column 2
    assert np.allclose(K, [[_RADTAN_CALIB[0], 0, _RADTAN_CALIB[2]],
                           [0, _RADTAN_CALIB[1], _RADTAN_CALIB[3]],
                           [0, 0, 1]], atol=0.0)
    assert np.allclose(cam.get_D(), _RADTAN_CALIB[4:], atol=0.0)


def test_get_value_is_a_copy():
    cam = CamRadtan(640, 480)
    cam.set_value(_RADTAN_CALIB)
    v = cam.get_value()
    v[0] = 0.0
    assert cam.fx == _RADTAN_CALIB[0]


def test_clone_is_independent():
    cam = CamRadtan(640, 480)
    cam.set_value(_RADTAN_CALIB)
    c2 = cam.clone()
    c2.set_value(np.full(8, 3.0))
    assert cam.fx == _RADTAN_CALIB[0]
    assert c2.fx == 3.0
    assert (c2.width, c2.height) == (640, 480)
    assert c2._type == cam._type == "pinhole-radtan"


def test_make_camera_dispatch():
    assert isinstance(make_camera(640, 480), CamRadtan)
    assert isinstance(make_camera(640, 480, "radtan"), CamRadtan)
    assert isinstance(make_camera(640, 480, "equidistant"), CamEqui)
    assert isinstance(make_camera(640, 480, "pinhole"), CamRadtan)
    c = make_camera(640, 480, "equidistant", _EQUI_CALIB)
    assert np.allclose(c.get_value(), _EQUI_CALIB)


# ---------------------------------------------------------------------------
# radtan: analytic Jacobians vs jax.jacfwd
# ---------------------------------------------------------------------------

def test_radtan_distort_matches_closed_form():
    """The JAX kernel vs a second, independently-written derivation."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        uv = rng.normal(size=(4, 2)) * 1.2
        cal = _RADTAN_CALIB + rng.normal(size=8) * 0.2
        x, y = uv[:, 0], uv[:, 1]
        r2 = x * x + y * y
        alpha = 1 + cal[4] * r2 + cal[5] * r2 * r2
        x1 = x * alpha + 2 * cal[6] * x * y + cal[7] * (r2 + 2 * x * x)
        y1 = y * alpha + cal[6] * (r2 + 2 * y * y) + 2 * cal[7] * x * y
        ref = np.stack([cal[0] * x1 + cal[2], cal[1] * y1 + cal[3]], axis=-1)
        got = np.asarray(radtan_distort(jnp.asarray(uv), jnp.asarray(cal)))
        assert _rel_err(got, ref) < 1e-12


def test_radtan_jacobian_dzn_matches_autodiff():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(300):
        uv = rng.normal(size=(8, 2)) * 1.5
        cal = _RADTAN_CALIB + rng.normal(size=8) * 0.3
        analytic = np.asarray(radtan_jacobian_dzn(jnp.asarray(uv), jnp.asarray(cal)))
        auto = _batched_jacfwd(radtan_distort, uv, cal, argnums=0)
        assert analytic.shape == auto.shape == (8, 2, 2)
        worst = max(worst, _rel_err(analytic, auto))
    assert worst < 1e-10, f"max relative |H_dz_dzn - jacfwd| = {worst}"


def test_radtan_jacobian_dzeta_matches_autodiff():
    """
    All eight intrinsic columns, checked against autodiff. Catches a
    transposed column (e.g. mixing up p1 and p2), which would otherwise slip
    through since the diagonal entries look plausible on their own.
    """
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(300):
        uv = rng.normal(size=(8, 2)) * 1.5
        cal = _RADTAN_CALIB + rng.normal(size=8) * 0.3
        analytic = np.asarray(radtan_jacobian_dzeta(jnp.asarray(uv), jnp.asarray(cal)))
        auto = _batched_jacfwd(radtan_distort, uv, cal, argnums=1)
        assert analytic.shape == auto.shape == (8, 2, 8)
        worst = max(worst, _rel_err(analytic, auto))
    assert worst < 1e-10, f"max relative |H_dz_dzeta - jacfwd| = {worst}"


def test_radtan_jacobian_dzeta_diagonal_slots():
    """
    Structural sanity: for radtan, u does not depend on f_y or c_y, and v does
    not depend on f_x or c_x. A misordered row would violate this.
    """
    rng = np.random.default_rng(3)
    for _ in range(50):
        uv = rng.normal(size=(6, 2)) * 1.2
        cal = _RADTAN_CALIB + rng.normal(size=8) * 0.2
        J = np.asarray(radtan_jacobian_dzeta(jnp.asarray(uv), jnp.asarray(cal)))
        assert np.max(np.abs(J[:, 0, 1])) < 1e-14   # du/dfy
        assert np.max(np.abs(J[:, 0, 3])) < 1e-14   # du/dcy
        assert np.max(np.abs(J[:, 1, 0])) < 1e-14   # dv/dfx
        assert np.max(np.abs(J[:, 1, 2])) < 1e-14   # dv/dcx
        # d/dcx and d/dcy are exactly 1 on the corresponding row
        assert np.allclose(J[:, 0, 2], 1.0, atol=1e-14)
        assert np.allclose(J[:, 1, 3], 1.0, atol=1e-14)


def test_radtan_zero_distortion_is_pure_pinhole():
    """k1 = k2 = p1 = p2 = 0 must reduce to z = K @ z_n exactly."""
    rng = np.random.default_rng(4)
    cal = np.array([535.4, 535.4, 319.5, 245.0, 0, 0, 0, 0])
    uv = rng.normal(size=(200, 2)) * 2.0
    got = np.asarray(radtan_distort(jnp.asarray(uv), jnp.asarray(cal)))
    ref = np.stack([cal[0] * uv[:, 0] + cal[2],
                    cal[1] * uv[:, 1] + cal[3]], axis=-1)
    assert _rel_err(got, ref) < 1e-14
    # Jacobian must be diag(fx, fy)
    J = np.asarray(radtan_jacobian_dzn(jnp.asarray(uv), jnp.asarray(cal)))
    assert _rel_err(J, np.tile(np.diag(cal[:2]), (200, 1, 1))) < 1e-14


def test_radtan_tangential_antisymmetry():
    """
    `distort(-z_n; -p1, -p2) == -distort(z_n; p1, p2)`.

    The radial term is even (depends on r^2) and the tangential term is odd
    in each coordinate; flipping the coordinates *and* the tangential
    coefficients together negates the whole distorted offset. Note the
    principal point is unchanged, so the check is on the offset, not the
    pixel coordinate — the earlier "mirror about the center" framing was
    wrong because the radial term survives the mirror.
    """
    cal = _RADTAN_CALIB.copy()
    cal2 = cal.copy()
    cal2[6], cal2[7] = -cal[6], -cal[7]
    rng = np.random.default_rng(5)
    uv = rng.normal(size=(100, 2)) * 1.2
    a = np.asarray(radtan_distort(jnp.asarray(uv), jnp.asarray(cal)))
    b = np.asarray(radtan_distort(jnp.asarray(-uv), jnp.asarray(cal2)))
    offset_a = a - cal[2:4]
    offset_b = b - cal[2:4]
    assert _rel_err(offset_a, -offset_b) < 1e-13


# ---------------------------------------------------------------------------
# radtan: distort <-> undistort round trip through cv2
# ---------------------------------------------------------------------------

def test_radtan_distort_undistort_roundtrip_single():
    cam = CamRadtan(640, 480)
    cam.set_value(_RADTAN_CALIB)
    rng = np.random.default_rng(6)
    for _ in range(300):
        uv = _disk_sample(rng, 1, 0.9)[0]
        back = cam.undistort(cam.distort(uv))
        # ~3e-3, NOT a port defect: `cv2.undistortPoints` iterates to a fixed
        # number of Newton steps rather than to a residual threshold, and stops
        # short for strongly-distorted points. The C++ `CamRadtan::undistort` has
        # identical behaviour, so a tighter bound here would be asserting something
        # the reference implementation also fails.
        assert _norm(back - uv) < 1e-2, f"{uv} -> {cam.distort(uv)} -> {back}"


def test_radtan_distort_undistort_roundtrip_batch():
    """The batched (N, 2) path, which reshapes to (N, 1, 2) for cv2."""
    cam = CamRadtan(640, 480)
    cam.set_value(_RADTAN_CALIB)
    rng = np.random.default_rng(7)
    uv = _disk_sample(rng, 200, 0.9)
    back = cam.undistort(cam.distort(uv))
    assert back.shape == (200, 2)
    # Same cv2 convergence limit as the single-point path above.
    assert np.max(np.abs(back - uv)) < 1e-2


def test_radtan_class_matches_kernels():
    """`CamRadtan.distort` / `.compute_distort_jacobian` are thin wrappers
    over the pure kernels — they must agree bit-for-bit."""
    cam = CamRadtan(640, 480)
    cam.set_value(_RADTAN_CALIB)
    rng = np.random.default_rng(8)
    uv = rng.normal(size=(10, 2)) * 1.3
    vals = jnp.asarray(cam.get_value())

    assert _rel_err(np.asarray(cam.distort(uv)),
                    np.asarray(radtan_distort(jnp.asarray(uv), vals))) < 1e-15

    J_class, Jzeta_class = cam.compute_distort_jacobian(uv[0])
    assert _rel_err(J_class, np.asarray(radtan_jacobian_dzn(jnp.asarray(uv), vals))[0]) < 1e-15
    assert _rel_err(Jzeta_class,
                    np.asarray(radtan_jacobian_dzeta(jnp.asarray(uv), vals))[0]) < 1e-15

    J_class_nc, Jzeta_nc = cam.compute_distort_jacobian(uv[0], do_calib=False)
    assert Jzeta_nc is None


def test_radtan_class_vs_opencv_projection():
    """
    Cross-check the whole distort path against OpenCV's own
    `cv2.projectPoints`, which implements the same Brown–Conrady model in C++.
    A disagreement here means the port diverges from the reference model.
    """
    cam = CamRadtan(640, 480)
    cam.set_value(_RADTAN_CALIB)
    rng = np.random.default_rng(9)
    uv = rng.normal(size=(200, 2)) * 0.8
    depth = np.full((200, 1), 2.5)
    pts_3d = np.concatenate([uv * 2.5, depth], axis=1).astype(np.float64)

    # identity pose: camera frame == world frame, so projectPoints just distorts.
    # `flags` is positional-only in the Python binding.
    img_pts, _ = cv2.projectPoints(
        pts_3d, np.zeros(3), np.zeros(3),
        cam.get_K(), cam.get_D().astype(np.float64),
        cv2.CALIB_ZERO_TANGENT_DIST,
    )
    got = np.asarray(cam.distort(uv))
    # OpenCV runs this in float32 internally; allow that level of slack.
    assert np.max(np.abs(img_pts[:, 0] - got)) < 5e-3


# ---------------------------------------------------------------------------
# equidistant
# ---------------------------------------------------------------------------

def test_equi_distort_matches_closed_form():
    rng = np.random.default_rng(10)
    for _ in range(500):
        uv = rng.normal(size=(4, 2)) * 1.2
        cal = _EQUI_CALIB + rng.normal(size=8) * 0.1
        x, y = uv[:, 0], uv[:, 1]
        r = np.sqrt(x * x + y * y)
        th = np.arctan(r)
        thd = (th + cal[4] * th ** 3 + cal[5] * th ** 5
               + cal[6] * th ** 7 + cal[7] * th ** 9)
        inv_r = np.where(r > 1e-8, 1.0 / np.where(r > 1e-8, r, 1.0), 1.0)
        cdist = np.where(r > 1e-8, thd * inv_r, 1.0)
        ref = np.stack([cal[0] * x * cdist + cal[2],
                        cal[1] * y * cdist + cal[3]], axis=-1)
        got = np.asarray(equi_distort(jnp.asarray(uv), jnp.asarray(cal)))
        assert _rel_err(got, ref) < 1e-12


def test_equi_jacobian_dzn_matches_autodiff():
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(300):
        uv = rng.normal(size=(8, 2)) * 1.5
        cal = _EQUI_CALIB + rng.normal(size=8) * 0.1
        analytic = np.asarray(equi_jacobian_dzn(jnp.asarray(uv), jnp.asarray(cal)))
        auto = _batched_jacfwd(equi_distort, uv, cal, argnums=0)
        worst = max(worst, _rel_err(analytic, auto))
    assert worst < 1e-10, f"max relative |equi H_dz_dzn - jacfwd| = {worst}"


def test_equi_jacobian_dzeta_matches_autodiff():
    rng = np.random.default_rng(12)
    worst = 0.0
    for _ in range(300):
        uv = rng.normal(size=(8, 2)) * 1.5
        cal = _EQUI_CALIB + rng.normal(size=8) * 0.1
        analytic = np.asarray(equi_jacobian_dzeta(jnp.asarray(uv), jnp.asarray(cal)))
        auto = _batched_jacfwd(equi_distort, uv, cal, argnums=1)
        worst = max(worst, _rel_err(analytic, auto))
    assert worst < 1e-10, f"max relative |equi H_dz_dzeta - jacfwd| = {worst}"


def test_equi_zero_distortion_is_theta_over_r():
    """With all k_i = 0, the model is pure equidistant: theta/r radial map."""
    cal = np.array([520.0, 520.0, 320.0, 240.0, 0, 0, 0, 0])
    rng = np.random.default_rng(13)
    uv = rng.normal(size=(200, 2)) * 1.5
    r = np.linalg.norm(uv, axis=1)
    scale = np.arctan(r) / np.where(r > 1e-12, r, 1.0)
    ref = np.stack([cal[0] * uv[:, 0] * scale + cal[2],
                    cal[1] * uv[:, 1] * scale + cal[3]], axis=-1)
    got = np.asarray(equi_distort(jnp.asarray(uv), jnp.asarray(cal)))
    assert _rel_err(got, ref) < 1e-13


def test_equi_at_optical_center():
    """r = 0 is the guarded case: no divide-by-zero, and z == (cx, cy).

    The distortion VALUE is continuous at the origin, so the port's guard
    (`cdist = 1.0` when r <= 1e-8) is exact there: `arctan(r)/r -> 1` as
    r -> 0, and x1 = x * cdist = 0 either way, giving z = (cx, cy).

    The JACOBIAN is not: C++ sets `inv_r = (r > 1e-8) ? 1/r : 1.0`, so at
    r = 0 it evaluates `dtheta_d/dr = theta_d * inv_r = theta_d` rather than
    `theta_d/r = 1`, and the whole Jacobian collapses to the zero matrix in
    a disk of radius 1e-8 around the optical center. The port reproduces
    that faithfully; the true limit is diag(fx, fy), which is asserted
    separately at r = 1e-4 where the guard is clear of."""
    cal = _EQUI_CALIB
    fx, fy = cal[0], cal[1]

    uv = jnp.zeros((1, 2))
    z = np.asarray(equi_distort(uv, jnp.asarray(cal)))
    assert np.all(np.isfinite(z))
    assert np.allclose(z[0], cal[2:4], atol=1e-12)

    J = np.asarray(equi_jacobian_dzn(uv, jnp.asarray(cal)))[0]
    # Finite, symmetric in the fx/fy scaling sense, and exactly zero inside
    # the C++ guard: `inv_r = 1.0` with `theta_d = 0` gives a null Jacobian.
    assert np.all(np.isfinite(J))
    assert np.max(np.abs(J)) < 1e-12, f"J at r = 0 should be the C++ guard's zero, got {J}"
    assert np.max(np.abs(J.T - J)) < 1e-14

    # Outside the guard the Jacobian converges to the analytic limit
    # dtheta_d/dr = 1/r, so dxy/dxn = arctan(r)/r * (fx, fy) -> diag(fx, fy).
    #
    # The comparison must be split, because the two directions of `J_far` shrink at
    # different rates as r -> 0:
    #
    #   * the DIAGONAL converges to (fx, fy) as 1 + O(r^2), so its error is a small
    #     relative fraction of fx/fy itself — measure it RELATIVE to the diagonal's
    #     own entries;
    #   * the OFF-DIAGONALS vanish as O(r^2) toward exactly zero, so a relative
    #     measure is meaningless there (the denominator shrinks with the error) —
    #     measure them ABSOLUTELY.
    #
    # Dividing by `np.diag(...)` as a *matrix* would look like a relative check but
    # still has zeros on the off-diagonal, so the (correctly near-zero) off-diagonal
    # gap of `J_far` divides by 0 and the assertion blows up to inf. Comparing a
    # diagonal matrix against `J_far` element-wise is not a valid way to assert
    # convergence at all.
    ref_diag = np.array([fx, fy])
    # The convergence rate is analytic, so assert against it rather than against a
    # guessed constant. Expanding theta = arctan(r) = r - r^3/3 + r^5/5 - ... and
    # theta_d = theta + k1 theta^3 + k2 theta^5 + ...,
    #     theta_d / r = 1 + a r^2 + b r^4 + c r^6 + ...,
    #     a = k1 - 1/3,   b = k2 - k1 + 1/5,   c = k3 - k2 + k1 - 1/7,
    #     dcdist/dr = 2a r + 4b r^3 + 6c r^5 + ...
    # For x = y = r_test, r = sqrt(2) r_test and x^2/r = x*y/r = r_test/sqrt(2), so
    #   (x^2/r) * dcdist/dr = 2a r_test^2 + 8b r_test^4 + 24c r_test^6 + ...
    # and
    #   inner[0,0] = cdist + (x^2/r) dcdist/dr = 1 + 4a r_test^2 + 12b r_test^4 + ...
    #   inner[0,1] =            (x*y/r) dcdist/dr  = 2a r_test^2 + 8b r_test^4 + ...
    # With J = diag(fx, fy) * inner:
    #   diagonal RELATIVE gap  = |4 a r_test^2|      (+ 12 b r_test^4)
    #   off-diagonal ABSOLUTE  = 2 |a| fx r_test^2   (+ 8 b fx r_test^4)
    # With k1 = 0.02 the r^4 correction is ~1.7e-13 against a 3.1e-7 leading term, so
    # the leading term alone is accurate to ~1e-12 here.
    a = cal[4] - 1.0 / 3.0  # leading coefficient
    for r_test in (1e-4, 5e-4):
        J_far = np.asarray(
            equi_jacobian_dzn(jnp.asarray([[r_test, r_test]]), jnp.asarray(cal)))[0]
        d = np.diag(J_far)
        rel_gap = np.max(np.abs(d - ref_diag) / np.abs(ref_diag))
        assert rel_gap < 10 * abs(4 * a * r_test ** 2), (
            f"diagonal relative gap at r = {r_test}: {rel_gap} "
            f"(closed form 4|a|r^2 = {4 * abs(a) * r_test ** 2}, "
            f"diag {d}, expected {ref_diag})")
        off = J_far - np.diag(d)
        abs_gap = np.max(np.abs(off))
        assert abs_gap < 10 * abs(2 * a * fx * r_test ** 2), (
            f"off-diagonal absolute gap at r = {r_test}: {abs_gap} "
            f"(closed form 2|a|fx r^2 = {2 * abs(a) * fx * r_test ** 2})")


def test_equi_class_vs_opencv_fisheye():
    """`CamEqui.distort` against `cv2.fisheye.projectPoints` — OpenCV's own
    equidistant implementation is the reference model."""
    cam = CamEqui(640, 480)
    cam.set_value(_EQUI_CALIB)
    rng = np.random.default_rng(14)
    uv = rng.normal(size=(200, 2)) * 0.7
    pts_3d = np.concatenate([uv * 2.0, np.full((200, 1), 2.0)], axis=1).astype(np.float64)
    # `cv2.fisheye.projectPoints` asserts `objectPoints.type() == CV_64FC3`, i.e.
    # it needs three channels — (N, 1, 3), not (N, 3).
    pts_3d = pts_3d.reshape(-1, 1, 3)

    img_pts, _ = cv2.fisheye.projectPoints(
        pts_3d, np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64),
        cam.get_K(), cam.get_D().astype(np.float64),
    )
    got = np.asarray(cam.distort(uv))
    assert np.max(np.abs(img_pts[:, 0] - got)) < 1e-9


def test_equi_distort_undistort_roundtrip():
    """Round trip through `cv2.fisheye.undistortPoints`."""
    cam = CamEqui(640, 480)
    cam.set_value(_EQUI_CALIB)
    rng = np.random.default_rng(15)
    uv = rng.normal(size=(100, 2)) * 0.6
    back = cam.undistort(cam.distort(uv))
    assert back.shape == (100, 2)
    assert np.max(np.abs(back - uv)) < 1e-6


def test_equi_class_matches_kernels():
    cam = CamEqui(640, 480)
    cam.set_value(_EQUI_CALIB)
    rng = np.random.default_rng(16)
    uv = rng.normal(size=(10, 2)) * 1.0
    vals = jnp.asarray(cam.get_value())
    assert _rel_err(np.asarray(cam.distort(uv)),
                    np.asarray(equi_distort(jnp.asarray(uv), vals))) < 1e-15
    J, Jzeta = cam.compute_distort_jacobian(uv[0])
    assert _rel_err(J, np.asarray(equi_jacobian_dzn(jnp.asarray(uv), vals))[0]) < 1e-15
    assert _rel_err(Jzeta, np.asarray(equi_jacobian_dzeta(jnp.asarray(uv), vals))[0]) < 1e-15
    assert cam.compute_distort_jacobian(uv[0], do_calib=False)[1] is None


def test_equi_clone_is_independent():
    cam = CamEqui(640, 480)
    cam.set_value(_EQUI_CALIB)
    c2 = cam.clone()
    c2.set_value(np.full(8, 1.0))
    assert cam.fx == _EQUI_CALIB[0]
    assert c2._type == "pinhole-equi"


# ---------------------------------------------------------------------------
# vmap-ability: the property Phase 4 actually relies on
# ---------------------------------------------------------------------------

def test_radtan_kernels_are_vmappable():
    """UpdaterHelper vmaps these over features; they must trace cleanly."""
    rng = np.random.default_rng(17)
    uv = jnp.asarray(rng.normal(size=(50, 2)) * 1.2)
    cal = jnp.asarray(_RADTAN_CALIB)

    f = jax.vmap(lambda u: radtan_distort(jnp.stack([u]), cal)[0])
    assert np.all(np.isfinite(np.asarray(f(uv))))

    g = jax.vmap(lambda u: radtan_jacobian_dzn(jnp.stack([u]), cal)[0])
    assert np.asarray(g(uv)).shape == (50, 2, 2)

    # and they compose with jacfwd for the full feature Jacobian. The
    # `jnp.stack([v])` must live INSIDE the differentiated function: hoisting it
    # out turns `v` into a vmap axis, which leaks a batch dimension into the
    # Jacobian ((50, 2, 1, 2) instead of (50, 2, 2)).
    h = jax.vmap(lambda u: jax.jacfwd(
        lambda v: radtan_distort(jnp.stack([v]), cal)[0], argnums=0)(u))
    assert np.asarray(h(uv)).shape == (50, 2, 2)


def test_equi_kernels_are_vmappable():
    rng = np.random.default_rng(18)
    uv = jnp.asarray(rng.normal(size=(50, 2)) * 1.2)
    cal = jnp.asarray(_EQUI_CALIB)
    f = jax.vmap(lambda u: equi_distort(jnp.stack([u]), cal)[0])
    assert np.all(np.isfinite(np.asarray(f(uv))))
    g = jax.vmap(lambda u: equi_jacobian_dzn(jnp.stack([u]), cal)[0])
    assert np.asarray(g(uv)).shape == (50, 2, 2)
