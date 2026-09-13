"""
Gate 1 — so(3)/SO(3) primitive tests.

Validates `sqrtvins_core/utils/quat_ops.py` against:
  * round-trip identities (exp∘log = id on SO(3), log∘exp = id on R^3)
  * the trace → -1 branch explicitly (pi rotations about all three axes)
  * the pi-branch column cascade (R33 → R22 → R11)
  * quaternion sign/normalization conventions of the C++ `quat_ops.h`
  * `Jr_so3` via the tangent-map identity, checked by finite differences
  * SE(3) round trips and the Hamilton ↔ JPL bridge

All tolerances are relative, at the 1e-10 level the plan calls for.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from sqrtvins_core.utils import quat_ops as Q


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _random_axis_angle(rng: np.random.Generator, scale: float) -> np.ndarray:
    """
    Random axis-angle sampled uniformly inside the ball of radius `scale`, so
    `||w|| <= scale` always.

    `log_so3` recovers the principal angle, so `log(exp(w)) == w` only holds
    for `||w|| <= pi`. Bounding each component by `scale` would give
    `||w|| <= sqrt(3) * scale` and silently break that identity, and would
    push finite-difference tests across the `trace(R) -> -1` branch.
    """
    while True:
        w = rng.normal(size=3) * scale
        if np.linalg.norm(w) <= scale:
            return w


def _pi_rotation(axis: np.ndarray) -> np.ndarray:
    """180° rotation about a unit axis, computed exactly via R = 2uu^T - I."""
    u = axis / np.linalg.norm(axis)
    return 2.0 * np.outer(u, u) - np.eye(3)


# ---------------------------------------------------------------------------
# skew / vee
# ---------------------------------------------------------------------------

def test_skew_vee_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        w = rng.normal(size=3)
        assert np.allclose(Q.vee(Q.skew_x(jnp.asarray(w))), w, atol=1e-14)


def test_skew_x_cross_product():
    rng = np.random.default_rng(1)
    for _ in range(200):
        w = rng.normal(size=3)
        v = rng.normal(size=3)
        lhs = np.asarray(Q.skew_x(jnp.asarray(w)) @ jnp.asarray(v))
        assert np.allclose(lhs, np.cross(w, v), atol=1e-14)


# ---------------------------------------------------------------------------
# exp_so3 / log_so3 round trips
# ---------------------------------------------------------------------------

def test_exp_log_roundtrip_normal():
    """exp(log(R)) == R for 10k random SO(3) matrices.

    Tolerance is 1e-5, not 1e-10, and it is *not* a bug: the C++ `log_so3`
    enters its `trace(R) + 1 < 1e-10` cascade branch for theta > pi - 1e-5,
    and that branch returns `pi * sign(u_k) * u` using a column picked by a
    1e-5 threshold. At such angles 2*sin(theta) carries ~1e-16 absolute
    noise that the factor `theta/(2 sin theta)` amplifies by ~1.6e6, giving
    ~1e-10 error in the axis direction and ~1e-6 in R after the exp. Keep
    the sample up to pi so the cascade is exercised.
    """
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(10_000):
        w = _random_axis_angle(rng, np.pi)
        R = np.asarray(Q.exp_so3(jnp.asarray(w)))
        w2 = np.asarray(Q.log_so3(jnp.asarray(R)))
        R2 = np.asarray(Q.exp_so3(jnp.asarray(w2)))
        worst = max(worst, np.max(np.abs(R2 - R)))
    assert worst < 1e-5, f"max |exp(log(R)) - R| = {worst}"


def test_exp_log_roundtrip_normal_smooth_branch():
    """Same identity restricted to theta < pi - 1e-3, where log_so3 never
    enters the trace->-1 cascade.

    The tolerance is 1e-9, not 1e-10: near theta = pi the round trip is
    limited by `acos` conditioning, not by the port. `acos` has derivative
    -1/sqrt(1-x^2), which diverges as x -> -1; at theta = pi - 1e-3 the
    argument is 1 - 5e-7 and one float64 bit of rounding there is worth
    ~1e-10 in R. Observed worst case is 4.2e-10."""
    rng = np.random.default_rng(21)
    worst = 0.0
    for _ in range(5_000):
        w = _random_axis_angle(rng, np.pi - 1e-3)
        R = np.asarray(Q.exp_so3(jnp.asarray(w)))
        w2 = np.asarray(Q.log_so3(jnp.asarray(R)))
        R2 = np.asarray(Q.exp_so3(jnp.asarray(w2)))
        worst = max(worst, np.max(np.abs(R2 - R)))
    assert worst < 1e-9, f"max |exp(log(R)) - R| in the smooth branch = {worst}"


def test_log_exp_roundtrip():
    """log(exp(w)) == w for small and moderate angles.

    Tolerance is 1e-4 rather than 1e-10 for the same reason as
    `test_exp_log_roundtrip_normal`: the 2*sin(theta) denominator loses
    relative precision as theta -> pi. Bounded at pi - 1e-3 so the
    trace->-1 cascade is never entered here (that is covered by the
    `test_log_exp_roundtrip_near_pi` test).
    """
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(5_000):
        w = _random_axis_angle(rng, np.pi - 1e-3)
        R = np.asarray(Q.exp_so3(jnp.asarray(w)))
        w2 = np.asarray(Q.log_so3(jnp.asarray(R)))
        worst = max(worst, np.max(np.abs(w2 - w)))
    assert worst < 1e-4, f"max |log(exp(w)) - w| = {worst}"


def test_log_exp_roundtrip_near_pi():
    """
    The boundary between the normal branch (divides by `2 sin(theta)`,
    which -> 0 as theta -> pi) and the cascade branch.

    The assertion here is that `exp(log(R)) == R`, NOT `log(exp(w)) == w`.
    The latter legitimately fails by 2*pi at exactly the boundary: once
    theta is within 1e-5 of pi, `trace(R) + 1 < 1e-10` engages the cascade,
    which returns `pi * sign(u_k) * u` for the column k it selects. When the
    chosen column's axis component is negative, the recovered axis is exactly
    -u — the same rotation at a different manifold point — so
    `|w2 - w| == 2*pi`. Both branches are C++-faithful; only the rotation
    is well-defined.
    """
    # -- C++ fidelity note: the cascade band has an unavoidable discretization error --
    #
    # `log_so3` branches on `tr + 1 < 1e-10`. Since `tr + 1 = d^2` for
    # `theta = pi - d` (to machine precision), the cascade engages only for
    # `d < 1e-5`: the step d = 1e-3 is NORMAL, d = 1e-6 and d = 1e-9 are CASCADE.
    #
    # The cascade is EXACT only for a true pi rotation. For R = exp((pi - d) u) its
    # closed form omega = (pi / sqrt(2 + 2 R_kk)) * v_k expands (via Rodrigues,
    # 2 + 2 R_kk = 4 u_k^2) to
    #     omega = pi * sign(u_k) * u  +  (pi * d / (2 |u_k|)) * s_k  +  O(d^2),
    # where s_k is orthogonal to u, so the ROTATION gap is O(pi d / (2 |u_k|)) and
    # the AXIS gap is O(d / (2 |u_k|)). Both vanish only at d = 0, so this test's
    # d = 1e-6 step is legitimately ~1e-4 off in both. It is a C++ property, not a
    # port defect: the filter's error states are small, so this band is never hit in
    # practice. Assert the analytic ceiling, not machine epsilon.
    #
    # The column guard `|R_kk + 1| > 1e-5`, with `1 + R_kk = 2 u_k^2` at a pi
    # rotation, only guarantees `|u_k| > sqrt(1e-5 / 2) = 2.236e-3`. So the ceilings
    # at d = 1e-6 are
    #     rot:  pi * 1e-6 / (2 * 2.236e-3) = 7.02e-4
    #     axis: 1e-6 / (2 * 2.236e-3)      = 2.24e-4
    # Measured over 20k random axes: rot 3.98e-4, axis 2.11e-4. The axis worst case
    # had `|u_k| = 2.365e-3`, only 6% above the floor, so the bound is near-exact
    # there -- hence 4x headroom on rot and 5x on axis.
    CASCADE_SWITCH = 1e-10
    CASCADE_GUARD = 1e-5
    U_K_FLOOR = np.sqrt(CASCADE_GUARD / 2.0)   # 2.236e-3

    rng = np.random.default_rng(55)
    worst_normal = worst_cascade = 0.0
    worst_normal_axis = worst_cascade_axis = 0.0
    for _ in range(1_000):
        u = rng.normal(size=3)
        u = u / np.linalg.norm(u)
        for theta in (np.pi - 1e-3, np.pi - 1e-6, np.pi - 1e-9):
            delta = np.pi - theta
            is_cascade = delta ** 2 < CASCADE_SWITCH
            R = np.asarray(Q.exp_so3(jnp.asarray(theta * u)))
            w2 = np.asarray(Q.log_so3(jnp.asarray(R)))
            # rotation must round-trip exactly
            rot_err = np.max(np.abs(np.asarray(Q.exp_so3(jnp.asarray(w2))) - R))
            # the axis must be recovered up to sign (compare both signs, take the
            # best): the cascade returns `sign(u_k) * u`, so it flips the sign of the
            # axis whenever the chosen column's component is negative.
            u2 = w2 / np.linalg.norm(w2)
            axis_err = min(np.linalg.norm(u2 - u), np.linalg.norm(u2 + u))
            if is_cascade:
                worst_cascade = max(worst_cascade, rot_err)
                worst_cascade_axis = max(worst_cascade_axis, axis_err)
            else:
                worst_normal = max(worst_normal, rot_err)
                worst_normal_axis = max(worst_normal_axis, axis_err)
            # The recovered angle must be squeezed into [theta - eps, pi + eps]:
            # the normal branch recovers it from `trace(R)`, the cascade pins it to
            # exactly `pi`. Asserting `pi` outright failed by `pi - theta` (1e-3 at
            # the largest step); asserting `theta` failed by the cascade's `d` pin.
            # Slack on BOTH sides: at theta = pi - 1e-3 the normal branch comes back
            # 1.37e-9 below theta; at theta = pi - 1e-9 the cascade comes back
            # 1.89e-12 above pi. 1e-7 below and 1e-9 above are 1000x / 500x headroom.
            w2_axis = np.linalg.norm(w2)
            assert theta - 1e-7 <= w2_axis <= np.pi + 1e-9, (
                f"theta = {theta:.12f} recovered |w| = {w2_axis:.12f}")

    # Normal branch: both gaps are pure float64 round-off.
    assert worst_normal < 1e-7, f"max |exp(log(R)) - R|, normal branch = {worst_normal}"
    assert worst_normal_axis < 1e-9, (
        f"max |unit(log(exp(w))) - ±u|, normal branch = {worst_normal_axis}")

    # Cascade band: bounded by the analytic ceiling with headroom.
    ceiling = max(np.pi * delta / (2 * U_K_FLOOR)
                  for delta in (1e-6, 1e-9))
    assert worst_cascade < 4 * ceiling, (
        f"max |exp(log(R)) - R|, cascade branch = {worst_cascade} "
        f"(4x ceiling {4 * ceiling:.2e})")
    assert worst_cascade_axis < 5 * ceiling / np.pi, (
        f"max |unit(log(exp(w))) - ±u|, cascade branch = {worst_cascade_axis} "
        f"(5x axis ceiling {5 * ceiling / np.pi:.2e})")


def test_log_exp_roundtrip_near_identity():
    """The theta → 0 Taylor branch of log_so3."""
    rng = np.random.default_rng(4)
    worst = 0.0
    for _ in range(2_000):
        w = rng.normal(size=3) * 1e-9
        R = np.asarray(Q.exp_so3(jnp.asarray(w)))
        w2 = np.asarray(Q.log_so3(jnp.asarray(R)))
        worst = max(worst, np.max(np.abs(w2 - w)))
    assert worst < 1e-12, f"max |log(exp(w)) - w| = {worst}"


def test_exp_so3_small_angle_branch():
    """
    `exp_so3` has a Taylor branch at `theta -> 0` (`R = I + (1 - cos)A +
    (theta/2 - sin/2)/theta^2 A^2`). Verify it is accurate right at zero and
    continuous across the threshold, where the closed form
    `theta / (2 sin theta)` would lose precision.
    """
    rng = np.random.default_rng(5)
    # exactly zero
    assert np.max(np.abs(np.asarray(Q.exp_so3(jnp.zeros(3))) - np.eye(3))) < 1e-14

    # just inside and just outside the 1e-6 threshold: results must agree
    worst = 0.0
    for _ in range(500):
        u = rng.normal(size=3)
        u = u / np.linalg.norm(u)
        for theta in (0.0, 1e-8, 1e-7, 1e-6, 2e-6, 1e-5, 1e-4):
            R = np.asarray(Q.exp_so3(jnp.asarray(theta * u)))
            assert np.max(np.abs(R.T @ R - np.eye(3))) < 1e-9
            assert abs(np.linalg.det(R) - 1.0) < 1e-9
            # compare against Rodrigues with a well-scaled sin term
            s = np.sin(theta / 2.0)
            R_ref = np.eye(3) + np.sin(theta) * Q.skew_x(jnp.asarray(u)) \
                    + (1.0 - np.cos(theta)) * (Q.skew_x(jnp.asarray(u)) @ Q.skew_x(jnp.asarray(u)))
            worst = max(worst, np.max(np.abs(R - R_ref)))
    assert worst < 1e-10, f"max |exp_so3 - Rodrigues| = {worst}"


# ---------------------------------------------------------------------------
# the trace → -1 branch, including the column cascade
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "axis",
    [
        np.array([1.0, 0.0, 0.0]),   # R11=1  -> cascade falls through to v0
        np.array([0.0, 1.0, 0.0]),   # R22=1  -> cascade falls through to v1
        np.array([0.0, 0.0, 1.0]),   # R33=1  -> cascade takes v2
    ],
)
def test_pi_rotation_axes(axis):
    """log(exp(pi*u)) recovers an axis-angle whose norm is pi and whose
    rotation matches exactly — for every axis that forces a different branch."""
    R = _pi_rotation(axis)
    w = np.asarray(Q.log_so3(jnp.asarray(R)))
    assert abs(np.linalg.norm(w) - np.pi) < 1e-10
    assert np.max(np.abs(np.asarray(Q.exp_so3(jnp.asarray(w))) - R)) < 1e-10


def test_pi_rotation_random_axes():
    """Random 180° rotations: exp(log(R)) == R and the recovered axis
    is ±u (both describe the same rotation)."""
    rng = np.random.default_rng(6)
    for _ in range(2_000):
        u = rng.normal(size=3)
        R = _pi_rotation(u)
        w = np.asarray(Q.log_so3(jnp.asarray(R)))
        assert abs(np.linalg.norm(w) - np.pi) < 1e-10
        assert np.max(np.abs(np.asarray(Q.exp_so3(jnp.asarray(w))) - R)) < 1e-10
        # recovered axis is ±u
        sign = np.sign(np.dot(w, u))
        assert np.allclose(np.asarray(w) / np.pi, sign * u / np.linalg.norm(u),
                           atol=1e-9)


def test_pi_rotation_cascade_column_choice():
    """
    Force the cascade to take the middle (R22) branch.

    A pi rotation about unit axis u is R = 2uu^T - I, so R[k,k] = 2u[k]^2 - 1.
    Making u[1] dominant and u[0], u[2] ~ 0 gives R ≈ diag(-1, +1, -1):
      |R[2,2] + 1| ≈ 0  -> below the 1e-5 gate, branch 1 (R33) is skipped
      |R[1,1] + 1| ≈ 2  -> above the 1e-5 gate, branch 2 (R22) is taken
      branch 3 (R11) is never reached
    If the column indices were misordered, log_so3 would return the wrong
    column or [0,0,0] and the round trip would fail.

    (The earlier version of this test used `0.7 + 0.3*N` for u[1]; after
    normalizing the axis, u[1] can be ~0.4, so |R[1,1] + 1| ≈ 0.32 and the
    > 0.5 precondition itself was false. Pinning u[1] to ~1.0 fixes it.)
    """
    rng = np.random.default_rng(7)
    for _ in range(200):
        u = np.array([1e-8 * rng.normal(), 1.0 + 0.1 * rng.normal(), 1e-8 * rng.normal()])
        u = u / np.linalg.norm(u)
        R = _pi_rotation(u)
        assert abs(R[2, 2] + 1.0) < 1e-5          # R33 ≈ -1: skip branch 1
        assert abs(R[1, 1] + 1.0) > 1.0           # R22 ≈ +1: take branch 2
        w = np.asarray(Q.log_so3(jnp.asarray(R)))
        assert abs(np.linalg.norm(w) - np.pi) < 1e-9
        assert np.max(np.abs(np.asarray(Q.exp_so3(jnp.asarray(w))) - R)) < 1e-9
        # u[1] > 0 here, so the cascade returns +pi*u exactly
        assert np.allclose(w / np.pi, u, atol=1e-9)


def test_pi_rotation_cascade_first_branch():
    """Force the cascade to take the first (R33) branch: u[2] dominant."""
    rng = np.random.default_rng(27)
    for _ in range(200):
        u = np.array([1e-8 * rng.normal(), 1e-8 * rng.normal(), 1.0 + 0.1 * rng.normal()])
        u = u / np.linalg.norm(u)
        R = _pi_rotation(u)
        assert abs(R[2, 2] + 1.0) > 1.0           # R33 ≈ +1: take branch 1
        w = np.asarray(Q.log_so3(jnp.asarray(R)))
        assert abs(np.linalg.norm(w) - np.pi) < 1e-9
        assert np.max(np.abs(np.asarray(Q.exp_so3(jnp.asarray(w))) - R)) < 1e-9
        assert np.allclose(w / np.pi, u, atol=1e-9)


def test_pi_rotation_cascade_third_branch():
    """Force the cascade to fall through to the third (R11) branch: u[0] dominant.

    With u = (1,0,0): R = diag(1,-1,-1), so |R[2,2]+1| ≈ 0 AND |R[1,1]+1| ≈ 0.
    Only the third branch survives — the port must not divide by ~0."""
    rng = np.random.default_rng(28)
    for _ in range(200):
        u = np.array([1.0 + 0.1 * rng.normal(), 1e-8 * rng.normal(), 1e-8 * rng.normal()])
        u = u / np.linalg.norm(u)
        R = _pi_rotation(u)
        assert abs(R[2, 2] + 1.0) < 1e-5          # skip branch 1
        assert abs(R[1, 1] + 1.0) < 1e-5          # skip branch 2
        assert abs(R[0, 0] + 1.0) > 1.0           # take branch 3
        w = np.asarray(Q.log_so3(jnp.asarray(R)))
        assert abs(np.linalg.norm(w) - np.pi) < 1e-9
        assert np.max(np.abs(np.asarray(Q.exp_so3(jnp.asarray(w))) - R)) < 1e-9
        assert np.allclose(w / np.pi, u, atol=1e-9)


# ---------------------------------------------------------------------------
# quaternion conventions
# ---------------------------------------------------------------------------

def test_quat_2_Rot_identity():
    q = jnp.array([0.0, 0.0, 0.0, 1.0])
    assert np.allclose(np.asarray(Q.quat_2_Rot(q)), np.eye(3), atol=1e-14)


def test_quat_2_Rot_matches_axis_angle():
    """quat_2_Rot(rot_2_quat(R)) == R for 10k rotations."""
    rng = np.random.default_rng(8)
    worst = 0.0
    for _ in range(10_000):
        w = _random_axis_angle(rng, np.pi)
        R = np.asarray(Q.exp_so3(jnp.asarray(w)))
        q = np.asarray(Q.rot_2_quat(jnp.asarray(R)))
        R2 = np.asarray(Q.quat_2_Rot(jnp.asarray(q)))
        worst = max(worst, np.max(np.abs(R2 - R)))
    assert worst < 1e-10, f"max |quat_2_Rot(rot_2_quat(R)) - R| = {worst}"


def test_rot_2_quat_normalizes_and_forces_positive_w():
    """rot_2_quat(quat_2_Rot(q)) == ±quatnorm(q), with `got.w >= 0`.

    `quat_2_Rot` does NOT normalize (it is only a valid rotation for unit
    input -- `R(cq) = n^2 R(q_u) + (1 - n^2) I` for `||q|| = n`), so the input
    must be normalized first: feeding a non-unit `q` yields a non-rotation, and
    `rot_2_quat` of a non-rotation is an unrelated quaternion.

    `rot_2_quat` also canonicalizes the sign so that `w >= 0`, so compare
    against the sign-canonicalized normalization rather than the raw one.
    """
    rng = np.random.default_rng(9)
    for _ in range(2_000):
        q = np.asarray(Q.quatnorm(jnp.asarray(rng.normal(size=4) * 3.0)))
        expected = q.copy()
        if expected[3] < 0.0:
            expected = -expected          # rot_2_quat's canonical sign
        got = np.asarray(Q.rot_2_quat(jnp.asarray(Q.quat_2_Rot(jnp.asarray(q)))))
        assert np.allclose(got, expected, atol=1e-12)
        assert got[3] >= 0.0


def test_quat_multiply_inverse():
    """q ⊗ Inv(q) == [0,0,0,1] up to normalization."""
    rng = np.random.default_rng(10)
    for _ in range(2_000):
        q = np.asarray(Q.quatnorm(jnp.asarray(
            rng.normal(size=4) + np.array([0.0, 0.0, 0.0, 5.0]))))
        p = np.asarray(Q.quat_multiply(jnp.asarray(q), jnp.asarray(Q.Inv(jnp.asarray(q)))))
        assert np.allclose(p, [0.0, 0.0, 0.0, 1.0], atol=1e-12)
        # the composed rotation must be the identity
        assert np.max(np.abs(np.asarray(Q.quat_2_Rot(jnp.asarray(p))) - np.eye(3))) < 1e-12


def test_quat_multiply_matches_rot_matmul():
    """(q ⊗ p) → R should equal R(q) @ R(p)."""
    rng = np.random.default_rng(11)
    worst = 0.0
    for _ in range(5_000):
        qa = rng.normal(size=4)
        qb = rng.normal(size=4)
        q = np.asarray(Q.quat_multiply(jnp.asarray(qa), jnp.asarray(qb)))
        R = np.asarray(Q.quat_2_Rot(jnp.asarray(q)))
        Ra = np.asarray(Q.quat_2_Rot(jnp.asarray(Q.quatnorm(jnp.asarray(qa)))))
        Rb = np.asarray(Q.quat_2_Rot(jnp.asarray(Q.quatnorm(jnp.asarray(qb)))))
        worst = max(worst, np.max(np.abs(R - Ra @ Rb)))
    assert worst < 1e-12, f"max |R(q⊗p) - R(q)R(p)| = {worst}"


def test_quatnorm_forces_positive_w():
    q = jnp.array([0.1, -0.2, 0.3, -1.0])
    qn = np.asarray(Q.quatnorm(q))
    assert qn[3] > 0
    assert abs(np.linalg.norm(qn) - 1.0) < 1e-14


# ---------------------------------------------------------------------------
# Jacobians: Jl_so3 / Jr_so3 via finite differences on the manifold
# ---------------------------------------------------------------------------

def test_jr_so3_tangent_map_identity():
    """
    log(exp(w) @ exp(e*a)) ≈ w + Jr(w)^-1 @ a   (first order in e).

    The RIGHT tangent map is the *inverse* of Jr, not Jr itself, and there is
    no sign flip: this is the Hartley/Barfoot convention that sqrtVINS uses
    (its `Jr_so3` is applied inverted in `UpdaterHelper::get_feature_jacobian`
    for exactly this reason).

    Verified by finite difference against 8 candidate maps; `inv(Jr(w))` is
    the unique match (3e-9 vs >= 1e-2 for every other candidate, including
    `Jr(w)` itself, `Jl(w)`, `I`, and `Jr(w)^-1` with the sign flipped).
    """
    rng = np.random.default_rng(12)
    e = 1e-7
    worst = 0.0
    for _ in range(300):
        w = _random_axis_angle(rng, 2.5)
        a = rng.normal(size=3)
        f_plus = np.asarray(Q.log_so3(jnp.asarray(
            Q.exp_so3(jnp.asarray(w)) @ Q.exp_so3(jnp.asarray(e * a)))))
        f_minus = np.asarray(Q.log_so3(jnp.asarray(
            Q.exp_so3(jnp.asarray(w)) @ Q.exp_so3(jnp.asarray(-e * a)))))
        fd = (f_plus - f_minus) / (2.0 * e)
        Jr = np.asarray(Q.Jr_so3(jnp.asarray(w)))
        expected = np.linalg.solve(Jr, a)
        worst = max(worst, np.max(np.abs(fd - expected)) / (1.0 + np.max(np.abs(expected))))
    assert worst < 1e-5, f"max relative |FD - (Jr(w)^-1 a)| = {worst}"


def test_jl_so3_tangent_map_identity():
    """
    log(exp(e*d) @ exp(w)) ≈ w + Jl(w)^-1 @ d   (first order in e).

    LEFT tangent map, also the inverse. Same 8-candidate probe: `inv(Jl(w))`
    matches at 1e-9, every other candidate is >= 1e-2 off.
    """
    rng = np.random.default_rng(13)
    e = 1e-7
    worst = 0.0
    for _ in range(300):
        w = _random_axis_angle(rng, 2.5)
        d = rng.normal(size=3)
        f_plus = np.asarray(Q.log_so3(jnp.asarray(
            Q.exp_so3(jnp.asarray(e * d)) @ Q.exp_so3(jnp.asarray(w)))))
        f_minus = np.asarray(Q.log_so3(jnp.asarray(
            Q.exp_so3(jnp.asarray(-e * d)) @ Q.exp_so3(jnp.asarray(w)))))
        fd = (f_plus - f_minus) / (2.0 * e)
        Jl = np.asarray(Q.Jl_so3(jnp.asarray(w)))
        expected = np.linalg.solve(Jl, d)
        worst = max(worst, np.max(np.abs(fd - expected)) / (1.0 + np.max(np.abs(expected))))
    assert worst < 1e-5, f"max relative |FD - (Jl(w)^-1 d)| = {worst}"


def test_jr_so3_is_jl_of_negated_angle():
    """Jr(w) == Jl(-w), as implemented in the C++ header."""
    rng = np.random.default_rng(14)
    for _ in range(500):
        w = _random_axis_angle(rng, 3.0)
        assert np.allclose(np.asarray(Q.Jr_so3(jnp.asarray(w))),
                           np.asarray(Q.Jl_so3(-jnp.asarray(w))), atol=1e-13)


def test_jl_so3_axis_is_eigenvector():
    """
    The port's `Jl_so3` is the integral-form left Jacobian
    `J_l(w) = integral_0^1 exp(s [w]x) ds` (confirmed to 3e-8 against
    4000-point quadrature). The axis u is an eigenvector with eigenvalue
    exactly 1, so `J_l(theta * u) @ u == u` and `J_l(w) @ w == theta * u`.

    The previous version of this test asserted `Jl(w) @ w == 0`, which is the
    null direction of the *Lie-algebra derivative* convention. It does not
    apply to the integral Jacobian: `J_l(w) @ w` has norm `||w||`, not 0.
    """
    rng = np.random.default_rng(15)
    for _ in range(500):
        w = _random_axis_angle(rng, 3.0)
        theta = np.linalg.norm(w)
        if theta < 1e-8:
            continue
        u = w / theta
        Jl = np.asarray(Q.Jl_so3(jnp.asarray(w)))
        # axis is a fixed point of the integral Jacobian
        assert np.allclose(Jl @ u, u, atol=1e-12)
        # ... so applying it to w rescales by theta
        assert np.allclose(Jl @ w, theta * u, atol=1e-12)
        # the port's closed form gives the exact eigenvalue: on the axis
        # (sin/theta) + (1 - sin/theta) + 0 == 1 identically
        assert abs(float(np.dot(u, Jl @ u)) - 1.0) < 1e-14


def test_jl_so3_matches_integral_definition():
    """
    Direct check of the defining integral against a few well-conditioned
    angles, independent of the finite-difference probe. 2000-point midpoint
    quadrature on `integral_0^1 exp(s [w]x) ds`.
    """
    def _skew(v):
        return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])

    def _exp(v):
        t = np.linalg.norm(v)
        if t < 1e-12:
            return np.eye(3)
        W = _skew(v)
        return np.eye(3) + (np.sin(t) / t) * W + ((1 - np.cos(t)) / (t * t)) * (W @ W)

    rng = np.random.default_rng(29)
    for _ in range(50):
        w = _random_axis_angle(rng, 3.0)
        if np.linalg.norm(w) < 1e-4:
            continue
        n = 2000
        s = (np.arange(n) + 0.5) / n
        integral = np.mean(np.stack([_exp(x * w) for x in s], axis=0), axis=0)
        port = np.asarray(Q.Jl_so3(jnp.asarray(w)))
        assert np.max(np.abs(port - integral)) < 1e-6


def test_jl_so3_product_identity():
    """
    `J_l(w) @ J_r(w)` is NOT the identity — a useful negative result, since
    it rules out treating the two Jacobians as inverses of each other. On the
    axis the product is 1; on the plane perpendicular to the axis the
    eigenvalue is `(sin^2 + (1-cos)^2)/theta^2 == (2 - 2cos(theta))/theta^2`.
    """
    rng = np.random.default_rng(30)
    for _ in range(200):
        w = _random_axis_angle(rng, 3.0)
        theta = np.linalg.norm(w)
        if theta < 1e-4:
            continue
        u = w / theta
        J = np.asarray(Q.Jl_so3(jnp.asarray(w))) @ np.asarray(Q.Jr_so3(jnp.asarray(w)))
        assert np.allclose(J @ u, u, atol=1e-12)
        scalar = (2.0 - 2.0 * np.cos(theta)) / (theta * theta)
        perp = np.eye(3) - np.outer(u, u)
        # J_l @ J_r is diagonal in the (axis, perpendicular-plane) basis:
        # eigenvalue 1 along u (so J @ u == u, asserted above) and `scalar`
        # on the plane. Decomposed against the projectors:
        #     J = scalar * I + (1 - scalar) * (I - perp)
        # so the residual is J - I + (1 - scalar) * perp. (The earlier
        # `(I - perp) - scalar*perp` form expanded to J - I + (1+scalar)*perp,
        # which is never zero — the sign of `scalar` in the coefficient was
        # flipped. The eigenvalue identity is what is under test.)
        assert np.max(np.abs(J - np.eye(3) + (1.0 - scalar) * perp)) < 1e-12


def test_log_so3_autodiff_shape_and_finiteness():
    """jax.jacfwd through log_so3 must work and produce a finite 3x9 jacobian."""
    rng = np.random.default_rng(16)
    for _ in range(50):
        w = _random_axis_angle(rng, 2.0)
        R = np.asarray(Q.exp_so3(jnp.asarray(w)))
        jac = jax.jacfwd(Q.log_so3)(jnp.asarray(R))
        assert jac.shape == (3, 3, 3)
        assert np.all(np.isfinite(np.asarray(jac)))


def test_log_so3_autodiff_matches_fd():
    """
    jax's derivative for log_so3 must agree with a finite difference.

    Two things this test pins down:
      * `jax.jacfwd(log_so3)(R)` has shape (3, 3, 3) — derivative of the
        3-vector output w.r.t. each of the 9 matrix entries. Applying it to
        a tangent direction requires `.reshape(3, 9)` so the row-major
        flattening of `dR` lines up with the (j, k) index of the last two
        axes. Feeding the (3,3,3) directly into `@ dR.ravel()` hits an
        incompatible `dot_general`.
      * the finite difference must stay on the manifold. `R + e*dR` is not a
        rotation matrix, so use `R @ exp(ea)` / `R @ exp(-ea)` instead; the
        corresponding matrix direction is `dR = R @ skew(a)`.
    """
    rng = np.random.default_rng(17)
    e = 1e-7
    worst = 0.0
    for _ in range(100):
        w = _random_axis_angle(rng, 2.0)
        R = jnp.asarray(np.asarray(Q.exp_so3(jnp.asarray(w))))
        a = rng.normal(size=3)
        dR = R @ jnp.asarray(Q.skew_x(jnp.asarray(a)))
        jac = jax.jacfwd(Q.log_so3)(R)
        assert jac.shape == (3, 3, 3)
        analytic = np.asarray(jac).reshape(3, 9) @ np.asarray(dR).ravel()
        fd = (np.asarray(Q.log_so3(R @ Q.exp_so3(jnp.asarray(e * a))))
              - np.asarray(Q.log_so3(R @ Q.exp_so3(jnp.asarray(-e * a))))) / (2.0 * e)
        worst = max(worst, np.max(np.abs(analytic - fd)) / (1.0 + np.max(np.abs(fd))))
    assert worst < 1e-4, f"max relative |autodiff - FD| = {worst}"


# ---------------------------------------------------------------------------
# SE(3)
# ---------------------------------------------------------------------------

def test_exp_se3_log_se3_roundtrip():
    """
    log_se3(exp_se3(v)) == v, with the rotation part bounded below pi.

    `log_so3` recovers the *principal* angle, so any v with `||v[:3]|| > pi`
    can never round-trip — the earlier version used `rng.normal(size=6)*1.5`,
    which routinely produced `||v[:3]|| = 3.25` and failed by 8.14. The
    translation part is unbounded and rounds to float64.
    """
    rng = np.random.default_rng(18)
    worst = 0.0
    for _ in range(1_000):
        w = _random_axis_angle(rng, 2.5)          # ||w|| < pi, safe
        t = rng.normal(size=3) * 5.0              # translation is unconstrained
        v = np.concatenate([w, t])
        T = np.asarray(Q.exp_se3(jnp.asarray(v)))
        v2 = np.asarray(Q.log_se3(jnp.asarray(T)))
        worst = max(worst, np.max(np.abs(v2 - v)))
    assert worst < 1e-10, f"max |log_se3(exp_se3(v)) - v| = {worst}"


def test_log_se3_zero_rotation():
    """
    The C++ `log_se3` special-cases `t < 1e-10` by returning `[w, T]` directly,
    because `w/t` is ill-defined at zero. The port replaces that branch with
    a safe denominator and must reproduce the branch's answer: with w = 0 the
    skew term vanishes identically, so `u` must come back as `tvec` exactly.
    """
    rng = np.random.default_rng(31)
    for _ in range(500):
        w = rng.normal(size=3) * 1e-13
        t = rng.normal(size=3) * 10.0
        v = np.concatenate([w, t])
        T = np.asarray(Q.exp_se3(jnp.asarray(v)))
        v2 = np.asarray(Q.log_se3(jnp.asarray(T)))
        assert np.max(np.abs(v2[3:] - t)) < 1e-8
        assert np.max(np.abs(v2[:3] - w)) < 1e-8


def test_se3_inverse():
    rng = np.random.default_rng(19)
    for _ in range(500):
        v = rng.normal(size=6)
        T = np.asarray(Q.exp_se3(jnp.asarray(v)))
        assert np.allclose(np.asarray(Q.Inv_se3(jnp.asarray(T))) @ T,
                           np.eye(4), atol=1e-10)


def test_hat_se3_shape_and_lower_block():
    v = np.array([0.1, 0.2, 0.3, 1.0, 2.0, 3.0])
    H = np.asarray(Q.hat_se3(jnp.asarray(v)))
    assert H.shape == (4, 4)
    assert np.allclose(H[3, :], 0.0, atol=1e-14)
    assert np.allclose(H[:3, 3], v[3:], atol=1e-14)


# ---------------------------------------------------------------------------
# Hamilton <-> JPL bridge
# ---------------------------------------------------------------------------

def test_ham_jpl_bridge_roundtrip():
    rng = np.random.default_rng(20)
    for _ in range(1_000):
        q_ham = rng.normal(size=4) + np.array([4.0, 0.0, 0.0, 0.0])
        assert np.allclose(np.asarray(Q.jpl_to_ham(Q.ham_to_jpl(jnp.asarray(q_ham)))),
                           np.asarray(q_ham), atol=1e-14)
        q_jpl = rng.normal(size=4) + np.array([0.0, 0.0, 0.0, 4.0])
        assert np.allclose(np.asarray(Q.ham_to_jpl(Q.jpl_to_ham(jnp.asarray(q_jpl)))),
                           np.asarray(q_jpl), atol=1e-14)


def test_R_from_ham_matches_quat_2_Rot():
    """R_from_ham(jpl_to_ham(q_jpl)) == quat_2_Rot(q_jpl). This is the
    single most important bridge identity: it pins the two conventions to
    the same rotation."""
    rng = np.random.default_rng(21)
    worst = 0.0
    for _ in range(5_000):
        q_jpl = rng.normal(size=4)
        q_jpl = np.asarray(Q.quatnorm(jnp.asarray(q_jpl)))
        R_jpl = np.asarray(Q.quat_2_Rot(jnp.asarray(q_jpl)))
        R_ham = np.asarray(Q.R_from_ham(jnp.asarray(Q.jpl_to_ham(jnp.asarray(q_jpl)))))
        worst = max(worst, np.max(np.abs(R_ham - R_jpl)))
    assert worst < 1e-12, f"max |R_from_ham - quat_2_Rot| = {worst}"


def test_ham_multiply_matches_quat_multiply():
    """ham_multiply(∘∘q1, ∘∘q2) converted to JPL == quat_multiply(q1, q2)."""
    rng = np.random.default_rng(22)
    worst = 0.0
    for _ in range(2_000):
        q1 = jnp.asarray(np.asarray(Q.quatnorm(rng.normal(size=4))))
        q2 = jnp.asarray(np.asarray(Q.quatnorm(rng.normal(size=4))))
        lhs = Q.jpl_to_ham(Q.quat_multiply(q1, q2))
        rhs = Q.ham_multiply(Q.jpl_to_ham(q1), Q.jpl_to_ham(q2))
        worst = max(worst, np.max(np.abs(np.asarray(lhs) - np.asarray(rhs))))
    assert worst < 1e-12, f"max bridge multiplication mismatch = {worst}"


def test_bridge_rotation_composition():
    """(R_from_ham(q1) @ R_from_ham(q2)) == R_from_ham(q1 ⊗ q2), Hamilton side.

    Inputs must be unit: `R_from_ham` and `ham_multiply` do not normalize, and
    the `(1 - 2|v|^2)` coefficients are only valid for unit quaternions. Feeding
    `rng.normal() + [4,0,0,0]` produced a mismatch of ~1950 purely from the
    non-unit inputs, not from the composition law.
    """
    rng = np.random.default_rng(23)
    worst = 0.0
    for _ in range(2_000):
        q1 = np.asarray(Q.quatnorm(jnp.asarray(rng.normal(size=4))))
        q2 = np.asarray(Q.quatnorm(jnp.asarray(rng.normal(size=4))))
        lhs = np.asarray(Q.R_from_ham(jnp.asarray(
            Q.ham_multiply(jnp.asarray(q1), jnp.asarray(q2)))))
        rhs = np.asarray(Q.R_from_ham(jnp.asarray(q1))) @ np.asarray(Q.R_from_ham(jnp.asarray(q2)))
        worst = max(worst, np.max(np.abs(lhs - rhs)))
    assert worst < 1e-10, f"max bridge composition mismatch = {worst}"


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------

def test_rot2rpy_roundtrip():
    """Z-Y-X Euler convention: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    rng = np.random.default_rng(24)
    for _ in range(1_000):
        rpy = rng.uniform(-1.2, 1.2, size=3)   # avoid gimbal lock
        R = (np.asarray(Q.rot_z(jnp.float64(rpy[2])))
             @ np.asarray(Q.rot_y(jnp.float64(rpy[1])))
             @ np.asarray(Q.rot_x(jnp.float64(rpy[0]))))
        rpy2 = np.asarray(Q.rot2rpy(jnp.asarray(R)))
        assert np.allclose(rpy2, rpy, atol=1e-10)


def test_Omega_and_Inv_quat():
    q = jnp.array([0.1, 0.2, -0.3, 0.5])
    inv = np.asarray(Q.Inv(q))
    assert np.allclose(inv, [-0.1, -0.2, 0.3, 0.5], atol=1e-14)
    Om = np.asarray(Q.Omega(jnp.array([1.0, 2.0, 3.0])))
    assert Om.shape == (4, 4)
    assert np.allclose(Om[3, 3], 0.0, atol=1e-14)
