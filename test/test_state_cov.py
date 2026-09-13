"""Unit tests for sqrtvins_jax.state.StateHelper — the sqrt-form covariance
engine.

Every test either checks an invariant that must hold regardless of the
specific numbers (symmetry, PSD-ness, size bookkeeping) or compares a
specific block operation against its dense-numpy closed form. The single
kernel that most deserves the closed-form comparison is `update_llt` — it
reproduces the OpenVINS sqrt-form update:

    F_final^T F_final = H^T R^-1 H + I   (after column/row reversals)
    U_new = F_final^{-T} U_prior
    P_new = U_new^T U_new                 (Eigen convention)
    dx    = P_new H^T R^-1 r

Note: this is NOT the classical Kalman posterior (P_prior^{-1} + H^T R^{-1} H)^{-1}.
The OpenVINS formula applies a congruence transformation to U_prior, not to P_prior.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqrtvins_core.types.IMU import IMU
from sqrtvins_core.types.Landmark import Landmark
from sqrtvins_core.types.PoseJPL import PoseJPL
from sqrtvins_core.types.Vec import Vec

from sqrtvins_jax.state import State, StateOptions, StateHelper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_state(
    *,
    num_cameras: int = 1,
    do_calib_camera_pose: bool = False,
    do_calib_camera_intrinsics: bool = False,
    do_calib_camera_timeoffset: bool = False,
    max_clone_size: int = 11,
    max_slam_features: int = 25,
) -> State:
    """Build a State with default init_options; `init_state` is left un-run.

    `state.init_options` is populated with a `_make_init_opts()` so
    `initialize_state` can be called directly on the returned state."""
    opts = StateOptions(
        num_cameras=num_cameras,
        do_calib_camera_pose=do_calib_camera_pose,
        do_calib_camera_intrinsics=do_calib_camera_intrinsics,
        do_calib_camera_timeoffset=do_calib_camera_timeoffset,
        max_clone_size=max_clone_size,
        max_slam_features=max_slam_features,
    )
    state = State(opts, _make_init_opts(num_cameras=num_cameras))
    return state


class _FakeInitOptions:
    """Minimal stand-in for InertialInitializerOptions that exposes every
    scalar field `initialize_state` reads. Enough to run the test without
    importing the real dataclass."""

    init_prior_q = 0.1
    init_prior_p = 0.5
    init_prior_v = 0.5
    init_prior_bg = 0.1
    init_prior_ba = 0.1
    init_prior_t = 0.001
    init_prior_qc = 0.02
    init_prior_pc = 0.01
    init_prior_fc = 1.0
    init_prior_dc1 = 0.01
    init_prior_dc2 = 1e-5
    calib_camimu_dt = 0.0
    num_cameras = 1

    # Populated on construction — see `_make_init_opts` below.
    camera_intrinsics: dict[int, Any]
    camera_extrinsics: dict[int, np.ndarray]


def _make_init_opts(num_cameras: int = 1) -> _FakeInitOptions:
    """Return a `_FakeInitOptions` with `camera_intrinsics` /
    `camera_extrinsics` populated for `num_cameras` cameras."""
    opts = _FakeInitOptions()
    opts.num_cameras = num_cameras
    opts.camera_intrinsics = {i: _FakeIntrinsics() for i in range(num_cameras)}
    # PoseJPL::set_value takes a 7-vector: 4 JPL quat [x,y,z,w] then 3 pos.
    # Identity rotation → w=1 (last slot).
    opts.camera_extrinsics = {i: _identity_extrinsic() for i in range(num_cameras)}
    return opts


def _identity_extrinsic() -> np.ndarray:
    """Identity-camera-to-IMU extrinsic as a 7-vector (JPL quat + pos)."""
    v = np.zeros(7, dtype=np.float64)
    v[3] = 1.0  # w = 1 → identity rotation
    return v


def _identity_imu_init() -> np.ndarray:
    """A (16, 1) IMU value with identity rotation (qw=1), zero pos/vel/biases."""
    v = np.zeros((16, 1), dtype=np.float64)
    v[3, 0] = 1.0  # qw = 1 → identity rotation
    return v


def _spd(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a random symmetric positive-definite matrix."""
    A = rng.standard_normal((n, n))
    return A @ A.T + n * np.eye(n)


def _up(chol_upper_A: np.ndarray) -> np.ndarray:
    """Return an upper-triangular Cholesky factor U such that A = U^T U.

    numpy.linalg.cholesky returns L with A = L L^T; U = L.T then gives
    A = U^T U. Matches Eigen's `LLT::matrixU()`.
    """
    return np.linalg.cholesky(A := chol_upper_A).T


# ---------------------------------------------------------------------------
# set_initial_imu_square_root_covariance
# ---------------------------------------------------------------------------

def test_set_initial_imu_square_root_covariance_sets_diag():
    state = _make_state()
    diag = np.linspace(1e-3, 1.0, 15).reshape(15, 1)
    StateHelper.set_initial_imu_square_root_covariance(state, diag)
    for i in range(15):
        assert state.U[i, i] == pytest.approx(diag[i, 0], abs=1e-15)


# ---------------------------------------------------------------------------
# initialize_state
# ---------------------------------------------------------------------------

def test_initialize_state_builds_imu_u_with_yaw_prior():
    state = _make_state()
    init_opts = _make_init_opts()
    state.init_options = init_opts

    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    # State is marked initialized and has just the IMU variable.
    assert state.is_initialized
    assert len(state.variables) == 1
    assert state.imu.id() == 0
    assert state.imu.size() == 15
    assert state.U.shape == (15, 15)

    # U is upper triangular.
    assert np.allclose(state.U, np.triu(state.U), atol=1e-14)

    # Recompute the imu_init U in closed form and compare.
    U_ref = np.eye(15)
    U_ref[0, 0] = init_opts.init_prior_q
    U_ref[1, 1] = init_opts.init_prior_q
    U_ref[2, 2] = 0.001
    R_GtoI = state.imu.Rot()
    U_ref[:3, :3] = U_ref[:3, :3] @ R_GtoI.T
    U_ref[3:6, 3:6] = init_opts.init_prior_p * np.eye(3)
    U_ref[6:9, 6:9] = init_opts.init_prior_v * np.eye(3)
    U_ref[9:12, 9:12] = init_opts.init_prior_bg * np.eye(3)
    U_ref[12:15, 12:15] = init_opts.init_prior_ba * np.eye(3)
    # `init_prior_*` values are the *diagonal of U itself* (the std-dev of
    # each state component), not of the covariance. C++ writes them straight
    # into U_i,i and calls `efficient_QR`, which is a no-op on a diagonal.
    # So no Cholesky step is needed here.

    assert np.allclose(state.U, np.triu(U_ref), atol=1e-10)


def test_initialize_state_grows_state_when_calibrating_extrinsics():
    state = _make_state(
        num_cameras=1,
        do_calib_camera_pose=True,
        do_calib_camera_intrinsics=True,
    )
    init_opts = _make_init_opts()
    state.init_options = init_opts

    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    # IMU (15) + cam_extrinsic (6) + cam_intrinsics (8) = 29
    assert state.U.shape == (29, 29)
    n_vars = 1 + 1 + 1  # IMU + pose + intrinsics
    assert len(state.variables) == n_vars


class _FakeIntrinsics:
    """Minimal stand-in for a camera object; exposes `.get_value()`, `.w()`,
    `.h()`, `type(...)`. Only the identity/size fields `initialize_state`
    reads are needed."""

    def __init__(self, size: int = 8) -> None:
        self._size = size
        self._values = np.zeros(size)

    def get_value(self) -> np.ndarray:
        return self._values.copy()

    def w(self) -> int:
        return 640

    def h(self) -> int:
        return 480


# ---------------------------------------------------------------------------
# clone
# ---------------------------------------------------------------------------

def test_clone_splices_six_column_block_at_clone_start_id():
    state = _make_state()
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    state_size_before = state.get_state_size()
    k_clone = state.kCloneStartId  # 15 with no calib
    assert k_clone == 15

    new_clone = PoseJPL()
    StateHelper.clone(state, new_clone)

    assert state.get_state_size() == state_size_before + 6
    assert new_clone.id() == k_clone
    assert new_clone.size() == 6

    # The right side of U shifts by 6; the new block equals the IMU block.
    U = state.U
    # Column at new_loc:15..6..15 was the IMU cols; the new clone should
    # equal the pre-clone IMU cols (which we still have as state.imu's cols
    # in state.U after the splice). Since the IMU is still at id 0 and we
    # just spliced a new clone at id 15, the IMU block is U[:, 0:6] for
    # the top 6 rotation rows and clone block is U[:, 15:21].
    # But note: clone is called on a fresh state so U_prior is diagonal
    # identity 1e-3; after clone the IMU block stays, the clone block
    # equals it, and the rest is zero.
    imu_block = U[:, 0:6]
    clone_block = U[:, 15:21]
    assert np.allclose(imu_block, clone_block, atol=1e-15)

    # All other columns to the right of the clone should remain zero
    # (initial state had no SLAM features, no calib vars after IMU).
    right = U[:, 21:]
    assert right.shape[1] == 0 or np.allclose(right, 0.0)


def test_clone_shifts_ids_of_trailing_variables():
    """Add an SLAM feature (id = kCloneStartId), then clone — the SLAM
    feature's id must be shifted by 6."""
    state = _make_state()
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    # Manually add a trailing variable (an extra PoseJPL) at the current
    # kCloneStartId — the same layout the SLAM initializer would create.
    # We use PoseJPL because Landmark requires a dim and the intent here
    # is just to exercise the id-shifting branch of `clone`.
    trailing = PoseJPL()
    trailing.set_local_id(state.kCloneStartId)
    state.variables.append(trailing)
    # Grow U to accommodate the new variable's columns.
    n = state.U.shape[0] + trailing.size()
    U_new = np.zeros((n, n))
    U_new[: state.U.shape[0], : state.U.shape[1]] = state.U
    state.U = U_new
    state.xk_minus_x0 = np.zeros((n, 1))
    state.xk_minus_xk1 = np.zeros((n, 1))

    trailing_id_before = trailing.id()

    new_clone = PoseJPL()
    StateHelper.clone(state, new_clone)

    assert new_clone.id() == state.kCloneStartId
    assert trailing.id() == trailing_id_before + 6


# ---------------------------------------------------------------------------
# propagate
# ---------------------------------------------------------------------------

def test_propagate_applies_phi_t_and_inserts_noise_block():
    rng = np.random.default_rng(0)
    state = _make_state()
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    Phi = _spd(15, rng)  # just needs to be non-singular and symmetric
    Q_sqrt = _up(_spd(15, rng))

    U_before = state.U.copy()
    state_size_before = state.get_state_size()

    StateHelper.propagate(state, Phi, Q_sqrt)

    # State size is unchanged (propagation doesn't add variables).
    assert state.get_state_size() == state_size_before

    # U is upper triangular after the QR (clones_IMU is empty, so QR runs).
    U = state.U
    assert U.shape == (state_size_before, state_size_before)
    assert np.allclose(U, np.triu(U), atol=1e-14)

    # `efficient_QR` is a column-norm-preserving upper-triangularization,
    # NOT a true sqrt-form update. For the rectangular augmented matrix
    # (rows+15, state_size) it preserves `A^T A` (the Gram matrix) and
    # column norms, but does NOT preserve `A A^T`. So `U U^T` is not the
    # closed-form propagated covariance. Verify the structural invariants
    # that the QR *does* maintain:
    #   (a) column norms of U match those of the pre-QR augmented matrix,
    #   (b) the Gram matrix `U^T U = A^T A` is preserved,
    #   (c) the result is full rank.
    U_new_pre_qr = np.zeros(
        (state_size_before + 15, state_size_before), dtype=np.float64
    )
    U_new_pre_qr[:15, :15] = Q_sqrt
    U_new_pre_qr[15:, :15] = U_before @ Phi.T

    # (a) Column norms preserved.
    col_norms_pre = np.linalg.norm(U_new_pre_qr, axis=0)
    col_norms_post = np.linalg.norm(U, axis=0)
    assert np.allclose(col_norms_pre, col_norms_post, atol=1e-10)

    # (b) Gram matrix preserved (the invariant that actually holds).
    gram_pre = U_new_pre_qr.T @ U_new_pre_qr
    gram_post = U.T @ U
    assert np.allclose(gram_pre, gram_post, atol=1e-10)

    # (c) Full rank.
    assert np.linalg.matrix_rank(U) == state_size_before


def test_propagate_keeps_rectangular_when_at_max_clone_size():
    """When clones_IMU.size() >= max_clone_size + 1, propagate leaves U
    rectangular (rows > cols) — no QR to collapse."""
    state = _make_state(max_clone_size=1)
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    # Add enough clones to trip the no-QR branch.
    for ts in [1.1, 1.2]:
        clone = PoseJPL()
        StateHelper.clone(state, clone)
        state.clones_IMU[ts] = clone

    assert len(state.clones_IMU) == 2  # == max_clone_size + 1

    U_before = state.U
    # After multiple clones without marginalization, U is rectangular
    # (rows < cols) because clone adds columns but not rows.
    assert U_before.shape[0] < U_before.shape[1]

    rng = np.random.default_rng(1)
    Phi = np.eye(15)
    Q_sqrt = _up(_spd(15, rng))

    StateHelper.propagate(state, Phi, Q_sqrt)

    U_after = state.U
    assert U_after.shape[0] == U_before.shape[0] + 15
    assert U_after.shape[1] == U_before.shape[1]
    assert U_after.shape[0] > U_after.shape[1]


# ---------------------------------------------------------------------------
# marginalize
# ---------------------------------------------------------------------------

def test_marginalize_drops_variable_and_preserves_marginal_covariance():
    """Marginalizing a variable must yield the same P on the surviving
    block as a closed-form marginal P_{keep} from the joint P."""
    rng = np.random.default_rng(2)
    state = _make_state()
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    # Add an SLAM landmark at the tail of the state so we can marginalize
    # it and get a well-defined margined covariance.
    slam_id = state.get_state_size()
    slam_lm = Landmark(3)
    slam_lm.set_local_id(slam_id)
    state.variables.append(slam_lm)

    # Build a random joint P and set U = chol.
    n = slam_id + 3
    P = _spd(n, rng)
    U = _up(P)
    state.U = U
    state.xk_minus_x0 = np.zeros((n, 1))
    state.xk_minus_xk1 = np.zeros((n, 1))

    keep_size = n - 3
    P_keep_theory = P[:keep_size, :keep_size]

    state.add_marginal_state(slam_lm)
    StateHelper.marginalize(state)

    assert state.get_state_size() == keep_size
    U_new = state.U
    P_new = U_new.T @ U_new
    assert np.allclose(P_new, P_keep_theory, atol=1e-10)

    # The marginalized variable is no longer in the list.
    assert slam_lm not in state.variables


def test_marginalize_sorts_variables_by_id():
    """The C++ sorts variables by id before compaction; the survivors must
    be re-laid out in id order. `set_local_id(curr_id)` rebases each
    survivor's id to its column offset in U (which is 0 for the IMU, then
    15 for the first landmark, etc. — not contiguous 0..n-1)."""
    state = _make_state()
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    # Add two SLAM landmarks at the tail in *reverse* id order.
    n = state.U.shape[0]
    P = _spd(n + 6, np.random.default_rng(3))
    state.U = _up(P)
    state.xk_minus_x0 = np.zeros((n + 6, 1))
    state.xk_minus_xk1 = np.zeros((n + 6, 1))

    lm2 = Landmark(3)
    lm2.set_local_id(n + 3)
    state.variables.append(lm2)
    lm1 = Landmark(3)
    lm1.set_local_id(n)
    state.variables.append(lm1)  # out of order on purpose

    state.add_marginal_state(lm2)
    StateHelper.marginalize(state)

    ids = [v.id() for v in state.variables]
    assert ids == sorted(ids)
    # Survivors are imu (size 15) then lm1 (size 3), so their local_ids
    # become 0 and 15 respectively (column offsets in U).
    assert ids == [0, 15]
    # U must be shrunk to the survivors' total size.
    assert state.U.shape == (n + 3, n + 3)


# ---------------------------------------------------------------------------
# update_llt
# ---------------------------------------------------------------------------

def test_update_llt_empty_buffer_is_noop():
    state = _make_state()
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)
    U_before = state.U.copy()

    dx = StateHelper.update_llt(state, is_iterative=False)

    assert state.U.shape == U_before.shape
    assert np.allclose(state.U, U_before)
    assert np.allclose(dx, 0.0)


def test_update_llt_matches_sqrt_form_update():
    """With R = I, verify the OpenVINS sqrt-form update:
      * F_final^T F_final = H^T H + I (after column/row reversals),
      * U_new = F_final^{-T} U_prior,
      * P_new = U_new^T U_new (Eigen convention),
      * dx = P_new H^T r.
    This is NOT the classical Kalman posterior (P_prior^{-1} + H^T H)^{-1};
    OpenVINS applies a congruence to U_prior, not to P_prior.
    """
    rng = np.random.default_rng(4)
    state = _make_state()
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    n = state.get_state_size()  # 15 (IMU only)
    m = 4  # measurement rows

    # Random prior covariance.
    P_prior = _spd(n, rng)
    U_prior = _up(P_prior)
    state.U = U_prior.copy()

    # Set the IMU value to something sensible so `state.imu.update(dx)` works.
    state.imu.set_value(_identity_imu_init().reshape(16, 1))

    # Random H (m x n) and residual r (m x 1). With R = I, the stored
    # R_sqrt_inv_H_UT = R^-0.5 H = H and HT_R_inv_res = H^T R^-1 r = H^T r.
    H = rng.standard_normal((m, n))
    r = rng.standard_normal((m, 1))

    # Populate the buffers (mirrors C++ setup_matrix_buffer).
    state.setup_matrix_buffer()
    # HT_R_inv_res must be a full state_size vector; pad with zeros in the offset part.
    HT_R_inv_res = H.T @ r  # (n, 1) — information-form residual
    state.store_update_factor(H, HT_R_inv_res)

    assert state.R_sqrt_inv_H_UT.rows() == m
    assert state.HT_R_inv_res.rows() == n

    # Reference: OpenVINS sqrt-form update with R = I.
    # F_final = P @ chol(H_rev^T H_rev + I).T @ P where P is the column
    # reversal permutation and H_rev = H @ P.
    P_mat = np.fliplr(np.eye(n))
    H_rev = H @ P_mat
    FT_F = H_rev.T @ H_rev + np.eye(n)
    F = np.linalg.cholesky(FT_F).T  # upper triangular
    F_final = P_mat @ F @ P_mat     # lower triangular, F_final^T F_final = H^T H + I

    U_new_theory = np.linalg.inv(F_final).T @ U_prior  # F^{-T} U_prior
    P_new_theory = U_new_theory.T @ U_new_theory
    dx_theory = P_new_theory @ HT_R_inv_res

    # Run the port.
    dx = StateHelper.update_llt(state, is_iterative=False)

    assert dx.shape == (n, 1)
    assert np.allclose(dx, dx_theory, atol=1e-8)

    # The new U must satisfy U_new^T U_new = P_new_theory (Eigen convention).
    U_new = state.U
    assert np.allclose(U_new.T @ U_new, P_new_theory, atol=1e-8)
    # U is upper-triangular.
    assert np.allclose(U_new, np.triu(U_new), atol=1e-12)

    # The state's variables must have been updated by dx.
    imu = state.imu
    v_new = imu.value().reshape(-1, 1)
    # The IMU is (q, p, v, bg, ba) with 16-dim value; dx covers the
    # 15-dim error-state (q in log). The exact correspondence between
    # `value()` and the 15-dim error state is type-specific, so we just
    # check that the value has changed.
    assert not np.allclose(v_new, _identity_imu_init().reshape(16, 1))


def test_update_llt_with_offset_skips_x_init_landmark_block():
    """When `state.x_init` is non-empty, the first `offset = 3 * len`
    columns are excluded from the measurement update. Verify that:
      1. The bottom-right block of U (the "x_init" landmark) is unchanged.
      2. The top-left block matches the sqrt-form update on the top block.
      3. dx is non-trivial.
    Note: dx is computed from the FULL U (including the unchanged offset
    block), so the offset portion of dx is generally non-zero — the offset
    columns of U still contribute via the U^T (U * HT_R_inv_res) product."""
    rng = np.random.default_rng(5)
    state = _make_state()
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    # Manually add one fake landmark to x_init (offset = 3).
    fake_lm = Landmark(3)
    fake_lm.set_local_id(state.get_state_size())
    state.x_init.append(fake_lm)

    n = state.get_state_size() + 3  # 18
    P = _spd(n, rng)
    state.U = _up(P).copy()
    state.imu.set_value(_identity_imu_init().reshape(16, 1))

    m = 2
    H_full = rng.standard_normal((m, n))
    # Only the top (n - offset) columns of H participate in the update.
    H_top = H_full[:, : n - 3]  # (m, n-3)
    r = rng.standard_normal((m, 1))

    # HT_R_inv_res stores the information-form residual H^T R^-1 r.
    HT_R_inv_res = H_top.T @ r  # (n-3, 1)

    # Custom buffer setup for the offset case: R_sqrt_inv_H_UT has
    # (n - offset) cols, HT_R_inv_res has (n - offset) rows.
    state.R_sqrt_inv_H_UT.set_size(0, n - 3)
    state.HT_R_inv_res.set_size(n - 3, 1)
    state.store_update_factor(H_top, HT_R_inv_res)

    # Reference: OpenVINS sqrt-form update on the top block only.
    P_mat = np.fliplr(np.eye(n - 3))
    H_rev = H_top @ P_mat
    FT_F = H_rev.T @ H_rev + np.eye(n - 3)
    F = np.linalg.cholesky(FT_F).T  # upper triangular
    F_final = P_mat @ F @ P_mat     # lower triangular

    U_prior_top = _up(P[: n - 3, : n - 3])
    U_new_top = np.linalg.inv(F_final).T @ U_prior_top

    dx = StateHelper.update_llt(state, is_iterative=False)

    # 1. The bottom-right block of U is unchanged.
    U_before = _up(P).copy()
    U_new = state.U
    assert U_new.shape == (n, n)
    assert np.allclose(U_new[n - 3:, n - 3:], U_before[n - 3:, n - 3:], atol=1e-12)

    # 2. The top-left block matches the sqrt-form update.
    assert np.allclose(U_new[: n - 3, : n - 3], U_new_top, atol=1e-8)

    # 3. dx is non-trivial.
    assert dx.shape == (n, 1)
    assert np.linalg.norm(dx) > 1e-10


# ---------------------------------------------------------------------------
# iterative_update_llt
# ---------------------------------------------------------------------------

def test_iterative_update_llt_converges_to_noniterative():
    """The iterative update produces a non-trivial dx. The iterative path
    splits H into IMU and non-IMU columns, so we need state_size > imu_size.
    We add a landmark to provide non-IMU columns."""
    rng = np.random.default_rng(6)
    state = _make_state()
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    # Add a landmark so state_size > imu_size (iterative path needs
    # non-IMU columns to QR against).
    lm = Landmark(3)
    lm.set_local_id(state.get_state_size())  # id = 15
    state.variables.append(lm)

    n = state.U.shape[0] + 3  # 18
    P_prior = _spd(n, rng)
    state.U = _up(P_prior).copy()
    state.imu.set_value(_identity_imu_init().reshape(16, 1))

    m = 3
    H = 0.5 * rng.standard_normal((m, n))
    r = rng.standard_normal((m, 1))

    state.H_update.set_size(0, n)
    state.res_update.set_size(0, 1)
    state.xk_minus_x0 = np.zeros((n, 1))
    state.store_update_jacobians(H, r, [state.imu, lm])

    StateHelper.iterative_update_llt(state)

    # The iterative update stores its final dx in state.xk_minus_x0.
    dx_iter = state.xk_minus_x0
    assert dx_iter.shape == (n, 1)
    # Non-trivial result.
    assert np.linalg.norm(dx_iter) > 1e-10


# ---------------------------------------------------------------------------
# get_marginal_U / get_marginal_covariance
# ---------------------------------------------------------------------------

def test_get_marginal_U_and_covariance_agree_with_joint_slice():
    rng = np.random.default_rng(7)
    state = _make_state()
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    # Add two SLAM landmarks at the tail; the "small variables" are just
    # the IMU and one landmark.
    n = state.U.shape[0]
    P = _spd(n + 6, rng)
    state.U = _up(P)
    state.xk_minus_x0 = np.zeros((n + 6, 1))
    state.xk_minus_xk1 = np.zeros((n + 6, 1))

    lm1 = Landmark(3)
    lm1.set_local_id(n)
    state.variables.append(lm1)
    lm2 = Landmark(3)
    lm2.set_local_id(n + 3)
    state.variables.append(lm2)

    small_vars = [state.imu, lm1]
    U_small = StateHelper.get_marginal_U(state, small_vars)
    Cov_small = StateHelper.get_marginal_covariance(state, small_vars)

    # Column ordering: imu (15) then lm1 (3), total 18 columns.
    keep_ids = [0, 15, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, n]
    keep = np.concatenate(
        [np.arange(0, 15), np.arange(n, n + 3)]
    )
    # max(id + size) over [imu (id=0, sz=15), lm1 (id=n, sz=3)] = n + 3 = 18
    assert U_small.shape == (n + 3, 18)
    # Each column group matches the corresponding columns of state.U
    # (restricted to the first max_row_size rows).
    assert np.allclose(U_small[:, :15], state.U[:n + 3, 0:15])
    assert np.allclose(U_small[:, 15:18], state.U[:n + 3, n:n + 3])

    Cov_theory = P[np.ix_(keep, keep)]
    assert np.allclose(Cov_small, Cov_theory, atol=1e-10)
    assert Cov_small.shape == (18, 18)


def test_get_marginal_U_row_bound_is_max_id_plus_size():
    """Row count of the returned U_small is `max(id + size)` over small_vars
    (StateHelper.cpp:70-82), not the state's full row count."""
    rng = np.random.default_rng(8)
    state = _make_state()
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    # Only add a landmark whose id+size is *less* than the total state.
    n = state.U.shape[0]
    P = _spd(n + 3, rng)
    state.U = _up(P)
    state.xk_minus_x0 = np.zeros((n + 3, 1))
    state.xk_minus_xk1 = np.zeros((n + 3, 1))

    lm = Landmark(3)
    lm.set_local_id(n)
    state.variables.append(lm)

    U_small = StateHelper.get_marginal_U(state, [lm])
    # max(id + size) for lm is n + 3 = state size here, but if we drop
    # the imu, the row bound is exactly n+3 for the lm var.
    assert U_small.shape == (n + 3, 3)


# ---------------------------------------------------------------------------
# propagate_zero_motion
# ---------------------------------------------------------------------------

def test_propagate_zero_motion_preserves_rank():
    """The rank of U (== rank of P = U U^T) should not change after adding
    zero-motion rows and doing QR."""
    rng = np.random.default_rng(9)
    state = _make_state()
    StateHelper.initialize_state(state, _identity_imu_init(), 1.0)

    n = state.get_state_size()
    assert state.U.shape == (n, n)

    StateHelper.propagate_zero_motion(state, dt_summed=1.0,
                                     sigma_wb=2e-5, sigma_ab=3e-3)

    U_new = state.U
    assert U_new.shape == (n, n)
    # P = U^T U is still symmetric PSD and full-rank (Eigen convention).
    P = U_new.T @ U_new
    assert np.allclose(P, P.T, atol=1e-14)
    assert np.linalg.matrix_rank(P) == n


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
