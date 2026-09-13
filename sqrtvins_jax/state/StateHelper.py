"""
StateHelper — JAX port of ov_srvins/src/state/StateHelper.cpp.

Static functions for the sqrt-form covariance engine: marginals, propagation,
cloning, the LLT update, and the (deferred) SLAM initialization pass.

Port notes
----------
* The C++ `U_` is a mutable Eigen matrix that's constantly resized/sliced.
  Python's numpy can't resize in place, so every kernel that changes the
  shape of `state.U` rebinds `state.U` to a new array. Callers must NOT
  hold a reference to `state.U` across these calls.
* C++ `Eigen::LLT<MatX> llt(FT_F.selfadjointView<Upper>()); MatX F =
  llt.matrixU();` maps to `np.linalg.cholesky(FT_F).T` (numpy returns L
  such that `A = L L^T`, so `U = L^T`).
* `.triangularView<Upper>()` on a matrix in an assignment target zeros the
  lower triangle; the Python equivalent is `np.triu`.
* The C++ `state->U_ = state->U_.triangularView<Eigen::Upper>()` after
  `efficient_QR` collapses a `(rows, cols)` matrix into a `(cols, cols)`
  upper-triangular one. This port mirrors it exactly.
* `initialize_state` needs `InertialInitializerOptions`. The header-only
  scalar fields are read directly from `state.init_options`; callers are
  expected to have built an options object with the corresponding attributes.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from scipy.linalg import solve_triangular
from scipy.stats import chi2 as _chi2_dist

from sqrtvins_core.types.IMU import IMU
from sqrtvins_core.types.Landmark import Landmark
from sqrtvins_core.types.PoseJPL import PoseJPL
from sqrtvins_core.types.Type import Type
from sqrtvins_core.types.Vec import Vec
from sqrtvins_core.utils.print import print_error

from ..utils.Helper import (
    efficient_QR,
    matrix_multiplier_ATA,
    reverse_mat,
    reverse_vec,
    triangular_matrix_inverse_solver,
    triangular_matrix_multiplier_LLT,
    triangular_matrix_multiplier_UU,
)
from .State import State


# ---------------------------------------------------------------------------
# Small cov helpers
# ---------------------------------------------------------------------------

def set_initial_imu_square_root_covariance(
    state: State, diagonal: np.ndarray
) -> None:
    """Set the diagonal of the top-left 15x15 block of U to `diagonal`
    (StateHelper.cpp:45-49). `diagonal` has shape (15, 1) or (15,).
    """
    assert state.imu.id() == 0, "set_initial_imu_square_root_covariance: IMU must be at id 0"
    d = np.asarray(diagonal, dtype=np.float64).reshape(-1)
    for i in range(15):
        state.U[i, i] = d[i]


def get_marginal_U(
    state: State, small_variables: List[Type]
) -> np.ndarray:
    """Extract the U columns corresponding to `small_variables`
    (StateHelper.cpp:65-91).

    Returns an (max_row_size, U_size) matrix where the columns come from
    the surviving variables' ids.
    """
    U_size = 0
    max_row_size = -1
    for var in small_variables:
        U_size += var.size()
        top = var.id() + var.size()
        if top > max_row_size:
            max_row_size = top

    Small_U = np.zeros((max_row_size, U_size), dtype=np.float64)

    current_id = 0
    for var in small_variables:
        Small_U[:, current_id:current_id + var.size()] = state.U[:max_row_size, var.id():var.id() + var.size()]
        current_id += var.size()
    return Small_U


def get_marginal_covariance(
    state: State, small_variables: List[Type]
) -> np.ndarray:
    """Return the marginal covariance of `small_variables` (StateHelper.cpp:51-63)."""
    Small_U = get_marginal_U(state, small_variables)
    state_size = 0
    for var in small_variables:
        state_size += var.size()
    Small_cov = np.zeros((state_size, state_size), dtype=np.float64)
    matrix_multiplier_ATA(Small_U, Small_cov)
    return Small_cov


def get_marginal_U_block(
    state: State,
    small_variables: List[Type],
    U_dense: np.ndarray,
    U_upper_tri: np.ndarray,
) -> None:
    """Split the marginal U into a dense (top) block and an upper-tri
    (bottom) block (StateHelper.cpp:93-129).

    Requires `state.U` to be square. `U_dense` and `U_upper_tri` are
    preallocated; they are overwritten in place.
    """
    assert state.U.shape[0] == state.U.shape[1], "U must be square"

    for i in range(len(small_variables) - 1):
        if small_variables[i].id() > small_variables[i + 1].id():
            print_error("get_marginal_U_block: small_variables must be in id ascending order")
            raise AssertionError("small_variables must be sorted by id")

    U_size = sum(v.size() for v in small_variables)
    last_var = small_variables[-1]
    max_non_zero_rows = last_var.id() + last_var.size()

    n_dense = max_non_zero_rows - U_size

    current_id = 0
    for var in small_variables:
        U_dense[:, current_id:current_id + var.size()] = state.U[:n_dense, var.id():var.id() + var.size()]
        U_upper_tri[:, current_id:current_id + var.size()] = state.U[n_dense:, var.id():var.id() + var.size()]
        current_id += var.size()


# ---------------------------------------------------------------------------
# Marginalization
# ---------------------------------------------------------------------------

def marginalize_old_clone(state: State) -> None:
    """Marginalize the oldest clone when we exceed the clone limit
    (StateHelper.cpp:131-138)."""
    if len(state.clones_IMU) > state.options.max_clone_size + 1:
        marginal_time = state.margtimestep()
        state.state_to_marg.append(state.clones_IMU[marginal_time])
        state.remove_timestamp(marginal_time)
        del state.clones_IMU[marginal_time]


def marginalize_slam(state: State) -> None:
    """Remove SLAM features with `should_marg` set and id > 4*max_aruco
    (StateHelper.cpp:140-155)."""
    threshold = 4 * state.options.max_aruco_features
    for feat_id, lm in list(state.features_SLAM.items()):
        if lm.should_marg and feat_id > threshold:
            state.state_to_marg.append(lm)
            del state.features_SLAM[feat_id]


def _sort_variables_by_id(variables: List[Type]) -> None:
    variables.sort(key=lambda v: v.id())


def marginalize(state: State) -> None:
    """Compact U by dropping variables in `state.state_to_marg`
    (StateHelper.cpp:157-198)."""
    marg_size = sum(m.size() for m in state.state_to_marg)

    _sort_variables_by_id(state.variables)

    rows = state.U.shape[0]
    old_state_size = state.U.shape[1]
    new_state_size = old_state_size - marg_size
    curr_id = 0

    # Copy surviving blocks to the left.
    # Python has no in-place row-shuffle like Eigen's block assign, so we
    # accumulate the new U into a new buffer (the block copies in the C++
    # do overlap; the C++ relies on Eigen's block eval semantics here).
    U_new = np.zeros((rows, new_state_size), dtype=np.float64)
    curr_id = 0
    for var in state.variables:
        if var in state.state_to_marg:
            continue
        src = var.id()
        sz = var.size()
        U_new[:, curr_id:curr_id + sz] = state.U[:, src:src + sz]
        var.set_local_id(curr_id)
        curr_id += sz

    # Drop marginalized vars from the list.
    keep = [v for v in state.variables if v not in state.state_to_marg]
    state.variables = keep

    # QR and reduce to upper triangular square.
    efficient_QR(U_new)
    state.U = np.triu(U_new[:new_state_size, :new_state_size])

    # Resize delta vectors.
    state.xk_minus_x0 = state.xk_minus_x0[:new_state_size]
    state.xk_minus_xk1 = state.xk_minus_xk1[:new_state_size]
    state.state_to_marg = []


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def propagate(
    state: State, Phi: np.ndarray, Q_sqrt: np.ndarray
) -> None:
    """Imu propagation + optional QR (StateHelper.cpp:200-233).

    `Phi` and `Q_sqrt` are 15x15 numpy arrays.
    """
    Phi = np.asarray(Phi, dtype=np.float64).reshape(15, 15)
    Q_sqrt = np.asarray(Q_sqrt, dtype=np.float64).reshape(15, 15)

    state_size = state.U.shape[1]
    rows = state.U.shape[0]
    imu_id = state.imu.id()
    remaining_id = imu_id + state.imu.size()  # imu.size() == 15

    U_new = np.zeros((rows + 15, state_size), dtype=np.float64)

    # Copy the pre-IMU block (only meaningful when imu.id() != 0, i.e. we
    # have calibration variables before the IMU).
    if imu_id != 0:
        U_new[15:15 + imu_id, :imu_id] = np.triu(state.U[:imu_id, :imu_id])

    # New IMU block: U[:, imu_id:remaining_id] * Phi^T
    U_new[15:15 + remaining_id, imu_id:imu_id + 15] = (
        state.U[:remaining_id, imu_id:imu_id + 15] @ Phi.T
    )

    # Remaining block (all clones, all other vars): copy verbatim.
    rest_cols = state_size - remaining_id
    if rest_cols > 0:
        U_new[15:15 + rows, remaining_id:state_size] = state.U[:, remaining_id:state_size]

    # Noise block
    U_new[:15, imu_id:imu_id + 15] = Q_sqrt

    if len(state.clones_IMU) < state.options.max_clone_size + 1:
        efficient_QR(U_new)
        U_new = np.triu(U_new[:state_size, :state_size])

    state.U = U_new


def clone(state: State, variable_to_clone: Type) -> Type:
    """Insert a 6-column clone at `state.kCloneStartId`, shifting all
    variables at `id >= new_loc` by `new_size` (StateHelper.cpp:235-273)."""

    new_size = variable_to_clone.size()
    state_size = state.U.shape[1]
    rows = state.U.shape[0]
    new_loc = state.kCloneStartId

    U_new = np.zeros((rows, state_size + new_size), dtype=np.float64)

    # Left of the clone is preserved (C++ conservativeResize keeps the
    # top-left block intact).
    if new_loc > 0:
        U_new[:, :new_loc] = state.U[:, :new_loc]

    # Right of the clone: shift by new_size.
    if new_loc < state_size:
        U_new[:, new_loc + new_size:] = state.U[:, new_loc:]

    # The new clone block: copy the IMU block.
    U_new[:, new_loc:new_loc + new_size] = state.U[:, state.imu.id():state.imu.id() + new_size]

    variable_to_clone.set_local_id(new_loc)

    # Shift ids of all variables at or after new_loc.
    for var in state.variables:
        if var.id() >= new_loc:
            var.set_local_id(var.id() + new_size)

    state.variables.append(variable_to_clone)
    state.U = U_new
    return variable_to_clone


def propagate_timeoffset(state: State, dnc_dt: np.ndarray) -> None:
    """Timeoffset Jacobian into the U block
    (StateHelper.cpp:793-807)."""
    dnc_dt = np.asarray(dnc_dt, dtype=np.float64).reshape(6, 1)
    kStartRow = 15
    if len(state.clones_IMU) <= state.options.max_clone_size + 1:
        kStartRow = 0

    # C++: state->clones_IMU.rbegin()->second — the LAST (most-recent) clone.
    pose = list(state.clones_IMU.values())[-1]
    dt_id = state.calib_dt_CAMtoIMU.id()

    # state->U_.block(dt_id + kStartRow, pose->id(), dt_id + 1, 6) += ...
    row_start = dt_id + kStartRow
    n_rows = dt_id + 1
    col_start = pose.id()
    U_new = state.U[row_start:row_start + n_rows, col_start:col_start + 6] + (
        state.U[row_start:row_start + n_rows, dt_id:dt_id + 1] @ dnc_dt.T
    )
    state.U[row_start:row_start + n_rows, col_start:col_start + 6] = U_new


def propagate_zero_motion(
    state: State, dt_summed: float, sigma_wb: float, sigma_ab: float
) -> None:
    """Add a zero-motion (biases-only) row block to U (StateHelper.cpp:809-821)."""
    U_new = np.zeros((6 + state.U.shape[0], state.U.shape[1]), dtype=np.float64)
    bg_id = state.imu.bg().id()
    ba_id = state.imu.ba().id()
    U_new[6:] = state.U
    U_new[:3, bg_id:bg_id + 3] = np.eye(3) * (dt_summed ** 0.5) * sigma_wb
    U_new[:3, ba_id:ba_id + 3] = np.eye(3) * (dt_summed ** 0.5) * sigma_ab
    efficient_QR(U_new)
    state.U = np.triu(U_new[:state.U.shape[1], :state.U.shape[1]])


def propagate_slam_anchor_feature(
    state: State,
    landmark: Landmark,
    Phi: np.ndarray,
    phi_order_OLD: List[Type],
) -> None:
    """Covariance propagation for an SLAM feature whose anchor just changed
    (StateHelper.cpp:823-872)."""
    Phi = np.asarray(Phi, dtype=np.float64)
    kFeatSize = landmark.size()
    start_row = 15

    phi_id_map: dict = {}
    current_it = 0
    for var in phi_order_OLD:
        phi_id_map[var] = current_it
        current_it += var.size()

    U_feat_new = np.zeros((state.U.shape[0], kFeatSize), dtype=np.float64)
    clone_A_new = state.clones_IMU[landmark.anchor_clone_timestamp]

    for var in phi_order_OLD:
        var_id = var.id()
        var_sz = var.size()

        if var_id == clone_A_new.id():
            # Top 6 rows: from the top 6 of U[:, var_id..].
            U_feat_new[:6, :] += (
                state.U[:6, var_id:var_id + var_sz]
                @ Phi[:, phi_id_map[var]:phi_id_map[var] + var_sz].T
            )
            if (state.imu.id() == 0
                and state.calib_dt_CAMtoIMU.id() == state.imu.id() + 1):
                U_feat_new[start_row:start_row + 16, :] += (
                    state.U[start_row:start_row + 16, var_id:var_id + var_sz]
                    @ Phi[:, phi_id_map[var]:phi_id_map[var] + var_sz].T
                )
                continue

        offset = 6 if var_id >= state.kCloneStartId else 0
        n_rows = var_sz + var_id - offset
        U_feat_new[start_row:start_row + n_rows, :] += (
            state.U[start_row:start_row + n_rows, var_id:var_id + var_sz]
            @ Phi[:, phi_id_map[var]:phi_id_map[var] + var_sz].T
        )

    state.U[:, landmark.id():landmark.id() + kFeatSize] = U_feat_new


# ---------------------------------------------------------------------------
# The LLT update kernels
# ---------------------------------------------------------------------------

def _cholesky_upper(A: np.ndarray) -> np.ndarray:
    """Return the U factor such that A = U^T U (matches Eigen's LLT::matrixU
    when A is provided as `selfadjointView<Upper>`)."""
    L = np.linalg.cholesky(A)
    return L.T


def update_llt(state: State, is_iterative: bool = False) -> np.ndarray:
    """The core sqrt-form update (StateHelper.cpp:275-355).

    Returns `dx_xkp1_minus_x0` — the state increment (a `(state_size, 1)`
    numpy array). In iterative mode the C++ computes H^T H dx and adds it to
    HT_R_inv_res before the LLT, then does the downdate/update dance.
    """
    if state.R_sqrt_inv_H_UT.rows() == 0:
        return np.zeros((state.U.shape[1], 1), dtype=np.float64)

    state_size = state.U.shape[1]
    offset = len(state.x_init) * 3

    # HT_R_inv_res is a full (state_size, 1) vector; the stored buffer only
    # fills the top `state_size - offset` rows (the rest stay zero).
    HT_R_inv_res = np.zeros((state_size, 1), dtype=np.float64)
    stored_HTR = state.HT_R_inv_res.get()
    n_stored = stored_HTR.shape[0]
    assert n_stored == state_size - offset, (
        f"update_llt: expected {state_size - offset} rows in HT_R_inv_res, got {n_stored}"
    )
    HT_R_inv_res[:n_stored] = stored_HTR

    # Copy R_sqrt_inv_H_UT so we don't alias the buffer.
    R_sqrt_inv_H_UT = state.R_sqrt_inv_H_UT.get().copy()

    # Reverse rows for the Cholesky.
    reverse_mat(R_sqrt_inv_H_UT)

    FT_F = np.zeros((state_size - offset, state_size - offset), dtype=np.float64)
    matrix_multiplier_ATA(R_sqrt_inv_H_UT, FT_F)
    diag = np.arange(FT_F.shape[0])
    FT_F[diag, diag] += 1.0

    F = _cholesky_upper(FT_F)

    # Reverse rows back.
    reverse_mat(F)
    # Reverse columns to obtain the upper-triangular view of F.
    reverse_mat(F, do_column=False)

    # Iterative: add H^T H dx.
    if is_iterative and state.H_update.rows() > 0:
        H_big = state.H_update.get()
        Hx0 = H_big @ state.xk_minus_x0
        HTHx0 = H_big.T @ Hx0
        HT_R_inv_res[:state_size - offset] += HTHx0[:state_size - offset]

    # Solve F^T U_topRight = U_topRight (in-place).
    # Eigen: `F.transpose().triangularView<Upper>().solveInPlace(state->U_.topRightCorner(...))`
    U = state.U
    # The top-right (state_size - offset) x offset corner:
    if offset > 0:
        top_right = U[:state_size - offset, state_size - offset:state_size].copy()
        top_right = solve_triangular(F.T, top_right, lower=False)
        U[:state_size - offset, state_size - offset:state_size] = top_right

    # Invert the top-left (state_size - offset) x (state_size - offset) block
    # by left-multiplying with F^{-T} (the C++ does this via
    # triangular_matrix_inverse_solver with F.transpose()).
    top_left = U[:state_size - offset, :state_size - offset].copy()
    triangular_matrix_inverse_solver(F.T, top_left)
    U[:state_size - offset, :state_size - offset] = top_left

    # Collapse to upper triangular.
    state.U = np.triu(U)

    # dx = U^T * (U * HT_R_inv_res)
    U_upper = state.U
    dx_xkp1_minus_x0 = U_upper.T @ (U_upper @ HT_R_inv_res)

    if is_iterative:
        # First downdate
        for var in state.variables:
            var_id = var.id()
            var_sz = var.size()
            var.update(-state.xk_minus_x0[var_id:var_id + var_sz].copy())

    state.xk_minus_x0 = dx_xkp1_minus_x0
    for var in state.variables:
        var_id = var.id()
        var_sz = var.size()
        var.update(state.xk_minus_x0[var_id:var_id + var_sz].copy())

    # Update camera objects if doing online intrinsic calibration.
    if state.options.do_calib_camera_intrinsics:
        for cam_id, intr in state.cam_intrinsics.items():
            cam = state.cam_intrinsics_cameras[cam_id]
            cam.set_value(intr.value())

    return dx_xkp1_minus_x0


def iterative_update_llt(state: State) -> None:
    """The null-space-projected iterative update (StateHelper.cpp:357-415).

    Uses the H_update buffer directly (not R_sqrt_inv_H_UT). Skipped when
    H_update is empty.
    """
    if state.H_update.rows() == 0:
        return

    state_size = state.U.shape[1]

    H_big = state.H_update.get().copy()
    r_big = state.res_update.get().copy()

    reverse_mat(H_big)

    # Get the block excluding the IMU columns (leftCols(state_size - 15)).
    H_big_block = H_big[:, :state_size - 15].copy()
    # efficient_QR in place; r_big needs to be a slice of the same height.
    efficient_QR(H_big_block, r_big)

    # Restore H_big with the QR-ed columns and the untouched IMU columns.
    H_big_new = np.zeros_like(H_big)
    H_big_new[:, :state_size - 15] = H_big_block
    H_big_new[:, state_size - 15:] = state.H_update.get()[:, state_size - 15:]
    r_big_new = r_big.copy()

    # The C++ uses get_block(0, 0, state_size, state_size) which pads the
    # buffer to (state_size, state_size). Do the same so the UU multiplier
    # gets square inputs.
    n_rows = min(H_big_new.shape[0], state_size)
    H_big_new_padded = np.zeros((state_size, state_size), dtype=np.float64)
    H_big_new_padded[:n_rows, :] = H_big_new[:n_rows, :]
    H_big_new = H_big_new_padded

    r_big_new_padded = np.zeros((state_size, 1), dtype=np.float64)
    r_big_new_padded[:n_rows, :] = r_big_new[:n_rows, :]
    r_big_new = r_big_new_padded

    # reverse_mat(H_big_new, false); reverse_mat(H_big_new); reverse_vec(r_big_new);
    reverse_mat(H_big_new, do_column=False)
    reverse_mat(H_big_new, do_column=True)
    reverse_vec(r_big_new)

    # r += H_lower * dx_prev
    H_lower = np.tril(H_big_new)
    r_big_new += H_lower @ state.xk_minus_x0

    # UH_T = U * H_big_new^T (via triangular matrix multiplier)
    UH_T = np.zeros((state_size, state_size), dtype=np.float64)
    triangular_matrix_multiplier_UU(state.U, H_big_new.T, UH_T)

    # S = UH_T^T * UH_T
    S = np.zeros((state_size, state_size), dtype=np.float64)
    triangular_matrix_multiplier_LLT(UH_T.T, S)
    diag = np.arange(state_size)
    S[diag, diag] += 1.0

    # dx = U^T * (U * (H_big_new^T.triangularUpper * solve_llt(S, r)))
    # H_big_new^T.triangularView<Upper> is just np.triu(H_big_new.T)
    S_inv_r = np.linalg.solve(S, r_big_new)
    rhs = np.triu(H_big_new.T) @ S_inv_r
    dx_xkp1_minus_x0 = state.U.T @ (state.U @ rhs)

    # Downdate, then update.
    for var in state.variables:
        var_id = var.id()
        var_sz = var.size()
        var.update(-state.xk_minus_x0[var_id:var_id + var_sz].copy())
    state.xk_minus_x0 = dx_xkp1_minus_x0
    for var in state.variables:
        var_id = var.id()
        var_sz = var.size()
        var.update(state.xk_minus_x0[var_id:var_id + var_sz].copy())

    if state.options.do_calib_camera_intrinsics:
        for cam_id, intr in state.cam_intrinsics.items():
            cam = state.cam_intrinsics_cameras[cam_id]
            cam.set_value(intr.value())


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def initialize_invertible(
    state: State,
    new_variable: Type,
    H_order: List[Type],
    H_R: np.ndarray,
    H_L: np.ndarray,
    res: np.ndarray,
    sigma_pix_inv: float,
) -> None:
    """Insert a new variable's invertible initialization factor into U
    (StateHelper.cpp:523-593)."""
    for var in state.variables:
        if var is new_variable:
            print_error("initialize_invertible: called on variable that is already in the state")
            raise AssertionError("new_variable already in state")

    H_R = np.asarray(H_R, dtype=np.float64)
    H_L = np.asarray(H_L, dtype=np.float64)
    res = np.asarray(res, dtype=np.float64).reshape(-1, 1)

    assert H_L.shape[0] == res.shape[0]
    assert H_L.shape[0] == H_R.shape[0]

    kStateSize = state.U.shape[1]
    kMeasSize = res.shape[0]

    U_HRT = np.zeros((kStateSize, kMeasSize), dtype=np.float64)

    # Build the H_id offsets into H_R.
    H_id: List[int] = []
    current_it = 0
    for meas_var in H_order:
        H_id.append(current_it)
        current_it += meas_var.size()

    # For each active var, M_i = sum U[var, meas_var] * H_R[:, H_id[i]]^T
    U = state.U
    for var in state.variables:
        M_i = np.zeros((var.size(), kMeasSize), dtype=np.float64)
        for i, meas_var in enumerate(H_order):
            sz = meas_var.size()
            v_id = var.id()
            m_id = meas_var.id()
            M_i += U[v_id:v_id + var.size(), m_id:m_id + sz] @ H_R[:, H_id[i]:H_id[i] + sz].T
        U_HRT[var.id():var.id() + var.size(), :] = M_i

    # H_Linv: solve H_L * X = I (lower-triangular)
    # Eigen: H_L.triangularView<Lower>().solveInPlace(H_Linv); H_Linv starts as I.
    H_Linv = np.eye(H_L.shape[0], dtype=np.float64)
    H_Linv = np.linalg.solve_triangular(H_L, H_Linv, lower=True)

    # dx of the existing state, if we're not yet initialized.
    dx = np.zeros(H_R.shape[1], dtype=np.float64)
    if not state.is_initialized:
        current_it = 0
        for var in H_order:
            dx[current_it:current_it + var.size()] = state.xk_minus_x0[var.id():var.id() + var.size()]
            current_it += var.size()

    new_variable.update(H_Linv @ (res.reshape(-1) + H_R @ dx))

    # Store the init factor (triangular and dense parts).
    tri_factor = (1.0 / sigma_pix_inv) * H_Linv.T  # (kMeasSize, kMeasSize)
    dense_factor = -U_HRT @ H_Linv.T  # (kStateSize, kMeasSize)
    state.store_init_factor(new_variable, tri_factor, dense_factor)


def initialize(
    state: State,
    new_variable: Type,
    H_order: List[Type],
    H_R: np.ndarray,
    H_L: np.ndarray,
    res: np.ndarray,
    chi_2_mult: float,
    sigma_pix_inv: float,
) -> bool:
    """Gate + insert a new SLAM landmark (StateHelper.cpp:417-521).

    Returns True if the landmark was accepted, False if the chi2 gate rejected
    it. `chi_2_mult == -1` disables the gate.
    """
    for var in state.variables:
        if var is new_variable:
            print_error("initialize: called on variable that is already in the state")
            raise AssertionError("new_variable already in state")

    H_R = np.asarray(H_R, dtype=np.float64)
    H_L = np.asarray(H_L, dtype=np.float64)
    res = np.asarray(res, dtype=np.float64).reshape(-1, 1)

    kNewVarSize = new_variable.size()
    kUpdateStateSize = H_R.shape[1]
    kUpdateMeasSize = H_R.shape[0] - kNewVarSize
    assert H_L.shape[1] == kNewVarSize

    # reverse_mat on H_L to make it lower triangular (row-reversal)
    reverse_mat(H_L)
    reverse_mat(H_R)
    efficient_QR(H_R, res, H_L)
    reverse_mat(H_R)

    # Extract init sub-blocks.
    Hx_init = H_R[:kNewVarSize, :].copy()
    Hf_init = H_L[:kNewVarSize, :].copy()
    res_init = res[:kNewVarSize, :].copy()

    # reverse_mat back for the init system.
    reverse_mat(Hf_init)
    reverse_mat(Hx_init, do_column=False)  # column-reverse
    reverse_mat(Hf_init, do_column=False)  # column-reverse
    reverse_vec(res_init.reshape(-1))  # in-place reversal of the 1D view

    # Null-space projected updating system.
    H_update = H_R[kNewVarSize:, :].copy()
    res_update = res[kNewVarSize:, :].copy()

    # Mahalanobis gate: split the marginal U into dense (top) + upper-tri
    # (bottom) blocks. The Python port returns correctly-sized arrays (the C++
    # `resizeLike` mutates the caller's buffers, which Python can't emulate).
    U_dense, U_tri = _compute_marginal_U_block(state, H_order)

    n_dense = U_dense.shape[0]
    n_tri = U_tri.shape[0]
    HUT = np.zeros((kUpdateMeasSize, n_dense + n_tri), dtype=np.float64)
    HUT[:, :n_dense] = H_update @ U_dense.T
    HUT[:, n_dense:] = H_update @ np.tril(U_tri.T)

    if chi_2_mult != -1:
        S = HUT @ HUT.T
        S_diag = S.shape[0]
        inv_sigma_sq = 1.0 / (sigma_pix_inv * sigma_pix_inv)
        for i in range(S_diag):
            S[i, i] += inv_sigma_sq
        chi2 = float(res_update.T @ np.linalg.solve(S, res_update))
        chi2_check = float(_chi2_dist.ppf(0.95, df=res.shape[0]))
        if chi2 > chi_2_mult * chi2_check:
            return False

    initialize_invertible(
        state, new_variable, H_order, Hx_init, Hf_init, res_init, sigma_pix_inv
    )

    if H_update.shape[0] > 0:
        RHTr = np.zeros((state.U.shape[1], 1), dtype=np.float64)
        local_id = 0
        for var in H_order:
            sz = var.size()
            v_id = var.id()
            RHTr[v_id:v_id + sz, :] += (
                H_update[:, local_id:local_id + sz].T @ res_update
            ) * (sigma_pix_inv * sigma_pix_inv)
            local_id += sz
        state.store_update_factor(HUT * sigma_pix_inv, RHTr)
        if not state.is_initialized:
            state.store_update_jacobians(
                H_update * sigma_pix_inv,
                res_update * sigma_pix_inv,
                H_order,
            )

    return True


def _compute_marginal_U_block(
    state: State, small_variables: List[Type]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (U_dense, U_upper_tri) as correctly-sized numpy arrays
    matching the C++ `get_marginal_U_block` semantics."""
    assert state.U.shape[0] == state.U.shape[1]

    U_size = sum(v.size() for v in small_variables)
    last_var = small_variables[-1]
    max_non_zero_rows = last_var.id() + last_var.size()
    n_dense = max_non_zero_rows - U_size

    U_dense = np.zeros((n_dense, U_size), dtype=np.float64)
    U_upper_tri = np.zeros((U_size, U_size), dtype=np.float64)

    current_id = 0
    for var in small_variables:
        sz = var.size()
        v_id = var.id()
        U_dense[:, current_id:current_id + sz] = state.U[:n_dense, v_id:v_id + sz]
        U_upper_tri[:, current_id:current_id + sz] = state.U[n_dense:, v_id:v_id + sz]
        current_id += sz
    return U_dense, U_upper_tri


def initialize_slam_in_U(state: State) -> None:
    """Fold the deferred x_init variables into U
    (StateHelper.cpp:595-644)."""
    if not state.x_init:
        return

    new_state_size = sum(v.size() for v in state.x_init)
    old_size = state.U.shape[1]
    curr_id = old_size

    U = np.zeros((old_size + new_state_size, old_size + new_state_size), dtype=np.float64)
    U[:curr_id, :curr_id] = state.U[:curr_id, :curr_id]
    if curr_id != old_size:
        U[:curr_id, curr_id:old_size] = state.U[:curr_id, curr_id:old_size]
        U[curr_id:old_size, curr_id:old_size] = state.U[curr_id:old_size, curr_id:old_size]

    # Shift ids.
    for var in state.variables:
        if var.id() >= curr_id:
            var.set_local_id(var.id() + new_state_size)

    factor_init_dense = state.factor_init_dense.get()
    U[:factor_init_dense.shape[0], curr_id:curr_id + factor_init_dense.shape[1]] = factor_init_dense

    factor_init_tri = state.factor_init_tri.get()
    for i, var in enumerate(state.x_init):
        var_size = var.size()
        # Upper triangular factor goes into (curr_id, curr_id).
        tri_block = factor_init_tri[3 * i:3 * i + var_size, :].copy()
        U[curr_id:curr_id + var_size, curr_id:curr_id + var_size] = np.triu(tri_block)
        var.set_local_id(curr_id)
        state.variables.append(var)
        curr_id += var_size

    state.U = U

    # Resize the delta vectors (zero-fill the new region).
    new_n = old_size + new_state_size
    for attr_name in ("xk_minus_x0", "xk_minus_xk1"):
        old = getattr(state, attr_name)
        new_v = np.zeros((new_n, 1), dtype=np.float64)
        n_copy = min(old.shape[0], new_n)
        new_v[:n_copy] = old[:n_copy]
        setattr(state, attr_name, new_v)


def initialize_state(state: State, imu_init: np.ndarray, timestamp: float) -> None:
    """Set up the initial filter state from an initializer's output
    (StateHelper.cpp:646-791).

    `imu_init` is a (16, 1) numpy array with the IMU value layout
    `[qx, qy, qz, qw, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]`.
    """
    # Clear.
    state.variables = []
    state.calib_IMUtoCAM = {}
    state.cam_intrinsics = {}
    state.cam_intrinsics_cameras = {}

    state.update_timestamp(timestamp)

    # Append IMU.
    state.imu = IMU()
    state.imu.set_local_id(0)
    v_init = np.asarray(imu_init, dtype=np.float64).reshape(-1)
    state.imu.set_value(v_init)
    state.imu.set_fej(v_init)
    state.variables.append(state.imu)

    current_id = state.imu.size()

    # Camera to IMU time offset.
    state.calib_dt_CAMtoIMU = Vec(1)
    if state.options.do_calib_camera_timeoffset:
        state.calib_dt_CAMtoIMU.set_local_id(current_id)
        state.variables.append(state.calib_dt_CAMtoIMU)
        current_id += state.calib_dt_CAMtoIMU.size()

    # Per-camera extrinsics and intrinsics.
    init_opts = state.init_options
    for i in range(state.options.num_cameras):
        pose = PoseJPL()
        intrin = Vec(8)
        state.calib_IMUtoCAM[i] = pose
        state.cam_intrinsics[i] = intrin

        if state.options.do_calib_camera_pose:
            pose.set_local_id(current_id)
            state.variables.append(pose)
            current_id += pose.size()
        if state.options.do_calib_camera_intrinsics:
            intrin.set_local_id(current_id)
            state.variables.append(intrin)
            current_id += intrin.size()

    # Set U.
    state.U = np.zeros((current_id, current_id), dtype=np.float64)
    U_imu_init = np.eye(15, dtype=np.float64)
    U_imu_init[0, 0] = init_opts.init_prior_q
    U_imu_init[1, 1] = init_opts.init_prior_q
    U_imu_init[2, 2] = 0.001  # small prior for global yaw
    R_GtoI = state.imu.Rot()
    U_imu_init[:3, :3] = U_imu_init[:3, :3] @ R_GtoI.T

    U_imu_init[3:6, 3:6] = init_opts.init_prior_p * np.eye(3)
    U_imu_init[6:9, 6:9] = init_opts.init_prior_v * np.eye(3)
    U_imu_init[9:12, 9:12] = init_opts.init_prior_bg * np.eye(3)
    U_imu_init[12:15, 12:15] = init_opts.init_prior_ba * np.eye(3)

    efficient_QR(U_imu_init)
    state.U[:15, :15] = np.triu(U_imu_init)

    # Calibration priors.
    if state.options.do_calib_camera_timeoffset:
        state.U[state.calib_dt_CAMtoIMU.id(), state.calib_dt_CAMtoIMU.id()] = (
            init_opts.init_prior_t
        )

    if state.options.do_calib_camera_pose:
        for i in range(state.options.num_cameras):
            pid = state.calib_IMUtoCAM[i].id()
            state.U[pid:pid + 3, pid:pid + 3] = init_opts.init_prior_qc * np.eye(3)
            state.U[pid + 3:pid + 6, pid + 3:pid + 6] = init_opts.init_prior_pc * np.eye(3)

    if state.options.do_calib_camera_intrinsics:
        for i in range(state.options.num_cameras):
            iid = state.cam_intrinsics[i].id()
            state.U[iid:iid + 4, iid:iid + 4] = init_opts.init_prior_fc * np.eye(4)
            state.U[iid + 4:iid + 6, iid + 4:iid + 6] = init_opts.init_prior_dc1 * np.eye(2)
            state.U[iid + 6:iid + 8, iid + 6:iid + 8] = init_opts.init_prior_dc2 * np.eye(2)

    # Set calibration values.
    camimu_dt = np.array([init_opts.calib_camimu_dt], dtype=np.float64)
    state.calib_dt_CAMtoIMU.set_value(camimu_dt)
    state.calib_dt_CAMtoIMU.set_fej(camimu_dt)

    for i in range(init_opts.num_cameras):
        cam_value = init_opts.camera_intrinsics[i].get_value()
        state.cam_intrinsics[i].set_value(cam_value)
        state.cam_intrinsics[i].set_fej(cam_value)
        state.calib_IMUtoCAM[i].set_value(init_opts.camera_extrinsics[i])
        state.calib_IMUtoCAM[i].set_fej(init_opts.camera_extrinsics[i])

        # Instantiate the matching camera object.
        cam_intrinsic = init_opts.camera_intrinsics[i]
        cam_type = type(cam_intrinsic).__name__
        w = cam_intrinsic.w()
        h = cam_intrinsic.h()
        if cam_type == "CamEqui":
            from sqrtvins_core.cam.CamEqui import CamEqui
            cam = CamEqui(w, h)
        else:
            from sqrtvins_core.cam.CamRadtan import CamRadtan
            cam = CamRadtan(w, h)
        state.cam_intrinsics_cameras[i] = cam
        cam.set_value(cam_value)

    # Grow xk_minus_x0 / xk_minus_xk1 to the new size.
    n = state.U.shape[1]
    state.xk_minus_x0 = np.zeros((n, 1), dtype=np.float64)
    state.xk_minus_xk1 = np.zeros((n, 1), dtype=np.float64)
    state.is_initialized = True


def get_factors_for_slam_feature(
    state: State, H_order: List[Type], H: np.ndarray, HUT: np.ndarray
) -> None:
    """Compute H * U^T into HUT in place (StateHelper.cpp:874-886).

    HUT starts zeroed and is +=-ed for each variable in H_order.
    """
    H = np.asarray(H, dtype=np.float64)
    HUT = np.asarray(HUT, dtype=np.float64)
    kMeasSize = H.shape[0]

    local_id = 0
    for var in H_order:
        sz = var.size()
        v_id = var.id()
        HUT[:, :v_id + sz] += H[:, local_id:local_id + sz] @ state.U[:, v_id:v_id + sz].T
        local_id += sz
