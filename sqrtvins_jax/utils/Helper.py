"""
Helper — JAX port of ov_srvins/src/utils/Helper.cpp.

Everything in this file is plain numpy float64, mirroring the C++ line by
line. The kernels that carry the filter's numerical weight (the Givens /
Householder QRs, `matrix_multiplier_ATA`, the blocked triangular solvers)
follow the C++ exactly — including the `u = -u` sign flips in `makeGivens`
and the `if (c0 >= 0) beta = -beta` sign flip in `makeHouseholder` that
Eigen carries. Dropping either of those silently gives the wrong
null-space direction for the MSCKF update.

The signatures match the C++:
  * `reverse_mat(A, do_column=True)` — C++ counterintuitive: `do_column=True`
    reverses **rows**; `do_column=False` reverses **columns**.
  * `efficient_QR(Hx, r=None, Hf=None)` — dispatch: Givens if `Hf` is given,
    Householder otherwise.
  * `triangular_matrix_inverse_solver(U_transpose, X, op_rows=32)` — the
    caller passes `F.transpose()`, so the *view's* upper triangle is the
    *original's* lower triangle.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------
# Givens rotation (ported from Eigen's src/Jacobi/Jacobi.h).
#
# Eigen's convention: `makeGivens(p, q)` returns (c, s) such that the
# rotation's *adjoint*, applied to the column [p; q], annihilates q and puts
# +/-sqrt(p^2 + q^2) into p. The adjoint is `JacobiRotation(c, -s)` — NOT a
# c/s swap (that's the transpose, a different operation for complex scalars).
# Two branches: `|p|>|q|` and `|q|>=|p|`, each with its own normalization and
# a sign flip on `u` when the dominant entry is negative.
#
# Numerical check: makeGivens(3, 4) -> (c=0.6, s=-0.8). The adjoint
# application `J^* [3; 4]` with J^* = [[c, -s], [s, c]] gives
#   new_p =  c*3 + (-s)*4 = 1.8 + 3.2 =  5 = sqrt(25)
#   new_q =  s*3 +  c*4   = -2.4 + 2.4 =  0
# --------------------------------------------------------------------------

def make_givens(p: float, q: float) -> Tuple[float, float]:
    """Return (c, s) for the Givens rotation that annihilates q.

    Port of Eigen::JacobiRotation<Scalar>::makeGivens(p, q, nullptr,
    false_type). The `u = -u` when p<0 (resp. q<0) is not cosmetic — it
    makes the rotation orthonormal in both branches, not just in the
    `|p|>|q|` one.
    """
    if q == 0.0:
        c = -1.0 if p < 0.0 else 1.0
        return c, 0.0
    if p == 0.0:
        s = 1.0 if q < 0.0 else -1.0
        return 0.0, s
    if abs(p) > abs(q):
        t = q / p
        u = np.sqrt(1.0 + t * t)
        if p < 0.0:
            u = -u
        c = 1.0 / u
        s = -t * c
        return c, s
    # else: |q| >= |p|
    t = p / q
    u = np.sqrt(1.0 + t * t)
    if q < 0.0:
        u = -u
    s = -1.0 / u
    c = -t * s
    return c, s


def apply_givens_adjoint(
    mat: np.ndarray,
    p: int,
    q: int,
    c: float,
    s: float,
    start_col: int,
    end_col: int,
) -> None:
    """Apply Eigen's `.adjoint()` Givens rotation in place to rows p and q
    of `mat` over columns `[start_col, end_col)`.

    Eigen's `JacobiRotation::adjoint()` returns `JacobiRotation(c, -s)` —
    the sign flip on `s`, not a c/s swap. The rotation itself applies as
        new_p =  c * x_p + (-s) * x_q
        new_q =  s * x_p +   c * x_q
    which, applied to `[p; q] = [3; 4]` with `(c, s) = (0.6, -0.8)`,
    gives `[5; 0]` — the subdiagonal entry is annihilated.
    """
    x_p = mat[p, start_col:end_col].copy()
    x_q = mat[q, start_col:end_col].copy()
    mat[p, start_col:end_col] = c * x_p - s * x_q
    mat[q, start_col:end_col] = s * x_p + c * x_q


# --------------------------------------------------------------------------
# Householder reflection (ported from Eigen's src/Householder/Householder.h).
# --------------------------------------------------------------------------

def make_householder(
    x: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    """Return (essential, tau, beta) for a Householder reflection on the
    vector `x`.

    Port of `MatrixBase::makeHouseholder(essential, tau, beta)` with the
    `if (real(c0) >= 0) beta = -beta` sign convention. `essential` is the
    `(n-1)`-vector of reflection coefficients; `tau` is the Householder
    scalar; `beta` is the resulting diagonal entry.

    Caller: `makeHouseholder(x)` is invoked on a column segment via
    `Hx.col(k).segment(k, op_rows)` in the C++ — in numpy that's
    `Hx[k:k+op_rows, k]`.
    """
    size = x.shape[0]
    tail_sq_norm = 0.0 if size == 1 else float(np.dot(x[1:], x[1:]))
    c0 = x[0]
    tol = np.finfo(np.float64).tiny
    if tail_sq_norm <= tol and abs(c0) <= tol:
        essential = np.zeros(max(size - 1, 0), dtype=np.float64)
        return essential, 0.0, c0
    beta = np.sqrt(c0 * c0 + tail_sq_norm)
    if c0 >= 0.0:
        beta = -beta
    essential = x[1:] / (c0 - beta)
    tau = (beta - c0) / beta
    return essential.astype(np.float64), float(tau), float(beta)


def apply_householder_on_left(
    block: np.ndarray,
    essential: np.ndarray,
    tau: float,
) -> None:
    """Apply a Householder reflection on the left of `block` in place.

    Port of `MatrixBase::applyHouseholderOnTheLeft(essential, tau,
    workspace)`. `block` is the (op_rows, cols-k) block whose top row is
    the pivot row. `essential` is the (op_rows-1)-vector from
    `make_householder`; `tau` is the Householder scalar.
    """
    if tau == 0.0:
        return
    row_view = block[0, :]
    bottom_view = block[1:, :]
    # tmp = essential^T @ bottom  (fresh, not a view)
    tmp = essential @ bottom_view
    tmp += row_view
    row_view -= tau * tmp
    bottom_view -= tau * np.outer(essential, tmp)


# --------------------------------------------------------------------------
# Row / column reversal (ported from Helper.h:117-123).
#
# The C++ signature is counterintuitive: `reverse_mat(A, do_column=True)`
# reverses *rows*; `do_column=False` reverses *columns*. Do not "fix"
# this — every call site in StateHelper.cpp relies on the convention.
# --------------------------------------------------------------------------

def reverse_mat(A: np.ndarray, do_column: bool = True) -> np.ndarray:
    """In-place reversal, matching C++ `reverse_mat(A, do_column=true)`.

    The C++ flag naming is a historical OpenVINS misnomer:

    - ``do_column=True`` (default): ``A.rowwise().reverseInPlace()``
      reverses elements **within each row**, i.e. reverses the
      **columns** (``A[:, ::-1]``, axis 1).
    - ``do_column=False``: ``A.colwise().reverseInPlace()`` reverses
      elements **within each column**, i.e. reverses the **rows**
      (``A[::-1, :]``, axis 0).

    The parameter name and its meaning are kept identical to the C++ so
    that the port of `update_llt` reads line-for-line the same. Uses
    `np.flip` + `copyto` because in-place slice assignment on a strided
    view silently no-ops.
    """
    if do_column:
        np.copyto(A, np.flip(A, axis=1))
    else:
        np.copyto(A, np.flip(A, axis=0))
    return A


def reverse_vec(v: np.ndarray) -> np.ndarray:
    np.copyto(v, np.flip(v))
    return v


# --------------------------------------------------------------------------
# Permutation matrix (ported from Helper.cpp:210-254).
#
# Groups rows of Hx by the index of the first nonzero column, then appends
# all-zero rows to the last group. Used to make the QR staircase explicit
# so the Householder / Givens sweeps don't have to search for the first
# nonzero entry at every step.
# --------------------------------------------------------------------------

def get_permutation_matrix(Hx: np.ndarray) -> np.ndarray:
    """Return the row permutation indices for `Hx`.

    Rows are grouped by the column of their first nonzero entry (threshold
    `np.finfo(float64).tiny` — the C++ uses `std::numeric_limits<double>::min()`,
    which is the smallest positive normal; numpy's `.min` is the most-negative
    finite value, which would treat every entry as nonzero). All-zero rows
    are appended to the `n_cols - 1` group (matching the C++ quirk — the
    extra group at index `n_cols` is unused). Returns a 1-D int64 array
    `idx` such that `P.transpose() @ Hx == Hx[idx, :]`.
    """
    n_rows, n_cols = Hx.shape
    row_ids: List[List[int]] = [[] for _ in range(n_cols + 1)]
    zero_block_ids: List[int] = []
    tol = np.finfo(np.float64).tiny
    for i in range(n_rows):
        is_zero_block = True
        for j in range(n_cols):
            if abs(Hx[i, j]) > tol:
                row_ids[j].append(i)
                is_zero_block = False
                break
        if is_zero_block:
            zero_block_ids.append(i)
    row_ids[n_cols - 1].extend(zero_block_ids)

    order_index = np.empty(n_rows, dtype=np.int64)
    curr_row_id = 0
    for group in row_ids:
        for rid in group:
            order_index[curr_row_id] = rid
            curr_row_id += 1
    return order_index


# --------------------------------------------------------------------------
# efficient_QR — three overloads, dispatched by signature.
#
# C++:
#   efficient_QR(Hx, r, Hf)     — Givens, MSCKF-style null-space projection
#   efficient_QR(Hx, r)         — Householder, with residual
#   efficient_QR(A)             — Householder, no residual
#   efficient_QR_givens(A)      — Givens variant (row-major copy + swap)
#
# We keep all four here; `efficient_QR(Hx, r=None, Hf=None)` dispatches
# on which optional args are supplied.
# --------------------------------------------------------------------------

def efficient_QR(Hx: np.ndarray, r: Optional[np.ndarray] = None,
                 Hf: Optional[np.ndarray] = None) -> None:
    """Dispatch to the right QR variant.

    * `efficient_QR(Hx, r, Hf)` — Givens, all three matrices.
    * `efficient_QR(Hx, r)`     — Householder with a residual column.
    * `efficient_QR(Hx)`        — Householder, no residual.

    In-place; callers must pass a writable numpy array.
    """
    assert Hx.shape[0] > 1, "efficient_QR: need rows > 1"
    if Hf is not None:
        assert r is not None
        assert Hx.shape[0] == r.shape[0]
        assert Hx.shape[0] == Hf.shape[0]
        _efficient_QR_givens_3arg(Hx, r, Hf)
    elif r is not None:
        assert Hx.shape[0] == r.shape[0]
        _efficient_QR_householder_with_r(Hx, r)
    else:
        _efficient_QR_householder(Hx)


def _efficient_QR_givens_3arg(Hx: np.ndarray, r: np.ndarray,
                              Hf: np.ndarray) -> None:
    """3-arg Givens QR (Helper.cpp:256-295). Used by MSCKF null-space
    projection: rows of Hx are the state Jacobian, Hf is the feature-state
    Jacobian, and the sweep annihilates Hf's subdiagonal while keeping Hx
    in step.

    The C++ permutes Hx's rows in place via `P.transpose().applyThisOnTheLeft`
    and then sweeps. Python can't rebind the caller's local, so we permute
    by copying the permuted rows back into the input buffers before the
    sweep.
    """
    idx = get_permutation_matrix(Hx)
    # Permute in place: overwrite caller's buffers with the permuted view.
    Hx_p = Hx[idx, :].copy()
    r_p = r[idx, :].copy()
    Hf_p = Hf[idx, :].copy()
    Hx[...] = Hx_p
    r[...] = r_p
    Hf[...] = Hf_p
    # -- sweep
    n_hf_rows, n_hf_cols = Hf.shape
    n_hx_cols = Hx.shape[1]
    tol = np.finfo(np.float64).tiny
    for n in range(n_hf_cols):
        for m in range(n_hf_rows - 1, n, -1):
            c, s = make_givens(Hf[m - 1, n], Hf[m, n])
            apply_givens_adjoint(Hf, m - 1, m, c, s, n, n_hf_cols)
            # Skip leading-zero columns of Hx before applying.
            start_col = n_hx_cols
            for sc in range(n_hx_cols):
                if abs(Hx[m - 1, sc]) > tol:
                    start_col = sc
                    break
            if start_col < n_hx_cols:
                apply_givens_adjoint(Hx, m - 1, m, c, s, start_col, n_hx_cols)
            apply_givens_adjoint(r, m - 1, m, c, s, 0, 1)


def _efficient_QR_householder_with_r(Hx: np.ndarray, r: np.ndarray) -> None:
    """Householder QR with a residual column (Helper.cpp:297-338)."""
    idx = get_permutation_matrix(Hx)
    n_rows, n_cols = Hx.shape
    total_op_cols = min(n_rows, n_cols)
    tol = np.finfo(np.float64).tiny
    # Apply permutation in place by copying the permuted slice back.
    Hx_perm = Hx[idx, :].copy()
    r_perm = r[idx, :].copy()
    Hx[...] = Hx_perm
    r[...] = r_perm
    for k in range(total_op_cols):
        # Find the first nonzero entry in column k starting from the bottom.
        j = n_rows - 1
        op_rows = 1
        for jj in range(n_rows - 1, k, -1):
            if abs(Hx[jj, k]) > tol:
                op_rows = jj - k + 1
                break
        if op_rows == 1:
            continue
        seg = Hx[k:k + op_rows, k]
        essential, tau, _beta = make_householder(seg)
        block = Hx[k:k + op_rows, k:n_cols]
        apply_householder_on_left(block, essential, tau)
        apply_householder_on_left(r[k:k + op_rows, :], essential, tau)


def _efficient_QR_householder(Hx: np.ndarray) -> None:
    """Householder QR on a single matrix (Helper.cpp:342-377)."""
    idx = get_permutation_matrix(Hx)
    n_rows, n_cols = Hx.shape
    total_op_cols = min(n_rows, n_cols)
    tol = np.finfo(np.float64).tiny
    Hx_perm = Hx[idx, :].copy()
    Hx[...] = Hx_perm
    for k in range(total_op_cols):
        j = n_rows - 1
        op_rows = 1
        for jj in range(n_rows - 1, k, -1):
            if abs(Hx[jj, k]) > tol:
                op_rows = jj - k + 1
                break
        if op_rows == 1:
            continue
        seg = Hx[k:k + op_rows, k]
        essential, tau, _beta = make_householder(seg)
        apply_householder_on_left(Hx[k:k + op_rows, k:n_cols], essential, tau)


def efficient_QR_givens(A: np.ndarray) -> np.ndarray:
    """Givens QR on a single matrix, row-major copy + swap (Helper.cpp:379-400).

    This variant zeroes already-zero entries before the sweep (the C++ does
    it explicitly to skip those iterations). Returns the new row-major
    array; the caller's `A` is left untouched.
    """
    assert A.shape[0] > 1, "efficient_QR_givens: need rows > 1"
    idx = get_permutation_matrix(A)
    A_tmp = np.ascontiguousarray(A[idx, :].copy(), dtype=np.float64)
    tol = np.finfo(np.float64).tiny
    n_rows, n_cols = A_tmp.shape
    for n in range(n_cols):
        for m in range(n_rows - 1, n, -1):
            if abs(A_tmp[m, n]) < tol:
                A_tmp[m, n] = 0.0
                continue
            c, s = make_givens(A_tmp[m - 1, n], A_tmp[m, n])
            apply_givens_adjoint(A_tmp, m - 1, m, c, s, n, n_cols)
    return A_tmp


# --------------------------------------------------------------------------
# matrix_multiplier_ATA (Helper.cpp:506-542).
#
# Computes A^T @ A via a blockwise rank-1-ish staircase. The internal
# permutation is mathematically invisible (P^T A)^T (P^T A) = A^T A, but
# the C++ writes the result into ATA[i:num_cols, i:num_cols] in a
# lower-right-corner structure and exploits sparsity via the exact
# `!= 0.0` compare. We mirror that faithfully.
# --------------------------------------------------------------------------

def matrix_multiplier_ATA(A: np.ndarray, ATA: np.ndarray) -> None:
    """Compute A^T @ A into `ATA` in place, using a staircase rank-1 update.

    Port of Helper.cpp:506-542. `A` is permuted row-wise first; then a
    sweep over columns `i` groups the nonzero rows of column `i` into a
    block, and `ATA[i:num_cols, i:num_cols] += sub^T @ sub`.
    """
    A_perm = A[get_permutation_matrix(A), :]
    n_rows, n_cols = A_perm.shape
    curr_row_id = 0
    op_rows = 0
    op_cols = 0
    for i in range(n_cols):
        if op_rows == 0:
            op_cols = n_cols - i
        # Count nonzero rows of column i starting from curr_row_id + op_rows.
        j = curr_row_id + op_rows
        while j < n_rows and A_perm[j, i] != 0.0:
            op_rows += 1
            j += 1
        if op_rows == 0:
            continue
        # sub = A_perm[curr_row_id : curr_row_id + op_rows, i : n_cols]
        sub = A_perm[curr_row_id:curr_row_id + op_rows, i:n_cols]
        # ATA[i : n_cols, i : n_cols] += sub^T @ sub  (symmetric, upper tri)
        ATA[i:n_cols, i:n_cols] += sub.T @ sub
        curr_row_id += op_rows
        op_rows = 0


# --------------------------------------------------------------------------
# triangular_matrix_inverse_solver (Helper.cpp:410-450).
#
# The caller passes `F.transpose()` — a Transpose view of `F`. In the
# update_llt call site, `F` has been reversed both rows and columns
# (`reverse_mat(F)` then `reverse_mat(F, false)`), which turns an
# upper-triangular Cholesky factor into a LOWER-triangular matrix whose
# entries run from the antidiagonal inward. Therefore `F.transpose()`
# is UPPER-triangular, and its upper-triangle equals itself.
#
# Eigen's `F.transpose().triangularView<Upper>()` accesses `F(j, i)` for
# i <= j — since F is lower-triangular, this picks up F's own lower
# triangle. For our usage (F lower after reversal), the solver's
# coefficient matrix is exactly `F.transpose()`.
#
# X is required to be square n×n (the caller passes
# `state->U_.topLeftCorner(n, n)`), because the subtraction step uses
# `X.bottomRightCorner(op_rows*(i+1), op_rows*(i+1))`, which is only a
# valid square block when X has n columns. X remains upper-triangular
# throughout: the diagonal-block solve preserves the structure, and the
# subtraction only touches X's upper-triangular region.
# --------------------------------------------------------------------------

def triangular_matrix_inverse_solver(
    U: np.ndarray,
    X: np.ndarray,
    op_rows: int = 32,
) -> None:
    """Blocked upper-triangular solve: `X = U^{-1} X` in place.

    Mirrors Helper.cpp:410-450. `U` is upper-triangular (this is the
    `F.transpose()` that `update_llt` passes); `X` is a square n×n
    matrix (the caller's `topLeftCorner`). Iterates from the bottom-right
    diagonal block up:

    * Solve the block-diagonal block:
      `X[s:s+op, s:s+op*(i+1)] = U[s:s+op, s:s+op]^{-1} * X[s:s+op, s:s+op*(i+1)]`.
    * Subtract the influence of the (now solved) bottom-right corner on
      the block immediately above:
      `X[s-op:s, s:s+op*(i+1)] -= U[s-op:s, s:s+op*(i+1)] @ X[s:n, s:n]`.
    * Decrement s by op, increment i.

    X remains upper-triangular throughout — both operations only write
    into its upper-triangular region.
    """
    rows = U.shape[0]
    assert rows == X.shape[0] == X.shape[1], (
        "triangular_matrix_inverse_solver requires square X (matches C++ "
        "caller's topLeftCorner). Got U=%s, X=%s" % (U.shape, X.shape)
    )
    row_start = rows - op_rows
    op_rows_first_block = rows - (rows // op_rows) * op_rows
    i = 0
    while True:
        if row_start >= 0:
            U_diag = U[row_start:row_start + op_rows, row_start:row_start + op_rows]
            X_solve = X[row_start:row_start + op_rows,
                        row_start:row_start + op_rows * (i + 1)]
            X_solve[:] = _solve_upper_inplace(U_diag, X_solve)
        else:
            if op_rows_first_block:
                U_diag = U[:op_rows_first_block, :op_rows_first_block]
                X_first = X[:op_rows_first_block, :rows]
                X_first[:] = _solve_upper_inplace(U_diag, X_first)
            break
        if row_start - op_rows >= 0:
            X[row_start - op_rows:row_start,
              row_start:row_start + op_rows * (i + 1)] -= (
                U[row_start - op_rows:row_start,
                  row_start:row_start + op_rows * (i + 1)]
                @ X[row_start:rows, row_start:rows]
            )
        else:
            if op_rows_first_block:
                X[:op_rows_first_block,
                  row_start:row_start + op_rows * (i + 1)] -= (
                    U[:op_rows_first_block,
                      row_start:row_start + op_rows * (i + 1)]
                    @ X[row_start:rows, row_start:rows]
                )
        row_start -= op_rows
        i += 1


def _solve_upper_inplace(U_block: np.ndarray, X_block: np.ndarray) -> np.ndarray:
    """Solve `U x = X` for an upper-triangular `U_block`, returning the
    result (X_block is overwritten in place as well)."""
    from scipy.linalg import solve_triangular
    return solve_triangular(U_block, X_block, lower=False, check_finite=False)


# --------------------------------------------------------------------------
# triangular_matrix_multiplier_UU (Helper.cpp:452-476).
#
# Computes `U1_U2 = U1 @ U2` for two upper-triangular matrices, blocked
# for cache locality. The C++ iterates by diagonal blocks: for each
# diagonal block (row_start, row_start, op_rows, op_rows), it computes
# the upper-triangular inner product and adds the off-diagonal
# contribution from the strictly-lower-right part of U1.
# --------------------------------------------------------------------------

def triangular_matrix_multiplier_UU(
    U1: np.ndarray,
    U2: np.ndarray,
    U1_U2: np.ndarray,
    op_rows: int = 32,
) -> None:
    """Compute `U1_U2 = U1 @ U2` for upper-triangular U1, U2 in place."""
    rows = U1.shape[0]
    row_start = 0
    while row_start != rows:
        cur_op = op_rows
        if row_start + cur_op > rows:
            cur_op = rows - row_start
        # U1_U2[row_start:row_start+cur_op, row_start:rows] =
        #   U1[row_start:row_start+cur_op, row_start:row_start+cur_op] (upper) @
        #   U2[row_start:row_start+cur_op, row_start:rows]
        U1_U2[row_start:row_start + cur_op, row_start:rows] = (
            U1[row_start:row_start + cur_op, row_start:row_start + cur_op]
            @ U2[row_start:row_start + cur_op, row_start:rows]
        )
        if rows - row_start - cur_op != 0:
            # Add the off-diagonal contribution:
            #   U1[row_start:row_start+cur_op, row_start+cur_op:rows] @
            #   U2[row_start+cur_op:rows, row_start+cur_op:rows] (upper)
            U1_U2[row_start:row_start + cur_op, row_start:rows - cur_op] += (
                U1[row_start:row_start + cur_op, row_start + cur_op:rows]
                @ U2[row_start + cur_op:rows, row_start + cur_op:rows]
            )
        row_start += cur_op


# --------------------------------------------------------------------------
# triangular_matrix_multiplier_LLT (Helper.cpp:478-504).
#
# Computes `LLT = L @ L^T` for lower-triangular L, blocked. The C++
# iterates by diagonal blocks of L: for each block, the left-outer
# contribution and the lower-outer contribution are added to the
# block's middle-rows x middle-cols.
# --------------------------------------------------------------------------

def triangular_matrix_multiplier_LLT(
    L: np.ndarray,
    LLT: np.ndarray,
    op_rows: int = 32,
) -> None:
    """Compute `LLT = L @ L^T` for lower-triangular L in place."""
    rows = L.shape[0]
    LT = L.T
    row_start = 0
    while row_start != rows:
        cur_op = op_rows
        if row_start + cur_op > rows:
            cur_op = rows - row_start
        if row_start != 0:
            # Left-outer contribution: L[row_start:, :row_start] @ L[:row_start, :row_start]^T
            LLT[row_start:row_start + cur_op, :row_start] = (
                L[row_start:row_start + cur_op, :row_start]
                @ LT[:row_start, :row_start]
            )
            # Middle-outer contribution:
            #   L[row_start:, :row_start] @ L[:row_start, row_start:row_start+cur_op]
            LLT[row_start:row_start + cur_op, row_start:row_start + cur_op] = (
                L[row_start:row_start + cur_op, :row_start]
                @ LT[:row_start, row_start:row_start + cur_op]
            )
        # Diagonal contribution (lower-triangular block).
        LLT[row_start:row_start + cur_op, row_start:row_start + cur_op] += (
            L[row_start:row_start + cur_op, row_start:row_start + cur_op]
            @ LT[row_start:row_start + cur_op, row_start:row_start + cur_op]
        )
        row_start += cur_op
    # The C++ finalizes with selfadjointView<Lower>: LLT = LLT.tril().
    # We mirror that: only the lower triangle is meaningful.
    LLT[:] = np.tril(LLT)


# --------------------------------------------------------------------------
# get_condition_number (Helper.cpp:402-408)
# --------------------------------------------------------------------------

def get_condition_number(A: np.ndarray) -> float:
    """Return the 2-norm condition number via BDCSVD (SVD in numpy)."""
    A64 = A.astype(np.float64)
    sv = np.linalg.svd(A64, compute_uv=False)
    return float(sv[0] / sv[-1])


# --------------------------------------------------------------------------
# gram_schmidt (Helper.cpp:164-192)
# --------------------------------------------------------------------------

def gram_schmidt(gravity_inI: np.ndarray, R_GtoI: np.ndarray) -> None:
    """Build an orthonormal (x, y, z) frame from a gravity direction.

    Mirrors Helper.cpp:164-192. `z_axis` is the gravity unit vector; the
    `x_axis` is chosen perpendicular to `z_axis` (picking the world
    frame axis that's least aligned with gravity), and `y_axis` completes
    the right-handed frame. Writes the 3 columns of `R_GtoI` in place.
    """
    z_axis = gravity_inI / np.linalg.norm(gravity_inI)
    # World frame basis (in the G frame): e1 = (1, 0, 0), e2 = (0, 1, 0).
    e1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    e2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    inner1 = float(np.dot(z_axis, e1))
    inner2 = float(np.dot(z_axis, e2))
    if abs(inner1) < abs(inner2):
        e = e1
    else:
        e = e2
    x_axis = np.cross(z_axis, e)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    R_GtoI[:, 0] = x_axis
    R_GtoI[:, 1] = y_axis
    R_GtoI[:, 2] = z_axis


# --------------------------------------------------------------------------
# get_gravity / get_gravity_Jacobian (Helper.cpp:544-561)
# --------------------------------------------------------------------------

def get_gravity(alpha: float, beta: float, mag: float) -> np.ndarray:
    """Return the gravity vector given two angles and a magnitude.

    Port of Helper.cpp:556-561:
        return Vec3(cos alpha * sin beta, sin alpha * sin beta, cos beta) * mag
    """
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    return np.array([ca * sb, sa * sb, cb], dtype=np.float64) * mag


def get_gravity_Jacobian(alpha: float, beta: float, mag: float) -> np.ndarray:
    """Return the 3x2 Jacobian of `get_gravity` w.r.t. (alpha, beta).

    Port of Helper.cpp:544-554.
    """
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    J = np.array(
        [
            [-sa * sb, ca * cb],
            [ ca * sb, sa * cb],
            [        0.0, -sb],
        ],
        dtype=np.float64,
    )
    return J * mag


# --------------------------------------------------------------------------
# visualize_matrix_structure (Helper.cpp) — debug only.
# --------------------------------------------------------------------------

def visualize_matrix_structure(A: np.ndarray) -> None:
    """Print a text map of the nonzero/zero structure of `A`."""
    tol = np.finfo(np.float64).tiny
    rows, cols = A.shape
    for i in range(rows):
        row = ""
        for j in range(cols):
            row += "#" if abs(A[i, j]) > tol else "."
        print(row)


# --------------------------------------------------------------------------
# select_imu_readings / interpolate_data — sensor utilities (Phase 4).
# Included here so that Phase 4 consumers can find them.
# --------------------------------------------------------------------------

def interpolate_data(
    imu_1: "ImuData",
    imu_2: "ImuData",
    timestamp: float,
) -> "ImuData":
    """Linearly interpolate between two IMU readings at `timestamp`.

    Port of Helper.cpp:691-704. `ImuData` is duck-typed — anything with
    `timestamp`, `am`, and `wm` attributes works.
    """
    from dataclasses import dataclass, fields
    try:
        # Reuse the caller's class if possible.
        cls = type(imu_1)
        fields_meta = {f.name: f for f in fields(cls)}
        if "timestamp" in fields_meta and "am" in fields_meta and "wm" in fields_meta:
            lambda_ = (timestamp - imu_1.timestamp) / (imu_2.timestamp - imu_1.timestamp)
            return cls(
                timestamp=timestamp,
                am=(1 - lambda_) * imu_1.am + lambda_ * imu_2.am,
                wm=(1 - lambda_) * imu_1.wm + lambda_ * imu_2.wm,
            )
    except Exception:
        pass
    # Fallback: return a plain dict-like namedtuple via SimpleNamespace.
    from types import SimpleNamespace
    lambda_ = (timestamp - imu_1.timestamp) / (imu_2.timestamp - imu_1.timestamp)
    return SimpleNamespace(
        timestamp=timestamp,
        am=(1 - lambda_) * np.asarray(imu_1.am, dtype=np.float64) + lambda_ * np.asarray(imu_2.am, dtype=np.float64),
        wm=(1 - lambda_) * np.asarray(imu_1.wm, dtype=np.float64) + lambda_ * np.asarray(imu_2.wm, dtype=np.float64),
    )


# A minimal stand-in for `ImuData` in the type annotation above. The real
# one lives in `sqrtvins_core/utils/sensor_data.py`; we import it lazily
# so this module doesn't have to depend on it.
from types import SimpleNamespace as ImuData  # noqa: E402
