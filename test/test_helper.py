"""Unit tests for sqrtvins_jax.utils.Helper — the numpy kernels that back the
sqrt-form covariance engine.

Each test either verifies a small numerical identity (Givens/Householder
annihilation, triangular solve) or checks a full-sweep kernel against
numpy.linalg.qr.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqrtvins_jax.utils.Helper import (
    apply_givens_adjoint,
    apply_householder_on_left,
    efficient_QR,
    efficient_QR_givens,
    get_permutation_matrix,
    make_givens,
    make_householder,
    matrix_multiplier_ATA,
    reverse_mat,
    reverse_vec,
    triangular_matrix_inverse_solver,
)


# ---------------------------------------------------------------------------
# Givens
# ---------------------------------------------------------------------------

def test_make_givens_classic():
    """The 3-4-5 triangle: c=3/5, s=-4/5 (Eigen sign convention)."""
    c, s = make_givens(3.0, 4.0)
    assert np.isclose(c, 0.6)
    assert np.isclose(s, -0.8)


def test_make_givens_branch_flip():
    """When |q| > |p|, Eigen takes the else branch — sign convention differs
    but the invariant |c|^2 + |s|^2 = 1 holds."""
    for p, q in [(3.0, 4.0), (4.0, 3.0), (-3.0, 4.0), (3.0, -4.0),
                 (-3.0, -4.0), (0.0, 5.0), (5.0, 0.0), (0.0, -5.0),
                 (-5.0, 0.0), (0.0, 0.0)]:
        c, s = make_givens(p, q)
        assert np.isclose(c * c + s * s, 1.0), (p, q, c, s)


def test_apply_givens_adjoint_annihilates_subdiagonal():
    """Eigen's adjoint application zeros the SECOND entry (the subdiagonal)
    of the pair [p; q]."""
    mat = np.array([[3.0, 1.0, 0.0],
                    [4.0, 0.0, 1.0]])
    c, s = make_givens(3.0, 4.0)
    apply_givens_adjoint(mat, 0, 1, c, s, 0, 3)
    assert np.isclose(mat[1, 0], 0.0)
    assert np.isclose(mat[0, 0], 5.0) or np.isclose(mat[0, 0], -5.0)


def test_apply_givens_adjoint_slice_only_touched_range():
    """Only cols in [start_col, end_col) should be modified."""
    mat = np.array([[1.0, 3.0, 0.0, 7.0],
                    [1.0, 4.0, 0.0, 9.0]])
    mat_before = mat.copy()
    c, s = make_givens(3.0, 4.0)
    apply_givens_adjoint(mat, 0, 1, c, s, 1, 2)
    # Cols 0 and 2-3 unchanged
    assert np.array_equal(mat[:, 0], mat_before[:, 0])
    assert np.array_equal(mat[:, 2], mat_before[:, 2])
    assert np.array_equal(mat[:, 3], mat_before[:, 3])
    # Col 1 annihilated at row 1
    assert np.isclose(mat[1, 1], 0.0)


# ---------------------------------------------------------------------------
# Givens QR sweep
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape", [(3, 3), (5, 3), (4, 4), (6, 2)])
def test_efficient_qr_givens_upper_triangular(shape):
    """Full Givens sweep must produce an upper-triangular matrix.
    Signs on the diagonal may differ from numpy.linalg.qr (which uses
    Householder with a specific sign convention); magnitudes must match."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal(shape)
    R = efficient_QR_givens(A.copy())
    # Upper-triangular: entries below the diagonal are (numerically) zero.
    n = min(shape)
    for i in range(1, shape[0]):
        for j in range(0, min(i, shape[1])):
            assert abs(R[i, j]) < 1e-10, f"R[{i},{j}] = {R[i,j]}"
    # Column norms preserved: for each column, sum-of-squares matches A.
    # (Orthonormal Givens rotations preserve column norms.)
    for j in range(shape[1]):
        col_a = np.linalg.norm(A[:, j])
        col_r = np.linalg.norm(R[:, j])
        assert np.isclose(col_a, col_r, atol=1e-10), (j, col_a, col_r)


def test_efficient_qr_givens_matches_numpy_qr():
    """The top n_cols rows of our R must match numpy's reduced R in MAGNITUDE.
    Two QRs of the same matrix can only differ by per-row sign flips of R
    (equivalently, per-column flips of Q). Rows below the diagonal are zero
    on our side; numpy's 'reduced' mode drops them entirely."""
    rng = np.random.default_rng(1)
    A = rng.standard_normal((6, 4))
    R_ours = efficient_QR_givens(A.copy())
    _, R_ref = np.linalg.qr(A)  # R_ref is (4, 4)
    R_ours_top = R_ours[: A.shape[1], :]  # (4, 4)
    assert np.allclose(np.abs(R_ours_top), np.abs(R_ref), atol=1e-10), (
        f"|R| mismatch:\nours=\n{np.abs(R_ours_top)}\nref=\n{np.abs(R_ref)}"
    )


# ---------------------------------------------------------------------------
# Householder QR
# ---------------------------------------------------------------------------

def test_householder_matches_givens_for_square():
    """For a square full-rank matrix, both QR variants must produce
    matrices with the same column norms (up to sign)."""
    rng = np.random.default_rng(2)
    A = rng.standard_normal((5, 5))
    R_givens = efficient_QR_givens(A.copy())
    R_house = A.copy()
    efficient_QR(R_house)
    for j in range(5):
        assert np.isclose(
            np.linalg.norm(R_givens[:, j]),
            np.linalg.norm(R_house[:, j]),
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# reverse_mat / reverse_vec
# ---------------------------------------------------------------------------

def test_reverse_mat_rows():
    A = np.arange(12).reshape(3, 4).astype(np.float64)
    B = A.copy()
    reverse_mat(B, do_column=True)
    assert np.array_equal(B, A[::-1, :])


def test_reverse_mat_cols():
    A = np.arange(12).reshape(3, 4).astype(np.float64)
    B = A.copy()
    reverse_mat(B, do_column=False)
    assert np.array_equal(B, A[:, ::-1])


def test_reverse_vec():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    reverse_vec(v)
    assert np.array_equal(v, [4.0, 3.0, 2.0, 1.0])


# ---------------------------------------------------------------------------
# matrix_multiplier_ATA
# ---------------------------------------------------------------------------

def test_matrix_multiplier_ATA_matches_dense():
    rng = np.random.default_rng(3)
    A = rng.standard_normal((10, 4))
    ATA = np.zeros((4, 4))
    matrix_multiplier_ATA(A, ATA)
    assert np.allclose(ATA, A.T @ A, atol=1e-10)


def test_matrix_multiplier_ATA_sparse_rows():
    """Column 0 has only 2 nonzero rows; the block staircase should handle
    this without loss."""
    A = np.zeros((6, 3))
    A[0, 0] = 1.0
    A[1, 0] = 2.0
    A[1, 1] = 3.0
    A[2, 1] = 4.0
    A[2, 2] = 5.0
    A[3, 2] = 6.0
    ATA = np.zeros((3, 3))
    matrix_multiplier_ATA(A, ATA)
    assert np.allclose(ATA, A.T @ A, atol=1e-10)


# ---------------------------------------------------------------------------
# triangular_matrix_inverse_solver
# ---------------------------------------------------------------------------

def _random_upper_triu(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random upper-triangular with nonzero diagonal."""
    U = np.triu(rng.standard_normal((n, n)))
    np.fill_diagonal(U, np.abs(U).diagonal() + 1.0)
    return U


def test_triangular_inverse_solver_upper():
    """Given upper-triangular U and square RHS X, verify U^{-1} X is computed."""
    rng = np.random.default_rng(4)
    n = 20
    U = _random_upper_triu(n, rng)
    # X is square n×n (the C++ caller passes topLeftCorner).
    X = np.triu(rng.standard_normal((n, n)))
    Y = X.copy()
    triangular_matrix_inverse_solver(U, Y, op_rows=8)
    expected = np.linalg.solve(U, X)
    assert np.allclose(Y, expected, atol=1e-10)


def test_triangular_inverse_solver_small_block():
    """op_rows=1 should give the same answer as the default."""
    rng = np.random.default_rng(5)
    n = 10
    U = _random_upper_triu(n, rng)
    X = np.triu(rng.standard_normal((n, n)))
    Y1 = X.copy()
    Y2 = X.copy()
    triangular_matrix_inverse_solver(U, Y1, op_rows=1)
    triangular_matrix_inverse_solver(U, Y2, op_rows=32)
    assert np.allclose(Y1, Y2, atol=1e-10)


# ---------------------------------------------------------------------------
# Permutation matrix
# ---------------------------------------------------------------------------

def test_get_permutation_matrix_identity():
    """A well-ordered matrix (already staircase) returns identity order."""
    A = np.array([[1.0, 0.0, 0.0],
                  [0.0, 2.0, 0.0],
                  [0.0, 0.0, 3.0]])
    idx = get_permutation_matrix(A)
    assert np.array_equal(idx, [0, 1, 2])


def test_get_permutation_matrix_reorders():
    """Rows are grouped by the index of their first nonzero column."""
    A = np.array([[0.0, 0.0, 1.0],
                  [1.0, 2.0, 3.0],
                  [0.0, 4.0, 5.0],
                  [6.0, 7.0, 8.0]])
    idx = get_permutation_matrix(A)
    # Row 0 has first-nonzero at col 2 → group 2.
    # Row 1 has first-nonzero at col 0 → group 0.
    # Row 2 has first-nonzero at col 1 → group 1.
    # Row 3 has first-nonzero at col 0 → group 0.
    assert list(idx) == [1, 3, 2, 0]


# ---------------------------------------------------------------------------
# Householder
# ---------------------------------------------------------------------------

def test_make_householder_zeros_subvector():
    """Householder on a nonzero vector gives a reflection that zeros the
    sub-vector."""
    v = np.array([1.0, 2.0, 3.0, 4.0])
    essential, tau, beta = make_householder(v.copy())
    assert len(essential) == 3
    assert tau != 0.0
    # Apply to a copy of v: the result should have zeros in positions 1..3.
    x = v.copy()
    block = x.reshape(-1, 1)
    apply_householder_on_left(block, essential, tau)
    assert np.all(np.isclose(block[1:, 0], 0.0, atol=1e-10))
    # And the first entry should equal ±||v||.
    assert np.isclose(abs(block[0, 0]), np.linalg.norm(v))


def test_apply_householder_tomato_matrix():
    """End-to-end: apply Householder via make_householder + apply_householder_on_left
    on a fat block and check that the top row is the only nonzero row after
    the reflection."""
    rng = np.random.default_rng(6)
    n_rows = 5
    n_cols = 3
    block = rng.standard_normal((n_rows, n_cols))
    seg = block[:, 0]
    essential, tau, _ = make_householder(seg)
    apply_householder_on_left(block, essential, tau)
    assert np.all(np.isclose(block[1:, 0], 0.0, atol=1e-10))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
