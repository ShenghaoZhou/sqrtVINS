"""
EigenMatrixBuffer — JAX port of ov_srvins/src/utils/EigenMatrixBuffer.h/.cpp.

A preallocated `max_rows × max_cols` numpy buffer with a `(rows, cols)` live
view. The C++ `get()` returns an `Eigen::Ref<MatX>` — a *view*, not a copy —
so callers that need a snapshot must `.copy()` it themselves. This Python
port keeps the exact same contract: `get()` returns a writable numpy slice
sharing memory with `buffer_`.

The `set_size(rows, cols)` call *shrinks the view* to that shape without
zeroing or reallocating anything; `reset()` zeros the current view and drops
`rows_` / `cols_` back to zero.
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple


class EigenMatrixBuffer:
    """Preallocated 2D matrix buffer with a live `(rows, cols)` view."""

    def __init__(self, max_rows: int = 0, max_cols: int = 0) -> None:
        self.max_rows_ = max_rows
        self.max_cols_ = max_cols
        self.buffer_ = np.zeros((max_rows, max_cols), dtype=np.float64)
        self.rows_ = 0
        self.cols_ = 0

    # ---------------------------------------------------------- size control
    def set_size(self, rows: int, cols: int) -> bool:
        """Shrink the live view to `(rows, cols)`. Returns False if out of range."""
        if rows > self.max_rows_ or cols > self.max_cols_:
            return False
        self.rows_ = rows
        self.cols_ = cols
        return True

    def rows(self) -> int:
        return self.rows_

    def cols(self) -> int:
        return self.cols_

    def reset(self) -> None:
        """Zero the current live block and clear the view."""
        if self.rows_ > 0 and self.cols_ > 0:
            self.buffer_[: self.rows_, : self.cols_].fill(0.0)
        self.rows_ = 0
        self.cols_ = 0

    # -------------------------------------------------------------- access
    def get(self) -> np.ndarray:
        """Writable *view* on the live `(rows_, cols_)` block — mirrors the
        `Eigen::Ref<MatX>` return; callers who need a snapshot must `.copy()`."""
        if self.rows_ == 0 and self.cols_ == 0:
            return self.buffer_[:0, :0]
        return self.buffer_[: self.rows_, : self.cols_]

    def get_rows(self, start_row: int, num_rows: int) -> np.ndarray:
        return self.buffer_[start_row : start_row + num_rows, : self.cols_]

    def get_block(
        self, start_row: int, start_col: int, num_rows: int, num_cols: int
    ) -> np.ndarray:
        return self.buffer_[
            start_row : start_row + num_rows, start_col : start_col + num_cols
        ]

    # ------------------------------------------------------------ appends
    def append_rows(self, mat: np.ndarray) -> None:
        mat = np.asarray(mat, dtype=np.float64)
        if self.rows_ + mat.shape[0] > self.max_rows_ or self.cols_ != mat.shape[1]:
            raise AssertionError(
                f"EigenMatrixBuffer.append_rows: overflow "
                f"({self.rows_}+{mat.shape[0]} > {self.max_rows_} or "
                f"cols {self.cols_} != {mat.shape[1]})"
            )
        self.buffer_[self.rows_ : self.rows_ + mat.shape[0], : self.cols_] = mat
        self.rows_ += mat.shape[0]

    def append_block_rows_with_order(
        self, mat: np.ndarray, order: List[Tuple[int, int]]
    ) -> None:
        """Append `mat`'s rows, shuffling its columns by the `(global_id, size)`
        order — used by `State.store_update_jacobians` to write a
        feature-Jacobian block into the covariance column order."""
        mat = np.asarray(mat, dtype=np.float64)
        if self.rows_ + mat.shape[0] > self.max_rows_ or mat.shape[1] > self.max_cols_:
            raise AssertionError(
                f"EigenMatrixBuffer.append_block_rows_with_order: overflow"
            )
        num_rows = mat.shape[0]
        curr_cols = 0
        for global_var_id, global_var_size in order:
            self.buffer_[
                self.rows_ : self.rows_ + num_rows,
                global_var_id : global_var_id + global_var_size,
            ] = mat[:, curr_cols : curr_cols + global_var_size]
            curr_cols += global_var_size
        if curr_cols != mat.shape[1]:
            raise AssertionError(
                f"EigenMatrixBuffer.append_block_rows_with_order: order size "
                f"{curr_cols} != mat cols {mat.shape[1]}"
            )
        self.rows_ += num_rows

    def append_left_rows(self, mat: np.ndarray) -> None:
        """Append rows to the LEFT (i.e. extend `rows_` while aligning at
        column 0) — used by `store_update_factor` for `R_sqrt_inv_H_UT_`."""
        mat = np.asarray(mat, dtype=np.float64)
        if self.rows_ + mat.shape[0] > self.max_rows_ or self.cols_ < mat.shape[1]:
            raise AssertionError(
                f"EigenMatrixBuffer.append_left_rows: overflow "
                f"({self.rows_}+{mat.shape[0]} > {self.max_rows_} or "
                f"cols {self.cols_} < {mat.shape[1]})"
            )
        self.buffer_[self.rows_ : self.rows_ + mat.shape[0], : mat.shape[1]] = mat
        self.rows_ += mat.shape[0]

    def append_top_cols_and_resize(self, mat: np.ndarray) -> None:
        """Append columns to the RIGHT and extend `rows_` to `max(rows_, mat.rows)`
        — used by `store_init_factor` for the dense init factor."""
        mat = np.asarray(mat, dtype=np.float64)
        if self.rows_ > self.max_rows_ or self.cols_ + mat.shape[1] > self.max_cols_:
            raise AssertionError(
                f"EigenMatrixBuffer.append_top_cols_and_resize: overflow"
            )
        self.buffer_[: mat.shape[0], self.cols_ : self.cols_ + mat.shape[1]] = mat
        self.rows_ = max(self.rows_, mat.shape[0])
        self.cols_ += mat.shape[1]
