"""
LandmarkMsckf — JAX port of ov_core/src/types/LandmarkMsckf.h.

Landmark variant that carries MSCKF-specific residuals and Jacobians, plus a
`feat_dx` delta used by `UpdaterMSCKF::update_features` to fold accumulated
residuals back into the landmark position.
"""

from __future__ import annotations

import numpy as np

from .Landmark import Landmark


class LandmarkMsckf(Landmark):
    """A 3D landmark for MSCKF (dim=3 by convention)."""

    def __init__(self, dim: int = 3):
        super().__init__(dim)
        # MSCKF-only fields, populated after the QR null-space projection
        # in UpdaterMSCKF.update().
        self.Hf_msckf: np.ndarray = np.zeros((3, dim), dtype=np.float64)
        self.Hx_msckf: np.ndarray = np.zeros((3, 0), dtype=np.float64)
        self.res_msckf: np.ndarray = np.zeros(3, dtype=np.float64)
        # Order of the state variables corresponding to Hx_msckf columns
        self.x_order_msckf: list = []
        # Accumulated landmark correction from the last MSCKF update
        self.feat_dx: np.ndarray = np.zeros(dim, dtype=np.float64)

    def store_msckf_jacobians(
        self,
        Hf: np.ndarray,
        Hx: np.ndarray,
        res: np.ndarray,
        x_order: list,
    ) -> None:
        self.Hf_msckf = np.asarray(Hf, dtype=np.float64)
        self.Hx_msckf = np.asarray(Hx, dtype=np.float64)
        self.res_msckf = np.asarray(res, dtype=np.float64)
        self.x_order_msckf = list(x_order)

    # -------------------------------------------------------------- clone
    def clone(self) -> "LandmarkMsckf":
        out = LandmarkMsckf(self._size)
        out._copy_landmark_metadata(self)
        out._id = self._id
        out._value = self._value.copy()
        out._fej = self._fej.copy()
        out.Hf_msckf = self.Hf_msckf.copy()
        out.Hx_msckf = self.Hx_msckf.copy()
        out.res_msckf = self.res_msckf.copy()
        out.x_order_msckf = list(self.x_order_msckf)
        out.feat_dx = self.feat_dx.copy()
        return out
