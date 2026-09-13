"""
Landmark — JAX port of ov_core/src/types/Landmark.h and Landmark.cpp.

A persistent SLAM feature. Subclasses Vec, but adds metadata (featid, anchor,
marginalization flags) and two representation-specific conversions:
  - `get_xyz(getfej)`: value -> global 3D position
  - `set_from_xyz(p_FinG, isfej)`: global 3D position -> value

The four supported representations:
  GLOBAL_3D / ANCHORED_3D                : 3-vector = position
  GLOBAL_FULL_INVERSE_DEPTH / ANCHORED_* : 3-vector = [theta, phi, 1/rho]
  ANCHORED_MSCKF_INVERSE_DEPTH           : 3-vector = [x/z, y/z, 1/z]
"""

from __future__ import annotations

import numpy as np

from .Vec import Vec
from .LandmarkRepresentation import Representation


class Landmark(Vec):
    def __init__(self, dim: int):
        super().__init__(dim)
        self.featid: int = -1
        self.unique_camera_id: int = -1
        self.anchor_cam_id: int = -1
        self.anchor_clone_timestamp: float = -1.0
        self.should_marg: bool = False
        self.update_fail_count: int = 0
        self.uv_norm_zero: np.ndarray = np.zeros(3, dtype=np.float64)
        self.uv_norm_zero_fej: np.ndarray = np.zeros(3, dtype=np.float64)
        self.feat_representation: Representation = Representation.GLOBAL_3D

    # -------------------------------------------------------------- update
    def update(self, dx: np.ndarray) -> None:
        """In place: `set_value(_value + dx)` (Landmark.h:91-94).

        Keeps the C++'s explicit override — which is deliberately just the
        plain additive rule, even though the docstring mentions selectively
        updating the FEJ. The check that would flip `should_marg` when an
        anchored inverse-depth representation's last component drops below
        1e-8 is commented out in the C++, so it stays dead here too.
        """
        dx = np.asarray(dx, dtype=np.float64).reshape(self._size)
        self.set_value(self._value + dx)

    # ----------------------------------------------------------- get / set
    def get_xyz(self, getfej: bool = False) -> np.ndarray:
        v = self._fej if getfej else self._value

        if self.feat_representation in (Representation.GLOBAL_3D,
                                        Representation.ANCHORED_3D):
            return np.asarray(v, dtype=np.float64).copy()

        if self.feat_representation in (Representation.GLOBAL_FULL_INVERSE_DEPTH,
                                        Representation.ANCHORED_FULL_INVERSE_DEPTH):
            # v = [theta, phi, 1/rho]; convert to [x, y, z]
            theta, phi, inv_rho = v[0], v[1], v[2]
            rho = 1.0 / inv_rho
            x = rho * np.cos(theta) * np.sin(phi)
            y = rho * np.sin(theta) * np.sin(phi)
            z = rho * np.cos(phi)
            return np.array([x, y, z], dtype=np.float64)

        if self.feat_representation == Representation.ANCHORED_MSCKF_INVERSE_DEPTH:
            # v = [x/z, y/z, 1/z]
            xz, yz, inv_z = v[0], v[1], v[2]
            inv_z_safe = inv_z if abs(inv_z) > 1e-12 else 1.0
            x = xz / inv_z_safe
            y = yz / inv_z_safe
            z = 1.0 / inv_z_safe
            return np.array([x, y, z], dtype=np.float64)

        raise ValueError(f"unknown representation: {self.feat_representation}")

    def set_from_xyz(self, p_FinG: np.ndarray, isfej: bool = False) -> None:
        p_FinG = np.asarray(p_FinG, dtype=np.float64).reshape(3)

        if self.feat_representation in (Representation.GLOBAL_3D,
                                        Representation.ANCHORED_3D):
            if isfej:
                self._fej = p_FinG.copy()
            else:
                self._value = p_FinG.copy()
            return

        if self.feat_representation in (Representation.GLOBAL_FULL_INVERSE_DEPTH,
                                        Representation.ANCHORED_FULL_INVERSE_DEPTH):
            # p_FinG -> [theta, phi, 1/rho]
            rho = 1.0 / np.linalg.norm(p_FinG)
            phi = np.arccos(rho * p_FinG[2])
            theta = np.arctan2(p_FinG[1], p_FinG[0])
            p_inv = np.array([theta, phi, rho], dtype=np.float64)
            if isfej:
                self._fej = p_inv
            else:
                self._value = p_inv
            return

        if self.feat_representation == Representation.ANCHORED_MSCKF_INVERSE_DEPTH:
            inv_z = 1.0 / p_FinG[2]
            p_inv = np.array([p_FinG[0] * inv_z, p_FinG[1] * inv_z, inv_z],
                             dtype=np.float64)
            if isfej:
                self._fej = p_inv
            else:
                self._value = p_inv
            return

        raise ValueError(f"unknown representation: {self.feat_representation}")

    # -------------------------------------------------------------- clone
    def clone(self) -> "Landmark":
        out = Landmark(self._size)
        self._copy_landmark_metadata(out)
        out._id = self._id
        out._value = self._value.copy()
        out._fej = self._fej.copy()
        return out

    # ------------------------------------------------------------ helpers
    def _copy_landmark_metadata(self, other: "Landmark") -> None:
        other.featid = self.featid
        other.unique_camera_id = self.unique_camera_id
        other.anchor_cam_id = self.anchor_cam_id
        other.anchor_clone_timestamp = self.anchor_clone_timestamp
        other.should_marg = self.should_marg
        other.update_fail_count = self.update_fail_count
        other.uv_norm_zero = self.uv_norm_zero.copy()
        other.uv_norm_zero_fej = self.uv_norm_zero_fej.copy()
        other.feat_representation = self.feat_representation
