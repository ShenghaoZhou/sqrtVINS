"""
JPLQuat — JAX port of ov_core/src/types/JPLQuat.h.

Quaternion stored as [x, y, z, w] (scalar LAST). Uses a *left*-multiplicative
error state:

    q ← quatnorm([0.5*dx, 1.0]) ⊗ q

where ⊗ is `quat_multiply`. See Trawny & Roumeliotis (2005) for the derivation.
`update` mutates in place and returns `None`, as `void JPLQuat::update` does.
"""

from __future__ import annotations

import numpy as np

from .Type import Type
from ..utils.quat_ops import quat_multiply as _np_quat_multiply, quatnorm as _np_quatnorm, quat_2_Rot as _np_quat_2_Rot


def quat_multiply(q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """JPL quaternion product; sign-flipped to w >= 0 and normalized."""
    return np.asarray(_np_quat_multiply(q, p), dtype=np.float64)


def quatnorm(q: np.ndarray) -> np.ndarray:
    """Normalize a JPL quaternion, sign-flipping so w >= 0."""
    return np.asarray(_np_quatnorm(q), dtype=np.float64)


def quat_2_Rot(q: np.ndarray) -> np.ndarray:
    """JPL quaternion [x,y,z,w] -> SO(3) rotation matrix."""
    return np.asarray(_np_quat_2_Rot(q), dtype=np.float64)


class JPLQuat(Type):
    """A 3-dof JPL quaternion. `value` is the 4-vector [x, y, z, w]."""

    def __init__(self):
        super().__init__(3)
        q0 = np.zeros(4, dtype=np.float64)
        q0[3] = 1.0
        self._value = q0
        self._fej = q0.copy()
        self._R = quat_2_Rot(q0)
        self._Rfej = self._R.copy()

    # ------------------------------------------------------------ accessors
    def Rot(self) -> np.ndarray:
        return self._R

    def Rot_fej(self) -> np.ndarray:
        return self._Rfej

    # --------------------------------------------------------------- update
    def update(self, dx: np.ndarray) -> None:
        """In place: `q <- quatnorm([0.5*dx, 1]) ⊗ q` (JPLQuat.h:131-143).

        `_fej` is untouched, and `set_value` recomputes the cached `_R` the
        way the C++ does.
        """
        dx = np.asarray(dx, dtype=np.float64).reshape(self._size)

        dq = np.zeros(4, dtype=np.float64)
        dq[:3] = 0.5 * dx
        dq[3] = 1.0
        dq = quatnorm(dq)

        self.set_value(quat_multiply(dq, self._value))

    def set_value(self, new_value: np.ndarray) -> None:
        q = np.asarray(new_value, dtype=np.float64).reshape(4)
        self._value = q.copy()
        self._R = quat_2_Rot(q)

    def set_fej(self, new_value: np.ndarray) -> None:
        q = np.asarray(new_value, dtype=np.float64).reshape(4)
        self._fej = q.copy()
        self._Rfej = quat_2_Rot(q)

    def clone(self) -> "JPLQuat":
        out = JPLQuat()
        out._id = self._id
        out._value = self._value.copy()
        out._fej = self._fej.copy()
        out._R = self._R.copy()
        out._Rfej = self._Rfej.copy()
        return out
