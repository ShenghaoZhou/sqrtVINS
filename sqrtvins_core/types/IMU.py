"""
IMU — JAX port of ov_core/src/types/IMU.h.

15-dof error state, ordered (q, p, v, bg, ba).
Value layout is `[qx, qy, qz, qw, px, py, pz, vx, vy, vz, bgx, bgy, bgz, bax, bay, baz]` (16-vector).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .Type import Type
from .Vec import Vec
from .PoseJPL import PoseJPL
from .JPLQuat import JPLQuat, quat_multiply, quatnorm


class IMU(Type):
    """The full IMU state: pose + velocity + gyro bias + accel bias."""

    def __init__(self):
        super().__init__(15)
        self._pose = PoseJPL()
        self._v = Vec(3)
        self._bg = Vec(3)
        self._ba = Vec(3)

        v = np.zeros(16, dtype=np.float64)
        v[3] = 1.0  # qw = 1
        self.set_value_internal(v)
        self.set_fej_internal(v.copy())

    # ------------------------------------------------------------- sub-ids
    def set_local_id(self, new_id: int) -> None:
        self._id = new_id
        self._pose.set_local_id(new_id)
        if new_id != -1:
            self._v.set_local_id(new_id + self._pose.size())
            self._bg.set_local_id(self._v.id() + self._v.size())
            self._ba.set_local_id(self._bg.id() + self._bg.size())

    # -------------------------------------------------------------- access
    def Rot(self) -> np.ndarray:
        return self._pose.Rot()

    def Rot_fej(self) -> np.ndarray:
        return self._pose.Rot_fej()

    def quat(self) -> np.ndarray:
        return self._pose.quat()

    def quat_fej(self) -> np.ndarray:
        return self._pose.quat_fej()

    def pos(self) -> np.ndarray:
        return self._pose.pos()

    def pos_fej(self) -> np.ndarray:
        return self._pose.pos_fej()

    def vel(self) -> np.ndarray:
        return self._v.value()

    def vel_fej(self) -> np.ndarray:
        return self._v.fej()

    def bias_g(self) -> np.ndarray:
        return self._bg.value()

    def bias_g_fej(self) -> np.ndarray:
        return self._bg.fej()

    def bias_a(self) -> np.ndarray:
        return self._ba.value()

    def bias_a_fej(self) -> np.ndarray:
        return self._ba.fej()

    def pose(self) -> PoseJPL:
        return self._pose

    def q(self) -> JPLQuat:
        return self._pose.q()

    def p(self) -> Vec:
        return self._pose.p()

    def v(self) -> Vec:
        return self._v

    def bg(self) -> Vec:
        return self._bg

    def ba(self) -> Vec:
        return self._ba

    # --------------------------------------------------------------- update
    def update(self, dx: np.ndarray) -> None:
        """In place: boxplus on `q`, additive on `p, v, bg, ba` (IMU.h:88-108).

        `Eigen::Matrix<DataType,16,1> newX = _value` in the C++ is a copy, and
        so is `newX = self._value.copy()` here — the in-place write into
        `_value` at the end of `set_value_internal` must not alias the source.
        `_fej` is untouched.
        """
        dx = np.asarray(dx, dtype=np.float64).reshape(self._size)

        newX = self._value.copy()

        dq = np.zeros(4, dtype=np.float64)
        dq[:3] = 0.5 * dx[:3]
        dq[3] = 1.0
        dq = quatnorm(dq)

        newX[:4] = quat_multiply(dq, self._pose.quat())
        newX[4:7] += dx[3:6]
        newX[7:10] += dx[6:9]
        newX[10:13] += dx[9:12]
        newX[13:16] += dx[12:15]

        self.set_value_internal(newX)

    # -------------------------------------------------------- set / clone
    def set_value(self, new_value: np.ndarray) -> None:
        self.set_value_internal(new_value)

    def set_fej(self, new_value: np.ndarray) -> None:
        self.set_fej_internal(new_value)

    def clone(self) -> "IMU":
        out = IMU()
        out._id = self._id
        out.set_value_internal(self._value.copy())
        out.set_fej_internal(self._fej.copy())
        return out

    def check_if_subvariable(self, check: Type) -> Optional[Type]:
        if check == self._pose:
            return self._pose
        sub = self._pose.check_if_subvariable(check)
        if sub is not None:
            return sub
        if check == self._v:
            return self._v
        if check == self._bg:
            return self._bg
        if check == self._ba:
            return self._ba
        return None

    # ---------------------------------------------------------- internals
    def set_value_internal(self, new_value: np.ndarray) -> None:
        v = np.asarray(new_value, dtype=np.float64).reshape(16)
        self._pose.set_value(v[:7])
        self._v.set_value(v[7:10])
        self._bg.set_value(v[10:13])
        self._ba.set_value(v[13:16])
        self._value = v.copy()

    def set_fej_internal(self, new_value: np.ndarray) -> None:
        v = np.asarray(new_value, dtype=np.float64).reshape(16)
        self._pose.set_fej(v[:7])
        self._v.set_fej(v[7:10])
        self._bg.set_fej(v[10:13])
        self._ba.set_fej(v[13:16])
        self._fej = v.copy()
