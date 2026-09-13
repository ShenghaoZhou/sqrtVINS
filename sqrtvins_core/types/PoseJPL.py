"""
PoseJPL — JAX port of ov_core/src/types/PoseJPL.h.

6-dof pose: JPLQuat + Vec(3) position. Error-state order is (q, p). The
value layout is `[x, y, z, w, px, py, pz]` (7-vector).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .Type import Type
from .Vec import Vec
from .JPLQuat import JPLQuat, quat_multiply, quatnorm


class PoseJPL(Type):
    """6-dof pose (orientation JPLQuat + position Vec(3))."""

    def __init__(self):
        super().__init__(6)
        self._q = JPLQuat()
        self._p = Vec(3)
        # value layout: [qx, qy, qz, qw, px, py, pz]
        v = np.zeros(7, dtype=np.float64)
        v[3] = 1.0
        self.set_value_internal(v)
        self.set_fej_internal(v.copy())

    # ------------------------------------------------------------- sub-ids
    def set_local_id(self, new_id: int) -> None:
        """Propagate the covariance id down to `_q` and `_p`.

        Mirrors `PoseJPL::set_local_id`: `p` gets `new_id + q->size()` when
        the pose is in the covariance (`new_id != -1`), else `new_id`.
        """
        self._id = new_id
        self._q.set_local_id(new_id)
        if new_id != -1:
            self._p.set_local_id(new_id + self._q.size())
        else:
            self._p.set_local_id(new_id)

    # -------------------------------------------------------------- access
    def Rot(self) -> np.ndarray:
        return self._q.Rot()

    def Rot_fej(self) -> np.ndarray:
        return self._q.Rot_fej()

    def quat(self) -> np.ndarray:
        return self._q.value()

    def quat_fej(self) -> np.ndarray:
        return self._q.fej()

    def pos(self) -> np.ndarray:
        return self._p.value()

    def pos_fej(self) -> np.ndarray:
        return self._p.fej()

    def q(self) -> JPLQuat:
        return self._q

    def p(self) -> Vec:
        return self._p

    # --------------------------------------------------------------- update
    def update(self, dx: np.ndarray) -> None:
        """In place: JPLQuat boxplus on `q`, additive on `p` (PoseJPL.h:84-101).

        Goes through `set_value_internal`, which pushes the new `q`/`p` into the
        sub-variables — "we update the sub-variables also", as the C++ comment
        says. `_fej` is untouched.
        """
        dx = np.asarray(dx, dtype=np.float64).reshape(self._size)

        newX = self._value.copy()

        dq = np.zeros(4, dtype=np.float64)
        dq[:3] = 0.5 * dx[:3]
        dq[3] = 1.0
        dq = quatnorm(dq)

        newX[:4] = quat_multiply(dq, self._q.value())
        newX[4:] += dx[3:]

        self.set_value_internal(newX)

    # -------------------------------------------------------- set / clone
    def set_value(self, new_value: np.ndarray) -> None:
        self.set_value_internal(new_value)

    def set_fej(self, new_value: np.ndarray) -> None:
        self.set_fej_internal(new_value)

    def clone(self) -> "PoseJPL":
        out = PoseJPL()
        out._id = self._id
        out.set_value_internal(self._value.copy())
        out.set_fej_internal(self._fej.copy())
        return out

    def check_if_subvariable(self, check: Type) -> Optional[Type]:
        if check == self._q:
            return self._q
        if check == self._p:
            return self._p
        return None

    # ---------------------------------------------------------- internals
    def set_value_internal(self, new_value: np.ndarray) -> None:
        v = np.asarray(new_value, dtype=np.float64).reshape(7)
        self._q.set_value(v[:4])
        self._p.set_value(v[4:])
        self._value = v.copy()

    def set_fej_internal(self, new_value: np.ndarray) -> None:
        v = np.asarray(new_value, dtype=np.float64).reshape(7)
        self._q.set_fej(v[:4])
        self._p.set_fej(v[4:])
        self._fej = v.copy()
