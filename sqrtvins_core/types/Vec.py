"""
Vec — JAX port of ov_core/src/types/Vec.h.
"""

from __future__ import annotations

import numpy as np

from .Type import Type


class Vec(Type):
    """A plain vector variable. `update` is additive."""

    def __init__(self, dim: int):
        super().__init__(dim)
        self._value = np.zeros(dim, dtype=np.float64)
        self._fej = np.zeros(dim, dtype=np.float64)

    def set_local_id(self, new_id: int) -> None:
        self._id = new_id

    def update(self, dx: np.ndarray) -> None:
        """In place: `set_value(_value + dx)` (Vec.h:63-65). `_fej` untouched."""
        dx = np.asarray(dx, dtype=np.float64).reshape(self._size)
        self.set_value(self._value + dx)

    def clone(self) -> "Vec":
        out = Vec(self._size)
        out._id = self._id
        out._value = self._value.copy()
        out._fej = self._fej.copy()
        return out
