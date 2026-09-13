"""
Base class for filter variables — JAX port of ov_core/src/types/Type.h.

This is a lightweight Python port. It uses numpy arrays (float64) for storage
so the reference implementation matches C++ exactly. JAX is pushed in only at
the Jacobian boundary (Phase 3), where the state is flattened into a padded
buffer.

The C++ class mutates in place: `update(dx)` is `void` and advances `_value`
through `set_value` (each subclass encodes its own boxplus). This port keeps
that contract, so `state->variables_` stays the single source of truth for
estimate values — `state.imu`, `state.clones_IMU[t]`, `state.calib_IMUtoCAM`
and `state.cam_intrinsics` all hold the *same objects* as `variables_`, and a
return-new-object `update` would silently desync any handle that was not
rebound after the call. `_fej` is deliberately left untouched by `update`,
exactly as in the C++.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Type:
    """
    Base class for a filter variable.

    Attributes:
      size:  number of error-state degrees of freedom.
      id:    index into the filter covariance. -1 means "not in the covariance".
      value: current best estimate (numpy float64).
      fej:   first-estimate (initial guess carried across the filter's lifetime).
    """

    def __init__(self, size: int):
        self._size: int = size
        self._id: int = -1
        self._value: np.ndarray = np.zeros(size, dtype=np.float64)
        self._fej: np.ndarray = np.zeros(size, dtype=np.float64)

    # ------------------------------------------------------------------ ids
    def set_local_id(self, new_id: int) -> None:
        self._id = new_id

    def id(self) -> int:
        return self._id

    def size(self) -> int:
        return self._size

    # ------------------------------------------------------------- estimates
    def value(self) -> np.ndarray:
        return self._value

    def fej(self) -> np.ndarray:
        return self._fej

    def set_value(self, new_value: np.ndarray) -> None:
        assert new_value.shape == (self._size, 1) or new_value.shape == (self._size,)
        v = np.asarray(new_value, dtype=np.float64).reshape(self._size)
        self._value = v.copy()

    def set_fej(self, new_value: np.ndarray) -> None:
        assert new_value.shape == (self._size, 1) or new_value.shape == (self._size,)
        f = np.asarray(new_value, dtype=np.float64).reshape(self._size)
        self._fej = f.copy()

    # ------------------------------------------------------------- dynamics
    def update(self, dx: np.ndarray) -> None:
        """Advance `_value` by `dx` in place, leaving `_fej` untouched.

        Mirrors `Type::update` — `void`, and the C++ `const` qualifier is what
        makes the mutation legal (the estimate is a `mutable` member).
        Subclasses override this to encode their specific boxplus
        (quaternion, vec, …). The return value is `None`, as in the C++; a
        caller that wants the new value reads it back with `value()`.
        """
        raise NotImplementedError

    def clone(self) -> "Type":
        """Copy the current value + fej into a fresh instance."""
        raise NotImplementedError

    def check_if_subvariable(self, check: "Type") -> Optional["Type"]:
        """Return `check` if it is a sub-variable of `self`, else None."""
        return None

    # ------------------------------------------------------------- utilities
    def _copy_to(self, other: "Type") -> None:
        assert isinstance(other, type(self))
        other._size = self._size
        other._id = self._id
        other._value = self._value.copy()
        other._fej = self._fej.copy()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self._size}, id={self._id})"
