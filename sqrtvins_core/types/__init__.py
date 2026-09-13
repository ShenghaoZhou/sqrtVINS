"""Type system — JAX port of ov_core/src/types/.

Every filter variable derives from `Type`. The base contract is:
  size()    -> int   : error-state dof
  id()      -> int   : position in the covariance; -1 means "not in covariance"
  value()   -> np.ndarray
  fej()     -> np.ndarray   (first-estimate, frozen at initialization)
  update(dx) -> Type       : boxplus; returns a new instance (JAX-friendly)
  clone()   -> Type
  set_local_id(int)        : propagate the covariance id down to sub-variables

The JAX hot loop (update_llt, feature Jacobian assembly) does not go through
these classes — it works on a flat padded numpy buffer plus side arrays.
These types are the readable "reference" layer for initialization, marginalization
bookkeeping, and the front-end feature contract.
"""

from .Type import Type
from .Vec import Vec
from .JPLQuat import JPLQuat
from .PoseJPL import PoseJPL
from .IMU import IMU
from .LandmarkRepresentation import Representation, is_relative_representation
from .Landmark import Landmark
from .LandmarkMsckf import LandmarkMsckf

__all__ = [
    "Type",
    "Vec",
    "JPLQuat",
    "PoseJPL",
    "IMU",
    "LandmarkRepresentation",
    "Representation",
    "is_relative_representation",
    "Landmark",
    "LandmarkMsckf",
]
