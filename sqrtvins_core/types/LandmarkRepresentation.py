"""
LandmarkRepresentation — JAX port of ov_core/src/types/LandmarkRepresentation.h.

Enum of the feature representation forms supported by the filter. Kept as a
Python `Enum` for readability; the C++ `static inline` helpers are module-level
functions.
"""

from __future__ import annotations

from enum import Enum


class Representation(Enum):
    GLOBAL_3D = 0
    GLOBAL_FULL_INVERSE_DEPTH = 1
    ANCHORED_3D = 2
    ANCHORED_FULL_INVERSE_DEPTH = 3
    ANCHORED_MSCKF_INVERSE_DEPTH = 4
    UNKNOWN = 5


def as_string(rep: Representation) -> str:
    return rep.name


def from_string(s: str) -> Representation:
    try:
        return Representation[s]
    except KeyError:
        return Representation.UNKNOWN


def is_relative_representation(rep: Representation) -> bool:
    return rep in (
        Representation.ANCHORED_3D,
        Representation.ANCHORED_FULL_INVERSE_DEPTH,
        Representation.ANCHORED_MSCKF_INVERSE_DEPTH,
    )
