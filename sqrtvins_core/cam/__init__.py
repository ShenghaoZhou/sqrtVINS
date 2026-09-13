"""
Camera models — JAX port of ov_core/src/cam/.

`CamBase`, `CamRadtan`, `CamEqui` are the estimator-facing classes (numpy
in/out, one point at a time). The `radtan_*` / `equi_*` functions are the
pure batched JAX kernels that `UpdaterHelper` vmaps over in Phase 4.

Distortion model selection mirrors `VioManagerOptions.cpp:212`:
`"equidistant"` → `CamEqui`, anything else → `CamRadtan`.
"""

from .CamBase import CamBase
from .CamEqui import (CamEqui, equi_distort, equi_jacobian_dzn,
                      equi_jacobian_dzeta)
from .CamRadtan import (CamRadtan, radtan_distort, radtan_jacobian_dzn,
                        radtan_jacobian_dzeta)


def make_camera(width: int, height: int, model: str = "radtan",
                values=None):
    """
    Build a camera by model name, matching `VioManagerOptions.cpp:211-214`.

    Args:
        width, height: image dimensions.
        model: "equidistant" for fisheye, anything else is Brown–Conrady.
        values: optional (8,) intrinsic vector; defaults to zeros.
    """
    cam = CamEqui(width, height) if model == "equidistant" else CamRadtan(width, height)
    if values is not None:
        cam.set_value(values)
    return cam


__all__ = [
    "CamBase", "CamRadtan", "CamEqui", "make_camera",
    "radtan_distort", "radtan_jacobian_dzn", "radtan_jacobian_dzeta",
    "equi_distort", "equi_jacobian_dzn", "equi_jacobian_dzeta",
]
