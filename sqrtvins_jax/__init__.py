"""JAX port of ov_srvins — the sqrt-form VIO estimator, mirroring 8d188bc.

Module layout mirrors the C++ `ov_srvins/src/` at commit `8d188bc`:

  state/          ↔ ov_srvins/src/state/       — State, StateHelper, StateOptions
  update/         ↔ ov_srvins/src/update/      — UpdaterMSCKF, UpdaterSLAM, … (Phase 4)
  initializer/    ↔ ov_srvins/src/initializer/ — InertialInitializer, … (Phase 5)
  core/           ↔ ov_srvins/src/core/        — VioManager (Phase 6)
  utils/          ↔ ov_srvins/src/utils/       — EigenMatrixBuffer, Helper, CameraPoseBuffer

**numpy over JAX for the covariance engine.**

`U_` is a real `(rows, cols)` float64 array whose shape changes on every clone
and marginalization. sqrt-form block surgery (column splices, staircase QRs,
`matrix_multiplier_ATA`) is fundamentally not `jax.jit`-friendly: each call
needs a re-trace, and the block indices themselves depend on runtime bookkeeping
(`state.imu.id()`, `kCloneStartId`, `state.x_init_.size()`). The kernels in
`sqrtvins_jax/utils/Helper.py` are therefore written as plain numpy float64,
mirroring the C++ line-for-line.

JAX stays where it has been proven: `sqrtvins_core/utils/quat_ops.py` for
so3 primitives, `sqrtvins_core/cam/` for the camera distortion Jacobians,
and Phase 4's `vmap` over feature rows in `UpdaterHelper.get_feature_jacobian_full`.
"""

from .utils import CameraPoseBuffer, EigenMatrixBuffer, Helper
from .state import State, StateOptions

__all__ = [
    "EigenMatrixBuffer",
    "CameraPoseBuffer",
    "Helper",
    "State",
    "StateOptions",
]
