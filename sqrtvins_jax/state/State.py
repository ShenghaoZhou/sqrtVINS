"""
State — JAX port of ov_srvins/src/state/State.h/.cpp.

The State object holds all current filter estimates plus the sqrt-form
upper-triangular covariance `U`. Estimates live *inside* each `Type` (see
`Type.update` docstring), so `variables_` is the single source of truth and
`imu`/`clones_IMU`/`calib_IMUtoCAM`/`cam_intrinsics` are aliases into it.

C++ attributes `_k_clone_start_id`, `_variables`, `_U`, `_R_sqrt_inv_H_UT`,
etc. are exposed as public `State` attributes here (Python has no `friend`
accessor pattern, and `StateHelper` needs direct mutation of `U` and the
`variables_` list). Naming convention: attributes whose C++ name ends in `_`
(`U_`, `variables_`, `x_init_`) drop the trailing underscore in Python.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from sqrtvins_core.types.IMU import IMU
from sqrtvins_core.types.Landmark import Landmark
from sqrtvins_core.types.LandmarkMsckf import LandmarkMsckf
from sqrtvins_core.types.PoseJPL import PoseJPL
from sqrtvins_core.types.Type import Type
from sqrtvins_core.types.Vec import Vec

from ..utils.CameraPoseBuffer import CameraPoseBuffer, PoseData
from ..utils.EigenMatrixBuffer import EigenMatrixBuffer
from .StateOptions import StateOptions


class State:
    """State of the filter (estimates + sqrt-form covariance)."""

    def __init__(self, options: StateOptions, init_options: Any) -> None:
        # Save our options
        self.options = options
        self.init_options = init_options

        # Setup buffer size
        k_marginal_offset = 50
        k_feat_size = 3
        k_max_state_size = (
            options.max_slam_features * k_feat_size
            + options.max_clone_size * 6
            + 15  # IMU
            + (1 if options.do_calib_camera_timeoffset else 0)
            + options.num_cameras
            * (options.num_cameras * 6 if options.do_calib_camera_pose else 0)
            + options.num_cameras
            * (options.num_cameras * 8 if options.do_calib_camera_intrinsics else 0)
            + k_marginal_offset
        )
        k_max_measurement_size = (
            2
            * options.num_cameras
            * (options.max_msckf_in_update + options.max_slam_features)
            * (options.max_clone_size + 1)
            + k_marginal_offset
        )
        k_max_slam_feat_state_size = options.max_slam_features * k_feat_size

        self.R_sqrt_inv_H_UT = EigenMatrixBuffer(
            k_max_measurement_size, k_max_state_size
        )
        self.HT_R_inv_res = EigenMatrixBuffer(k_max_state_size, 1)

        self.factor_init_dense = EigenMatrixBuffer(
            k_max_state_size, k_max_slam_feat_state_size
        )
        self.factor_init_tri = EigenMatrixBuffer(k_max_slam_feat_state_size, k_feat_size)

        self.H_update = EigenMatrixBuffer(k_max_measurement_size, k_max_state_size)
        self.res_update = EigenMatrixBuffer(k_max_measurement_size, 1)

        self.timestamp: float = -1.0
        self.is_initialized: bool = False

        # Camera pose buffers
        self.cam_pose_buffer = CameraPoseBuffer(
            options.max_clone_size + 2, options.num_cameras
        )
        self.cam_pose_fej_buffer = CameraPoseBuffer(
            options.max_clone_size + 2, options.num_cameras
        )

        # Starting ID for clones
        self.kCloneStartId = (
            15  # IMU
            + (1 if options.do_calib_camera_timeoffset else 0)
            + options.num_cameras * (6 if options.do_calib_camera_pose else 0)
            + options.num_cameras * (8 if options.do_calib_camera_intrinsics else 0)
        )

        # Vector of variables
        self.variables: List[Type] = []

        # Square root of the covariance (upper-triangular during cloning)
        self.U: np.ndarray = np.zeros((0, 0), dtype=np.float64)

        current_id = 0
        # Append the imu to the state and covariance
        self.imu = IMU()
        self.imu.set_local_id(current_id)
        self.variables.append(self.imu)
        current_id += self.imu.size()

        # Camera to IMU time offset
        self.calib_dt_CAMtoIMU = Vec(1)
        if options.do_calib_camera_timeoffset:
            self.calib_dt_CAMtoIMU.set_local_id(current_id)
            self.variables.append(self.calib_dt_CAMtoIMU)
            current_id += self.calib_dt_CAMtoIMU.size()

        # Loop through each camera and create extrinsic and intrinsics
        self.calib_IMUtoCAM: Dict[int, PoseJPL] = {}
        self.cam_intrinsics: Dict[int, Vec] = {}
        self.cam_intrinsics_cameras: Dict[int, Any] = {}

        for i in range(options.num_cameras):
            pose = PoseJPL()
            intrin = Vec(8)
            self.calib_IMUtoCAM[i] = pose
            self.cam_intrinsics[i] = intrin

            if options.do_calib_camera_pose:
                pose.set_local_id(current_id)
                self.variables.append(pose)
                current_id += pose.size()
            if options.do_calib_camera_intrinsics:
                intrin.set_local_id(current_id)
                self.variables.append(intrin)
                current_id += intrin.size()

        # Finally initialize our covariance to small value
        self.U = 1e-3 * np.eye(current_id, dtype=np.float64)

        # Finally, set some of our priors for our calibration parameters
        if options.do_calib_camera_timeoffset:
            i0 = self.calib_dt_CAMtoIMU.id()
            self.U[i0, i0] = 0.01
        if options.do_calib_camera_pose:
            for i in range(options.num_cameras):
                i0 = self.calib_IMUtoCAM[i].id()
                self.U[i0:i0 + 3, i0:i0 + 3] = 0.005 * np.eye(3)
                self.U[i0 + 3:i0 + 6, i0 + 3:i0 + 6] = 0.015 * np.eye(3)
        if options.do_calib_camera_intrinsics:
            for i in range(options.num_cameras):
                i0 = self.cam_intrinsics[i].id()
                self.U[i0:i0 + 4, i0:i0 + 4] = 1.0 * np.eye(4)
                self.U[i0 + 4:i0 + 8, i0 + 4:i0 + 8] = 0.005 * np.eye(4)

        # SLAM / MSCKF features
        self.features_SLAM: Dict[int, Landmark] = {}
        self.features_MSCKF: Dict[int, LandmarkMsckf] = {}

        # Clones (timestamp -> PoseJPL). Python dicts preserve insertion order,
        # but the C++ uses std::map sorted by key, so `margtimestep()` uses
        # `min(self.clones_IMU)` below.
        self.clones_IMU: Dict[float, PoseJPL] = {}

        # Store the initialization values
        self.x_init: List[Type] = []

        # Store x_{k}-x_{k-1} and x_{k}-x_{0}
        self.xk_minus_xk1: np.ndarray = np.zeros((current_id, 1), dtype=np.float64)
        self.xk_minus_x0: np.ndarray = np.zeros((current_id, 1), dtype=np.float64)

        # Vector of variables to be marginalized
        self.state_to_marg: List[Type] = []

    # ------------------------------------------------------------ helpers
    def margtimestep(self) -> float:
        """Timestep that will be marginalized next (the oldest clone)."""
        assert len(self.clones_IMU) > 0
        return min(self.clones_IMU)

    def clear(self, clean_msckf_state: bool = False) -> None:
        if clean_msckf_state:
            self.features_MSCKF.clear()

        # Clean the buffer
        self.HT_R_inv_res.reset()
        self.R_sqrt_inv_H_UT.reset()
        if not self.is_initialized:
            self.H_update.reset()
            self.res_update.reset()

        # Clear the init buffers
        self.x_init.clear()
        self.factor_init_tri.reset()
        self.factor_init_dense.reset()

    def get_x_squared_norm(self) -> float:
        x_norm_sq = 0.0
        for var in self.variables:
            v = var.value()
            x_norm_sq += float(v @ v)
        for var in self.features_MSCKF.values():
            v = var.value()
            x_norm_sq += float(v @ v)
        return x_norm_sq

    def get_dx_squared_norm(self) -> float:
        dx_norm_sq = float(self.xk_minus_xk1 @ self.xk_minus_xk1)
        for feat_msckf in self.features_MSCKF.values():
            dx = feat_msckf.feat_dx
            dx_norm_sq += float(dx @ dx)
        return dx_norm_sq

    def get_residual_squared_norm(self) -> float:
        res_norm = 0.0
        for feat_msckf in self.features_MSCKF.values():
            r = feat_msckf.res_msckf
            res_norm += float(r @ r)
        r_up = self.res_update.get()
        res_norm += float(r_up @ r_up)
        return res_norm

    def get_clone_pose(self, query_timestamp: float) -> Optional[PoseJPL]:
        if query_timestamp in self.clones_IMU:
            return self.clones_IMU[query_timestamp]
        elif query_timestamp == self.timestamp:
            return self.imu.pose()
        else:
            return None

    def resize_U_to_square(self) -> None:
        """Resize the covariance matrix to be square (zero-fill new region)."""
        state_size = self.U.shape[1]
        new_U = np.zeros((state_size, state_size), dtype=np.float64)
        n = min(self.U.shape[0], state_size)
        m = min(self.U.shape[1], state_size)
        new_U[:n, :m] = self.U[:n, :m]
        self.U = new_U
        # Zero-fill the delta vectors, preserving existing data
        new_x0 = np.zeros((state_size, 1), dtype=np.float64)
        n0 = min(self.xk_minus_x0.shape[0], state_size)
        new_x0[:n0] = self.xk_minus_x0[:n0]
        self.xk_minus_x0 = new_x0
        new_x1 = np.zeros((state_size, 1), dtype=np.float64)
        n1 = min(self.xk_minus_xk1.shape[0], state_size)
        new_x1[:n1] = self.xk_minus_xk1[:n1]
        self.xk_minus_xk1 = new_x1

    def calculate_clone_poses(self, fej: bool = False) -> None:
        """Calculate the clone poses to avoid recalculation.

        For each (calib, clone) pair, compute the resulting camera pose in G
        and write it to the appropriate pose buffer.
        """
        target_buffer = self.cam_pose_fej_buffer if fej else self.cam_pose_buffer
        for calib_id, calib in self.calib_IMUtoCAM.items():
            # Follow OpenVINS convention here, we don't FEJ calibration.
            R_ItoC = calib.Rot()
            p_IinC = calib.pos()
            for clone_ts, clone in self.clones_IMU.items():
                if fej:
                    R_GtoI = clone.Rot_fej()
                    p_IinG = clone.pos_fej()
                else:
                    R_GtoI = clone.Rot()
                    p_IinG = clone.pos()
                cam_pose = target_buffer.get_buffer_unsafe(calib_id, clone_ts)
                cam_pose.R_GtoC = R_ItoC @ R_GtoI
                cam_pose.p_CinG = -R_GtoI.T @ p_IinC + p_IinG

    def calculate_clone_poses_fej(self) -> None:
        self.calculate_clone_poses(True)

    def erase_feat(self, feat_id: int) -> None:
        self.features_MSCKF.pop(feat_id, None)

    def store_init_factor(
        self,
        new_var: Type,
        tri_factor: np.ndarray,
        dense_factor: np.ndarray,
    ) -> None:
        self.x_init.append(new_var)
        self.factor_init_tri.append_rows(tri_factor)
        self.factor_init_dense.append_top_cols_and_resize(dense_factor)

    def store_update_factor(
        self,
        R_sinv_H_UT: np.ndarray,
        R_inv_res: np.ndarray,
    ) -> None:
        self.R_sqrt_inv_H_UT.append_left_rows(R_sinv_H_UT)
        # noalias() +=
        cur = self.HT_R_inv_res.get()
        cur += R_inv_res

    def store_update_jacobians(
        self,
        Hx: np.ndarray,
        res: np.ndarray,
        x_order: List[Type],
    ) -> None:
        order: List[tuple] = [(var.id(), var.size()) for var in x_order]
        self.H_update.append_block_rows_with_order(Hx, order)
        self.res_update.append_rows(res)

    def get_state_size(self) -> int:
        return int(self.U.shape[1])

    def setup_matrix_buffer(self) -> None:
        state_size = self.get_state_size()
        k_feat_size = 3
        self.R_sqrt_inv_H_UT.set_size(0, state_size)
        self.HT_R_inv_res.set_size(state_size, 1)
        self.factor_init_tri.set_size(0, k_feat_size)
        self.H_update.set_size(0, state_size)
        self.res_update.set_size(0, 1)

    def update_timestamp(self, new_time: float) -> None:
        self.timestamp = new_time
        self.cam_pose_buffer.add_timestamp(new_time)
        self.cam_pose_fej_buffer.add_timestamp(new_time)

    def remove_timestamp(self, old_time: float) -> None:
        self.cam_pose_buffer.remove_timestamp(old_time)
        self.cam_pose_fej_buffer.remove_timestamp(old_time)

    def get_xk_minus_xk1(self) -> np.ndarray:
        return self.xk_minus_xk1

    def add_marginal_state(self, var: Type) -> None:
        self.state_to_marg.append(var)
