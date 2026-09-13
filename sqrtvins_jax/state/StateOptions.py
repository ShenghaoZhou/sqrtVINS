"""
StateOptions — JAX port of ov_srvins/src/state/StateOptions.h.

Every field and every YAML key mirrors the C++ struct line-for-line so the
existing `config/*.yml` files parse unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from sqrtvins_core.types.LandmarkRepresentation import (
    Representation,
    as_string,
    from_string,
)
from sqrtvins_core.utils.print import print_debug


@dataclass
class StateOptions:
    """Struct which stores all our filter options.

    The `print()` method doubles as the YAML loader (as in the C++). It is
    called with a `YamlParser` in the config-load path and with `None` in the
    print-only path.
    """

    do_fej: bool = True
    imu_avg: bool = False
    use_rk4_integration: bool = True
    do_calib_camera_pose: bool = False
    do_calib_camera_intrinsics: bool = False
    do_calib_camera_timeoffset: bool = False
    max_clone_size: int = 11
    max_slam_features: int = 25
    max_slam_in_update: int = 1000
    max_msckf_in_update: int = 1000
    max_aruco_features: int = 1024
    num_cameras: int = 1
    feat_rep_msckf: Representation = Representation.GLOBAL_3D
    feat_rep_slam: Representation = Representation.GLOBAL_3D
    feat_rep_aruco: Representation = Representation.GLOBAL_3D

    def print(self, parser: Optional[Any] = None) -> None:
        if parser is not None:
            # parse_config returns the parsed value; assign back.
            self.do_fej = parser.parse_config("use_fej", self.do_fej)
            self.imu_avg = parser.parse_config("use_imuavg", self.imu_avg)
            self.use_rk4_integration = parser.parse_config(
                "use_rk4int", self.use_rk4_integration
            )
            self.do_calib_camera_pose = parser.parse_config(
                "calib_cam_extrinsics", self.do_calib_camera_pose
            )
            self.do_calib_camera_intrinsics = parser.parse_config(
                "calib_cam_intrinsics", self.do_calib_camera_intrinsics
            )
            self.do_calib_camera_timeoffset = parser.parse_config(
                "calib_cam_timeoffset", self.do_calib_camera_timeoffset
            )
            self.max_clone_size = parser.parse_config(
                "max_clones", self.max_clone_size
            )
            self.max_slam_features = parser.parse_config(
                "max_slam", self.max_slam_features
            )
            self.max_slam_in_update = parser.parse_config(
                "max_slam_in_update", self.max_slam_in_update
            )
            self.max_msckf_in_update = parser.parse_config(
                "max_msckf_in_update", self.max_msckf_in_update
            )
            self.max_aruco_features = parser.parse_config(
                "num_aruco", self.max_aruco_features
            )
            self.num_cameras = parser.parse_config(
                "max_cameras", self.num_cameras
            )
            rep1 = as_string(self.feat_rep_msckf)
            rep1 = parser.parse_config("feat_rep_msckf", rep1)
            self.feat_rep_msckf = from_string(rep1)
            rep2 = as_string(self.feat_rep_slam)
            rep2 = parser.parse_config("feat_rep_slam", rep2)
            self.feat_rep_slam = from_string(rep2)
            rep3 = as_string(self.feat_rep_aruco)
            rep3 = parser.parse_config("feat_rep_aruco", rep3)
            self.feat_rep_aruco = from_string(rep3)

        print_debug(f"  - use_fej: {int(self.do_fej)}")
        print_debug(f"  - use_imuavg: {int(self.imu_avg)}")
        print_debug(f"  - use_rk4int: {int(self.use_rk4_integration)}")
        print_debug(f"  - calib_cam_extrinsics: {int(self.do_calib_camera_pose)}")
        print_debug(f"  - calib_cam_intrinsics: {int(self.do_calib_camera_intrinsics)}")
        print_debug(f"  - calib_cam_timeoffset: {int(self.do_calib_camera_timeoffset)}")
        print_debug(f"  - max_clones: {self.max_clone_size}")
        print_debug(f"  - max_slam: {self.max_slam_features}")
        print_debug(f"  - max_slam_in_update: {self.max_slam_in_update}")
        print_debug(f"  - max_msckf_in_update: {self.max_msckf_in_update}")
        print_debug(f"  - max_aruco: {self.max_aruco_features}")
        print_debug(f"  - max_cameras: {self.num_cameras}")
        print_debug(f"  - feat_rep_msckf: {as_string(self.feat_rep_msckf)}")
        print_debug(f"  - feat_rep_slam: {as_string(self.feat_rep_slam)}")
        print_debug(f"  - feat_rep_aruco: {as_string(self.feat_rep_aruco)}")
