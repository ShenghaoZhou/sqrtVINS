/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
 * Copyright (C) 2018-2026 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program. If not, see
 * <https://www.gnu.org/licenses/>.
 */

#ifndef OV_SRVINS_COLMAPMAPADAPTER_H
#define OV_SRVINS_COLMAPMAPADAPTER_H

#include <Eigen/Eigen>

#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"
#include "utils/DataType.h"

namespace ov_core {
class CamBase;
}

namespace ov_srvins {

/**
 * @brief Conversions between SqrtVINS/OpenVINS and colmap conventions.
 *
 * Convention mapping (the IMU is the colmap rig's reference sensor):
 *  - OpenVINS stores a clone as (R_GtoI, p_IinG): rotation taking global
 *    coordinates into the IMU frame, and the IMU origin in global.
 *    colmap stores rig_from_world = [R | t] with x_rig = R * x_world + t,
 *    hence R = R_GtoI and t = -R_GtoI * p_IinG.
 *  - OpenVINS stores the camera extrinsic as (R_ItoC, p_IinC): IMU-frame
 *    coordinates into the camera frame, IMU origin expressed in the camera
 *    frame. This is exactly colmap's sensor_from_rig (cam_from_rig).
 *  - colmap quaternions are Hamilton (w first in the API, Eigen storage);
 *    OpenVINS uses JPL. All conversions go through rotation matrices, never
 *    raw quaternion arrays.
 *  - colmap keypoints are distorted pixel coordinates, matching the uv
 *    measurements stored in the OpenVINS feature database.
 */
namespace colmap_adapter {

/// Clone/IMU pose (R_GtoI, p_IinG) to colmap rig_from_world
colmap::Rigid3d rig_from_world(const Mat3 &R_GtoI, const Vec3 &p_IinG);

/// Inverse of rig_from_world(): recover (R_GtoI, p_IinG)
void rig_from_world_to_clone(const colmap::Rigid3d &rig_from_world,
                             Mat3 &R_GtoI, Vec3 &p_IinG);

/// Camera extrinsic (R_ItoC, p_IinC) to colmap sensor_from_rig (cam_from_imu)
colmap::Rigid3d sensor_from_rig(const Mat3 &R_ItoC, const Vec3 &p_IinC);

/// Compose cam_from_world = sensor_from_rig * rig_from_world as a 3x4
/// projection matrix (for multiview triangulation)
Eigen::Matrix3x4d cam_from_world_matrix(const colmap::Rigid3d &sensor_from_rig,
                                        const colmap::Rigid3d &rig_from_world);

/// Build a colmap camera from an ov camera model + intrinsics vector
/// (fx fy cx cy + 4 distortion coefficients). CamRadtan maps to colmap
/// OPENCV, CamEqui to OPENCV_FISHEYE.
colmap::Camera camera_from_ov(colmap::camera_t cam_id, ov_core::CamBase &cam,
                              const VecX &intrinsics);

} // namespace colmap_adapter
} // namespace ov_srvins

#endif // OV_SRVINS_COLMAPMAPADAPTER_H
