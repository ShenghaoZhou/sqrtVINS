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

#include "backend/ColmapMapAdapter.h"

#include "cam/CamEqui.h"
#include "cam/CamRadtan.h"
#include "colmap/sensor/models.h"

namespace ov_srvins {
namespace colmap_adapter {

colmap::Rigid3d rig_from_world(const Mat3 &R_GtoI, const Vec3 &p_IinG) {
  const Eigen::Matrix3d R = R_GtoI.cast<double>();
  const Eigen::Vector3d p = p_IinG.cast<double>();
  // x_imu = R_GtoI * x_world + t  =>  t = -R_GtoI * p_IinG
  return colmap::Rigid3d(Eigen::Quaterniond(R), -R * p);
}

void rig_from_world_to_clone(const colmap::Rigid3d &rig_from_world, Mat3 &R_GtoI,
                             Vec3 &p_IinG) {
  const Eigen::Matrix3d R = rig_from_world.rotation().toRotationMatrix();
  // p_IinG = -R^T * t
  const Eigen::Vector3d p = -(R.transpose() * rig_from_world.translation());
  R_GtoI = R.cast<DataType>();
  p_IinG = p.cast<DataType>();
}

colmap::Rigid3d sensor_from_rig(const Mat3 &R_ItoC, const Vec3 &p_IinC) {
  // x_cam = R_ItoC * x_imu + p_IinC: already the cam_from_rig transform
  return colmap::Rigid3d(Eigen::Quaterniond(R_ItoC.cast<double>()),
                         p_IinC.cast<double>());
}

Eigen::Matrix3x4d
cam_from_world_matrix(const colmap::Rigid3d &sensor_from_rig,
                      const colmap::Rigid3d &rig_from_world) {
  const colmap::Rigid3d cam_from_world = sensor_from_rig * rig_from_world;
  return cam_from_world.ToMatrix();
}

colmap::Camera camera_from_ov(colmap::camera_t cam_id, ov_core::CamBase &cam,
                              const VecX &intrinsics) {
  // OpenVINS intrinsics layout: fx fy cx cy + 4 distortion coefficients.
  // CamRadtan (k1 k2 p1 p2) matches colmap OPENCV; CamEqui (k1 k2 k3 k4)
  // matches colmap OPENCV_FISHEYE.
  const bool is_fisheye =
      dynamic_cast<ov_core::CamEqui *>(&cam) != nullptr;
  const colmap::CameraModelId model_id =
      is_fisheye ? colmap::CameraModelId::kOpenCVFisheye
                 : colmap::CameraModelId::kOpenCV;

  colmap::Camera camera = colmap::Camera::CreateFromModelId(
      cam_id, model_id, intrinsics(0), cam.w(), cam.h());
  camera.params.resize(8);
  for (int i = 0; i < 8; i++) {
    camera.params[i] = intrinsics(i);
  }
  camera.has_prior_focal_length = true;
  return camera;
}

} // namespace colmap_adapter
} // namespace ov_srvins
