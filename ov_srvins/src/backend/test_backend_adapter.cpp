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

/**
 * @brief Round-trip and smoke tests for the colmap BA backend (Phase 1).
 *
 *  1. Pose conversion round trip (clone <-> rig_from_world)
 *  2. Camera model consistency (ov CamRadtan distort vs colmap OPENCV)
 *  3. Synthetic stereo-rig BA: noise-free observations from perturbed
 *     initial values must re-converge to ~zero reprojection error.
 */

#include <cassert>
#include <cstdio>
#include <random>
#include <type_traits>

#include "backend/BackendSystem.h"
#include "backend/ColmapMapAdapter.h"
#include "cam/CamRadtan.h"

#include "colmap/scene/reconstruction.h"

using namespace ov_srvins;

#define CHECK_TRUE(cond, msg)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("[FAIL] %s:%d: %s\n", __FILE__, __LINE__, msg);                   \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

static int test_pose_roundtrip() {
  // DataType may be float: pick a tolerance matching the storage precision
  const double tol =
      std::is_same<DataType, float>::value ? 1e-5 : 1e-10;
  std::mt19937 gen(42);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int trial = 0; trial < 100; trial++) {
    const Eigen::Vector3d axis(nd(gen), nd(gen), nd(gen));
    if (axis.norm() < 1e-6)
      continue;
    const Eigen::Matrix3d R_true =
        Eigen::AngleAxisd(0.3 * trial / 100.0, axis.normalized())
            .toRotationMatrix();
    const Eigen::Vector3d p_true(nd(gen), nd(gen), nd(gen));

    const Mat3 R_in = R_true.cast<DataType>();
    const Vec3 p_in = p_true.cast<DataType>();
    Mat3 R_out;
    Vec3 p_out;
    colmap_adapter::rig_from_world_to_clone(
        colmap_adapter::rig_from_world(R_in, p_in), R_out, p_out);

    CHECK_TRUE((R_out.cast<double>() - R_true).norm() < tol,
               "rotation round trip mismatch");
    CHECK_TRUE((p_out.cast<double>() - p_true).norm() < tol,
               "position round trip mismatch");
  }
  printf("[PASS] pose round trip\n");
  return EXIT_SUCCESS;
}

static int test_camera_consistency() {
  // fx fy cx cy k1 k2 p1 p2
  VecX intrinsics(8);
  intrinsics << 400, 405, 320, 240, -0.05, 0.01, 0.001, -0.002;
  ov_core::CamRadtan ov_cam(640, 480);
  ov_cam.set_value(intrinsics);

  colmap::Camera colmap_cam =
      colmap_adapter::camera_from_ov(0, ov_cam, intrinsics);
  CHECK_TRUE(colmap_cam.model_id == colmap::CameraModelId::kOpenCV,
             "model mapping should be OPENCV");
  CHECK_TRUE(colmap_cam.width == 640 && colmap_cam.height == 480,
             "image size mismatch");

  std::mt19937 gen(7);
  std::uniform_real_distribution<double> ud(-0.8, 0.8);
  for (int trial = 0; trial < 100; trial++) {
    // point in camera frame, in front of the camera
    const Eigen::Vector3d p_cam(ud(gen), ud(gen), 2.0 + trial * 0.01);
    const Vec2 uv_norm = (p_cam.hnormalized()).cast<DataType>();
    const Vec2 uv_ov = ov_cam.distort(uv_norm);
    const auto uv_colmap = colmap_cam.ImgFromCam(p_cam);
    CHECK_TRUE(uv_colmap.has_value(), "colmap projection failed");
    CHECK_TRUE((uv_ov.cast<double>() - *uv_colmap).norm() <
                 (std::is_same<DataType, float>::value ? 1e-3 : 1e-6),
               "ov/colmap projection mismatch");
  }
  printf("[PASS] camera projection consistency\n");
  return EXIT_SUCCESS;
}

/// Assemble a synthetic stereo-rig reconstruction (same wiring pattern as
/// BackendSystem::build_reconstruction) with perturbed poses/points
static int test_synthetic_ba(const std::string &traj_out) {
  constexpr size_t kNumFrames = 20;
  constexpr size_t kNumPoints = 200;
  constexpr int kWidth = 640, kHeight = 480;

  std::mt19937 gen(123);
  std::uniform_real_distribution<double> ud(-1.0, 1.0);
  std::normal_distribution<double> noise(0.0, 1.0);

  // ground truth rig: IMU reference, cam0 identity, cam1 shifted by baseline
  const colmap::sensor_t imu_sensor(colmap::SensorType::IMU, 1);
  const colmap::sensor_t cam0_sensor(colmap::SensorType::CAMERA, 0);
  const colmap::sensor_t cam1_sensor(colmap::SensorType::CAMERA, 1);
  const colmap::Rigid3d cam1_from_rig(Eigen::Quaterniond::Identity(),
                                      Eigen::Vector3d(0.1, 0, 0));

  colmap::Reconstruction recon;

  colmap::Camera camera = colmap::Camera::CreateFromModelId(
      0, colmap::CameraModelId::kOpenCV, 400, kWidth, kHeight);
  camera.params = {400, 405, 320, 240, 0, 0, 0, 0};
  recon.AddCamera(camera);
  camera.camera_id = 1;
  recon.AddCamera(camera);

  // rig references the cameras: they must be added first
  colmap::Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(imu_sensor);
  rig.AddSensor(cam0_sensor, colmap::Rigid3d());
  rig.AddSensor(cam1_sensor, cam1_from_rig);
  recon.AddRig(rig);

  // ground truth: straight line motion with a gentle sway in rotation
  std::vector<colmap::Rigid3d> gt_rig_from_world;
  for (size_t i = 0; i < kNumFrames; i++) {
    const Eigen::Matrix3d R_ItoG =
        Eigen::AngleAxisd(0.02 * i, Eigen::Vector3d::UnitY())
            .toRotationMatrix();
    const Eigen::Vector3d p_IinG(0.2 * i, 0.05 * std::sin(0.3 * i), 0);
    gt_rig_from_world.emplace_back(Eigen::Quaterniond(R_ItoG.transpose()),
                                   -R_ItoG.transpose() * p_IinG);
  }

  // ground truth points in a volume ahead of the trajectory
  std::vector<Eigen::Vector3d> gt_points;
  for (size_t j = 0; j < kNumPoints; j++) {
    gt_points.emplace_back(2.0 * j / kNumPoints + ud(gen), 2.0 * ud(gen),
                           6.0 + 4.0 * (j % 7) / 7.0 + ud(gen));
  }

  // frames, images, and noise-free observations from perturbed values
  std::vector<double> timestamps;
  colmap::point3D_t next_point_id = 1;
  colmap::image_t next_image_id = 1;
  for (size_t i = 0; i < kNumFrames; i++) {
    const colmap::frame_t frame_id = i + 1;
    timestamps.push_back(0.05 * i);

    // perturbed initial pose (keep frame 1 exact: it is the constant anchor)
    colmap::Rigid3d init = gt_rig_from_world[i];
    if (i > 0) {
      const Eigen::Vector3d rot_noise(0.002 * noise(gen),
                                      0.002 * noise(gen),
                                      0.002 * noise(gen));
      const Eigen::Quaterniond q_noise(
          1.0, rot_noise.x() / 2, rot_noise.y() / 2, rot_noise.z() / 2);
      init.rotation() = (q_noise.normalized() * init.rotation()).normalized();
      init.translation() +=
          0.03 * Eigen::Vector3d(noise(gen), noise(gen), noise(gen));
    }

    colmap::Frame frame;
    frame.SetFrameId(frame_id);
    frame.SetRigId(1);
    frame.SetRigFromWorld(init);

    std::vector<colmap::Image> images;
    std::vector<std::vector<std::pair<colmap::point3D_t, colmap::point2D_t>>>
        obs_per_image(2);
    for (size_t cam_id = 0; cam_id < 2; cam_id++) {
      const colmap::image_t image_id = next_image_id++;
      const colmap::Rigid3d cam_from_world =
          (cam_id == 0 ? colmap::Rigid3d() : cam1_from_rig) *
          gt_rig_from_world[i];
      std::vector<colmap::Point2D> points2D;
      for (size_t j = 0; j < kNumPoints; j++) {
        const Eigen::Vector3d p_cam = cam_from_world * gt_points[j];
        if (p_cam.z() < 0.5)
          continue;
        const auto uv = recon.Camera(cam_id).ImgFromCam(p_cam);
        if (!uv.has_value() || (*uv)(0) < 0 || (*uv)(0) >= kWidth ||
            (*uv)(1) < 0 || (*uv)(1) >= kHeight)
          continue;
        const colmap::point3D_t point_id = next_point_id + j;
        obs_per_image[cam_id].emplace_back(point_id, points2D.size());
        colmap::Point2D p2d;
        p2d.xy = *uv;
        points2D.push_back(p2d);
      }
      frame.AddDataId(colmap::data_t(
          colmap::sensor_t(colmap::SensorType::CAMERA, cam_id), image_id));
      colmap::Image image;
      image.SetImageId(image_id);
      image.SetCameraId(cam_id);
      image.SetFrameId(frame_id);
      image.SetPoints2D(points2D);
      images.push_back(std::move(image));
    }
    recon.AddFrame(frame);
    for (auto &image : images) {
      recon.AddImage(std::move(image));
    }

    // register observations into shared 3D points (perturbed initial values)
    for (size_t cam_id = 0; cam_id < 2; cam_id++) {
      const colmap::image_t image_id =
          next_image_id - 2 + static_cast<colmap::image_t>(cam_id);
      for (const auto &[point_id, idx] : obs_per_image[cam_id]) {
        if (!recon.ExistsPoint3D(point_id)) {
          colmap::Point3D point;
          point.xyz =
              gt_points[point_id - next_point_id] +
              0.1 * Eigen::Vector3d(noise(gen), noise(gen), noise(gen));
          recon.AddPoint3D(point_id, point);
        }
        recon.Image(image_id).SetPoint3DForPoint2D(idx, point_id);
        recon.Point3D(point_id).track.AddElement(image_id, idx);
      }
    }
  }
  next_point_id += kNumPoints;

  printf("[INFO] synthetic map: %d images, %zu points, %d observations\n",
         static_cast<int>(recon.NumImages()), recon.NumPoints3D(),
         static_cast<int>(recon.ComputeNumObservations()));

  BackendOptions opts;
  opts.loss_scale = 0;             // noise-free observations
  opts.max_reproj_error_px = -1;   // no pruning for the smoke test
  opts.print_summary = true;
  BackendSystem backend(opts, NoiseManager(), 9.81);
  BackendSummary summary =
      backend.solve_and_export(recon, timestamps, traj_out);

  printf("[INFO] reproj error: before=%.4f px, after=%.6f px\n",
         summary.mean_reproj_error_before, summary.mean_reproj_error_after);
  CHECK_TRUE(summary.solved, "BA solve failed");
  CHECK_TRUE(summary.mean_reproj_error_after <
                 summary.mean_reproj_error_before,
             "BA did not reduce reprojection error");
  CHECK_TRUE(summary.mean_reproj_error_after < 0.05,
             "BA did not re-converge on noise-free observations");

  // refined poses should be (near) ground truth
  double max_trans_err = 0;
  for (size_t i = 0; i < kNumFrames; i++) {
    const auto &refined = recon.Frame(i + 1).RigFromWorld();
    const Eigen::Vector3d p_gt =
        -(gt_rig_from_world[i].rotation().toRotationMatrix().transpose() *
          gt_rig_from_world[i].translation());
    const Eigen::Vector3d p_ref =
        -(refined.rotation().toRotationMatrix().transpose() *
          refined.translation());
    max_trans_err = std::max(max_trans_err, (p_gt - p_ref).norm());
  }
  printf("[INFO] max translation error vs GT: %.6f m\n", max_trans_err);
  CHECK_TRUE(max_trans_err < 0.02, "refined poses far from ground truth");
  printf("[PASS] synthetic stereo-rig BA\n");
  return EXIT_SUCCESS;
}

int main() {
  if (test_pose_roundtrip() != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (test_camera_consistency() != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (test_synthetic_ba("/tmp/test_backend_adapter_traj.txt") != EXIT_SUCCESS)
    return EXIT_FAILURE;
  printf("[PASS] all backend adapter tests\n");
  return EXIT_SUCCESS;
}
