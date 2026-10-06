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

#include "backend/BackendSystem.h"

#include <algorithm>
#include <cfloat>
#include <fstream>
#include <iomanip>
#include <unordered_map>

#include "backend/ColmapMapAdapter.h"
#include "cam/CamBase.h"
#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/State.h"
#include "types/PoseJPL.h"
#include "utils/print.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/geometry/triangulation.h"
#include "colmap/scene/projection.h"
#include "colmap/scene/reconstruction.h"

namespace ov_srvins {

/// colmap id of the IMU reference sensor (ids are namespaced per sensor type)
static constexpr colmap::sensor_t kImuSensorId(colmap::SensorType::IMU, 1);

BackendSystem::BackendSystem(const BackendOptions &opts) : opts_(opts) {}

void BackendSystem::record_observations(double timestamp,
                                        ov_core::FeatureDatabase &db) {
  // Start a keyframe only every keyframe_stride calls; intermediate frames
  // are not recorded at all (offline decimation)
  const bool is_keyframe = (record_count_ % opts_.keyframe_stride) == 0;
  record_count_++;
  if (!is_keyframe)
    return;

  Keyframe kf;
  kf.timestamp = timestamp;
  for (const auto &feat_pair : db.get_internal_data()) {
    const auto &feat = feat_pair.second;
    for (const auto &cam_pair : feat->timestamps) {
      const size_t cam_id = cam_pair.first;
      const auto &times = cam_pair.second;
      const auto it = std::find(times.begin(), times.end(), timestamp);
      if (it == times.end())
        continue;
      const size_t idx = std::distance(times.begin(), it);
      const auto &uv = feat->uvs.at(cam_id).at(idx);
      kf.obs.push_back({feat->featid, cam_id, uv.cast<double>()});
    }
  }
  keyframes_.push_back(std::move(kf));
  keyframe_pending_pose_ = true;
}

bool BackendSystem::record_pose(const State &state) {
  if (!keyframe_pending_pose_)
    return false;
  keyframe_pending_pose_ = false;
  if (state.clones_IMU.empty())
    return false;

  const auto &clone = state.clones_IMU.rbegin()->second;
  Keyframe &kf = keyframes_.back();
  if (std::abs(state.clones_IMU.rbegin()->first - kf.timestamp) > 1e-9) {
    PRINT_ERROR(YELLOW
                "[BACKEND]: keyframe pose/observation timestamp mismatch "
                "(%.9f vs %.9f), dropping keyframe\n" RESET,
                state.clones_IMU.rbegin()->first, kf.timestamp);
    return false;
  }
  kf.R_GtoI = clone->Rot();
  kf.p_IinG = clone->pos();
  kf.has_pose = true;
  return true;
}

std::unique_ptr<colmap::Reconstruction>
BackendSystem::build_reconstruction(const std::vector<Keyframe> &keyframes_in,
                                    State &state,
                                    std::vector<double> &frame_timestamps) {
  auto recon = std::make_unique<colmap::Reconstruction>();

  // --- cameras + rig (IMU reference sensor, cameras as rig members) ---
  for (const auto &cam_pair : state.cam_intrinsics_cameras) {
    const size_t cam_id = cam_pair.first;
    recon->AddCamera(colmap_adapter::camera_from_ov(
        cam_id, *cam_pair.second, state.cam_intrinsics.at(cam_id)->value()));
  }

  colmap::Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(kImuSensorId);
  for (const auto &calib_pair : state.calib_IMUtoCAM) {
    const size_t cam_id = calib_pair.first;
    rig.AddSensor(colmap::sensor_t(colmap::SensorType::CAMERA, cam_id),
                  colmap_adapter::sensor_from_rig(calib_pair.second->Rot(),
                                                  calib_pair.second->pos()));
  }
  recon->AddRig(rig);

  // Drop pose-less/empty keyframes up front so frame ids stay consecutive
  std::vector<const Keyframe *> keyframes;
  for (const auto &kf : keyframes_in) {
    if (!kf.has_pose || kf.obs.empty())
      continue;
    keyframes.push_back(&kf);
  }

  // --- frames + images (one frame per keyframe, one image per camera) -------
  // Tracks are collected in the same pass: the point2D index of an
  // observation is its insertion rank within the (keyframe, camera) image,
  // which we track exactly while filling each image's points2D.
  struct TrackObs {
    int kf_idx;
    size_t cam_id;
    colmap::image_t image_id;
    colmap::point2D_t point2D_idx;
    Eigen::Vector2d uv;
  };
  std::unordered_map<size_t, std::vector<TrackObs>> tracks;

  frame_timestamps.clear();
  colmap::image_t next_image_id = 1;
  for (size_t i = 0; i < keyframes.size(); i++) {
    const Keyframe &kf = *keyframes[i];
    const colmap::frame_t frame_id = i + 1;
    frame_timestamps.push_back(kf.timestamp);

    colmap::Frame frame;
    frame.SetFrameId(frame_id);
    frame.SetRigId(1);
    frame.SetRigFromWorld(colmap_adapter::rig_from_world(kf.R_GtoI, kf.p_IinG));

    // collect this keyframe's observations per camera (insertion order =
    // point2D index order)
    std::map<size_t, std::vector<colmap::Point2D>> points_per_cam;
    std::unordered_map<size_t, colmap::image_t> image_id_of_cam;
    for (const auto &ob : kf.obs) {
      auto it = image_id_of_cam.find(ob.cam_id);
      if (it == image_id_of_cam.end()) {
        it = image_id_of_cam.emplace(ob.cam_id, next_image_id++).first;
      }
      auto &points = points_per_cam[ob.cam_id];
      const colmap::point2D_t idx = points.size();
      colmap::Point2D p2d;
      p2d.xy = ob.uv;
      points.push_back(p2d);
      tracks[ob.feat_id].push_back(
          {static_cast<int>(i), ob.cam_id, it->second, idx, ob.uv});
    }

    std::vector<colmap::Image> images;
    for (auto &cam_points : points_per_cam) {
      const size_t cam_id = cam_points.first;
      const colmap::image_t image_id = image_id_of_cam.at(cam_id);
      frame.AddDataId(colmap::data_t(
          colmap::sensor_t(colmap::SensorType::CAMERA, cam_id), image_id));
      colmap::Image image;
      image.SetImageId(image_id);
      image.SetCameraId(cam_id);
      image.SetFrameId(frame_id);
      image.SetName("kf" + std::to_string(frame_id) + "_cam" +
                    std::to_string(cam_id));
      image.SetPoints2D(cam_points.second);
      images.push_back(std::move(image));
    }

    recon->AddFrame(frame); // registered automatically (pose is set)
    for (auto &image : images) {
      recon->AddImage(std::move(image));
    }
  }

  // --- triangulate surviving tracks and add them as Point3D -----------------
  std::vector<Eigen::Matrix3x4d> cams_from_world;
  std::vector<Eigen::Vector2d> points_norm;
  for (const auto &track_pair : tracks) {
    const size_t feat_id = track_pair.first;
    const auto &obs = track_pair.second;
    if (obs.size() < opts_.min_track_length)
      continue;

    cams_from_world.clear();
    points_norm.clear();
    bool valid = true;
    for (const auto &ob : obs) {
      const Keyframe &kf = *keyframes[ob.kf_idx];
      const auto &calib = state.calib_IMUtoCAM.at(ob.cam_id);
      cams_from_world.push_back(colmap_adapter::cam_from_world_matrix(
          colmap_adapter::sensor_from_rig(calib->Rot(), calib->pos()),
          colmap_adapter::rig_from_world(kf.R_GtoI, kf.p_IinG)));
      // undistort to normalized coordinates for triangulation
      auto cam_it = state.cam_intrinsics_cameras.find(ob.cam_id);
      if (cam_it == state.cam_intrinsics_cameras.end()) {
        valid = false;
        break;
      }
      const Vec2 uv_norm = cam_it->second->undistort(ob.uv.cast<DataType>());
      points_norm.push_back(uv_norm.cast<double>());
    }
    if (!valid)
      continue;

    Eigen::Vector3d xyz;
    const colmap::span<const Eigen::Matrix3x4d> poses_span(
        cams_from_world.data(), cams_from_world.size());
    const colmap::span<const Eigen::Vector2d> points_span(points_norm.data(),
                                                          points_norm.size());
    if (!colmap::TriangulateMultiViewPoint(poses_span, points_span, &xyz))
      continue;

    // reject degenerate triangulations: behind a camera or inconsistent
    // with the filter poses (loose gate; BA's robust loss refines the rest)
    double err2_sum = 0;
    bool bad = false;
    for (const auto &ob : obs) {
      const Keyframe &kf = *keyframes[ob.kf_idx];
      const auto &calib = state.calib_IMUtoCAM.at(ob.cam_id);
      const colmap::Rigid3d cam_from_world =
          colmap_adapter::sensor_from_rig(calib->Rot(), calib->pos()) *
          colmap_adapter::rig_from_world(kf.R_GtoI, kf.p_IinG);
      const double e2 = colmap::CalculateSquaredReprojectionError(
          ob.uv, xyz, cam_from_world, recon->Camera(ob.cam_id));
      if (e2 == DBL_MAX) { // behind the camera
        bad = true;
        break;
      }
      err2_sum += e2;
    }
    const double tri_thr2 =
        opts_.max_triang_error_px * opts_.max_triang_error_px;
    if (bad || err2_sum / obs.size() > tri_thr2) {
      num_triang_rejected_++;
      continue;
    }

    // keep the feature id in the point3D id (0 is invalid in colmap)
    const colmap::point3D_t point3D_id = feat_id + 1;
    colmap::Point3D point3D;
    point3D.xyz = xyz;
    recon->AddPoint3D(point3D_id, point3D);

    for (const auto &ob : obs) {
      auto &image = recon->Image(ob.image_id);
      image.SetPoint3DForPoint2D(ob.point2D_idx, point3D_id);
      recon->Point3D(point3D_id)
          .track.AddElement(ob.image_id, ob.point2D_idx);
    }
  }

  return recon;
}

BackendSummary BackendSystem::solve_and_export(
    colmap::Reconstruction &recon, const std::vector<double> &frame_timestamps,
    const std::string &traj_out_path) {
  BackendSummary summary;
  summary.num_keyframes = static_cast<int>(frame_timestamps.size());
  summary.num_images = static_cast<int>(recon.NumImages());
  summary.num_points = static_cast<int>(recon.NumPoints3D());
  summary.num_observations = recon.ComputeNumObservations();
  if (recon.NumRegFrames() < 2 || recon.NumPoints3D() == 0) {
    PRINT_ERROR(RED "[BACKEND]: not enough registered frames/points for BA\n"
                    RESET);
    return summary;
  }

  // errors are cached on the points: refresh before reading
  recon.UpdatePoint3DErrors();
  summary.mean_reproj_error_before = recon.ComputeMeanReprojectionError();

  // --- BA configuration: vision-only refinement of frame poses + points ----
  colmap::BundleAdjustmentConfig config;
  for (const auto &image_pair : recon.Images()) {
    config.AddImage(image_pair.first);
  }
  for (const auto &cam_pair : recon.Cameras()) {
    config.SetConstantCamIntrinsics(cam_pair.first);
    config.SetConstantSensorFromRigPose(
        colmap::sensor_t(colmap::SensorType::CAMERA, cam_pair.first));
  }
  // anchor the gauge to the filter frame: oldest keyframe pose constant
  config.SetConstantRigFromWorldPose(1);

  colmap::BundleAdjustmentOptions ba_options;
  ba_options.refine_focal_length = false;
  ba_options.refine_principal_point = false;
  ba_options.refine_extra_params = false;
  ba_options.refine_sensor_from_rig = false;
  ba_options.refine_rig_from_world = true;
  ba_options.refine_points3D = true;
  ba_options.print_summary = false;
  if (opts_.loss_scale > 0) {
    ba_options.ceres->loss_function_type =
        colmap::CeresLossFunctionType::CAUCHY;
    ba_options.ceres->loss_function_scale = opts_.loss_scale;
  }
  ba_options.ceres->solver_options.max_num_iterations =
      opts_.max_num_iterations;
  ba_options.ceres->solver_options.num_threads = opts_.num_threads;

  auto adjuster =
      colmap::CreateDefaultCeresBundleAdjuster(ba_options, config, recon);
  auto ba_summary = adjuster->Solve();
  summary.solved = ba_summary->IsSolutionUsable();
  if (opts_.print_summary) {
    PRINT_INFO("[BACKEND]: first solve: %s",
               ba_summary->BriefReport().c_str());
  }
  if (!summary.solved)
    return summary;

  recon.UpdatePoint3DErrors();
  summary.mean_reproj_error_after = recon.ComputeMeanReprojectionError();

  // --- prune outlier points, then (optionally) re-solve --------------------
  if (opts_.max_reproj_error_px > 0) {
    const double thr2 = opts_.max_reproj_error_px * opts_.max_reproj_error_px;
    std::vector<colmap::point3D_t> to_delete;
    for (const auto &point_pair : recon.Points3D()) {
      const auto &point = point_pair.second;
      double err2_sum = 0;
      for (const auto &el : point.track.Elements()) {
        const auto &image = recon.Image(el.image_id);
        const auto &camera = recon.Camera(image.CameraId());
        const auto &frame = recon.Frame(image.FrameId());
        const auto &rig = recon.Rig(frame.RigId());
        const colmap::Rigid3d cam_from_world =
            rig.SensorFromRig(camera.SensorId()) * frame.RigFromWorld();
        err2_sum += colmap::CalculateSquaredReprojectionError(
            image.Point2D(el.point2D_idx).xy, point.xyz, cam_from_world,
            camera);
      }
      if (err2_sum / point.track.Length() > thr2)
        to_delete.push_back(point_pair.first);
    }
    summary.num_pruned_points = static_cast<int>(to_delete.size());
    for (const auto id : to_delete) {
      recon.DeletePoint3D(id);
    }

    if (opts_.refine_after_pruning && !to_delete.empty()) {
      adjuster =
          colmap::CreateDefaultCeresBundleAdjuster(ba_options, config, recon);
      ba_summary = adjuster->Solve();
      summary.solved = ba_summary->IsSolutionUsable();
      if (opts_.print_summary) {
        PRINT_INFO("[BACKEND]: prune+resolve: %s",
                   ba_summary->BriefReport().c_str());
      }
    }
    recon.UpdatePoint3DErrors();
    summary.mean_reproj_error_final = recon.ComputeMeanReprojectionError();
  }

  // --- export the refined trajectory in the filter's output format ---------
  if (!traj_out_path.empty()) {
    std::ofstream out(traj_out_path);
    out << std::setprecision(9) << std::fixed;
    for (size_t i = 0; i < frame_timestamps.size(); i++) {
      const colmap::frame_t frame_id = i + 1;
      if (!recon.ExistsFrame(frame_id))
        continue;
      Mat3 R_GtoI;
      Vec3 p_IinG;
      colmap_adapter::rig_from_world_to_clone(
          recon.Frame(frame_id).RigFromWorld(), R_GtoI, p_IinG);
      const Eigen::Quaternion<DataType> q_ItoG(R_GtoI.transpose());
      out << frame_timestamps[i] << " " << p_IinG(0) << " " << p_IinG(1) << " "
          << p_IinG(2) << " " << q_ItoG.x() << " " << q_ItoG.y() << " "
          << q_ItoG.z() << " " << q_ItoG.w() << "\n";
    }
  }
  return summary;
}

BackendSummary BackendSystem::run_offline_ba(State &state,
                                             const std::string &traj_out_path) {
  num_triang_rejected_ = 0;
  std::vector<double> frame_timestamps;
  auto recon = build_reconstruction(keyframes_, state, frame_timestamps);
  PRINT_INFO("[BACKEND]: built reconstruction: %zu keyframes, %d images, "
             "%zu points (%d triangulations rejected)\n",
             frame_timestamps.size(), static_cast<int>(recon->NumImages()),
             recon->NumPoints3D(), num_triang_rejected_);
  return solve_and_export(*recon, frame_timestamps, traj_out_path);
}

} // namespace ov_srvins
