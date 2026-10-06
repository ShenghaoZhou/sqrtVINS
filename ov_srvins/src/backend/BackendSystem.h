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

#ifndef OV_SRVINS_BACKENDSYSTEM_H
#define OV_SRVINS_BACKENDSYSTEM_H

#include <cstddef>
#include <string>
#include <vector>

#include <Eigen/Eigen>

#include "backend/BackendOptions.h"
#include "utils/DataType.h"

namespace colmap {
class Reconstruction;
}

namespace ov_core {
class FeatureDatabase;
}

namespace ov_srvins {

class State;

/// Outcome of an offline BA run (costs in px for interpretability)
struct BackendSummary {
  bool solved = false;
  int num_keyframes = 0;
  int num_images = 0;
  int num_points = 0;
  int num_observations = 0;
  /// Mean reprojection error [px] with the filter poses (pre-BA)
  double mean_reproj_error_before = -1;
  /// Mean reprojection error [px] after the first solve
  double mean_reproj_error_after = -1;
  /// Mean reprojection error [px] after pruning + re-solve (if enabled)
  double mean_reproj_error_final = -1;
  int num_pruned_points = 0;
};

/**
 * @brief Bundle-adjustment backend (Phase 1: offline, vision-only).
 *
 * Records keyframe poses and feature observations during the filter run
 * (record_observations BEFORE the visual update, record_pose AFTER it),
 * then assembles a colmap::Reconstruction — one rig whose reference sensor
 * is the IMU, one frame per keyframe, one image per camera, one Point3D per
 * surviving feature track — and refines it with colmap's Ceres bundle
 * adjuster. Camera intrinsics and cam-IMU extrinsics are held constant; the
 * oldest keyframe pose is held constant to anchor the gauge to the filter
 * frame so trajectories stay comparable.
 */
class BackendSystem {
public:
  /// One 2D measurement of a feature track at a keyframe
  struct Observation {
    size_t feat_id;
    size_t cam_id;
    Eigen::Vector2d uv; // distorted pixels (as stored in the feature db)
  };

  /// Recorded keyframe: filter pose + all track observations at that time
  struct Keyframe {
    double timestamp = -1;
    bool has_pose = false;
    Mat3 R_GtoI = Mat3::Identity();
    Vec3 p_IinG = Vec3::Zero();
    std::vector<Observation> obs;
  };

  explicit BackendSystem(const BackendOptions &opts);

  /// Snapshot feature observations at `timestamp` (call BEFORE the visual
  /// update, after feed_camera, while tracks consumed by the update are
  /// still in the database). Starts a new keyframe every
  /// BackendOptions::keyframe_stride calls.
  void record_observations(double timestamp, ov_core::FeatureDatabase &db);

  /// Attach the latest clone pose to the pending keyframe (call AFTER the
  /// visual update). Returns false if no keyframe is pending.
  bool record_pose(const State &state);

  size_t num_keyframes() const { return keyframes_.size(); }

  /// Assemble the reconstruction, triangulate landmarks, run BA, and write
  /// the refined trajectory (same format as the filter output:
  /// timestamp px py pz qx qy qz qw) to traj_out_path.
  BackendSummary run_offline_ba(State &state, const std::string &traj_out_path);

  /// Solve + prune (+ re-solve) + export on an already-assembled
  /// reconstruction. Exposed for unit tests with synthetic maps.
  /// frame_timestamps must be ordered by ascending frame id (1..N).
  BackendSummary solve_and_export(colmap::Reconstruction &recon,
                                  const std::vector<double> &frame_timestamps,
                                  const std::string &traj_out_path);

private:
  /// Build the colmap reconstruction from the recorded keyframes (poses,
  /// stereo rig, triangulated tracks). Keyframes without a pose are dropped.
  std::unique_ptr<colmap::Reconstruction>
  build_reconstruction(const std::vector<Keyframe> &keyframes, State &state,
                       std::vector<double> &frame_timestamps);

  BackendOptions opts_;
  std::vector<Keyframe> keyframes_;
  int num_triang_rejected_ = 0;
  int record_count_ = 0;
  bool keyframe_pending_pose_ = false;
};

} // namespace ov_srvins

#endif // OV_SRVINS_BACKENDSYSTEM_H
