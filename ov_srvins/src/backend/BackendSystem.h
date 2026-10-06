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

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <Eigen/Eigen>

#include "backend/BackendOptions.h"
#include "backend/RelativePoseFactor.h"
#include "utils/DataType.h"
#include "utils/NoiseManager.h"
#include "utils/sensor_data.h"

namespace colmap {
class Reconstruction;
class CeresBundleAdjuster;
}

namespace ov_core {
class FeatureDatabase;
class CamBase;
}

namespace ov_srvins {

class State;
struct ImuFactorData;

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

/// IMU constraints for the BA solve (Phase 2): per-keyframe velocity/bias
/// blocks plus the preintegrated factors between consecutive keyframes.
/// The `sb` storage backs the Ceres parameter blocks and must stay alive
/// (and un-reallocated) across Solve().
struct ImuConstraints {
  Eigen::Vector3d gravity = Eigen::Vector3d(0, 0, 9.81);
  /// per-keyframe [v_IinG(3) bg(3) ba(3)] blocks (initial values from the
  /// filter; updated in place by the solver)
  std::vector<std::array<double, 9>> sb;
  /// factor between keyframe k and k+1 (size = num keyframes - 1)
  std::vector<ImuFactorData> factors;
};

/**
 * @brief Bundle-adjustment backend (offline: vision + IMU).
 *
 * Records keyframe poses/velocities/biases and feature observations during
 * the filter run (record_observations BEFORE the visual update, record_pose
 * AFTER it), then assembles a colmap::Reconstruction — one rig whose
 * reference sensor is the IMU, one frame per keyframe, one image per camera,
 * one Point3D per surviving feature track — and refines it with colmap's
 * Ceres bundle adjuster. Camera intrinsics and cam-IMU extrinsics are held
 * constant; the oldest keyframe pose is held constant to anchor the gauge to
 * the filter frame so trajectories stay comparable. When IMU data has been
 * fed (feed_imu), preintegrated IMU factors between consecutive keyframes
 * are injected into the same Ceres problem (Phase 2).
 */
class BackendSystem {
public:
  /// One 2D measurement of a feature track at a keyframe
  struct Observation {
    size_t feat_id;
    size_t cam_id;
    Eigen::Vector2d uv; // distorted pixels (as stored in the feature db)
  };

  /// Recorded keyframe: filter pose + velocity/bias snapshot + all track
  /// observations at that time
  struct Keyframe {
    double timestamp = -1;
    bool has_pose = false;
    Mat3 R_GtoI = Mat3::Identity();
    Vec3 p_IinG = Vec3::Zero();
    Vec3 v_IinG = Vec3::Zero();
    Vec3 bg = Vec3::Zero();
    Vec3 ba = Vec3::Zero();
    std::vector<Observation> obs;
  };

  BackendSystem(const BackendOptions &opts, const NoiseManager &imu_noises,
                double gravity_mag);
  /// Stops the online worker thread (if running)
  ~BackendSystem();

  /// Append IMU readings (sorted by timestamp) for the preintegrated
  /// factors. Feed the same stream that goes to the estimator.
  void feed_imu(const std::vector<ov_core::ImuData> &msgs);

  /// Snapshot the (constant) calibration from the state: camera models,
  /// intrinsics, cam-IMU extrinsics and the cam-IMU time offset. Called
  /// automatically on the first record_pose / run_offline_ba; the online
  /// worker needs it to build reconstructions without touching State.
  void cache_calibration(const State &state);

  /// --- online fixed-lag windowed BA (Phase 2b) ----------------------------

  bool online_enabled() const { return opts_.online_enabled; }

  /// Backend options (read-only)
  const BackendOptions &options() const { return opts_; }

  /// Number of windowed solves completed so far (monotonic; poll to detect
  /// a fresh get_refined_poses() publication)
  int online_solve_count() const { return online_solve_count_.load(); }

  /// Latest refined poses from the online windowed solves, keyed by keyframe
  /// timestamp (clone convention: R_GtoI, p_IinG). A keyframe's estimate is
  /// refreshed every solve while it is inside the window.
  std::map<double, std::pair<Mat3, Vec3>> get_refined_poses();

  /// Write the latest refined poses to a file (same format as the filter
  /// output: timestamp px py pz qx qy qz qw)
  void export_online_trajectory(const std::string &path);

  /// --- loop-closure constraints (Phase 3c) ---------------------------------

  /// Register a relative-pose (loop-closure) constraint between two
  /// keyframes, keyed by their camera timestamps. The constraint is injected
  /// into every subsequent solve (online windowed and offline) whose frame
  /// set contains both keyframes. R_ItoJ maps i-IMU coordinates to j-IMU
  /// coordinates, p_JinI is the origin of j expressed in i, and cov is the
  /// 6x6 covariance of the [theta, p] measurement.
  ///
  /// Retrieval (place recognition) is intentionally out of scope here:
  /// colmap-lite excludes the feature/retrieval modules and the frontend
  /// keeps no descriptors, so constraints are expected from an external
  /// loop-closure module calling this method (thread-safe).
  void add_loop_constraint(double timestamp_i, double timestamp_j,
                           const Eigen::Matrix3d &R_ItoJ,
                           const Eigen::Vector3d &p_JinI,
                           const Eigen::Matrix<double, 6, 6> &cov);

  /// Number of registered loop constraints
  size_t num_loop_constraints() const {
    std::lock_guard<std::mutex> lk(record_mtx_);
    return loop_constraints_.size();
  }

  /// Snapshot feature observations at `timestamp` (call BEFORE the visual
  /// update, after feed_camera, while tracks consumed by the update are
  /// still in the database). Starts a new keyframe every
  /// BackendOptions::keyframe_stride calls.
  void record_observations(double timestamp, ov_core::FeatureDatabase &db);

  /// Attach the latest clone pose to the pending keyframe (call AFTER the
  /// visual update). Returns false if no keyframe is pending.
  bool record_pose(const State &state);

  size_t num_keyframes() const {
    std::lock_guard<std::mutex> lk(record_mtx_);
    return keyframes_.size();
  }

  /// Assemble the reconstruction, triangulate landmarks, run BA, and write
  /// the refined trajectory (same format as the filter output:
  /// timestamp px py pz qx qy qz qw) to traj_out_path.
  BackendSummary run_offline_ba(State &state, const std::string &traj_out_path);

  /// Solve + prune (+ re-solve) + export on an already-assembled
  /// reconstruction. Exposed for unit tests with synthetic maps.
  /// frame_timestamps must be ordered by ascending frame id (1..N).
  /// If `imu` is given, its factors/parameter blocks are injected into the
  /// Ceres problem before each solve (it must outlive the call).
  /// max_iterations/max_solver_time override the option budgets when
  /// positive (used by the online worker); prune=false skips outlier
  /// pruning + re-solve.
  BackendSummary solve_and_export(colmap::Reconstruction &recon,
                                  const std::vector<double> &frame_timestamps,
                                  const std::string &traj_out_path,
                                  ImuConstraints *imu = nullptr,
                                  int max_iterations = -1,
                                  double max_solver_time = 0,
                                  bool prune = true);

  /// Build the IMU constraints over a set of keyframes (must be the same
  /// keyframes/ordering that enter the reconstruction). Returns nullptr if
  /// IMU factors are disabled or the data is insufficient. Static so unit
  /// tests can build constraints for synthetic problems.
  static std::unique_ptr<ImuConstraints> build_imu_constraints(
      const std::vector<Keyframe> &keyframes,
      const std::vector<ov_core::ImuData> &imu_data, double t_cam_to_imu,
      const NoiseManager &imu_noises, double gravity_mag, bool enabled);

private:
  /// Build the colmap reconstruction from a set of keyframes (poses, stereo
  /// rig, triangulated tracks) using the cached calibration. Keyframes
  /// without a pose are dropped; `used_keyframes` (if given) receives the
  /// keyframes that made it in, in frame-id order.
  std::unique_ptr<colmap::Reconstruction>
  build_reconstruction(const std::vector<Keyframe> &keyframes,
                       std::vector<double> &frame_timestamps,
                       std::vector<Keyframe> *used_keyframes = nullptr);

  /// Add the velocity/bias parameter blocks and preintegrated IMU residual
  /// blocks to the adjuster's Ceres problem (one factor per consecutive
  /// keyframe pair).
  void inject_imu_factors(colmap::CeresBundleAdjuster &adjuster,
                          colmap::Reconstruction &recon,
                          ImuConstraints &imu);

  /// One registered loop constraint (measurement + whitening)
  struct LoopConstraint {
    double timestamp_i = -1, timestamp_j = -1; // keyframe (camera) times
    RelativePoseFactorData factor;
  };

  /// Inject registered loop-closure residual blocks whose endpoints both
  /// exist in the current frame set (matched via frame_timestamps, which is
  /// ordered by ascending frame id 1..N). Called by solve_and_export.
  void inject_loop_factors(colmap::CeresBundleAdjuster &adjuster,
                           colmap::Reconstruction &recon,
                           const std::vector<double> &frame_timestamps);

  /// Online worker: repeatedly solves the sliding window of the newest
  /// BackendOptions::window_size keyframes every window_solve_stride new
  /// keyframes, publishing refined poses into refined_poses_.
  void online_worker();

  BackendOptions opts_;
  NoiseManager imu_noises_;
  Eigen::Vector3d gravity_ = Eigen::Vector3d(0, 0, 9.81);

  // recording (written by the filter thread, read by the online worker
  // under record_mtx_)
  mutable std::mutex record_mtx_;
  std::vector<Keyframe> keyframes_;
  std::vector<ov_core::ImuData> imu_data_;
  int num_triang_rejected_ = 0;
  int record_count_ = 0;
  bool keyframe_pending_pose_ = false;

  // cached calibration (constant after construction of State)
  bool calib_ready_ = false;
  std::map<size_t, std::shared_ptr<ov_core::CamBase>> calib_cameras_;
  std::map<size_t, VecX> calib_intrinsics_;
  std::map<size_t, Mat3> calib_R_ItoC_;
  std::map<size_t, Vec3> calib_p_IinC_;
  double t_cam_to_imu_ = 0.0;

  // online worker state (worker_stop_/solved_through_keyframe_ live under
  // record_mtx_; refined_poses_ has its own mutex since it is written by
  // the worker and read by the filter thread)
  std::thread worker_;
  std::condition_variable worker_cv_;
  bool worker_stop_ = false;
  size_t solved_through_keyframe_ = 0; // keyframes_ size at last solve
  std::mutex results_mtx_;
  std::map<double, std::pair<Mat3, Vec3>> refined_poses_;

  // registered loop-closure constraints (under record_mtx_; copied out
  // before each solve so solving never holds the lock)
  std::vector<LoopConstraint> loop_constraints_;
  std::atomic<int> online_solve_count_{0}; // incremented by the worker
  double online_solve_ms_sum_ = 0;
  double online_solve_ms_max_ = 0;
};

} // namespace ov_srvins

#endif // OV_SRVINS_BACKENDSYSTEM_H
