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

#ifndef OV_SRVINS_BACKENDOPTIONS_H
#define OV_SRVINS_BACKENDOPTIONS_H

#include <cstddef>
#include <string>

namespace ov_srvins {

/**
 * @brief Options for the colmap/Ceres bundle-adjustment backend.
 *
 * Offline (post-run) BA over recorded keyframes: vision-only reprojection
 * factors (Phase 1) plus preintegrated IMU factors between consecutive
 * keyframes (Phase 2, enabled by default).
 */
struct BackendOptions {

  /// Master switch (parsed as "backend_enabled")
  bool enabled = false;

  /// Record a keyframe every Nth visually-updated frame (1 = every frame).
  /// Decimates the offline BA problem size; observations are still taken
  /// from all tracks alive at the keyframe time.
  int keyframe_stride = 5;

  /// Minimum number of keyframe observations for a track to enter BA
  size_t min_track_length = 3;

  /// Triangulated points whose mean reprojection error under the filter
  /// poses exceeds this [px] are rejected before BA (kills degenerate
  /// triangulations that would poison the problem and the error metrics)
  double max_triang_error_px = 10.0;

  /// Points with a post-BA reprojection error above this threshold [px] are
  /// pruned before the (optional) second solve
  double max_reproj_error_px = 5.0;

  /// Run a second BA solve after outlier pruning
  bool refine_after_pruning = true;

  /// Add preintegrated IMU factors between consecutive keyframes (with
  /// per-keyframe velocity/bias blocks) to the BA problem. Requires IMU
  /// data to be fed via BackendSystem::feed_imu during the run.
  bool use_imu_factors = true;

  /// --- online fixed-lag windowed BA (Phase 2b) ----------------------------

  /// Run a background thread that continuously bundle-adjusts a sliding
  /// window of the most recent keyframes. The refined poses are exposed via
  /// BackendSystem::get_refined_poses() for filter feedback (Phase 3).
  bool online_enabled = false;

  /// Number of keyframes in the sliding window
  int window_size = 15;

  /// Solve every Nth new keyframe
  int window_solve_stride = 2;

  /// Iteration budget for a windowed solve (small: the window is warm-started
  /// from the filter poses and the previous solve)
  int window_max_iterations = 15;

  /// Wall-clock budget per windowed solve [s] (0 = no limit)
  double window_max_solver_time = 0.05;

  /// Write each windowed solve's refined poses and velocity/bias blocks back
  /// into the recorded keyframes, so the next solve starts from the previous
  /// solution on the overlap instead of the filter snapshots
  bool window_warm_start = true;

  /// Soft gauge prior on the oldest window keyframe pose [m] / [rad]. When
  /// both are positive the constant anchor is replaced by a prior factor
  /// centered on the keyframe's current value (the previous solve's refined
  /// pose once warm starting has run), so loop closures and past refinements
  /// can move the window seam. <= 0 restores the constant anchor.
  double window_prior_sigma_pos = 0.05;
  double window_prior_sigma_ori = 0.02;

  /// Prior on the oldest window keyframe's [v, bg, ba] block, centered on its
  /// current value [m/s, rad/s, m/s^2]. Anchors the leading edge of the IMU
  /// factor chain so biases cannot drift solve-to-solve. <= 0 disables each.
  double window_prior_sigma_vel = 0.5;
  double window_prior_sigma_bg = 0.05;
  double window_prior_sigma_ba = 0.2;

  /// --- filter feedback (Phase 3) -------------------------------------------

  /// Feed the refined window poses back into the filter as soft pose
  /// measurements on the matching clones (requires online_enabled)
  bool feedback_enabled = false;

  /// Feedback position measurement noise [m] per axis. Conservative (large)
  /// on purpose: the window information overlaps the filter's own visual
  /// measurements, so small values double-count information.
  double feedback_sigma_pos = 0.05;

  /// Feedback orientation measurement noise [rad] per axis
  double feedback_sigma_ori = 0.02;

  /// Per-clone gate on the whitened 6-dof residual squared norm; feedback
  /// for clones above this is dropped (protects against bad window solves)
  double feedback_gate_chi2 = 50.0;

  /// Ceres solver settings
  int max_num_iterations = 100;
  int num_threads = 4;

  /// Cauchy robust loss scale [px] on reprojection residuals (0 = disabled)
  double loss_scale = 1.0;

  /// Print the Ceres/colmap solver summary to stdout
  bool print_summary = true;
};

} // namespace ov_srvins

#endif // OV_SRVINS_BACKENDOPTIONS_H
