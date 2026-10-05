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

#ifndef OV_SRVINS_INITRUNNER_H
#define OV_SRVINS_INITRUNNER_H

#include <deque>
#include <future>
#include <memory>

#include "VinsOptions.h"

namespace ov_srvins {

class SqrtEstimator;
class Frontend;
class InertialInitializer;
class State;

/**
 * @brief Drives system initialization, one call per camera frame.
 *
 * With init_async = false (default) this is exactly the synchronous path:
 * InertialInitializer::initialize() on the live state, then
 * finalize_initialization() on success.
 *
 * With init_async = true, the dynamic-initialization solve runs on a
 * background thread over SNAPSHOTS (shadow state, private propagator fed
 * from a locked IMU copy, deep-copied feature database) while the main
 * thread keeps tracking features and buffering IMU. When the solve
 * finishes, the shadow state is committed at a camera frame boundary: the
 * estimator's state is swapped for the shadow one (nothing else needs
 * repointing - the frontend holds no state pointer; selection rules take
 * the current state per call), the camera times recorded while the solve
 * was in flight are replayed through propagate() to rebuild the clone
 * window, and finalize_initialization() runs.
 *
 * Threading contract: the main thread owns the live State / Propagator /
 * Frontend exclusively; the background thread only touches its private
 * snapshot objects. try_initialize() is main-thread only.
 *
 * Static initialization always stays synchronous (it is cheap), and
 * offline deterministic runs should keep init_async = false: with async
 * commit, the frame at which initialization is observed depends on
 * thread scheduling.
 */
class InitRunner {
public:
  InitRunner(const VinsOptions &params,
             std::shared_ptr<SqrtEstimator> estimator,
             std::shared_ptr<Frontend> frontend,
             std::shared_ptr<InertialInitializer> initializer);

  ~InitRunner();

  /**
   * @brief Attempt (or continue) initialization for this camera frame
   * @param cam_time Current camera timestamp (only used on the async path,
   * to record frames for the post-commit clone replay; the synchronous
   * path works from the propagator/tracker buffers and ignores it)
   * @param wait_for_jerk If true, wait for a "jerk" before static init
   * @return True on the frame where initialization has completed (and, in
   * the async case, been committed)
   */
  bool try_initialize(double cam_time, bool wait_for_jerk);

private:
  /// Launch the dynamic solve on snapshot copies (main thread, IDLE only)
  void launch_shadow_solve();

  /// Commit a successfully solved shadow state (main thread, PENDING only)
  void commit_shadow_state();

  /// Build a fresh state with calibration copied (deep) from the live one
  std::shared_ptr<State> make_shadow_state();

  /// System parameters (copy, like SqrtEstimator)
  VinsOptions params_;

  std::shared_ptr<SqrtEstimator> estimator_;
  std::shared_ptr<Frontend> frontend_;
  std::shared_ptr<InertialInitializer> initializer_;

  /// True while a shadow solve is in flight
  bool solve_pending_ = false;

  /// The in-flight solve and the shadow state it mutates
  std::future<bool> solve_future_;
  std::shared_ptr<State> shadow_state_;

  /// Camera timestamps recorded while the solve is in flight (main thread
  /// only); replayed through propagate() at commit to rebuild clones
  std::deque<double> pending_cam_times_;
};

} // namespace ov_srvins

#endif // OV_SRVINS_INITRUNNER_H
