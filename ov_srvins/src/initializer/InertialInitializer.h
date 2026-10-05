/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
 * Copyright (C) 2018-2026 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2019 Kevin Eckenhoff
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

#ifndef OV_SRVINS_INERTIALINITIALIZER_H
#define OV_SRVINS_INERTIALINITIALIZER_H

#include "InertialInitializerOptions.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializerOptions.h"
#include "initializer/dynamic/DynamicInitializer.h"
#include "initializer/static/StaticInitializer.h"
#include "update/UpdaterOptions.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "utils/sensor_data.h"

namespace ov_srvins {

/**
 * @brief Initializer for visual-inertial system.
 *
 * This will try to do both dynamic and state initialization of the state.
 * The user can request to wait for a jump in our IMU readings (i.e. device is
 * picked up) or to initialize as soon as possible.
 */
class InertialInitializer {

public:
  /**
   * @brief Default constructor
   * @param params_ Parameters loaded from either ROS or CMDLINE
   * @param db Feature tracker database with all features in it
   * @param propagator Propagator shared with VIO
   */
  explicit InertialInitializer(
      const InertialInitializerOptions &params,
      std::shared_ptr<ov_core::FeatureDatabase> db,
      std::shared_ptr<ov_srvins::Propagator> propagator,
      const UpdaterOptions &msckf_options, const UpdaterOptions &slam_options,
      const ov_core::FeatureInitializerOptions &feat_init_options);

  /**
   * @brief Try to get the initialized system
   * @param state VIO state to be initialized
   * @param wait_for_jerk If true we will wait for a "jerk"
   * @return True if we have successfully initialized our system
   */

  bool initialize(std::shared_ptr<ov_srvins::State> &state, bool wait_for_jerk);

  /// Which initializer the readiness/jerk logic selected
  enum class InitMethod { NONE, STATIC, DYNAMIC };

  /**
   * @brief Run the readiness checks (window size, disparity) and jerk logic
   * to select an initialization method, WITHOUT running the solve.
   *
   * This performs the same database scans/cleanup as initialize() and is
   * cheap. If it returns true, the caller MUST run the selected method
   * (run_static / run_dynamic, or nothing for NONE) and then call
   * finish_attempt() exactly once to update the jerk bookkeeping.
   *
   * @param wait_for_jerk If true we will wait for a "jerk"
   * @param method Output: selected initialization method
   * @return False if the system is not ready to attempt initialization
   * (window not full or disparity check failed); true if an attempt should
   * be made (method may still be NONE when init must be skipped)
   */
  bool choose_method(bool wait_for_jerk, InitMethod &method);

  /// Run the static initializer (call only after choose_method -> STATIC)
  bool run_static(std::shared_ptr<ov_srvins::State> &state) {
    return init_static_->initialize(state, prev_static_timestamp_);
  }

  /// Run the dynamic initializer (call only after choose_method -> DYNAMIC)
  bool run_dynamic(std::shared_ptr<ov_srvins::State> &state) {
    return init_dynamic_->initialize(state);
  }

  /// Update the jerk bookkeeping after an attempt (see choose_method)
  void finish_attempt();

  /// Oldest camera time of the init window used by the last choose_method
  double last_oldest_win_time() const { return last_oldest_win_time_; }

protected:
  /// Initialization parameters
  InertialInitializerOptions params_;

  /// Feature tracker database with all features in it
  std::shared_ptr<ov_core::FeatureDatabase> db_;

  // Propagator shared with VIO
  std::shared_ptr<ov_srvins::Propagator> propagator_;

  // Options forwarded from VIO
  UpdaterOptions msckf_options_;
  UpdaterOptions slam_options_;
  ov_core::FeatureInitializerOptions feat_init_options_;

  // Note: updaterMSCKF and updaterSLAM are now static functions

  /// Static initialization helper class
  std::unique_ptr<StaticInitializer> init_static_;

  /// Dynamic initialization helper class
  std::unique_ptr<DynamicInitializer> init_dynamic_;

  // Record if the platform is static previously
  bool is_static_prev_ = false;

  // Record the previous static timestamp
  double prev_static_timestamp_ = -1;

  // Cached by choose_method for finish_attempt / the async runner
  bool last_is_still_ = false;
  double last_latest_cam_time_ = -1;
  double last_oldest_win_time_ = -1;
};

} // namespace ov_srvins

#endif // OV_SRVINS_INERTIALINITIALIZER_H
