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

#ifndef OV_SRVINS_SQRTESTIMATOR_H
#define OV_SRVINS_SQRTESTIMATOR_H

#include <memory>
#include <vector>
#include "VioManagerOptions.h"
#include "utils/sensor_data.h"

namespace ov_core {
class Feature;
}

namespace ov_srvins {

class State;
class Propagator;
class UpdaterZeroVelocity;

/**
 * @brief Core class that handles the Square-Root EKF estimation logic.
 *
 * This class encapsulates the state estimation process, including IMU propagation,
 * zero-velocity updates, and visual feature updates (MSCKF and SLAM).
 */
class SqrtEstimator {
public:
  /**
   * @brief Constructor
   * @param params_ System parameters
   */
  SqrtEstimator(VioManagerOptions &params_);

  /**
   * @brief Feed IMU data to the estimator (propagator and ZUPT)
   * @param message IMU data
   * @param oldest_time Oldest time to keep in buffers
   */
  void feed_imu(const ov_core::ImuData &message, double oldest_time);

  /**
   * @brief Try to perform a zero-velocity update
   * @param timestamp Target timestamp
   * @param has_moved_since_zupt Flag indicating if the system has moved
   * @return True if a ZUPT update was performed
   */
  bool try_zupt(double timestamp, bool has_moved_since_zupt);

  /**
   * @brief Propagate the state forward and add a new clone
   * @param timestamp Target timestamp
   * @return True if propagation was successful
   */
  bool propagate(double timestamp);

  /**
   * @brief Perform the full state update with visual features
   * @param message Camera data
   * @param featsup_MSCKF MSCKF features
   * @param feats_slam_UPDATE SLAM features for update
   * @param feats_slam_DELAYED SLAM features for delayed initialization
   */
  void update(std::vector<std::shared_ptr<ov_core::Feature>> &featsup_MSCKF,
              std::vector<std::shared_ptr<ov_core::Feature>> &feats_slam_UPDATE,
              std::vector<std::shared_ptr<ov_core::Feature>> &feats_slam_DELAYED);

  /**
   * @brief Set the feature database for ZUPT updates
   * @param db Feature database from trackers
   */
  void set_zupt_database(std::shared_ptr<ov_core::FeatureDatabase> db);

  /// Accessor for the state
  std::shared_ptr<State> get_state() { return state; }

  /// Accessor for the propagator
  std::shared_ptr<Propagator> get_propagator() { return propagator; }

  /// Accessor for the ZUPT updater
  std::shared_ptr<UpdaterZeroVelocity> get_updater_zupt() { return updaterZUPT; }

private:
  /// Perform marginalization of old states and features
  void handle_marginalization();

  /// Manager parameters
  VioManagerOptions params;

  /// Our master state object
  std::shared_ptr<State> state;

  /// Propagator of our state
  std::shared_ptr<Propagator> propagator;

  /// Our zero velocity tracker
  std::shared_ptr<UpdaterZeroVelocity> updaterZUPT;

  /// Timing points for internal stats (mirrored from VioManager for now)
  std::chrono::steady_clock::time_point rT4, rT5, rT6, rT7, rT8, rT9;
};

} // namespace ov_srvins

#endif // OV_SRVINS_SQRTESTIMATOR_H
