/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
 * Copyright (C) 2018-2026 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2023 Patrick Geneva
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

#ifndef OV_SRVINS_VIOMANAGER_H
#define OV_SRVINS_VIOMANAGER_H

#include <Eigen/StdVector>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <cstddef>
#include <memory>
#include <mutex>
#include <vector>

#include "VioManagerOptions.h"

namespace ov_core {
struct ImuData;
struct CameraData;
class TrackBase;
class FeatureInitializer;
class Feature;
} // namespace ov_core

namespace ov_srvins {
class InertialInitializer;
class State;
class StateHelper;
class UpdaterMSCKF;
class UpdaterSLAM;
class UpdaterZeroVelocity;
class Propagator;
class SqrtEstimator;
} // namespace ov_srvins

namespace ov_srvins {

/**
 * @brief Core class that manages the entire system
 *
 * This class coordinates between trackers, the state estimator, and the initializer.
 * It handles the high-level workflow of the system.
 */
class VioManager {

public:
  /**
   * @brief Default constructor, will load all configuration variables
   * @param params_ Parameters loaded from either ROS or CMDLINE
   */
  VioManager(VioManagerOptions &params_);

  /// Our state estimator
  std::shared_ptr<SqrtEstimator> estimator;

  /// Our master state object (shortcut to estimator->state)
  std::shared_ptr<State> state;

  /**
   * @brief Feed function for inertial data
   * @param message Contains our timestamp and inertial information
   */
  void feed_measurement_imu(const ov_core::ImuData &message);

  /**
   * @brief Feed function for camera measurements
   * @param message Contains our timestamp, images, and camera ids
   */
  void feed_measurement_camera(const ov_core::CameraData &message) {
    track_image_and_update(message);
  }

  /// If we are initialized or not
  bool initialized() { return is_initialized_vio; }

  /// Timestamp that the system was initialized at
  double initialized_time() { return startup_time; }

  /// Accessor for current system parameters
  VioManagerOptions get_params() { return params; }

  /// Accessor to get the current state
  std::shared_ptr<State> get_state() { return state; }

  /// Accessor to get the current propagator
  std::shared_ptr<Propagator> get_propagator();

  /// Get a nice visualization image of what tracks we have
  cv::Mat get_historical_viz_image();

  /// Returns 3d SLAM features in the global frame
  std::vector<Vec3> get_features_SLAM();

  /// Returns 3d ARUCO features in the global frame
  std::vector<Vec3> get_features_ARUCO();

  /// Returns 3d features used in the last update in global frame
  std::vector<Vec3> get_good_features_MSCKF() { return good_features_MSCKF; }

protected:
  /**
   * @brief Given a new set of camera images, this will track them.
   * @param message Contains our timestamp, images, and camera ids
   */
  void track_image_and_update(const ov_core::CameraData &message);

  /**
   * @brief Helper to handle the logic of sorting features into different
   * categories (lost, marg, slam, etc) for update.
   */
  void process_measurements_rules(
      double timestamp, const std::vector<int> &sensor_ids,
      std::vector<std::shared_ptr<ov_core::Feature>> &featsup_MSCKF,
      std::vector<std::shared_ptr<ov_core::Feature>> &feats_slam_UPDATE,
      std::vector<std::shared_ptr<ov_core::Feature>> &feats_slam_DELAYED);

  /**
   * @brief This will do the propagation and feature updates to the state
   * @param message Contains our timestamp, images, and camera ids
   */
  void do_feature_propagate_update(const ov_core::CameraData &message);

  /**
   * @brief This function will try to initialize the state.
   * @param message Contains our timestamp, images, and camera ids
   * @return True if we have successfully initialized
   */
  bool try_to_initialize(const ov_core::CameraData &message);

  /// Manager parameters
  VioManagerOptions params;

  /// Our sparse feature tracker (klt or descriptor)
  std::shared_ptr<ov_core::TrackBase> trackFEATS;

  /// Our aruoc tracker
  std::shared_ptr<ov_core::TrackBase> trackARUCO;

  /// State initializer
  std::shared_ptr<ov_srvins::InertialInitializer> initializer;

  /// Propagator of our state
  std::shared_ptr<Propagator> propagator;

  /// Our zero velocity tracker
  std::shared_ptr<UpdaterZeroVelocity> updaterZUPT;

  /// Boolean if we are initialized or not
  bool is_initialized_vio = false;

  /// This is the queue of measurement times that have come in since we starting
  /// doing initialization
  std::vector<double> camera_queue_init;
  std::mutex camera_queue_init_mtx;

  // Timing statistic file and variables
  std::ofstream of_statistics;
  std::chrono::steady_clock::time_point rT1, rT2, rT3, rT4, rT5, rT6, rT7, rT8, rT9;

  // Track how much distance we have traveled
  double timelastupdate = -1;
  DataType distance = 0;

  // Startup time of the filter
  double startup_time = -1;

  // Threads and their atomics
  std::atomic<bool> thread_init_running, thread_init_success;

  // If we did a zero velocity update
  bool did_zupt_update = false;
  bool has_moved_since_zupt = false;

  // Good features that where used in the last update (used in visualization)
  std::vector<Vec3> good_features_MSCKF;
};

} // namespace ov_srvins

#endif // OV_SRVINS_VIOMANAGER_H
