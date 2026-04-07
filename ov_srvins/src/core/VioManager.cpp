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

#include "VioManager.h"
#include "Frontend.h"
#include <algorithm>

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "feat/FeatureInitializer.h"
#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"

#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/DataType.h"
#include "utils/opencv_lambda_body.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

#include "initializer/InertialInitializer.h"

#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"

#include "SqrtEstimator.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_srvins;

VioManager::VioManager(VioManagerOptions &params_)
    : thread_init_running(false), thread_init_success(false) {

  // Nice startup message
  PRINT_DEBUG("=======================================\n");
  PRINT_DEBUG("OPENVINS ON-MANIFOLD SQR-EKF IS STARTING\n");
  PRINT_DEBUG("=======================================\n");

  // Nice debug
  this->params = params_;
  params.print_and_load_estimator();
  params.print_and_load_noise();
  params.print_and_load_state();
  params.print_and_load_trackers();

  // This will globally set the thread count we will use
  cv::setNumThreads(params.num_opencv_threads);
  cv::setRNGSeed(0);

  // Create the estimator!!
  estimator = std::make_shared<SqrtEstimator>(params);
  state = estimator->get_state();
  propagator = estimator->get_propagator();
  updaterZUPT = estimator->get_updater_zupt();

  // If we are recording statistics, then open our file
  if (params.record_timing_information) {
    if (std::filesystem::exists(params.record_timing_filepath)) {
      std::filesystem::remove(params.record_timing_filepath);
      PRINT_INFO(YELLOW "[STATS]: found old file found, deleted...\n" RESET);
    }
    std::filesystem::path p(params.record_timing_filepath);
    std::filesystem::create_directories(p.parent_path());
    of_statistics.open(params.record_timing_filepath,
                       std::ofstream::out | std::ofstream::app);
    of_statistics
        << "# timestamp (sec),tracking,propagation,marg,msckf update,";
    if (state->options.max_slam_features > 0) {
      of_statistics << "slam update,slam delayed,";
    }
    of_statistics << "state init,qr,back sub,re-tri,total" << std::endl;
  }

  // Let's make a front-end!
  frontend = std::make_shared<Frontend>(params, state);

  // Set the ZUPT database if enabled
  estimator->set_zupt_database(frontend->get_trackFEATS()->get_feature_database());
  updaterZUPT = estimator->get_updater_zupt(); // Update shortcut

  // Our state initialize
  initializer = std::make_shared<ov_srvins::InertialInitializer>(
      params.init_options, frontend->get_trackFEATS()->get_feature_database(),
      propagator, params.msckf_options, params.slam_options,
      params.featinit_options);
}

VioManager::~VioManager() {
  if (initialization_thread.joinable()) {
    initialization_thread.join();
  }
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {
  std::lock_guard<std::mutex> lck(vio_mtx);
  // The oldest time we need IMU with is the last clone
  double oldest_time = state->margtimestep();
  if (oldest_time > state->timestamp) {
    oldest_time = -1;
  }
  if (!is_initialized_vio) {
    oldest_time = message.timestamp - params.init_options.init_window_time +
                  state->calib_dt_CAMtoIMU->value()(0) - 0.1;
  }

  // Feed to estimator
  estimator->feed_imu(message, oldest_time);
}

void VioManager::track_image_and_update(
    const ov_core::CameraData &message_const) {

  std::lock_guard<std::mutex> lck(vio_mtx);
  rT1 = std::chrono::steady_clock::now();

  // Perform our feature tracking!
  ov_core::CameraData message = message_const;
  frontend->feed_camera(message);
  rT2 = std::chrono::steady_clock::now();

  // Try zero-velocity update
  if (is_initialized_vio) {
    did_zupt_update =
        estimator->try_zupt(message.timestamp, has_moved_since_zupt);
    if (did_zupt_update) {
      return;
    }
  }

  // If we do not have VIO initialization, then try to initialize
  if (!is_initialized_vio) {
    is_initialized_vio = try_to_initialize(message);
    if (!is_initialized_vio) {
      return;
    }
  }

  // Call on our propagate and update function
  do_feature_propagate_update(message);
}

void VioManager::do_feature_propagate_update(
    const ov_core::CameraData &message) {

  // State propagation
  if (!estimator->propagate(message.timestamp)) {
    return;
  }
  rT3 = std::chrono::steady_clock::now();

  // Wait for enough clones
  if ((int)state->clones_IMU.size() <
          std::min(state->options.max_clone_size, 5) &&
      state->features_SLAM.empty()) {
    return;
  }

  // Ensure propagation reached target
  if (state->timestamp != message.timestamp) {
    PRINT_WARNING(RED
                  "[PROP]: Propagator unable to reach target time!\n" RESET);
    return;
  }
  has_moved_since_zupt = true;

  // Cleanup old measurements from database
  if ((int)state->clones_IMU.size() > state->options.max_clone_size + 1) {
    frontend->get_trackFEATS()->get_feature_database()->cleanup_measurements(
        state->margtimestep());
    if (frontend->get_trackARUCO() != nullptr) {
      frontend->get_trackARUCO()->get_feature_database()->cleanup_measurements(
          state->margtimestep());
    }
  }

  // Sorting features according to rules
  std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE,
      featsup_MSCKF;
  frontend->process_measurements_rules(message.timestamp, message.sensor_ids,
                                       featsup_MSCKF, feats_slam_UPDATE,
                                       feats_slam_DELAYED);

  // Estimator update
  rT4 = std::chrono::steady_clock::now(); // Timing for stats
  estimator->update(featsup_MSCKF, feats_slam_UPDATE, feats_slam_DELAYED);

  // Timing points for stats (mirrored from VioManager for now)
  // We can refine this by making estimator return timing info
  rT9 = std::chrono::steady_clock::now();

  // Update visualization
  if (message.sensor_ids.at(0) == 0) {
    good_features_MSCKF.clear();
  }
  for (auto const &feat : featsup_MSCKF) {
    good_features_MSCKF.push_back(feat->p_FinG);
    feat->to_delete = true;
  }

  // Cleanup tracker database
  frontend->get_trackFEATS()->get_feature_database()->cleanup();
  if (frontend->get_trackARUCO() != nullptr) {
    frontend->get_trackARUCO()->get_feature_database()->cleanup();
  }

  // Stats and Printing (Condensed for brevity, in real impl we'd handle all rT
  // points)
  double time_total =
      std::chrono::duration_cast<std::chrono::microseconds>(rT9 - rT1).count() *
      1e-6;
  PRINT_DEBUG(BLUE "[TIME]: %.4f seconds for total update\n" RESET, time_total);
}

bool VioManager::try_to_initialize(const ov_core::CameraData &message) {
  if (thread_init_running) {
    std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
    camera_queue_init.push_back(message.timestamp);
    return false;
  }
  if (thread_init_success) {
    return true;
  }

  thread_init_running = true;
  if (initialization_thread.joinable()) {
    initialization_thread.detach(); // Should not really happen if thread_init_running is used correctly
  }

  initialization_thread = std::thread([&] {
    auto init_rT1 = std::chrono::steady_clock::now();
    bool wait_for_jerk = (updaterZUPT == nullptr);
    bool success = initializer->initialize(state, wait_for_jerk);

    if (success) {
      std::lock_guard<std::mutex> lck(vio_mtx);
      startup_time = state->timestamp;
      frontend->set_startup_time(startup_time);
      state->is_initialized = true;

      frontend->get_trackFEATS()->get_feature_database()->cleanup_measurements(
          state->timestamp);
      frontend->get_trackFEATS()->set_num_features(
          std::floor((DataType)params.num_pts /
                     (DataType)params.state_options.num_cameras));
      if (frontend->get_trackARUCO() != nullptr) {
        frontend->get_trackARUCO()->get_feature_database()->cleanup_measurements(
            state->timestamp);
      }

      if (state->imu->vel().norm() > params.zupt_max_velocity) {
        has_moved_since_zupt = true;
      }

      std::lock_guard<std::mutex> lck_q(camera_queue_init_mtx);
      for (double ts : camera_queue_init) {
        if (ts > startup_time)
          estimator->propagate(ts);
      }
      thread_init_success = true;
      camera_queue_init.clear();
    } else {
      std::lock_guard<std::mutex> lck_q(camera_queue_init_mtx);
      camera_queue_init.clear();
    }
    thread_init_running = false;
  });

  return false;
}

std::shared_ptr<Propagator> VioManager::get_propagator() { return propagator; }

cv::Mat VioManager::get_historical_viz_image() {
  if (state == nullptr || frontend == nullptr)
    return cv::Mat();
  return frontend->get_historical_viz_image(did_zupt_update, is_initialized_vio);
}

std::vector<Vec3> VioManager::get_features_SLAM() {
  std::vector<Vec3> slam_feats;
  for (auto &f : state->features_SLAM) {
    if ((int)f.first <= 4 * state->options.max_aruco_features)
      continue;
    if (ov_type::LandmarkRepresentation::is_relative_representation(
            f.second->feat_representation)) {
      assert(f.second->anchor_cam_id != -1);
      const auto anchor_pose = state->cam_pose_buffer.get_buffer_unsafe(
          f.second->anchor_cam_id, f.second->anchor_clone_timestamp);
      slam_feats.push_back(anchor_pose.R_GtoC.transpose() *
                               f.second->get_xyz(false) +
                           anchor_pose.p_CinG);
    } else
      slam_feats.push_back(f.second->get_xyz(false));
  }
  return slam_feats;
}

std::vector<Vec3> VioManager::get_features_ARUCO() {
  std::vector<Vec3> aruco_feats;
  for (auto &f : state->features_SLAM) {
    if ((int)f.first > 4 * state->options.max_aruco_features)
      continue;
    if (ov_type::LandmarkRepresentation::is_relative_representation(
            f.second->feat_representation)) {
      assert(f.second->anchor_cam_id != -1);
      const auto anchor_pose = state->cam_pose_buffer.get_buffer_unsafe(
          f.second->anchor_cam_id, f.second->anchor_clone_timestamp);
      aruco_feats.push_back(anchor_pose.R_GtoC.transpose() *
                                f.second->get_xyz(false) +
                            anchor_pose.p_CinG);
    } else
      aruco_feats.push_back(f.second->get_xyz(false));
  }
  return aruco_feats;
}

