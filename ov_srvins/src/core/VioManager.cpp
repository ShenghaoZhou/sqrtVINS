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

  // Let's make a feature extractor
  int init_max_features =
      std::floor((float)params.init_options.init_max_features /
                 (float)params.state_options.num_cameras);
  if (params.use_klt) {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackKLT(
        state->cam_intrinsics_cameras, init_max_features,
        state->options.max_aruco_features, params.use_stereo,
        params.histogram_method, params.fast_threshold, params.grid_x,
        params.grid_y, params.min_px_dist, params.ransac_th));
  } else {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackDescriptor(
        state->cam_intrinsics_cameras, init_max_features,
        state->options.max_aruco_features, params.use_stereo,
        params.histogram_method, params.fast_threshold, params.grid_x,
        params.grid_y, params.min_px_dist, params.knn_ratio));
  }

  // Set the ZUPT database if enabled
  estimator->set_zupt_database(trackFEATS->get_feature_database());
  updaterZUPT = estimator->get_updater_zupt(); // Update shortcut

  // Initialize our aruco tag extractor
  if (params.use_aruco) {
    trackARUCO = std::shared_ptr<TrackBase>(new TrackAruco(
        state->cam_intrinsics_cameras, state->options.max_aruco_features,
        params.use_stereo, params.histogram_method, params.downsize_aruco));
  }

  // Our state initialize
  initializer = std::make_shared<ov_srvins::InertialInitializer>(
      params.init_options, trackFEATS->get_feature_database(), propagator,
      params.msckf_options, params.slam_options, params.featinit_options);
}

void VioManager::feed_measurement_imu(const ov_core::ImuData &message) {
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

  rT1 = std::chrono::steady_clock::now();

  // Assert we have valid measurement data and ids
  assert(!message_const.sensor_ids.empty());
  assert(message_const.sensor_ids.size() == message_const.images.size());

  // Downsample if needed
  ov_core::CameraData message = message_const;
  for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras;
       i++) {
    cv::Mat img = message.images.at(i);
    cv::Mat mask = message.masks.at(i);
    cv::Mat img_temp, mask_temp;
    cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
    message.images.at(i) = img_temp;
    cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
    message.masks.at(i) = mask_temp;
  }

  // Perform our feature tracking!
  trackFEATS->feed_new_camera(message);
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
    trackFEATS->get_feature_database()->cleanup_measurements(
        state->margtimestep());
    if (trackARUCO != nullptr) {
      trackARUCO->get_feature_database()->cleanup_measurements(
          state->margtimestep());
    }
  }

  // Sorting features according to rules
  std::vector<std::shared_ptr<Feature>> feats_slam_DELAYED, feats_slam_UPDATE,
      featsup_MSCKF;
  process_measurements_rules(message.timestamp, message.sensor_ids, featsup_MSCKF,
                             feats_slam_UPDATE, feats_slam_DELAYED);

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
  trackFEATS->get_feature_database()->cleanup();
  if (trackARUCO != nullptr) {
    trackARUCO->get_feature_database()->cleanup();
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
  std::thread thread([&] {
    auto init_rT1 = std::chrono::steady_clock::now();
    bool wait_for_jerk = (updaterZUPT == nullptr);
    bool success = initializer->initialize(state, wait_for_jerk);

    if (success) {
      startup_time = state->timestamp;
      state->is_initialized = true;

      trackFEATS->get_feature_database()->cleanup_measurements(
          state->timestamp);
      trackFEATS->set_num_features(
          std::floor((DataType)params.num_pts /
                     (DataType)params.state_options.num_cameras));
      if (trackARUCO != nullptr) {
        trackARUCO->get_feature_database()->cleanup_measurements(
            state->timestamp);
      }

      if (state->imu->vel().norm() > params.zupt_max_velocity) {
        has_moved_since_zupt = true;
      }

      std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
      for (double ts : camera_queue_init) {
        if (ts > startup_time)
          estimator->propagate(ts);
      }
      thread_init_success = true;
      camera_queue_init.clear();
    } else {
      std::lock_guard<std::mutex> lck(camera_queue_init_mtx);
      camera_queue_init.clear();
    }
    thread_init_running = false;
  });

  if (!params.use_multi_threading_subs) {
    thread.join();
  } else {
    thread.detach();
  }
  return false;
}

std::shared_ptr<Propagator> VioManager::get_propagator() { return propagator; }

cv::Mat VioManager::get_historical_viz_image() {
  if (state == nullptr || trackFEATS == nullptr)
    return cv::Mat();
  std::vector<size_t> highlighted_ids;
  for (const auto &feat : state->features_SLAM)
    highlighted_ids.push_back(feat.first);
  std::string overlay = (did_zupt_update) ? "zvupt" : "";
  overlay = (!is_initialized_vio) ? "init" : overlay;
  cv::Mat img_history;
  trackFEATS->display_history(img_history, 255, 255, 0, 255, 255, 255,
                              highlighted_ids, overlay);
  if (trackARUCO != nullptr)
    trackARUCO->display_history(img_history, 0, 255, 255, 255, 255, 255,
                                highlighted_ids, overlay);
  return img_history;
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

void VioManager::process_measurements_rules(
    double timestamp, const std::vector<int> &sensor_ids,
    std::vector<std::shared_ptr<Feature>> &featsup_MSCKF,
    std::vector<std::shared_ptr<Feature>> &feats_slam_UPDATE,
    std::vector<std::shared_ptr<Feature>> &feats_slam_DELAYED) {

  std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_maxtracks;
  std::vector<std::shared_ptr<Feature>> feats_slam;
  feats_lost =
      trackFEATS->get_feature_database()->features_not_containing_newer(
          state->timestamp, false, true);

  if ((int)state->clones_IMU.size() == state->options.max_clone_size + 1 ||
      (int)state->clones_IMU.size() > 5) {
    feats_marg = trackFEATS->get_feature_database()->features_containing(
        state->margtimestep(), false, true);
    if (trackARUCO != nullptr &&
        timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(
          state->margtimestep(), false, true);
    }
  }

  auto it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    bool found_cam = false;
    for (const auto &pair : (*it1)->uvs)
      if (std::find(sensor_ids.begin(), sensor_ids.end(),
                    pair.first) != sensor_ids.end()) {
        found_cam = true;
        break;
      }
    if (found_cam)
      it1++;
    else
      it1 = feats_lost.erase(it1);
  }

  it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) !=
        feats_marg.end())
      it1 = feats_lost.erase(it1);
    else
      it1++;
  }

  auto it2 = feats_marg.begin();
  while (it2 != feats_marg.end()) {
    bool reached_max = false;
    for (const auto &cams : (*it2)->timestamps)
      if ((int)cams.second.size() > state->options.max_clone_size) {
        reached_max = true;
        break;
      }
    if (reached_max) {
      feats_maxtracks.push_back(*it2);
      it2 = feats_marg.erase(it2);
    } else
      it2++;
  }

  int curr_aruco_tags = 0;
  for (auto &f : state->features_SLAM)
    if ((int)f.second->featid <= 4 * state->options.max_aruco_features)
      curr_aruco_tags++;

  if (state->options.max_slam_features > 0 &&
      timestamp - startup_time >= params.dt_slam_delay &&
      (int)state->features_SLAM.size() <
          state->options.max_slam_features + curr_aruco_tags) {
    int amount_to_add = (state->options.max_slam_features + curr_aruco_tags) -
                        (int)state->features_SLAM.size();
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size())
                           ? (int)feats_maxtracks.size()
                           : amount_to_add;
    if (valid_amount > 0) {
      feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount,
                        feats_maxtracks.end());
      feats_maxtracks.erase(feats_maxtracks.end() - valid_amount,
                            feats_maxtracks.end());
    }
  }

  for (auto &landmark : state->features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat =
          trackARUCO->get_feature_database()->get_feature(
              landmark.second->featid);
      if (feat != nullptr)
        feats_slam.push_back(feat);
    }
    std::shared_ptr<Feature> feat =
        trackFEATS->get_feature_database()->get_feature(
            landmark.second->featid);
    if (feat != nullptr)
      feats_slam.push_back(feat);
    bool current_unique_cam =
        std::find(sensor_ids.begin(), sensor_ids.end(),
                  landmark.second->unique_camera_id) !=
        sensor_ids.end();
    if (feat == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
    if (landmark.second->update_fail_count > 1)
      landmark.second->should_marg = true;
  }

  for (auto const &f : feats_slam) {
    if (state->features_SLAM.find(f->featid) != state->features_SLAM.end())
      feats_slam_UPDATE.push_back(f);
    else
      feats_slam_DELAYED.push_back(f);
  }

  featsup_MSCKF = feats_lost;
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(),
                       feats_marg.end());
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(),
                       feats_maxtracks.end());

  auto compare_feat = [](const std::shared_ptr<Feature> &a,
                         const std::shared_ptr<Feature> &b) -> bool {
    size_t asize = 0, bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

  if ((int)featsup_MSCKF.size() > state->options.max_msckf_in_update)
    featsup_MSCKF.erase(featsup_MSCKF.begin(),
                        featsup_MSCKF.end() -
                            state->options.max_msckf_in_update);
}
