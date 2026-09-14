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

/**
 * @brief SqrtVINS (square-root EKF) formulation of the EuRoC runner.
 *
 * Orchestrates the system exactly like the Python bindings script
 * (run_euroc.py / run_tum_vi.py): SqrtEstimator + Frontend +
 * InertialInitializer with batched IMU feeding, ZUPT handling and the fused
 * propagate/select/update/cleanup step.
 *
 * Kept in its own translation unit (with module-pinned include dirs) because
 * the sqrt and full-covariance modules expose colliding header paths.
 */

#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "euroc_common.h"
#include "core/Frontend.h"
#include "core/SqrtEstimator.h"
#include "core/VioManagerOptions.h"
#include "initializer/InertialInitializer.h"
#include "utils/print.h"
#include "utils/yaml_parse.h"

using namespace ov_core;
using namespace ov_srvins;

int run_euroc_sqrt(const EurocRunOptions &opt, const std::vector<ImuReading> &imu_data,
                   const std::vector<CamReading> &cam_data) {
  //===================================================================================
  // Setup options and system (mirrors the Python orchestration)
  //===================================================================================
  auto parser = std::make_shared<YamlParser>(opt.config_path);
  VioManagerOptions params;
  params.print_and_load(parser);
  params.cv_backend = opt.cv_backend;
  // Repeatability settings used by the original ros1_serial_msckf
  cv::setNumThreads(params.num_opencv_threads);
  cv::setRNGSeed(0);

  auto estimator = std::make_shared<SqrtEstimator>(params);
  auto frontend = std::make_shared<Frontend>(params, estimator->get_state());
  auto initializer = std::make_shared<InertialInitializer>(
      params.init_options,
      frontend->get_trackFEATS()->get_feature_database(),
      estimator->get_propagator(), params.msckf_options, params.slam_options,
      params.featinit_options);

  frontend->set_startup_time(cam_data.front().timestamp);

  //===================================================================================
  // Processing loop
  //===================================================================================
  std::ofstream outfile(opt.output_path);
  if (!outfile.is_open()) {
    PRINT_ERROR(RED "[ERROR]: unable to open output file %s\n" RESET,
                opt.output_path.c_str());
    return EXIT_FAILURE;
  }
  outfile << std::setprecision(9) << std::fixed;

  cv::Mat zero_mask;
  size_t imu_idx = 0;
  int processed = 0;
  bool has_moved_since_zupt = false;
  auto t_start = std::chrono::steady_clock::now();

  for (const auto &cam : cam_data) {
    double curr_cam_time = cam.timestamp;

    // Feed IMU measurements up to this camera time in one batched call.
    // Mirror the original ROS pipeline exactly: a camera is only processed
    // once the IMU clock has passed cam_time + dt, so the buffer must end at
    // the FIRST sample strictly past cam_time + dt (the initializer's window
    // and the propagator's final integration interval both depend on it).
    double t_off = estimator->get_state()->calib_dt_CAMtoIMU->value()(0);
    size_t k = imu_idx;
    while (k < imu_data.size() && imu_data.at(k).timestamp <= curr_cam_time + t_off)
      k++;
    if (k < imu_data.size() && imu_data.at(k).timestamp > curr_cam_time + t_off)
      k++;
    if (k > imu_idx) {
      std::vector<ImuData> msgs(k - imu_idx);
      for (size_t i = imu_idx; i < k; i++) {
        msgs[i - imu_idx].timestamp = imu_data.at(i).timestamp;
        msgs[i - imu_idx].wm = imu_data.at(i).wm.cast<DataType>();
        msgs[i - imu_idx].am = imu_data.at(i).am.cast<DataType>();
      }
      estimator->feed_imu_batch(msgs);
      imu_idx = k;
    }

    // Load stereo images
    cv::Mat img0 = cv::imread(opt.dataset_path + "/mav0/cam0/data/" +
                                  cam.filename_cam0,
                              cv::IMREAD_GRAYSCALE);
    cv::Mat img1 = cv::imread(opt.dataset_path + "/mav0/cam1/data/" +
                                  cam.filename_cam1,
                              cv::IMREAD_GRAYSCALE);
    if (img0.empty() || img1.empty()) {
      processed++;
      continue;
    }

    CameraData message;
    message.timestamp = curr_cam_time;
    message.sensor_ids = {0, 1};
    message.images.push_back(img0);
    message.images.push_back(img1);
    if (zero_mask.empty())
      zero_mask = cv::Mat::zeros(img0.rows, img0.cols, CV_8UC1);
    message.masks.push_back(zero_mask);
    message.masks.push_back(zero_mask.clone());
    frontend->feed_camera(message);

    auto state = estimator->get_state();

    // Check for initialization
    if (!state->is_initialized) {
      // Mirror VioManager::try_to_initialize: without ZUPT we must wait for a
      // jerk before the static initializer will use the stationary window
      if (initializer->initialize(state, !params.try_zupt)) {
        PRINT_INFO("VIO Initialized at %.4f!\n", curr_cam_time);
        frontend->set_startup_time(curr_cam_time);
        // Post-init bookkeeping from VioManager::try_to_initialize
        frontend->get_trackFEATS()->get_feature_database()->cleanup_measurements(
            state->timestamp);
        frontend->get_trackFEATS()->set_num_features(
            std::floor((double)params.num_pts /
                       (double)params.state_options.num_cameras));
        if (state->imu->vel().norm() > params.zupt_max_velocity)
          has_moved_since_zupt = true;
      }
      processed++;
      continue;
    }

    // Try a zero-velocity update
    if (estimator->try_zupt(curr_cam_time, has_moved_since_zupt)) {
      processed++;
      continue;
    }

    // Propagation, feature selection, update, and database cleanup
    // (mirrors the fused propagate_and_update of the Python bindings)
    bool did_update = false;
    if (estimator->propagate(curr_cam_time)) {
      bool too_early = (int)state->clones_IMU.size() <
                           std::min(state->options.max_clone_size, 5) &&
                       state->features_SLAM.empty();
      if (!too_early && state->timestamp == curr_cam_time) {
        // Capture the marginalization time and cleanup gate BEFORE the update:
        // the update will marginalize the clone at this time, so measurements
        // up to it can be dropped only AFTER selection has used them
        bool do_cleanup = (int)state->clones_IMU.size() >
                          state->options.max_clone_size + 1;
        double marg_time = state->margtimestep();
        std::vector<std::shared_ptr<Feature>> featsup_MSCKF, feats_slam;
        frontend->process_measurements_rules(curr_cam_time, message.sensor_ids,
                                             featsup_MSCKF, feats_slam);
        estimator->update(featsup_MSCKF, feats_slam);
        if (do_cleanup) {
          frontend->get_trackFEATS()
              ->get_feature_database()
              ->cleanup_measurements(marg_time);
          if (frontend->get_trackARUCO() != nullptr)
            frontend->get_trackARUCO()
                ->get_feature_database()
                ->cleanup_measurements(marg_time);
        }
        frontend->get_trackFEATS()->get_feature_database()->cleanup();
        if (frontend->get_trackARUCO() != nullptr)
          frontend->get_trackARUCO()->get_feature_database()->cleanup();
        did_update = true;
      }
    }
    if (did_update)
      has_moved_since_zupt = true;

    // Record state (OpenVINS estimate format: p_IinG and JPL q_GtoI)
    state = estimator->get_state();
    if (state->is_initialized) {
      Eigen::Matrix<DataType, 3, 1> pos = state->imu->pos();
      Eigen::Matrix<DataType, 3, 3> Rot = state->imu->Rot(); // R_GtoI
      Eigen::Quaternion<DataType> quat(Rot.transpose());     // R_ItoG as quat
      outfile << state->timestamp << " " << pos(0) << " " << pos(1) << " "
              << pos(2) << " " << quat.x() << " " << quat.y() << " "
              << quat.z() << " " << quat.w() << std::endl;
    }

    processed++;
    if (processed % 200 == 0) {
      auto now = std::chrono::steady_clock::now();
      double secs =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start)
              .count() *
          1e-3;
      PRINT_INFO("processed %d frames (%.1fs wall)\n", processed, secs);
      t_start = now;
    }
  }

  outfile.close();
  PRINT_INFO("done: wrote %d frames to %s\n", processed,
             opt.output_path.c_str());
  return EXIT_SUCCESS;
}
