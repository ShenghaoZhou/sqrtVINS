/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/**
 * @brief Original OpenVINS (full-covariance EKF) formulation of the EuRoC
 * runner.
 *
 * Feeds IMU and stereo camera measurements natively through the upstream
 * ov_msckf::VioManager (no ROS, no Python bindings).
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
#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "types/IMU.h"
#include "utils/print.h"
#include "utils/yaml_parse.h"

using namespace ov_core;
using namespace ov_msckf;

int run_euroc_full(const EurocRunOptions &opt, const std::vector<ImuReading> &imu_data,
                   const std::vector<CamReading> &cam_data) {
  //===================================================================================
  // Setup options and system
  //===================================================================================
  auto parser = std::make_shared<YamlParser>(opt.config_path);
  VioManagerOptions params;
  params.print_and_load(parser);
  params.cv_backend = opt.cv_backend;
  params.num_opencv_threads = 4;
  VioManager sys(params);

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
  auto t_start = std::chrono::steady_clock::now();

  for (const auto &cam : cam_data) {
    double curr_cam_time = cam.timestamp;

    // Feed all IMU readings up to this camera timestamp
    while (imu_idx < imu_data.size() && imu_data.at(imu_idx).timestamp <= curr_cam_time) {
      const auto &imu = imu_data.at(imu_idx);
      ImuData message;
      message.timestamp = imu.timestamp;
      message.wm = imu.wm.cast<DataType>();
      message.am = imu.am.cast<DataType>();
      sys.feed_measurement_imu(message);
      imu_idx++;
    }

    // Load stereo images
    cv::Mat img0 = cv::imread(opt.dataset_path + "/mav0/cam0/data/" + cam.filename_cam0,
                              cv::IMREAD_GRAYSCALE);
    cv::Mat img1 = cv::imread(opt.dataset_path + "/mav0/cam1/data/" + cam.filename_cam1,
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
    sys.feed_measurement_camera(message);

    // Record state (OpenVINS estimate format: p_IinG and JPL q_GtoI)
    auto state = sys.get_state();
    if (state != nullptr && state->_timestamp != -1 && sys.initialized()) {
      Eigen::Vector3d pos = state->_imu->pos().cast<double>();
      Eigen::Matrix3d Rot = state->_imu->Rot().cast<double>(); // R_GtoI
      Eigen::Quaterniond quat(Rot.transpose());                // R_ItoG as quat
      outfile << state->_timestamp << " " << pos(0) << " " << pos(1) << " " << pos(2) << " " << quat.x() << " "
              << quat.y() << " " << quat.z() << " " << quat.w() << std::endl;
    }

    processed++;
    if (processed % 200 == 0) {
      auto now = std::chrono::steady_clock::now();
      double secs = std::chrono::duration_cast<std::chrono::milliseconds>(now - t_start).count() * 1e-3;
      PRINT_INFO("processed %d frames (%.1fs wall)\n", processed, secs);
      t_start = now;
    }
  }

  outfile.close();
  PRINT_INFO("done: wrote %d frames to %s\n", processed, opt.output_path.c_str());
  return EXIT_SUCCESS;
}
