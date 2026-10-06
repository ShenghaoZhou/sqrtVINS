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
#include "image_prefetcher.h"
#include "core/Frontend.h"
#include "core/InitRunner.h"
#include "core/Pipeline.h"
#include "core/SqrtEstimator.h"
#include "core/System.h"
#include "core/VinsOptions.h"
#ifdef SQRTVINS_BACKEND
#include "backend/BackendSystem.h"
#endif
#include "state/State.h"
#include "types/IMU.h"
#include "utils/Profiler.h"
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
  VinsOptions params;
  params.print_and_load(parser);
  params.cv_backend = opt.cv_backend;
  // Repeatability settings used by the original ros1_serial_msckf
  cv::setNumThreads(params.num_opencv_threads);
  cv::setRNGSeed(0);

  auto sys = System::create(params);
  auto estimator = sys.estimator;
  auto frontend = sys.frontend;
  auto init_runner = sys.init_runner;

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
  auto t_start = std::chrono::steady_clock::now();

  // Decode upcoming stereo pairs on a worker thread (PNG decode is the
  // largest replay cost and otherwise serializes with tracking + update)
  StereoImagePrefetcher prefetcher(opt.dataset_path, cam_data);

  for (const auto &cam : cam_data) {
    double curr_cam_time = cam.timestamp;

    // Feed IMU measurements up to this camera time in one batched call
    // (core/Pipeline.h: feeding policy shared with the Python driver)
    double t_off = estimator->get_state()->calib_dt_CAMtoIMU->value()(0);
    size_t k = imu_batch_end(imu_data, imu_idx, curr_cam_time + t_off,
                             [](const ImuReading &r) { return r.timestamp; });
    if (k > imu_idx) {
      std::vector<ImuData> msgs(k - imu_idx);
      for (size_t i = imu_idx; i < k; i++) {
        msgs[i - imu_idx].timestamp = imu_data.at(i).timestamp;
        msgs[i - imu_idx].wm = imu_data.at(i).wm.cast<DataType>();
        msgs[i - imu_idx].am = imu_data.at(i).am.cast<DataType>();
      }
      {
        SRVINS_PROFILE("drv.feed_imu_batch");
        estimator->feed_imu_batch(msgs);
      }
#ifdef SQRTVINS_BACKEND
      if (sys.backend) {
        sys.backend->feed_imu(msgs);
      }
#endif
      imu_idx = k;
    }

    // Fetch the next prefetched stereo pair (decoded off the critical path)
    cv::Mat img0, img1;
    {
      SRVINS_PROFILE("drv.imread");
      StereoImagePrefetcher::Frame frame = prefetcher.next();
      img0 = frame.img0;
      img1 = frame.img1;
    }
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
    {
      SRVINS_PROFILE("drv.feed_camera_track");
      frontend->feed_camera(message);
    }

    auto state = estimator->get_state();

    // Check for initialization (synchronous, or async shadow solve when
    // init_async is enabled; without ZUPT we must wait for a jerk before
    // the static initializer will use the stationary window)
    if (!state->is_initialized) {
      if (init_runner->try_initialize(curr_cam_time, !params.try_zupt)) {
        PRINT_INFO("VIO Initialized at %.4f!\n", curr_cam_time);
      }
      processed++;
      continue;
    }

    // Try a zero-velocity update
    if (estimator->try_zupt(curr_cam_time)) {
      processed++;
      continue;
    }

    // Propagation, feature selection, update, and database cleanup
    {
      SRVINS_PROFILE("drv.process_frame");
#ifdef SQRTVINS_BACKEND
      // snapshot observations BEFORE the update consumes tracks from the db
      if (sys.backend) {
        sys.backend->record_observations(
            curr_cam_time,
            *frontend->get_trackFEATS()->get_feature_database());
      }
#endif
      process_frame(*estimator, *frontend, message);
#ifdef SQRTVINS_BACKEND
      if (sys.backend) {
        sys.backend->record_pose(*estimator->get_state());
      }
#endif
    }

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

#ifdef SQRTVINS_BACKEND
  // Offline bundle adjustment over the recorded keyframes (Phase 1)
  if (sys.backend) {
    const std::string ba_path = opt.output_path + ".ba";
    BackendSummary summary =
        sys.backend->run_offline_ba(*estimator->get_state(), ba_path);
    PRINT_INFO(
        "[BACKEND]: keyframes=%d points=%d obs=%d solved=%d reproj "
        "before=%.3f after=%.3f final=%.3f px (pruned %d), trajectory -> %s\n",
        summary.num_keyframes, summary.num_points, summary.num_observations,
        (int)summary.solved, summary.mean_reproj_error_before,
        summary.mean_reproj_error_after, summary.mean_reproj_error_final,
        summary.num_pruned_points, ba_path.c_str());
  }
#endif

  StageProfiler::instance().report();
  return EXIT_SUCCESS;
}
