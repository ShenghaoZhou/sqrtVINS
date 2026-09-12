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
 * @brief C++ runner for EuRoC-format datasets (asl_dataset layout).
 *
 * Feeds IMU and stereo camera measurements natively through VioManager,
 * bypassing the Python bindings, and writes the estimated trajectory in the
 * standard OpenVINS estimate format (timestamp p_IinG q_GtoI):
 *   timestamp(s), tx, ty, tz, qx, qy, qz, qw
 *
 * Usage:
 *   run_euroc <dataset_path> <config.yaml> <out.csv> [max_frames] [cv_backend]
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "utils/print.h"
#include "utils/yaml_parse.h"

using namespace ov_core;
using namespace ov_srvins;

struct CamReading {
  double timestamp;
  std::string filename_cam0;
  std::string filename_cam1;
};

struct ImuReading {
  double timestamp;
  Eigen::Matrix<DataType, 3, 1> wm, am;
};

static std::vector<std::vector<std::string>>
load_csv(const std::string &path) {
  std::vector<std::vector<std::string>> rows;
  std::ifstream file(path);
  if (!file.is_open()) {
    PRINT_ERROR(RED "[ERROR]: unable to open file %s\n" RESET, path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    std::vector<std::string> cells;
    std::string cell;
    std::stringstream ss(line);
    while (std::getline(ss, cell, ',')) {
      // strip whitespace and carriage returns
      cell.erase(0, cell.find_first_not_of(" \t\r\n"));
      cell.erase(cell.find_last_not_of(" \t\r\n") + 1);
      cells.push_back(cell);
    }
    rows.push_back(cells);
  }
  return rows;
}

// Find the column index in a EuRoC csv header (headers like
// "#timestamp [ns]" are matched by their first token)
static int column_of(const std::vector<std::string> &header,
                     const std::string &name) {
  for (size_t i = 0; i < header.size(); i++) {
    std::string first = header.at(i).substr(0, header.at(i).find(' '));
    if (first == name)
      return (int)i;
  }
  PRINT_ERROR(RED "[ERROR]: column %s not found\n" RESET, name.c_str());
  std::exit(EXIT_FAILURE);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("usage: run_euroc <dataset_path> <config.yaml> <out.csv> "
           "[max_frames] [cv_backend]\n");
    return EXIT_FAILURE;
  }
  std::string dataset_path = argv[1];
  std::string config_path = argv[2];
  std::string output_path = argv[3];
  int max_frames = (argc > 4) ? std::atoi(argv[4]) : 100000;
  std::string cv_backend = (argc > 5) ? argv[5] : "opencv";

  //===================================================================================
  // Load dataset
  //===================================================================================
  auto imu_rows = load_csv(dataset_path + "/mav0/imu0/data.csv");
  auto cam0_rows = load_csv(dataset_path + "/mav0/cam0/data.csv");
  auto cam1_rows = load_csv(dataset_path + "/mav0/cam1/data.csv");

  int imu_ts_col = column_of(imu_rows.at(0), "#timestamp");
  int imu_wm_col = column_of(imu_rows.at(0), "w_RS_S_x");
  int imu_am_col = column_of(imu_rows.at(0), "a_RS_S_x");
  std::vector<ImuReading> imu_data;
  for (size_t i = 1; i < imu_rows.size(); i++) {
    const auto &r = imu_rows.at(i);
    ImuReading reading;
    reading.timestamp = std::stod(r.at(imu_ts_col)) * 1e-9;
    for (int j = 0; j < 3; j++) {
      reading.wm(j) = std::stod(r.at(imu_wm_col + j));
      reading.am(j) = std::stod(r.at(imu_am_col + j));
    }
    imu_data.push_back(reading);
  }

  std::vector<CamReading> cam_data;
  {
    int cam0_ts = column_of(cam0_rows.at(0), "#timestamp");
    int cam0_fn = column_of(cam0_rows.at(0), "filename");
    int cam1_ts = column_of(cam1_rows.at(0), "#timestamp");
    int cam1_fn = column_of(cam1_rows.at(0), "filename");
    // intersect timestamps of the two cameras (both sorted)
    size_t i0 = 1, i1 = 1;
    while (i0 < cam0_rows.size() && i1 < cam1_rows.size()) {
      double t0 = std::stod(cam0_rows.at(i0).at(cam0_ts)) * 1e-9;
      double t1 = std::stod(cam1_rows.at(i1).at(cam1_ts)) * 1e-9;
      if (std::abs(t0 - t1) < 1e-6) {
        CamReading reading;
        reading.timestamp = t0;
        reading.filename_cam0 = cam0_rows.at(i0).at(cam0_fn);
        reading.filename_cam1 = cam1_rows.at(i1).at(cam1_fn);
        cam_data.push_back(reading);
        i0++;
        i1++;
      } else if (t0 < t1) {
        i0++;
      } else {
        i1++;
      }
    }
  }

  //===================================================================================
  // Setup options and system
  //===================================================================================
  auto parser = std::make_shared<YamlParser>(config_path);
  VioManagerOptions params;
  params.print_and_load(parser);
  params.cv_backend = cv_backend;
  VioManager sys(params);

  //===================================================================================
  // Processing loop
  //===================================================================================
  std::ofstream outfile(output_path);
  if (!outfile.is_open()) {
    PRINT_ERROR(RED "[ERROR]: unable to open output file %s\n" RESET,
                output_path.c_str());
    return EXIT_FAILURE;
  }
  outfile << std::setprecision(9) << std::fixed;

  cv::Mat zero_mask;
  size_t imu_idx = 0;
  int processed = 0;
  auto t_start = std::chrono::steady_clock::now();

  for (const auto &cam : cam_data) {
    if (processed >= max_frames)
      break;

    // feed all imu readings up to this camera timestamp
    while (imu_idx < imu_data.size() &&
           imu_data.at(imu_idx).timestamp <= cam.timestamp) {
      const auto &imu = imu_data.at(imu_idx);
      ImuData message;
      message.timestamp = imu.timestamp;
      message.wm = imu.wm;
      message.am = imu.am;
      sys.feed_measurement_imu(message);
      imu_idx++;
    }

    // load stereo images
    cv::Mat img0 = cv::imread(dataset_path + "/mav0/cam0/data/" +
                                  cam.filename_cam0,
                              cv::IMREAD_GRAYSCALE);
    cv::Mat img1 = cv::imread(dataset_path + "/mav0/cam1/data/" +
                                  cam.filename_cam1,
                              cv::IMREAD_GRAYSCALE);
    if (img0.empty() || img1.empty()) {
      processed++;
      continue;
    }

    CameraData message;
    message.timestamp = cam.timestamp;
    message.sensor_ids = {0, 1};
    message.images.push_back(img0);
    message.images.push_back(img1);
    if (zero_mask.empty())
      zero_mask = cv::Mat::zeros(img0.rows, img0.cols, CV_8UC1);
    message.masks.push_back(zero_mask);
    message.masks.push_back(zero_mask.clone());
    sys.feed_measurement_camera(message);

    // record state (OpenVINS estimate format: p_IinG and JPL q_GtoI)
    auto state = sys.get_state();
    if (state != nullptr && state->is_initialized) {
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
  PRINT_INFO("done: wrote %d frames to %s\n", processed, output_path.c_str());
  return EXIT_SUCCESS;
}
