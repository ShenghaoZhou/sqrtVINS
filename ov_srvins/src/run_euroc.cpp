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
 * @brief C++ runner for EuRoC-format datasets (asl_dataset layout) supporting
 * BOTH filter formulations, selected with a flag:
 *
 *   --estimator sqrt   SqrtVINS square-root EKF formulation (default)
 *   --estimator full   original OpenVINS full-covariance EKF formulation
 *
 * Writes the estimated trajectory in the OpenVINS estimate format
 * (timestamp p_IinG q_GtoI): timestamp(s), tx, ty, tz, qx, qy, qz, qw
 *
 * Usage:
 *   run_euroc <dataset_path> <config.yaml> <out.csv> [max_frames]
 *             [cv_backend] [--estimator sqrt|full]
 *
 * The two formulations are implemented in separate translation units
 * (run_euroc_sqrt.cpp / run_euroc_full.cpp) because their headers expose
 * colliding include paths; this file stays module-independent.
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "euroc_common.h"
#include "utils/colors.h"
#include "utils/print.h"

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

// Defined in run_euroc_sqrt.cpp (SqrtVINS square-root EKF formulation)
int run_euroc_sqrt(const EurocRunOptions &opt,
                   const std::vector<ImuReading> &imu_data,
                   const std::vector<CamReading> &cam_data);

// Defined in run_euroc_full.cpp (original OpenVINS full-covariance EKF)
int run_euroc_full(const EurocRunOptions &opt,
                   const std::vector<ImuReading> &imu_data,
                   const std::vector<CamReading> &cam_data);

int main(int argc, char **argv) {
  //===================================================================================
  // Parse arguments: positional (dataset, config, out, max_frames,
  // cv_backend) plus the --estimator formulation flag
  //===================================================================================
  std::string estimator = "sqrt";
  double bag_start = 0.0;
  std::vector<std::string> positional;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--estimator" || arg == "--formulation") {
      if (i + 1 >= argc) {
        PRINT_ERROR(RED "[ERROR]: %s requires a value (sqrt|full)\n" RESET,
                    arg.c_str());
        return EXIT_FAILURE;
      }
      estimator = argv[++i];
    } else if (arg.rfind("--estimator=", 0) == 0) {
      estimator = arg.substr(std::string("--estimator=").size());
    } else if (arg == "--sqrt") {
      estimator = "sqrt";
    } else if (arg == "--full") {
      estimator = "full";
    } else if (arg == "--bag_start") {
      if (i + 1 >= argc) {
        PRINT_ERROR(RED "[ERROR]: --bag_start requires a value\n" RESET);
        return EXIT_FAILURE;
      }
      bag_start = std::atof(argv[++i]);
    } else if (arg.rfind("--bag_start=", 0) == 0) {
      bag_start = std::atof(arg.c_str() + std::string("--bag_start=").size());
    } else {
      positional.push_back(arg);
    }
  }
  std::transform(estimator.begin(), estimator.end(), estimator.begin(),
                 ::tolower);
  if (estimator != "sqrt" && estimator != "full") {
    PRINT_ERROR(RED "[ERROR]: unknown estimator '%s' (expected sqrt or full)\n" RESET,
                estimator.c_str());
    return EXIT_FAILURE;
  }
  if (positional.size() < 3) {
    printf("usage: run_euroc <dataset_path> <config.yaml> <out.csv> "
           "[max_frames] [cv_backend] [--estimator sqrt|full]\n");
    return EXIT_FAILURE;
  }
  EurocRunOptions opt;
  opt.dataset_path = positional.at(0);
  opt.config_path = positional.at(1);
  opt.output_path = positional.at(2);
  opt.max_frames = (positional.size() > 3) ? std::atoi(positional.at(3).c_str())
                                           : 100000;
  opt.cv_backend =
      (positional.size() > 4) ? positional.at(4) : "opencv";
  PRINT_INFO("formulation: %s\n", estimator.c_str());

  //===================================================================================
  // Load dataset
  //===================================================================================
  auto imu_rows = load_csv(opt.dataset_path + "/mav0/imu0/data.csv");
  auto cam0_rows = load_csv(opt.dataset_path + "/mav0/cam0/data.csv");
  auto cam1_rows = load_csv(opt.dataset_path + "/mav0/cam1/data.csv");

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
  if (cam_data.size() > (size_t)opt.max_frames)
    cam_data.resize((size_t)opt.max_frames);

  //===================================================================================
  // Skip the first bag_start seconds (like a rosbag view starting at
  // bag_begin + bag_start): drop everything before that time
  //===================================================================================
  opt.bag_start = bag_start;
  if (opt.bag_start > 0 && !imu_data.empty() && !cam_data.empty()) {
    double t_begin =
        std::min(imu_data.front().timestamp, cam_data.front().timestamp);
    double t0 = t_begin + opt.bag_start;
    imu_data.erase(std::remove_if(imu_data.begin(), imu_data.end(),
                                  [&](const ImuReading &m) {
                                    return m.timestamp < t0;
                                  }),
                   imu_data.end());
    cam_data.erase(std::remove_if(cam_data.begin(), cam_data.end(),
                                  [&](const CamReading &c) {
                                    return c.timestamp < t0;
                                  }),
                   cam_data.end());
    PRINT_INFO("bag_start: skipping to t = %.3f (%zu imu, %zu cam left)\n", t0,
               imu_data.size(), cam_data.size());
  }

  //===================================================================================
  // Run the selected formulation
  //===================================================================================
  if (estimator == "sqrt")
    return run_euroc_sqrt(opt, imu_data, cam_data);
  return run_euroc_full(opt, imu_data, cam_data);
}
