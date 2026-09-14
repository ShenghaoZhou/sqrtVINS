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
 */

/**
 * @brief Shared dataset types for the EuRoC runner.
 *
 * NOTE: this header must stay independent of the estimator modules (it may
 * only use Eigen and the STL) because the sqrt and full-covariance modules
 * expose colliding header paths and are therefore compiled in separate
 * translation units.
 */

#pragma once

#include <string>
#include <vector>

#include <Eigen/Eigen>

/// Options for one EuRoC run (shared by both formulations)
struct EurocRunOptions {
  std::string dataset_path;
  std::string config_path;
  std::string output_path;
  std::string cv_backend = "opencv";
  int max_frames = 100000;
  /// Seconds to skip from the start of the dataset (rosbag --start style),
  /// matching the bag_start values used by the original launch files
  double bag_start = 0.0;
};

/// One stereo camera reading (timestamp-intersected cam0/cam1 pair)
struct CamReading {
  double timestamp;
  std::string filename_cam0;
  std::string filename_cam1;
};

/// One IMU reading
struct ImuReading {
  double timestamp;
  Eigen::Vector3d wm, am;
};
