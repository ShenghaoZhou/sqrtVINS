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

#ifndef OV_SRVINS_STATE_IMU_HANDLER_H
#define OV_SRVINS_STATE_IMU_HANDLER_H

#include <memory>
#include <mutex>
#include <vector>
#include <functional>

#include <Eigen/Eigen>

#include "utils/sensor_data.h"

namespace ov_srvins {

/**
 * @brief Handles IMU data storage and management for the propagator
 */
class IMUHandler {

public:
  /**
   * @brief Stores incoming inertial readings
   * @param message Contains our timestamp and inertial information
   * @param oldest_time Time that we can discard measurements before (in IMU clock)
   */
  void feed_imu(const ov_core::ImuData &message, double oldest_time = -1);

  /**
   * @brief This will remove any IMU measurements that are older then the given
   * measurement time
   * @param oldest_time Time that we can discard measurements before (in IMU clock)
   */
  void clean_old_imu_measurements(double oldest_time);

  /**
   * @brief Gets a copy of the IMU data
   * @param imu_data Vector to fill with IMU data
   */
  void get_imu_data(std::vector<ov_core::ImuData> &imu_data);

  /**
   * @brief Gets a shared pointer to the IMU data (use with caution)
   * @return Shared pointer to the IMU data vector
   */
  std::shared_ptr<std::vector<ov_core::ImuData>> get_imu_data();

  /**
   * @brief Executes a function with access to the IMU data under lock
   * @param f Function to execute with const reference to imu_data
   */
  template<typename Func>
  void with_imu_data(Func&& f) {
    std::lock_guard<std::mutex> lck(imu_data_mtx_);
    f(imu_data_);
  }

private:
  /// Our history of IMU messages (time, angular, linear)
  std::vector<ov_core::ImuData> imu_data_;

  /// Mutex for thread-safe access to IMU data
  std::mutex imu_data_mtx_;
};

} // namespace ov_srvins

#endif // OV_SRVINS_STATE_IMU_HANDLER_H