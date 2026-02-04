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

#include "IMUHandler.h"

namespace ov_srvins {

void IMUHandler::feed_imu(const ov_core::ImuData &message, double oldest_time) {
  // Append it to our vector
  {
    std::lock_guard<std::mutex> lck(imu_data_mtx_);
    imu_data_.emplace_back(message);
  }

  // Clean old measurements
  if (oldest_time > 0) {
    clean_old_imu_measurements(oldest_time);
  }
}

void IMUHandler::clean_old_imu_measurements(double oldest_time) {
  std::lock_guard<std::mutex> lck(imu_data_mtx_);
  auto it0 = imu_data_.begin();
  while (it0 != imu_data_.end()) {
    if (it0->timestamp < oldest_time) {
      it0 = imu_data_.erase(it0);
    } else {
      ++it0;
    }
  }
}

void IMUHandler::get_imu_data(std::vector<ov_core::ImuData> &imu_data) {
  std::lock_guard<std::mutex> lck(imu_data_mtx_);
  imu_data = imu_data_;
}

std::shared_ptr<std::vector<ov_core::ImuData>> IMUHandler::get_imu_data() {
  return std::shared_ptr<std::vector<ov_core::ImuData>>(&imu_data_,
                                                        [](auto *) {});
}

} // namespace ov_srvins