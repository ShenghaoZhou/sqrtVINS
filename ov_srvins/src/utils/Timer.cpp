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




#include "Timer.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

namespace ov_srvins {

void Timer::tic() {
  time_start_ = std::chrono::steady_clock::now();
}

void Timer::toc(const std::string &func_name, bool output) {
  time_end_ = std::chrono::steady_clock::now();
  double dt = std::chrono::duration_cast<std::chrono::microseconds>(time_end_ - time_start_).count() * 1e-3;
  duration_ += dt;
  if (output) {
    std::cout << func_name << " takes " << std::setprecision(5) << dt << " ms"
              << std::endl;
  }
}

void Timer::report(const std::string &func_name) {
  std::cout << func_name << " takes " << std::setprecision(5) << duration_
            << " ms" << std::endl;
}

void Timer::reset() { duration_ = 0.0; }

} // namespace ov_srvins
