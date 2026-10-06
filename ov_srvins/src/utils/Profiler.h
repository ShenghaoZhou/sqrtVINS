/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
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

/// Lightweight per-stage wall-clock profiler for bottleneck analysis.
/// Enabled by defining SRVINS_PROFILER; otherwise all macros compile away.

#ifndef OV_SRVINS_PROFILER_H
#define OV_SRVINS_PROFILER_H

#include <chrono>
#include <cstdio>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ov_srvins {

class StageProfiler {
public:
  struct Stats {
    double total_s = 0.0;
    long count = 0;
  };

  static StageProfiler &instance() {
    static StageProfiler inst;
    return inst;
  }

  void add(const std::string &name, double seconds) {
    std::lock_guard<std::mutex> lck(mtx_);
    stats_[name].total_s += seconds;
    stats_[name].count++;
  }

  void report() const {
    std::vector<std::pair<std::string, Stats>> rows(stats_.begin(),
                                                    stats_.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto &a, const auto &b) {
                return a.second.total_s > b.second.total_s;
              });
    printf("\n===== StageProfiler report =====\n");
    printf("%-32s %12s %10s %12s\n", "stage", "total (s)", "calls", "avg (ms)");
    for (const auto &row : rows) {
      printf("%-32s %12.3f %10ld %12.3f\n", row.first.c_str(),
             row.second.total_s, row.second.count,
             row.second.count > 0 ? 1e3 * row.second.total_s / row.second.count
                                  : 0.0);
    }
  }

private:
  mutable std::mutex mtx_;
  std::map<std::string, Stats> stats_;
};

class ScopedTimer {
public:
  explicit ScopedTimer(const char *name)
      : name_(name), t0_(std::chrono::steady_clock::now()) {}
  ~ScopedTimer() {
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                                t0_)
                      .count();
    StageProfiler::instance().add(name_, secs);
  }

private:
  const char *name_;
  std::chrono::steady_clock::time_point t0_;
};

} // namespace ov_srvins

#define SRVINS_CONCAT_(a, b) a##b
#define SRVINS_CONCAT(a, b) SRVINS_CONCAT_(a, b)
#ifdef SRVINS_PROFILER
#define SRVINS_PROFILE(name)                                                   \
  ov_srvins::ScopedTimer SRVINS_CONCAT(srvins_timer_, __LINE__)(name)
#else
#define SRVINS_PROFILE(name)
#endif

#endif // OV_SRVINS_PROFILER_H
