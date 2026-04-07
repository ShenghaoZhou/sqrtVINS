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

#ifndef OV_SRVINS_FRONTEND_H
#define OV_SRVINS_FRONTEND_H

#include <cstddef>
#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
#include "VioManagerOptions.h"
#include "state/State.h"
#include "track/TrackBase.h"

namespace ov_core {
  class Feature;
}

namespace ov_srvins {

class Frontend {
public:
  Frontend(VioManagerOptions &params_, std::shared_ptr<State> state_);

  void feed_camera(ov_core::CameraData &message);

  void process_measurements_rules(
      double timestamp, const std::vector<int> &sensor_ids,
      std::vector<std::shared_ptr<ov_core::Feature>> &featsup_MSCKF,
      std::vector<std::shared_ptr<ov_core::Feature>> &feats_slam_UPDATE,
      std::vector<std::shared_ptr<ov_core::Feature>> &feats_slam_DELAYED);

  std::shared_ptr<ov_core::TrackBase> get_trackFEATS() { return trackFEATS; }
  std::shared_ptr<ov_core::TrackBase> get_trackARUCO() { return trackARUCO; }

  cv::Mat get_historical_viz_image(bool did_zupt, bool is_init);

  void set_startup_time(double t) { startup_time = t; }

private:
  VioManagerOptions &params;
  std::shared_ptr<State> state;
  std::shared_ptr<ov_core::TrackBase> trackFEATS;
  std::shared_ptr<ov_core::TrackBase> trackARUCO;
  double startup_time = -1;
};

} // namespace ov_srvins

#endif // OV_SRVINS_FRONTEND_H
