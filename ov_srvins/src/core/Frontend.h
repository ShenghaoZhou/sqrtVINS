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
#include <utility>
#include <vector>
#include <opencv2/opencv.hpp>
#include "VinsOptions.h"
#include "state/State.h"
#include "track/TrackBase.h"

namespace ov_core {
  class Feature;
}

namespace ov_srvins {

class Frontend {
public:
  /**
   * @brief Constructor
   *
   * The state is only read at construction time (camera models); the
   * frontend does NOT retain it. Selection rules and visualization take the
   * current state per call, so a state swap (async init commit) can never
   * leave the frontend pointing at a stale object.
   */
  Frontend(const VinsOptions &params_, std::shared_ptr<State> state_);

  void feed_camera(ov_core::CameraData &message);

  // NOTE: feats_slam is NOT yet separated into UPDATE/DELAYED - the split
  // must happen after StateHelper::marginalize_slam (inside the estimator
  // update), matching VioManager/open_vins ordering
  void process_measurements_rules(
      const std::shared_ptr<State> &state, double timestamp,
      const std::vector<int> &sensor_ids,
      std::vector<std::shared_ptr<ov_core::Feature>> &featsup_MSCKF,
      std::vector<std::shared_ptr<ov_core::Feature>> &feats_slam);

  std::shared_ptr<ov_core::TrackBase> get_trackFEATS() { return trackFEATS; }
  std::shared_ptr<ov_core::TrackBase> get_trackARUCO() { return trackARUCO; }

  /**
   * @brief Replace the CV backend of the feature tracker.
   *
   * Forwards to TrackBase::set_cv_backend, which only KLT-style trackers
   * support. Call before the first feed_camera, e.g. to inject a backend
   * dispatching to Python (see the pysqrtvins bindings).
   */
  void set_cv_backend(std::shared_ptr<ov_core::vision::CVBackend> backend) {
    trackFEATS->set_cv_backend(std::move(backend));
  }

  cv::Mat get_historical_viz_image(const std::shared_ptr<State> &state,
                                   bool did_zupt, bool is_init);

  void set_startup_time(double t) { startup_time = t; }

private:
  /// System parameters (copy, like SqrtEstimator/InitRunner: a reference
  /// member would dangle if the caller's options object dies first, which
  /// the Python bindings cannot guard against)
  VinsOptions params;
  std::shared_ptr<ov_core::TrackBase> trackFEATS;
  std::shared_ptr<ov_core::TrackBase> trackARUCO;
  double startup_time = -1;
};

} // namespace ov_srvins

#endif // OV_SRVINS_FRONTEND_H
