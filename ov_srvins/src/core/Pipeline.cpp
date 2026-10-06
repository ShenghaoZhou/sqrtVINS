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

#include "Pipeline.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "track/TrackBase.h"
#include "types/Landmark.h"
#include "types/LandmarkRepresentation.h"
#include "utils/print.h"

#include "Frontend.h"
#include "SqrtEstimator.h"
#include "VinsOptions.h"
#include "state/State.h"
#include "utils/Profiler.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_srvins;

bool ov_srvins::process_frame(SqrtEstimator &estimator, Frontend &frontend,
                              const ov_core::CameraData &message) {
  auto state = estimator.get_state();

  // State propagation
  bool prop_ok;
  {
    SRVINS_PROFILE("pipe.propagate");
    prop_ok = estimator.propagate(message.timestamp);
  }
  if (!prop_ok) {
    return false;
  }

  // Wait for enough clones
  if ((int)state->clones_IMU.size() <
          std::min(state->options.max_clone_size, 5) &&
      state->features_SLAM.empty()) {
    return false;
  }

  // Ensure propagation reached target
  if (state->timestamp != message.timestamp) {
    PRINT_WARNING(RED
                  "[PROP]: Propagator unable to reach target time!\n" RESET);
    return false;
  }
  estimator.notify_moved();

  // Capture the marginalization time and cleanup gate BEFORE the update: the
  // update marginalizes the clone at this time, so measurements up to it can
  // be dropped only AFTER selection has used them
  bool do_cleanup =
      (int)state->clones_IMU.size() > state->options.max_clone_size + 1;
  double marg_time = state->margtimestep();

  // Sorting features according to rules
  std::vector<std::shared_ptr<Feature>> feats_slam, featsup_MSCKF;
  {
    SRVINS_PROFILE("pipe.feature_selection");
    frontend.process_measurements_rules(state, message.timestamp,
                                        message.sensor_ids, featsup_MSCKF,
                                        feats_slam);
  }

  // Estimator update
  {
    SRVINS_PROFILE("pipe.estimator_update");
    estimator.update(featsup_MSCKF, feats_slam);
  }

  // Cleanup measurements at the pre-update marginalization time
  if (do_cleanup) {
    frontend.get_trackFEATS()->get_feature_database()->cleanup_measurements(
        marg_time);
    if (frontend.get_trackARUCO() != nullptr) {
      frontend.get_trackARUCO()->get_feature_database()->cleanup_measurements(
          marg_time);
    }
  }

  // Mark consumed MSCKF features for deletion: they have been fused into the
  // filter and must never be re-selected (selection only skips to_delete
  // features, it does not remove them from the database)
  for (auto const &feat : featsup_MSCKF) {
    feat->to_delete = true;
  }

  // Cleanup tracker database
  frontend.get_trackFEATS()->get_feature_database()->cleanup();
  if (frontend.get_trackARUCO() != nullptr) {
    frontend.get_trackARUCO()->get_feature_database()->cleanup();
  }
  return true;
}

void ov_srvins::finalize_initialization(SqrtEstimator &estimator,
                                        Frontend &frontend,
                                        const VinsOptions &params) {
  auto state = estimator.get_state();

  frontend.set_startup_time(state->timestamp);

  // Drop feature measurements from before the initialization time
  frontend.get_trackFEATS()->get_feature_database()->cleanup_measurements(
      state->timestamp);
  if (frontend.get_trackARUCO() != nullptr) {
    frontend.get_trackARUCO()->get_feature_database()->cleanup_measurements(
        state->timestamp);
  }

  // Rebalance the tracker budget across cameras now that we are live
  frontend.get_trackFEATS()->set_num_features(
      std::floor((double)params.num_pts /
                 (double)params.state_options.num_cameras));

  // If already moving at init, ZUPT-at-beginning must not fire again
  if (state->imu->vel().norm() > params.zupt_max_velocity) {
    estimator.notify_moved();
  }
}

bool ov_srvins::is_aruco_landmark(const State &state, size_t featid) {
  return (int)featid <= 4 * state.options.max_aruco_features;
}

/// Global positions of the SLAM landmarks selected by `want_aruco`
static std::vector<Vec3> collect_landmarks(const std::shared_ptr<State> &state,
                                           bool want_aruco) {
  std::vector<Vec3> feats;
  for (auto &f : state->features_SLAM) {
    if (ov_srvins::is_aruco_landmark(*state, f.first) != want_aruco)
      continue;
    if (ov_type::LandmarkRepresentation::is_relative_representation(
            f.second->feat_representation)) {
      assert(f.second->anchor_cam_id != -1);
      const auto anchor_pose = state->cam_pose_buffer.get_buffer_unsafe(
          f.second->anchor_cam_id, f.second->anchor_clone_timestamp);
      feats.push_back(anchor_pose.R_GtoC.transpose() *
                          f.second->get_xyz(false) +
                      anchor_pose.p_CinG);
    } else
      feats.push_back(f.second->get_xyz(false));
  }
  return feats;
}

std::vector<Vec3>
ov_srvins::get_features_SLAM(const std::shared_ptr<State> &state) {
  return collect_landmarks(state, false);
}

std::vector<Vec3>
ov_srvins::get_features_ARUCO(const std::shared_ptr<State> &state) {
  return collect_landmarks(state, true);
}
