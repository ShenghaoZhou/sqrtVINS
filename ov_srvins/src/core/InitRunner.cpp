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

#include "InitRunner.h"

#include <chrono>

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "track/TrackBase.h"
#include "utils/print.h"

#include "Frontend.h"
#include "Pipeline.h"
#include "SqrtEstimator.h"
#include "initializer/dynamic/DynamicInitializer.h"
#include "initializer/InertialInitializer.h"
#include "state/Propagator.h"
#include "state/State.h"

using namespace ov_core;
using namespace ov_srvins;

InitRunner::InitRunner(const VinsOptions &params,
                       std::shared_ptr<SqrtEstimator> estimator,
                       std::shared_ptr<Frontend> frontend,
                       std::shared_ptr<InertialInitializer> initializer)
    : params_(params), estimator_(std::move(estimator)),
      frontend_(std::move(frontend)), initializer_(std::move(initializer)) {}

InitRunner::~InitRunner() {
  // Never leave the background solve running past our lifetime; it only
  // touches its private snapshot objects, so letting it finish is safe
  if (solve_future_.valid()) {
    solve_future_.wait();
  }
}

bool InitRunner::try_initialize(double cam_time, bool wait_for_jerk) {
  auto state = estimator_->get_state();

  //====================================================================
  // Synchronous path (default): identical to calling
  // InertialInitializer::initialize() + finalize_initialization()
  //====================================================================
  if (!params_.init_options.init_async) {
    if (initializer_->initialize(state, wait_for_jerk)) {
      finalize_initialization(*estimator_, *frontend_, params_);
      return true;
    }
    return false;
  }

  //====================================================================
  // Async path
  //====================================================================
  if (!solve_pending_) {
    // Readiness / jerk decision on the main thread (cheap, keeps the live
    // database trimmed and the disparity info fresh)
    InertialInitializer::InitMethod method;
    if (!initializer_->choose_method(wait_for_jerk, method)) {
      return false;
    }
    if (method == InertialInitializer::InitMethod::STATIC) {
      // Static init is cheap: run it synchronously on the live state
      bool success = initializer_->run_static(state);
      initializer_->finish_attempt();
      if (success) {
        state->is_initialized = true;
        finalize_initialization(*estimator_, *frontend_, params_);
        return true;
      }
      return false;
    }
    if (method == InertialInitializer::InitMethod::NONE) {
      initializer_->finish_attempt();
      return false;
    }
    // DYNAMIC: launch the expensive solve on snapshots and keep going
    PRINT_INFO(YELLOW "[init]: launching asynchronous dynamic initialization "
                      "solve\n" RESET);
    launch_shadow_solve();
    pending_cam_times_.push_back(cam_time);
    return false;
  }

  // Solve in flight: record the frame for the post-commit clone replay
  pending_cam_times_.push_back(cam_time);
  if (solve_future_.wait_for(std::chrono::seconds(0)) !=
      std::future_status::ready) {
    return false;
  }

  // Resolve the attempt (jerk bookkeeping must advance exactly once per
  // attempt, as in the synchronous path)
  bool success = solve_future_.get();
  solve_pending_ = false;
  initializer_->finish_attempt();
  estimator_->unpin_imu_trim();
  if (!success) {
    PRINT_WARNING(YELLOW "[init]: asynchronous dynamic initialization failed, "
                         "will retry\n" RESET);
    pending_cam_times_.clear();
    return false;
  }

  commit_shadow_state();
  finalize_initialization(*estimator_, *frontend_, params_);
  return true;
}

void InitRunner::launch_shadow_solve() {
  auto live_state = estimator_->get_state();

  // Shadow state with deep-copied calibration (the solve refines calib
  // values in place; it must not alias the live state's variables)
  shadow_state_ = make_shadow_state();

  // Private propagator fed from a locked copy of the IMU buffer
  std::vector<ImuData> imu_snapshot;
  estimator_->get_propagator()->get_imu_data(imu_snapshot);
  auto shadow_propagator = std::make_shared<Propagator>(
      params_.imu_noises, params_.gravity_mag);
  for (const auto &m : imu_snapshot) {
    shadow_propagator->feed_imu(m);
  }

  // Deep copy of the feature database (trackers keep appending to the live
  // one) plus the disparity map the keyframing step reads
  auto live_db = frontend_->get_trackFEATS()->get_feature_database();
  auto shadow_db = std::make_shared<FeatureDatabase>();
  for (const auto &feat : live_db->get_internal_data()) {
    shadow_db->insert_feature(feat.first,
                              std::make_shared<Feature>(*feat.second));
  }
  shadow_db->map_disp = live_db->map_disp;

  // Pin the IMU trim floor so the buffer keeps everything the post-commit
  // catch-up propagation will need (oldest keyframe, IMU clock, margin)
  double pin_time = initializer_->last_oldest_win_time() +
                    params_.calib_camimu_dt - 0.05;
  estimator_->pin_imu_trim(pin_time);

  // Copy everything the solve needs; the background thread must not touch
  // any live object
  auto init_options = params_.init_options;
  auto msckf_options = params_.msckf_options;
  auto slam_options = params_.slam_options;
  auto feat_init_options = params_.featinit_options;
  auto shadow_state = shadow_state_;
  solve_future_ = std::async(
      std::launch::async,
      [shadow_state, shadow_db, shadow_propagator, init_options,
       msckf_options, slam_options, feat_init_options]() {
        DynamicInitializer solver(init_options, shadow_db, shadow_propagator,
                                  msckf_options, slam_options,
                                  feat_init_options);
        // initialize() takes a non-const shared_ptr ref; hand it a local
        // copy (same underlying State object)
        auto state = shadow_state;
        return solver.initialize(state);
      });
  solve_pending_ = true;
}

void InitRunner::commit_shadow_state() {
  shadow_state_->is_initialized = true;
  estimator_->swap_state(shadow_state_);
  // The frontend holds no state pointer (selection rules take it per call),
  // so nothing else needs repointing here.

  // Rebuild the clone window over the frames tracked while the solve was
  // in flight (the pinned IMU buffer guarantees the data is still there)
  for (double t : pending_cam_times_) {
    if (t > shadow_state_->timestamp &&
        !estimator_->propagate(t)) {
      PRINT_WARNING(YELLOW "[init]: catch-up propagation unable to reach "
                           "%.4f (state at %.4f)\n" RESET,
                    t, shadow_state_->timestamp);
    }
  }
  pending_cam_times_.clear();
}

std::shared_ptr<State> InitRunner::make_shadow_state() {
  auto live = estimator_->get_state();
  auto shadow =
      std::make_shared<State>(params_.state_options, params_.init_options);

  // Timeoffset from camera to IMU
  shadow->calib_dt_CAMtoIMU->set_value(live->calib_dt_CAMtoIMU->value());
  shadow->calib_dt_CAMtoIMU->set_fej(live->calib_dt_CAMtoIMU->value());

  // Camera models are read-only during the solve and can be shared; the
  // per-camera calibration VARIABLES must be independent copies
  shadow->cam_intrinsics_cameras = live->cam_intrinsics_cameras;
  for (int i = 0; i < shadow->options.num_cameras; i++) {
    shadow->cam_intrinsics.at(i)->set_value(
        live->cam_intrinsics.at(i)->value());
    shadow->cam_intrinsics.at(i)->set_fej(
        live->cam_intrinsics.at(i)->value());
    shadow->calib_IMUtoCAM.at(i)->set_value(
        live->calib_IMUtoCAM.at(i)->value());
    shadow->calib_IMUtoCAM.at(i)->set_fej(
        live->calib_IMUtoCAM.at(i)->value());
  }
  return shadow;
}
