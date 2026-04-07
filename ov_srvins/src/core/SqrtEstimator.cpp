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

#include "SqrtEstimator.h"

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "update/UpdaterMSCKF.h"
#include "update/UpdaterSLAM.h"
#include "update/UpdaterZeroVelocity.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

using namespace ov_core;
using namespace ov_type;
using namespace ov_srvins;

SqrtEstimator::SqrtEstimator(VioManagerOptions &params_) : params(params_) {
  // Create the state!!
  state = std::make_shared<State>(params.state_options, params.init_options);

  // Timeoffset from camera to IMU
  VecX temp_camimu_dt;
  temp_camimu_dt.resize(1);
  temp_camimu_dt(0) = params.calib_camimu_dt;
  state->calib_dt_CAMtoIMU->set_value(temp_camimu_dt);
  state->calib_dt_CAMtoIMU->set_fej(temp_camimu_dt);

  // Loop through and load each of the cameras
  state->cam_intrinsics_cameras = params.camera_intrinsics;
  for (int i = 0; i < state->options.num_cameras; i++) {
    state->cam_intrinsics.at(i)->set_value(
        params.camera_intrinsics.at(i)->get_value());
    state->cam_intrinsics.at(i)->set_fej(
        params.camera_intrinsics.at(i)->get_value());
    state->calib_IMUtoCAM.at(i)->set_value(params.camera_extrinsics.at(i));
    state->calib_IMUtoCAM.at(i)->set_fej(params.camera_extrinsics.at(i));
  }

  // Initialize our state propagator
  propagator =
      std::make_shared<Propagator>(params.imu_noises, params.gravity_mag);

  // If we are using zero velocity updates, then create the updater
  // Note: VioManager will set its feature database as we won't have it here
  if (params.try_zupt) {
    updaterZUPT = std::make_shared<UpdaterZeroVelocity>(
        params.zupt_options, params.imu_noises, nullptr, propagator,
        params.gravity_mag, params.zupt_max_velocity,
        params.zupt_noise_multiplier, params.zupt_max_disparity);
  }
}

void SqrtEstimator::feed_imu(const ov_core::ImuData &message,
                             double oldest_time) {
  propagator->feed_imu(message, oldest_time);

  // ZUPT feeder
  if (updaterZUPT != nullptr) {
    updaterZUPT->feed_imu(message, oldest_time);
  }
}

bool SqrtEstimator::try_zupt(double timestamp, bool &has_moved_since_zupt) {
  if (updaterZUPT == nullptr)
    return false;

  // Check if we have moved since the last ZUPT update
  if (params.zupt_only_at_beginning && has_moved_since_zupt)
    return false;

  // No ZUPT if slamming
  if (!state->features_SLAM.empty())
    return false;

  bool did_zupt_update = false;
  // If the same state time, use the previous timestep decision
  if (state->timestamp != timestamp) {
    state->setup_matrix_buffer();
    did_zupt_update = updaterZUPT->try_update(state, timestamp);
  }

  if (did_zupt_update) {
    assert(state->timestamp == timestamp);
    // Cleanup old imu data if ZUPT update successful
    double cleanup_time =
        timestamp + state->calib_dt_CAMtoIMU->value()(0) - 0.10;
    propagator->clean_old_imu_measurements(cleanup_time);
    updaterZUPT->clean_old_imu_measurements(cleanup_time);
    return true;
  }

  return false;
}

bool SqrtEstimator::propagate(double timestamp) {
  // Return if the camera measurement is out of order
  if (state->timestamp > timestamp) {
    PRINT_WARNING(YELLOW "image received out of order, unable to do anything "
                         "(prop dt = %3f)\n" RESET,
                  (timestamp - state->timestamp));
    return false;
  }

  // Propagate the state forward to the current update time
  if (state->timestamp != timestamp) {
    propagator->propagate_and_clone(state, timestamp);
  }
  return true;
}

void SqrtEstimator::update(
    std::vector<std::shared_ptr<ov_core::Feature>> &featsup_MSCKF,
    std::vector<std::shared_ptr<ov_core::Feature>> &feats_slam_UPDATE,
    std::vector<std::shared_ptr<ov_core::Feature>> &feats_slam_DELAYED) {

  // First do anchor change if we are about to lose an anchor pose
  state->calculate_clone_poses();
  if (state->options.do_fej) {
    state->calculate_clone_poses_fej();
  }

  // Handle marginalization of old clone and features
  handle_marginalization();

  // Perform the actual updates
  state->setup_matrix_buffer();
  UpdaterMSCKF::update(state, featsup_MSCKF, params.msckf_options,
                       params.featinit_options);

  UpdaterSLAM::update(state, feats_slam_UPDATE, params.slam_options,
                      params.aruco_options);

  UpdaterSLAM::delayed_init(state, feats_slam_DELAYED, params.slam_options,
                            params.aruco_options, params.featinit_options);

  // Final factorization and state update
  StateHelper::initialize_slam_in_U(state);
  StateHelper::update_llt(state);
  state->clear(true);
}

void SqrtEstimator::set_zupt_database(
    std::shared_ptr<ov_core::FeatureDatabase> db) {
  if (updaterZUPT != nullptr) {
    updaterZUPT = std::make_shared<UpdaterZeroVelocity>(
        params.zupt_options, params.imu_noises, db, propagator,
        params.gravity_mag, params.zupt_max_velocity,
        params.zupt_noise_multiplier, params.zupt_max_disparity);
  }
}

void SqrtEstimator::handle_marginalization() {
  // Lets marginalize out all old SLAM features here
  StateHelper::marginalize_slam(state);

  // Anchor change
  UpdaterSLAM::change_anchors(state);

  // Marginalize the oldest clone if needed
  StateHelper::marginalize_old_clone(state);
  StateHelper::marginalize(state);
}
