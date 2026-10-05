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

#ifndef OV_SRVINS_PIPELINE_H
#define OV_SRVINS_PIPELINE_H

#include <memory>
#include <vector>

#include "utils/DataType.h"
#include "utils/sensor_data.h"

namespace ov_srvins {

class SqrtEstimator;
class Frontend;
class State;
class VioManagerOptions;

/**
 * @brief Canonical per-camera-frame filter step (replaces VioManager).
 *
 * Runs the full MSCKF sequence: propagate to the frame time, wait for enough
 * clones, select features with the frontend rules, apply the estimator update,
 * then clean the tracker databases. Consumed MSCKF features are marked
 * to_delete before cleanup so they can never be re-selected (feature selection
 * only skips deleted features, it does not remove them from the database).
 *
 * Ordering contract:
 *  - margtimestep()/cleanup gate are captured BEFORE the update (the update
 *    marginalizes that clone)
 *  - cleanup_measurements(marg_time) runs only AFTER feature selection
 *
 * @param estimator The square-root filter
 * @param frontend  The vision frontend (trackers + selection rules)
 * @param message   Camera message (only timestamp and sensor_ids are used)
 * @return True if a visual update was performed
 */
bool process_frame(SqrtEstimator &estimator, Frontend &frontend,
                   const ov_core::CameraData &message);

/**
 * @brief Post-initialization bookkeeping (replaces VioManager).
 *
 * Sets the frontend startup time, drops pre-init feature measurements,
 * rebalance the per-camera feature budget, and marks the estimator as moved
 * if the initial velocity is already above the ZUPT threshold.
 *
 * @param estimator The square-root filter (state must be initialized)
 * @param frontend  The vision frontend
 * @param params    System parameters
 */
void finalize_initialization(SqrtEstimator &estimator, Frontend &frontend,
                             const VioManagerOptions &params);

/// Global positions of active SLAM landmarks (excluding aruco tags)
std::vector<Vec3> get_features_SLAM(const std::shared_ptr<State> &state);

/// Global positions of active aruco tag landmarks
std::vector<Vec3> get_features_ARUCO(const std::shared_ptr<State> &state);

} // namespace ov_srvins

#endif // OV_SRVINS_PIPELINE_H
