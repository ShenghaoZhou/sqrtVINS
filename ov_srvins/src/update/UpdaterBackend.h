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

#ifndef OV_SRVINS_UPDATERBACKEND_H
#define OV_SRVINS_UPDATERBACKEND_H

#include <map>
#include <memory>
#include <utility>

#include "utils/DataType.h"

namespace ov_srvins {

class State;

/**
 * @brief Soft pose feedback from the windowed-BA backend into the filter.
 *
 * Each refined keyframe pose from BackendSystem::get_refined_poses() whose
 * timestamp matches a clone still in the sliding window is applied as a
 * direct 6-dof pose measurement on that clone (identity Jacobian, fixed
 * conservative noise), using the same square-root update machinery as the
 * other updaters (whitened residual + HUT factor + update_llt).
 *
 * The backend window is anchored to the filter frame (oldest window keyframe
 * held constant), so the refined poses live in the filter's global frame and
 * can be fused without an alignment step. Because the window information
 * overlaps the filter's own visual measurements, the feedback noise should
 * be chosen conservatively (large) to avoid double counting.
 */
class UpdaterBackend {
public:
  /**
   * @brief Apply backend-refined poses as soft pose updates on matching clones
   * @param state Filter state (clones_IMU is matched against the timestamps)
   * @param refined_poses Refined keyframe poses keyed by keyframe (camera)
   * timestamp, as (R_GtoI, p_IinG) — clone convention
   * @param sigma_ori Orientation measurement noise [rad] (per axis)
   * @param sigma_pos Position measurement noise [m] (per axis)
   * @param gate_chi2 Per-clone gate on the whitened 6-dof residual squared
   * norm; clones above it are skipped (protects against bad window solves)
   * @return Number of clones that received the update
   */
  static int update(std::shared_ptr<State> state,
                    const std::map<double, std::pair<Mat3, Vec3>>
                        &refined_poses,
                    DataType sigma_ori, DataType sigma_pos,
                    DataType gate_chi2);
};

} // namespace ov_srvins

#endif // OV_SRVINS_UPDATERBACKEND_H
