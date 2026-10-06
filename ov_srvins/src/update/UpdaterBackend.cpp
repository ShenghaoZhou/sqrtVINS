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

#include "UpdaterBackend.h"

#include <algorithm>
#include <numeric>
#include <vector>

#include "state/State.h"
#include "state/StateHelper.h"
#include "types/PoseJPL.h"
#include "utils/Helper.h"
#include "utils/print.h"
#include "utils/quat_ops.h"

using namespace ov_core;
using namespace ov_srvins;
using namespace ov_type;

int UpdaterBackend::update(
    std::shared_ptr<State> state,
    const std::map<double, std::pair<Mat3, Vec3>> &refined_poses,
    DataType sigma_ori, DataType sigma_pos, DataType gate_chi2) {

  if (!state->is_initialized || refined_poses.empty()) {
    return 0;
  }

  const DataType w_th = 1.0 / sigma_ori;
  const DataType w_p = 1.0 / sigma_pos;

  // Match refined keyframe poses to clones still in the filter window.
  // Skip the newest clone (at the current state time): its pose is a copy of
  // the *current* IMU pose, which this update does not touch — moving the
  // clone alone would split the twin. Such a keyframe is picked up on a later
  // call once the filter has advanced past it.
  std::vector<std::shared_ptr<Type>> Hx_order;
  std::vector<Eigen::Matrix<DataType, 6, 1>> res_blocks;
  int num_gated = 0;
  for (const auto &kv : state->clones_IMU) {
    const double t = kv.first;
    if (t >= state->timestamp - 1e-9) {
      continue;
    }
    // Tolerant timestamp lookup (both keys derive from camera timestamps)
    auto it = refined_poses.lower_bound(t - 1e-6);
    if (it == refined_poses.end() || it->first > t + 1e-6) {
      continue;
    }
    const Mat3 &R_ba = it->second.first;
    const Vec3 &p_ba = it->second.second;

    // Whitened residual, consistent with the filter's update rule
    // dx = P H^T R^-1 res (res must linearize as +H * dx for state error dx):
    //   ori: log(R_est * R_ba^T) = +dth  (JPL left-mult convention
    //        R_true = (I-[dth]x) R_est, matching the ZUPT relative-pose rows)
    //   pos: p_ba - p_est = +dp          (p_true = p_est + dp)
    // both with H = +w * I.
    Eigen::Matrix<DataType, 6, 1> r;
    r.block(0, 0, 3, 1) = w_th * log_so3(kv.second->Rot() * R_ba.transpose());
    r.block(3, 0, 3, 1) = w_p * (p_ba - kv.second->pos());

    // Per-clone gate: drop wildly inconsistent feedback (bad window solve)
    if (r.squaredNorm() > gate_chi2) {
      num_gated++;
      continue;
    }
    Hx_order.push_back(kv.second);
    res_blocks.push_back(r);
  }
  if (Hx_order.empty()) {
    return 0;
  }

  // get_marginal_U_block requires the measured variables in ascending id
  // order. In this state layout the newest clone has the LOWEST id (clones
  // are inserted at kCloneStartId and older ones shifted right), i.e. ids
  // decrease with timestamp, so sort explicitly.
  std::vector<size_t> sort_idx(Hx_order.size());
  std::iota(sort_idx.begin(), sort_idx.end(), 0);
  std::sort(sort_idx.begin(), sort_idx.end(), [&](size_t a, size_t b) {
    return Hx_order.at(a)->id() < Hx_order.at(b)->id();
  });
  {
    std::vector<std::shared_ptr<Type>> sorted_vars;
    std::vector<Eigen::Matrix<DataType, 6, 1>> sorted_res;
    for (size_t k : sort_idx) {
      sorted_vars.push_back(Hx_order.at(k));
      sorted_res.push_back(res_blocks.at(k));
    }
    Hx_order = std::move(sorted_vars);
    res_blocks = std::move(sorted_res);
  }

  // Stacked system: 6 rows per clone, block-diagonal H (identity Jacobians
  // for the direct pose measurement, whitened by the noise sigmas)
  const int n = (int)Hx_order.size();
  MatX H = MatX::Zero(6 * n, 6 * n);
  VecX res = VecX::Zero(6 * n);
  for (int k = 0; k < n; k++) {
    H.block(6 * k, 6 * k, 3, 3) = w_th * Mat3::Identity();
    H.block(6 * k + 3, 6 * k + 3, 3, 3) = w_p * Mat3::Identity();
    res.block(6 * k, 0, 6, 1) = res_blocks.at(k);
  }

  // Compose the update factor against the marginal square root of the
  // measured columns (same pattern as the MSCKF updater; lossless for any
  // ascending-id subset since U is upper triangular)
  MatX U_dense, U_tri;
  StateHelper::get_marginal_U_block(state, Hx_order, U_dense, U_tri);
  MatX HUT = MatX::Zero(H.rows(), U_dense.rows() + U_tri.rows());
  HUT.leftCols(U_dense.rows()).noalias() = H * U_dense.transpose();
  HUT.rightCols(U_tri.rows()).noalias() =
      H * U_tri.transpose().triangularView<Eigen::Lower>();

  // Accumulate H^T R^-1 res per variable (R = I after whitening)
  VecX RHTr = VecX::Zero(state->get_state_size());
  int local_id = 0;
  for (const auto &var : Hx_order) {
    RHTr.block(var->id(), 0, var->size(), 1).noalias() +=
        H.block(0, local_id, H.rows(), var->size()).transpose() * res;
    local_id += var->size();
  }

  state->setup_matrix_buffer();
  state->store_update_factor(HUT, RHTr);
  StateHelper::update_llt(state);
  state->clear();

  PRINT_DEBUG("[BACKEND-FB]: pose feedback on %d clones (%d gated)\n", n,
              num_gated);
  return n;
}
