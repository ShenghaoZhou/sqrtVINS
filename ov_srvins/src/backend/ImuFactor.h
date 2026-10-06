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

#ifndef OV_SRVINS_IMUFACTOR_H
#define OV_SRVINS_IMUFACTOR_H

#include <Eigen/Eigen>

#include <ceres/ceres.h>

#include "backend/ImuPreintegration.h"

namespace ov_srvins {

/// Constant data of one preintegrated IMU factor between keyframes i and j
struct ImuFactorData {
  Eigen::Matrix3d dR;          // R_GtoI_i * R_GtoI_j^T
  Eigen::Vector3d dv, dp;      // integrated deltas in the IMU_i frame
  Eigen::Vector3d bg_lin, ba_lin; // biases at integration time
  Eigen::Vector3d gravity;     // world-frame gravity vector (subtracted in dynamics)
  double dt = 0.0;             // t_j - t_i
  // first-order bias-correction Jacobians (blocks of the preintegration
  // error-state Jacobian wrt the biases at t0)
  Eigen::Matrix3d J_th_bg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_v_bg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_v_ba = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_p_bg = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d J_p_ba = Eigen::Matrix3d::Zero();
  /// Square-root information (L^-1 of the preintegration covariance LLT),
  /// residual ordering [theta(0:3), p(3:6), v(6:9), bg(9:12), ba(12:15)]
  Eigen::Matrix<double, 15, 15> sqrt_info =
      Eigen::Matrix<double, 15, 15>::Identity();

  /// Fill from a completed preintegration (computes sqrt_info internally)
  static ImuFactorData FromPreintegration(const ImuPreintegration &pre,
                                          const Eigen::Vector3d &gravity) {
    ImuFactorData d;
    d.dR = pre.dR;
    d.dv = pre.dv;
    d.dp = pre.dp;
    d.bg_lin = pre.bg_lin;
    d.ba_lin = pre.ba_lin;
    d.gravity = gravity;
    d.dt = pre.sum_dt;
    d.J_th_bg = pre.jac.block<3, 3>(ImuPreintegration::O_TH,
                                    ImuPreintegration::O_BG);
    d.J_v_bg =
        pre.jac.block<3, 3>(ImuPreintegration::O_V, ImuPreintegration::O_BG);
    d.J_v_ba =
        pre.jac.block<3, 3>(ImuPreintegration::O_V, ImuPreintegration::O_BA);
    d.J_p_bg =
        pre.jac.block<3, 3>(ImuPreintegration::O_P, ImuPreintegration::O_BG);
    d.J_p_ba =
        pre.jac.block<3, 3>(ImuPreintegration::O_P, ImuPreintegration::O_BA);
    // square-root information from the preintegration covariance
    Eigen::Matrix<double, 15, 15> cov = pre.cov;
    Eigen::LLT<Eigen::Matrix<double, 15, 15>> llt(cov);
    if (llt.info() != Eigen::Success) {
      cov += 1e-12 * Eigen::Matrix<double, 15, 15>::Identity();
      llt.compute(cov);
    }
    Eigen::Matrix<double, 15, 15> L = llt.matrixL();
    d.sqrt_info = Eigen::Matrix<double, 15, 15>::Identity();
    L.triangularView<Eigen::Lower>().solveInPlace(d.sqrt_info);
    return d;
  }
};

/**
 * @brief Ceres autodiff IMU factor between two consecutive keyframes.
 *
 * Parameter blocks (sizes 7, 9, 7, 9):
 *  - pose_i / pose_j: colmap rig_from_world [qx qy qz qw tx ty tz] where the
 *    rotation is R_GtoI (Hamilton, world->IMU) and t = -R_GtoI * p_IinG
 *  - sb_i / sb_j: [v_IinG(3) bg(3) ba(3)] velocity and biases in the world /
 *    IMU frames
 *
 * The 15-dim residual [r_th; r_p; r_v; r_bg; r_ba] compares the relative
 * motion implied by the states with the preintegrated deltas (first-order
 * bias corrected) and is whitened by ImuFactorData::sqrt_info.
 */
struct ImuFactor {
  explicit ImuFactor(const ImuFactorData &data) : data_(data) {}

  template <typename T>
  bool operator()(const T *const pose_i, const T *const sb_i,
                  const T *const pose_j, const T *const sb_j,
                  T *residuals) const {
    // poses: colmap rig_from_world (rotation = R_GtoI, t = -R_GtoI p_IinG)
    const Eigen::Quaternion<T> q_i(pose_i[3], pose_i[0], pose_i[1], pose_i[2]);
    const Eigen::Quaternion<T> q_j(pose_j[3], pose_j[0], pose_j[1], pose_j[2]);
    const Eigen::Matrix<T, 3, 1> t_i(pose_i[4], pose_i[5], pose_i[6]);
    const Eigen::Matrix<T, 3, 1> t_j(pose_j[4], pose_j[5], pose_j[6]);
    const Eigen::Matrix<T, 3, 3> R_i = q_i.toRotationMatrix();
    const Eigen::Matrix<T, 3, 3> R_j = q_j.toRotationMatrix();
    const Eigen::Matrix<T, 3, 1> p_i = -R_i.transpose() * t_i;
    const Eigen::Matrix<T, 3, 1> p_j = -R_j.transpose() * t_j;

    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_i(sb_i);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg_i(sb_i + 3);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba_i(sb_i + 6);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> v_j(sb_j);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg_j(sb_j + 3);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba_j(sb_j + 6);

    // first-order correction of the integrated deltas for bias motion off
    // the linearization point
    const Eigen::Matrix<T, 3, 1> dbg = bg_i - data_.bg_lin.cast<T>();
    const Eigen::Matrix<T, 3, 1> dba = ba_i - data_.ba_lin.cast<T>();
    const Eigen::Matrix<T, 3, 3> dR_c =
        data_.dR.cast<T>() * ExpSO3(data_.J_th_bg.cast<T>() * dbg);
    const Eigen::Matrix<T, 3, 1> dv_c =
        data_.dv.cast<T>() + data_.J_v_bg.cast<T>() * dbg +
        data_.J_v_ba.cast<T>() * dba;
    const Eigen::Matrix<T, 3, 1> dp_c =
        data_.dp.cast<T>() + data_.J_p_bg.cast<T>() * dbg +
        data_.J_p_ba.cast<T>() * dba;

    const T dt = T(data_.dt);
    const Eigen::Matrix<T, 3, 1> g = data_.gravity.cast<T>();

    Eigen::Matrix<T, 15, 1> r;
    // rotation: dR should equal R_GtoI_i * R_GtoI_j^T
    r.template block<3, 1>(0, 0) =
        LogSO3(dR_c.transpose() * (R_i * R_j.transpose()));
    // position: p_j = p_i + v_i dt - 0.5 g dt^2 + R_ItoG_i dp
    r.template block<3, 1>(3, 0) =
        R_i * (p_j - p_i - v_i * dt + T(0.5) * g * dt * dt) - dp_c;
    // velocity: v_j = v_i - g dt + R_ItoG_i dv
    r.template block<3, 1>(6, 0) = R_i * (v_j - v_i + g * dt) - dv_c;
    // bias random walks
    r.template block<3, 1>(9, 0) = bg_j - bg_i;
    r.template block<3, 1>(12, 0) = ba_j - ba_i;

    Eigen::Map<Eigen::Matrix<T, 15, 1>> r_out(residuals);
    r_out = data_.sqrt_info.cast<T>() * r;
    return true;
  }

private:
  ImuFactorData data_;
};

} // namespace ov_srvins

#endif // OV_SRVINS_IMUFACTOR_H
