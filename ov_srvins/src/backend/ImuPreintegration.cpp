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

#include "backend/ImuPreintegration.h"

#include "utils/Helper.h"

namespace ov_srvins {

Eigen::Matrix3d JrSO3(const Eigen::Vector3d &w) {
  const double th = w.norm();
  Eigen::Matrix3d W;
  W << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
  if (th < 1e-10) {
    return Eigen::Matrix3d::Identity() - 0.5 * W;
  }
  const double a = (1.0 - std::cos(th)) / (th * th);
  const double b = (th - std::sin(th)) / (th * th * th);
  return Eigen::Matrix3d::Identity() - a * W + b * W * W;
}

ImuPreintegration::ImuPreintegration(double sigma_w, double sigma_a,
                                     double sigma_wb, double sigma_ab,
                                     const Vec3d &bg_lin, const Vec3d &ba_lin)
    : bg_lin(bg_lin), ba_lin(ba_lin),
      sigma_w_2_(sigma_w * sigma_w), sigma_a_2_(sigma_a * sigma_a),
      sigma_wb_2_(sigma_wb * sigma_wb), sigma_ab_2_(sigma_ab * sigma_ab) {}

void ImuPreintegration::reset() {
  dR.setIdentity();
  dv.setZero();
  dp.setZero();
  cov.setZero();
  jac.setIdentity();
  sum_dt = 0.0;
}

bool ImuPreintegration::integrate(
    const std::vector<ov_core::ImuData> &imu_data, double t0, double t1) {
  if (t1 <= t0)
    return false;
  const std::vector<ov_core::ImuData> data =
      select_imu_readings(imu_data, t0, t1, false);
  if (data.size() < 2)
    return false;

  reset();
  for (size_t i = 0; i + 1 < data.size(); i++) {
    const double dt = data.at(i + 1).timestamp - data.at(i).timestamp;
    step(dt, data.at(i).wm.cast<double>(), data.at(i).am.cast<double>(),
         data.at(i + 1).wm.cast<double>(), data.at(i + 1).am.cast<double>());
  }
  sum_dt = data.back().timestamp - data.front().timestamp;
  return true;
}

void ImuPreintegration::step(double dt, const Vec3d &w0m, const Vec3d &a0m,
                             const Vec3d &w1m, const Vec3d &a1m) {
  const Vec3d w = 0.5 * (w0m + w1m) - bg_lin;
  const Vec3d a0 = a0m - ba_lin;
  const Vec3d a1 = a1m - ba_lin;

  const Vec3d wdt = w * dt;
  const Mat3d A = ExpSO3(wdt);       // rotation increment this step
  const Mat3d Jr = JrSO3(wdt);       // right Jacobian at the increment
  const Mat3d R0 = dR;
  const Mat3d R1 = dR * A;
  const Vec3d u = 0.5 * (R0 * a0 + R1 * a1); // specific force in I_i frame

  // --- error-state transition (right-multiplicative rotation error) -------
  // theta' = A^T theta - Jr dt dbg
  // p' = p + v dt + 0.5 u dt^2
  // v' = v + u dt
  // biases: identity (random walk enters through the noise matrix)
  Mat15 F = Mat15::Identity();
  F.block<3, 3>(O_TH, O_TH) = A.transpose();
  F.block<3, 3>(O_TH, O_BG) = -Jr * dt;

  Mat3d a0_skew, a1_skew;
  a0_skew << 0, -a0(2), a0(1), a0(2), 0, -a0(0), -a0(1), a0(0), 0;
  a1_skew << 0, -a1(2), a1(1), a1(2), 0, -a1(0), -a1(1), a1(0), 0;
  const Mat3d d_du_dth = -0.5 * (R0 * a0_skew + R1 * a1_skew * A.transpose());
  const Mat3d d_du_dba = -0.5 * (R0 + R1);

  F.block<3, 3>(O_P, O_TH) = dt * dt * d_du_dth;      // 0.5 dt^2 * (-0.5 ...)
  F.block<3, 3>(O_P, O_TH) *= 0.5;
  F.block<3, 3>(O_P, O_V) = Mat3d::Identity() * dt;
  F.block<3, 3>(O_P, O_BA) = 0.5 * dt * dt * d_du_dba;

  F.block<3, 3>(O_V, O_TH) = dt * d_du_dth;
  F.block<3, 3>(O_V, O_BA) = dt * d_du_dba;

  // --- noise mapping, n = [n_g, n_a, n_bg, n_ba] ---------------------------
  Eigen::Matrix<double, 15, 12> G = Eigen::Matrix<double, 15, 12>::Zero();
  G.block<3, 3>(O_TH, 0) = -Jr * dt;
  G.block<3, 3>(O_P, 3) = 0.5 * dt * dt * d_du_dba;
  G.block<3, 3>(O_V, 3) = dt * d_du_dba;
  G.block<3, 3>(O_BG, 6) = Mat3d::Identity();
  G.block<3, 3>(O_BA, 9) = Mat3d::Identity();

  Eigen::Matrix<double, 12, 12> Qd = Eigen::Matrix<double, 12, 12>::Zero();
  Qd.block<3, 3>(0, 0) = (sigma_w_2_ / dt) * Mat3d::Identity();
  Qd.block<3, 3>(3, 3) = (sigma_a_2_ / dt) * Mat3d::Identity();
  Qd.block<3, 3>(6, 6) = (sigma_wb_2_ * dt) * Mat3d::Identity();
  Qd.block<3, 3>(9, 9) = (sigma_ab_2_ * dt) * Mat3d::Identity();

  cov = F * cov * F.transpose() + G * Qd * G.transpose();
  jac = F * jac;

  // --- mean propagation -----------------------------------------------------
  dp += dv * dt + 0.5 * u * dt * dt;
  dv += u * dt;
  dR = R1;
}

} // namespace ov_srvins
