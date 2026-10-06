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

#ifndef OV_SRVINS_IMUPREINTEGRATION_H
#define OV_SRVINS_IMUPREINTEGRATION_H

#include <vector>

#include <Eigen/Eigen>

#include "utils/sensor_data.h"

namespace ov_srvins {

/**
 * @brief On-manifold IMU preintegration between two keyframes (Forster et
 * al., "On-Manifold Preintegration for Real-Time Visual-Inertial Odometry").
 *
 * Integrates gyro/accel measurements relative to the first keyframe's IMU
 * frame, producing the deltas
 *   dR = R_GtoI_i * R_GtoI_j^T  (= R_ItoG_i^T * R_ItoG_j)
 *   dv, dp (expressed in the IMU_i frame)
 * along with their 15x15 covariance and the 15x15 error-state Jacobian wrt
 * the error state at t0 — its bias columns are the first-order correction
 * Jacobians used when the optimizer moves the biases off their
 * linearization values.
 *
 * Error-state ordering: [theta(0:3), p(3:6), v(6:9), bg(9:12), ba(12:15)].
 * Mid-point integration with the same dynamics/noise discretization as the
 * filter's Propagator, in double precision (the backend is offline).
 */
class ImuPreintegration {
public:
  enum : int {
    O_TH = 0,
    O_P = 3,
    O_V = 6,
    O_BG = 9,
    O_BA = 12,
    O_DIM = 15
  };

  using Mat15 = Eigen::Matrix<double, 15, 15>;
  using Vec3d = Eigen::Vector3d;
  using Mat3d = Eigen::Matrix3d;

  /**
   * @brief Construct with noise densities and linearization biases
   * @param sigma_w Gyro white noise density (rad/s/sqrt(Hz))
   * @param sigma_a Accel white noise density (m/s^2/sqrt(Hz))
   * @param sigma_wb Gyro bias random walk density
   * @param sigma_ab Accel bias random walk density
   * @param bg_lin Gyro bias at integration time
   * @param ba_lin Accel bias at integration time
   */
  ImuPreintegration(double sigma_w, double sigma_a, double sigma_wb,
                    double sigma_ab, const Vec3d &bg_lin, const Vec3d &ba_lin);

  /**
   * @brief Integrate the measurements covering (t0, t1]
   *
   * Boundary readings are linearly interpolated (same helper as the filter
   * propagator), so t0/t1 need not coincide with sample times.
   *
   * @param imu_data Full IMU history (sorted by timestamp)
   * @param t0 Start time (IMU clock)
   * @param t1 End time (IMU clock)
   * @return False if fewer than two usable readings bracket the interval
   */
  bool integrate(const std::vector<ov_core::ImuData> &imu_data, double t0,
                 double t1);

  /// Reset deltas/covariance/Jacobian to identity (keeps noises/biases)
  void reset();

  // --- integrated quantities (valid after integrate()) ---
  Mat3d dR = Mat3d::Identity();
  Vec3d dv = Vec3d::Zero();
  Vec3d dp = Vec3d::Zero();
  Mat15 cov = Mat15::Zero();
  /// Jacobian of the final error state wrt the error state at t0; bias
  /// columns serve as first-order bias-correction Jacobians
  Mat15 jac = Mat15::Identity();
  double sum_dt = 0.0;

  const Vec3d bg_lin;
  const Vec3d ba_lin;

private:
  /// One mid-point integration step (mean + covariance + Jacobian)
  void step(double dt, const Vec3d &w0m, const Vec3d &a0m, const Vec3d &w1m,
            const Vec3d &a1m);

  double sigma_w_2_, sigma_a_2_, sigma_wb_2_, sigma_ab_2_;
};

/// SO(3) exponential map (rotation matrix), templated for ceres::Jet
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 3>
ExpSO3(const Eigen::MatrixBase<Derived> &w_in) {
  using T = typename Derived::Scalar;
  const Eigen::Matrix<T, 3, 1> w = w_in;
  const T th = w.norm();
  Eigen::Matrix<T, 3, 3> W;
  W << T(0), -w(2), w(1), w(2), T(0), -w(0), -w(1), w(0), T(0);
  Eigen::Matrix<T, 3, 3> R = Eigen::Matrix<T, 3, 3>::Identity() + W;
  if (th < T(1e-8)) {
    return R; // first order is enough at this scale
  }
  const T a = sin(th) / th;
  const T b = (T(1) - cos(th)) / (th * th);
  return Eigen::Matrix<T, 3, 3>::Identity() + a * W + b * W * W;
}

/// SO(3) logarithm map, templated for ceres::Jet
template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, 3, 1>
LogSO3(const Eigen::MatrixBase<Derived> &R_in) {
  using T = typename Derived::Scalar;
  const Eigen::Matrix<T, 3, 3> R = R_in;
  Eigen::Matrix<T, 3, 1> v;
  v << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1);
  T cos_th = (R.trace() - T(1)) * T(0.5);
  // clamp into [-1, 1] without std::min/max (Jet-safe)
  cos_th = (cos_th > T(1)) ? T(1) : ((cos_th < T(-1)) ? T(-1) : cos_th);
  const T th = acos(cos_th);
  if (th < T(1e-8)) {
    return T(0.5) * v;
  }
  return (th / (T(2) * sin(th))) * v;
}

/// Right Jacobian of SO(3) (double; used for covariance/Jacobian propagation)
Eigen::Matrix3d JrSO3(const Eigen::Vector3d &w);

} // namespace ov_srvins

#endif // OV_SRVINS_IMUPREINTEGRATION_H
