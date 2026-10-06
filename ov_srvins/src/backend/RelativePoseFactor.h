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

#ifndef OV_SRVINS_RELATIVEPOSEFACTOR_H
#define OV_SRVINS_RELATIVEPOSEFACTOR_H

#include <Eigen/Eigen>

#include "backend/ImuPreintegration.h"

namespace ov_srvins {

/// Data for one relative-pose (loop-closure) constraint between keyframes i
/// and j: the measured relative pose of j expressed in i's IMU frame, plus
/// the 6x6 whitening (L^-1 of the measurement covariance LLT).
struct RelativePoseFactorData {
  Eigen::Matrix3d R_ItoJ = Eigen::Matrix3d::Identity(); // i-coords -> j-coords
  Eigen::Vector3d p_JinI = Eigen::Vector3d::Zero();     // origin of j in i
  Eigen::Matrix<double, 6, 6> sqrt_info =
      Eigen::Matrix<double, 6, 6>::Identity();

  /// Build from a relative-pose covariance [theta, p] (jittered LLT fallback
  /// to a weakly-informative factor, mirroring ImuFactorData)
  static RelativePoseFactorData
  FromMeasurement(const Eigen::Matrix3d &R_ItoJ, const Eigen::Vector3d &p_JinI,
                  const Eigen::Matrix<double, 6, 6> &cov) {
    RelativePoseFactorData d;
    d.R_ItoJ = R_ItoJ;
    d.p_JinI = p_JinI;
    Eigen::Matrix<double, 6, 6> C = cov;
    Eigen::LLT<Eigen::Matrix<double, 6, 6>> llt(C);
    if (llt.info() != Eigen::Success) {
      C.diagonal().array() += 1e-12;
      llt.compute(C);
      if (llt.info() != Eigen::Success) {
        d.sqrt_info.setIdentity();
        return d;
      }
    }
    d.sqrt_info = llt.matrixL();
    d.sqrt_info.triangularView<Eigen::Lower>().solveInPlace(d.sqrt_info);
    return d;
  }
};

/**
 * @brief Ceres functor for a relative-pose constraint between two colmap
 * frame pose blocks (rig_from_world, [qx qy qz qw tx ty tz], rotation =
 * R_GtoI, t = -R_GtoI * p_IinG).
 *
 * Predicted relative pose: R_pred = R_j * R_i^T, p_pred = R_j * (p_j - p_i).
 * Residual (whitened, order [theta, p], same convention as ImuFactor):
 *   r_th = Log(R_meas^T * R_pred)
 *   r_p  = p_pred - p_meas
 */
struct RelativePoseFactor {
  explicit RelativePoseFactor(const RelativePoseFactorData &data)
      : data_(data) {}

  template <typename T>
  bool operator()(const T *const pose_i, const T *const pose_j,
                  T *residuals) const {
    const Eigen::Quaternion<T> q_i(pose_i[3], pose_i[0], pose_i[1], pose_i[2]);
    const Eigen::Quaternion<T> q_j(pose_j[3], pose_j[0], pose_j[1], pose_j[2]);
    const Eigen::Matrix<T, 3, 3> R_i = q_i.toRotationMatrix();
    const Eigen::Matrix<T, 3, 3> R_j = q_j.toRotationMatrix();
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_i(pose_i + 4);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t_j(pose_j + 4);
    const Eigen::Matrix<T, 3, 1> p_i = -R_i.transpose() * t_i;
    const Eigen::Matrix<T, 3, 1> p_j = -R_j.transpose() * t_j;

    const Eigen::Matrix<T, 3, 3> R_pred = R_j * R_i.transpose();
    const Eigen::Matrix<T, 3, 1> p_pred = R_j * (p_j - p_i);

    Eigen::Matrix<T, 6, 1> r;
    r.template block<3, 1>(0, 0) =
        LogSO3(data_.R_ItoJ.cast<T>().transpose() * R_pred);
    r.template block<3, 1>(3, 0) = p_pred - data_.p_JinI.cast<T>();

    Eigen::Map<Eigen::Matrix<T, 6, 1>> r_out(residuals);
    r_out = data_.sqrt_info.cast<T>() * r;
    return true;
  }

private:
  const RelativePoseFactorData &data_;
};

} // namespace ov_srvins

#endif // OV_SRVINS_RELATIVEPOSEFACTOR_H
