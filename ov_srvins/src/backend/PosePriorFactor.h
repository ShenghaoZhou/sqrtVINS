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

#ifndef OV_SRVINS_POSEPRIORFACTOR_H
#define OV_SRVINS_POSEPRIORFACTOR_H

#include <Eigen/Eigen>

#include "backend/ImuPreintegration.h"

namespace ov_srvins {

/// Data for one absolute-pose prior on a keyframe: the prior pose in clone
/// convention (R_GtoI, p_IinG) plus the 6x6 whitening of [theta, p].
struct PosePriorFactorData {
  Eigen::Matrix3d R_GtoI = Eigen::Matrix3d::Identity(); // prior rotation
  Eigen::Vector3d p_IinG = Eigen::Vector3d::Zero();     // prior position
  Eigen::Matrix<double, 6, 6> sqrt_info =
      Eigen::Matrix<double, 6, 6>::Identity();

  /// Build from per-axis sigmas [rad] / [m] (isotropic per component group)
  static PosePriorFactorData FromPose(const Eigen::Matrix3d &R_GtoI,
                                      const Eigen::Vector3d &p_IinG,
                                      double sigma_ori, double sigma_pos) {
    PosePriorFactorData d;
    d.R_GtoI = R_GtoI;
    d.p_IinG = p_IinG;
    d.sqrt_info.setZero();
    d.sqrt_info.diagonal().head<3>().setConstant(1.0 / sigma_ori);
    d.sqrt_info.diagonal().tail<3>().setConstant(1.0 / sigma_pos);
    return d;
  }
};

/**
 * @brief Ceres functor for an absolute-pose prior on one colmap frame pose
 * block (rig_from_world, [qx qy qz qw tx ty tz], rotation = R_GtoI,
 * t = -R_GtoI * p_IinG).
 *
 * Residual (whitened, order [theta, p], same convention as ImuFactor and
 * RelativePoseFactor):
 *   r_th = Log(R_prior^T * R)
 *   r_p  = p - p_prior
 *
 * Used in the windowed solve as a soft gauge anchor on the oldest window
 * keyframe, replacing the hard constant pose so loop closures and past
 * refinements can move the window seam. The data is held by value so the
 * factor owns its whitening (unlike RelativePoseFactor, whose data outlives
 * the solve by construction).
 */
struct PosePriorFactor {
  explicit PosePriorFactor(const PosePriorFactorData &data) : data_(data) {}

  template <typename T>
  bool operator()(const T *const pose, T *residuals) const {
    const Eigen::Quaternion<T> q(pose[3], pose[0], pose[1], pose[2]);
    const Eigen::Matrix<T, 3, 3> R = q.toRotationMatrix();
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> t(pose + 4);
    const Eigen::Matrix<T, 3, 1> p = -R.transpose() * t;

    Eigen::Matrix<T, 6, 1> r;
    r.template block<3, 1>(0, 0) =
        LogSO3(data_.R_GtoI.cast<T>().transpose() * R);
    r.template block<3, 1>(3, 0) = p - data_.p_IinG.cast<T>();

    Eigen::Map<Eigen::Matrix<T, 6, 1>> r_out(residuals);
    r_out = data_.sqrt_info.cast<T>() * r;
    return true;
  }

private:
  const PosePriorFactorData data_;
};

/**
 * @brief Ceres functor for a diagonal prior on a keyframe's 9-dim
 * velocity/bias block [v_IinG(3) bg(3) ba(3)] (the ImuConstraints::sb
 * layout). Residual: r = W * (sb - sb_prior) with W diagonal, one sigma per
 * component group. Anchors the leading edge of the IMU factor chain in the
 * windowed solve.
 */
struct SbPriorFactor {
  /// sb_prior: [v bg ba]; sigmas: [sigma_vel, sigma_bg, sigma_ba]
  SbPriorFactor(const Eigen::Matrix<double, 9, 1> &sb_prior, double sigma_vel,
                double sigma_bg, double sigma_ba)
      : sb_prior_(sb_prior) {
    w_.head<3>().setConstant(1.0 / sigma_vel);
    w_.segment<3>(3).setConstant(1.0 / sigma_bg);
    w_.tail<3>().setConstant(1.0 / sigma_ba);
  }

  template <typename T>
  bool operator()(const T *const sb, T *residuals) const {
    Eigen::Map<Eigen::Matrix<T, 9, 1>> r_out(residuals);
    r_out = w_.cast<T>().cwiseProduct(
        Eigen::Map<const Eigen::Matrix<T, 9, 1>>(sb) - sb_prior_.cast<T>());
    return true;
  }

private:
  const Eigen::Matrix<double, 9, 1> sb_prior_;
  Eigen::Matrix<double, 9, 1> w_;
};

} // namespace ov_srvins

#endif // OV_SRVINS_POSEPRIORFACTOR_H
