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

/**
 * @brief Tests for the Phase-2 IMU backend pieces.
 *
 *  1. SO(3) Exp/Log round trips (incl. small-angle branches)
 *  2. Preintegration accuracy vs fine-step ground-truth integration
 *  3. First-order bias-Jacobian correction vs re-integration
 *  4. Synthetic visual-inertial BA: stereo rig + IMU factors injected into
 *     the colmap problem must recover ground truth from perturbed init
 */

#include <cstdio>
#include <random>
#include <vector>

#include "backend/BackendSystem.h"
#include "backend/ImuFactor.h"
#include "backend/ImuPreintegration.h"

#include "colmap/scene/reconstruction.h"

using namespace ov_srvins;

#define CHECK_TRUE(cond, msg)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("[FAIL] %s:%d: %s\n", __FILE__, __LINE__, msg);                   \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

namespace {

/// Smooth analytic ground-truth motion shared by the tests
struct TrueMotion {
  const Eigen::Vector3d g{0, 0, 9.81};

  // position and its derivatives (world frame)
  Eigen::Vector3d p(double t) const {
    return {0.5 * t + 0.2 * std::sin(1.0 * t), 0.3 * std::sin(0.8 * t),
            1.0 + 0.2 * std::sin(0.5 * t)};
  }
  Eigen::Vector3d v(double t) const {
    return {0.5 + 0.2 * std::cos(1.0 * t), 0.24 * std::cos(0.8 * t),
            0.1 * std::cos(0.5 * t)};
  }
  Eigen::Vector3d a(double t) const {
    return {-0.2 * std::sin(1.0 * t), -0.192 * std::sin(0.8 * t),
            -0.05 * std::sin(0.5 * t)};
  }

  // orientation R_ItoG from smooth roll/pitch/yaw schedules
  Eigen::Matrix3d C(double t) const {
    const double roll = 0.05 * std::sin(0.9 * t);
    const double pitch = 0.10 * std::sin(0.7 * t);
    const double yaw = 0.20 * std::sin(0.4 * t);
    return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
           Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY())
               .toRotationMatrix() *
           Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
  }

  // body angular velocity via central differencing of C (accurate to ~1e-8)
  Eigen::Vector3d w(double t) const {
    const double h = 1e-5;
    const Eigen::Matrix3d dC = (C(t + h) - C(t - h)) / (2 * h);
    const Eigen::Matrix3d W = C(t).transpose() * dC;
    return {W(2, 1), W(0, 2), W(1, 0)};
  }

  // true specific force in the body frame
  Eigen::Vector3d acc_body(double t) const {
    return C(t).transpose() * (a(t) + g);
  }
};

std::vector<ov_core::ImuData>
make_imu(const TrueMotion &motion, double t0, double t1, double rate,
         const Eigen::Vector3d &bg, const Eigen::Vector3d &ba) {
  std::vector<ov_core::ImuData> data;
  const double dt = 1.0 / rate;
  // pad the interval so boundary interpolation has bracketing samples
  for (double t = t0 - dt; t <= t1 + dt + 1e-9; t += dt) {
    ov_core::ImuData msg;
    msg.timestamp = t;
    msg.wm = (motion.w(t) + bg).cast<DataType>();
    msg.am = (motion.acc_body(t) + ba).cast<DataType>();
    data.push_back(msg);
  }
  return data;
}

const NoiseManager kNoises; // defaults match the EuRoC config

} // namespace

static int test_exp_log() {
  std::mt19937 gen(5);
  std::normal_distribution<double> nd(0.0, 1.0);
  for (int trial = 0; trial < 200; trial++) {
    Eigen::Vector3d w(nd(gen), nd(gen), nd(gen));
    if (trial < 100) {
      // keep |w| < pi so Log returns the same branch (log is 2pi-periodic)
      w *= 3.0 / w.norm();
    } else {
      w *= 1e-10; // exercise the small-angle branch
    }
    const Eigen::Matrix3d R = ExpSO3(w);
    const Eigen::Vector3d w_back = LogSO3(R);
    CHECK_TRUE((w_back - w).norm() < 1e-10, "Exp/Log round trip mismatch");
    CHECK_TRUE(std::abs(R.determinant() - 1.0) < 1e-10, "Exp not orthonormal");
  }
  printf("[PASS] SO(3) Exp/Log\n");
  return EXIT_SUCCESS;
}

static int test_preintegration_accuracy() {
  const TrueMotion motion;
  const Eigen::Vector3d bg_true(0.01, -0.008, 0.005);
  const Eigen::Vector3d ba_true(0.15, -0.10, 0.20);
  const double t0 = 0.3, t1 = 1.3;

  const auto imu = make_imu(motion, t0, t1, 200.0, bg_true, ba_true);
  ImuPreintegration pre(kNoises.sigma_w, kNoises.sigma_a, kNoises.sigma_wb,
                        kNoises.sigma_ab, bg_true, ba_true);
  CHECK_TRUE(pre.integrate(imu, t0, t1), "preintegration failed");

  // ground-truth relative motion (analytic states, gravity subtracted)
  const double dt = t1 - t0;
  const Eigen::Matrix3d dR_gt = motion.C(t0).transpose() * motion.C(t1);
  const Eigen::Vector3d dv_gt =
      motion.C(t0).transpose() * (motion.v(t1) - motion.v(t0) + motion.g * dt);
  const Eigen::Vector3d dp_gt =
      motion.C(t0).transpose() *
      (motion.p(t1) - motion.p(t0) - motion.v(t0) * dt +
       0.5 * motion.g * dt * dt);

  const double err_R = LogSO3(pre.dR.transpose() * dR_gt).norm();
  const double err_v = (pre.dv - dv_gt).norm();
  const double err_p = (pre.dp - dp_gt).norm();
  printf("[INFO] preintegration errors: rot=%.2e rad, vel=%.2e m/s, pos=%.2e "
         "m (dt=%.2fs)\n",
         err_R, err_v, err_p, dt);
  CHECK_TRUE(err_R < 5e-4, "rotation delta inaccurate");
  CHECK_TRUE(err_v < 5e-3, "velocity delta inaccurate");
  CHECK_TRUE(err_p < 5e-3, "position delta inaccurate");
  CHECK_TRUE(std::abs(pre.sum_dt - dt) < 1e-9, "integration time mismatch");

  // covariance sanity: positive definite and plausible scale
  Eigen::LLT<Eigen::Matrix<double, 15, 15>> llt(pre.cov);
  CHECK_TRUE(llt.info() == Eigen::Success, "covariance not positive definite");
  CHECK_TRUE(pre.cov(ImuPreintegration::O_P, ImuPreintegration::O_P) > 1e-12,
             "position variance implausibly small");
  CHECK_TRUE(pre.cov(ImuPreintegration::O_P, ImuPreintegration::O_P) < 1e-2,
             "position variance implausibly large");
  printf("[PASS] preintegration accuracy\n");
  return EXIT_SUCCESS;
}

static int test_bias_jacobians() {
  const TrueMotion motion;
  const Eigen::Vector3d bg_true(0.01, -0.008, 0.005);
  const Eigen::Vector3d ba_true(0.15, -0.10, 0.20);
  const double t0 = 0.3, t1 = 1.3;

  const auto imu = make_imu(motion, t0, t1, 200.0, bg_true, ba_true);

  // reference: integrated at the true biases
  ImuPreintegration pre_ref(kNoises.sigma_w, kNoises.sigma_a, kNoises.sigma_wb,
                            kNoises.sigma_ab, bg_true, ba_true);
  CHECK_TRUE(pre_ref.integrate(imu, t0, t1), "reference integration failed");

  // perturbed linearization point: deltas are wrong, but the first-order
  // correction back to the true biases should recover them
  const Eigen::Vector3d dbg(-0.004, 0.003, -0.005);
  const Eigen::Vector3d dba(-0.03, 0.025, -0.04);
  ImuPreintegration pre(kNoises.sigma_w, kNoises.sigma_a, kNoises.sigma_wb,
                        kNoises.sigma_ab, bg_true + dbg, ba_true + dba);
  CHECK_TRUE(pre.integrate(imu, t0, t1), "perturbed integration failed");

  const Eigen::Matrix3d J_th_bg = pre.jac.block<3, 3>(
      ImuPreintegration::O_TH, ImuPreintegration::O_BG);
  const Eigen::Matrix3d J_v_bg = pre.jac.block<3, 3>(ImuPreintegration::O_V,
                                                     ImuPreintegration::O_BG);
  const Eigen::Matrix3d J_v_ba = pre.jac.block<3, 3>(ImuPreintegration::O_V,
                                                     ImuPreintegration::O_BA);
  const Eigen::Matrix3d J_p_bg = pre.jac.block<3, 3>(ImuPreintegration::O_P,
                                                     ImuPreintegration::O_BG);
  const Eigen::Matrix3d J_p_ba = pre.jac.block<3, 3>(ImuPreintegration::O_P,
                                                     ImuPreintegration::O_BA);

  // correction uses delta = b_true - b_lin
  const Eigen::Vector3d corr_bg = -dbg, corr_ba = -dba;
  const Eigen::Matrix3d dR_corr = pre.dR * ExpSO3(J_th_bg * corr_bg);
  const Eigen::Vector3d dv_corr =
      pre.dv + J_v_bg * corr_bg + J_v_ba * corr_ba;
  const Eigen::Vector3d dp_corr =
      pre.dp + J_p_bg * corr_bg + J_p_ba * corr_ba;

  const double err_R = LogSO3(dR_corr.transpose() * pre_ref.dR).norm();
  const double err_v = (dv_corr - pre_ref.dv).norm();
  const double err_p = (dp_corr - pre_ref.dp).norm();
  printf("[INFO] bias-correction residuals: rot=%.2e rad, vel=%.2e m/s, "
         "pos=%.2e m\n",
         err_R, err_v, err_p);
  CHECK_TRUE(err_R < 1e-3, "first-order rotation correction off");
  CHECK_TRUE(err_v < 1e-2, "first-order velocity correction off");
  CHECK_TRUE(err_p < 1e-2, "first-order position correction off");
  printf("[PASS] bias Jacobian correction\n");
  return EXIT_SUCCESS;
}

/// Synthetic visual-inertial BA: perturbed poses/velocities/biases/points,
/// noise-free IMU, 0.3 px vision noise -> must re-converge to ground truth
static int test_synthetic_vio_ba() {
  constexpr size_t kNumFrames = 30;
  constexpr double kDt = 0.1; // 10 Hz keyframes
  constexpr int kWidth = 640, kHeight = 480;
  const TrueMotion motion;
  const Eigen::Vector3d bg_true(0.01, -0.008, 0.005);
  const Eigen::Vector3d ba_true(0.15, -0.10, 0.20);

  std::mt19937 gen(99);
  std::normal_distribution<double> noise(0.0, 1.0);
  std::uniform_real_distribution<double> ud(-1.0, 1.0);

  // --- rig: IMU reference, cam0 identity, cam1 on a 0.1 m baseline --------
  const colmap::sensor_t imu_sensor(colmap::SensorType::IMU, 1);
  const colmap::sensor_t cam0_sensor(colmap::SensorType::CAMERA, 0);
  const colmap::sensor_t cam1_sensor(colmap::SensorType::CAMERA, 1);
  const colmap::Rigid3d cam1_from_rig(Eigen::Quaterniond::Identity(),
                                      Eigen::Vector3d(0.1, 0, 0));

  colmap::Reconstruction recon;
  colmap::Camera camera = colmap::Camera::CreateFromModelId(
      0, colmap::CameraModelId::kOpenCV, 400, kWidth, kHeight);
  camera.params = {400, 405, 320, 240, 0, 0, 0, 0};
  recon.AddCamera(camera);
  camera.camera_id = 1;
  recon.AddCamera(camera);

  colmap::Rig rig;
  rig.SetRigId(1);
  rig.AddRefSensor(imu_sensor);
  rig.AddSensor(cam0_sensor, colmap::Rigid3d());
  rig.AddSensor(cam1_sensor, cam1_from_rig);
  recon.AddRig(rig);

  // --- ground-truth states -------------------------------------------------
  std::vector<colmap::Rigid3d> gt_rig_from_world;
  std::vector<Eigen::Vector3d> gt_vel;
  std::vector<double> timestamps;
  for (size_t i = 0; i < kNumFrames; i++) {
    const double t = 0.05 + kDt * i;
    timestamps.push_back(t);
    const Eigen::Matrix3d C = motion.C(t);
    gt_rig_from_world.emplace_back(Eigen::Quaterniond(C.transpose()),
                                   -C.transpose() * motion.p(t));
    gt_vel.push_back(motion.v(t));
  }

  // landmarks ahead of the trajectory (camera looks along world +z)
  std::vector<Eigen::Vector3d> gt_points;
  for (size_t j = 0; j < 250; j++) {
    gt_points.emplace_back(1.5 + 1.5 * ud(gen), 1.8 * ud(gen),
                           4.0 + 3.0 * (j % 9) / 9.0 + 0.5 * ud(gen));
  }

  // --- keyframes with perturbed init (also drives ImuConstraints) ----------
  std::vector<BackendSystem::Keyframe> keyframes(kNumFrames);
  std::vector<Eigen::Vector3d> init_pos(kNumFrames);
  for (size_t i = 0; i < kNumFrames; i++) {
    BackendSystem::Keyframe &kf = keyframes[i];
    kf.timestamp = timestamps[i];
    kf.has_pose = true;
    colmap::Rigid3d init = gt_rig_from_world[i];
    if (i > 0) { // frame 1 is the constant gauge anchor: keep it exact
      const Eigen::Quaterniond q_noise(
          1.0, 0.005 * noise(gen), 0.005 * noise(gen), 0.005 * noise(gen));
      init.rotation() = (q_noise.normalized() * init.rotation()).normalized();
      init.translation() +=
          0.05 * Eigen::Vector3d(noise(gen), noise(gen), noise(gen));
    }
    const Eigen::Matrix3d R_init = init.rotation().toRotationMatrix();
    kf.R_GtoI = R_init.cast<DataType>();
    kf.p_IinG = (-R_init.transpose() * init.translation()).cast<DataType>();
    init_pos[i] = kf.p_IinG.cast<double>();
    kf.v_IinG = (gt_vel[i] + 0.05 * Eigen::Vector3d(noise(gen), noise(gen),
                                                    noise(gen)))
                    .cast<DataType>();
    kf.bg = (bg_true + 0.002 * Eigen::Vector3d(noise(gen), noise(gen),
                                               noise(gen)))
                .cast<DataType>();
    kf.ba = (ba_true + 0.02 * Eigen::Vector3d(noise(gen), noise(gen),
                                              noise(gen)))
                .cast<DataType>();
  }

  // --- frames, images, noisy observations ----------------------------------
  colmap::point3D_t next_point_id = 1;
  colmap::image_t next_image_id = 1;
  for (size_t i = 0; i < kNumFrames; i++) {
    const colmap::frame_t frame_id = i + 1;
    colmap::Frame frame;
    frame.SetFrameId(frame_id);
    frame.SetRigId(1);
    frame.SetRigFromWorld(colmap::Rigid3d(
        Eigen::Quaterniond(keyframes[i].R_GtoI.cast<double>()),
        -keyframes[i].R_GtoI.cast<double>() * keyframes[i].p_IinG.cast<double>()));

    std::vector<colmap::Image> images;
    std::vector<std::vector<std::pair<colmap::point3D_t, colmap::point2D_t>>>
        obs_per_image(2);
    for (size_t cam_id = 0; cam_id < 2; cam_id++) {
      const colmap::image_t image_id = next_image_id++;
      const colmap::Rigid3d cam_from_world =
          (cam_id == 0 ? colmap::Rigid3d() : cam1_from_rig) *
          gt_rig_from_world[i];
      std::vector<colmap::Point2D> points2D;
      for (size_t j = 0; j < gt_points.size(); j++) {
        const Eigen::Vector3d p_cam = cam_from_world * gt_points[j];
        if (p_cam.z() < 0.5)
          continue;
        const auto uv = recon.Camera(cam_id).ImgFromCam(p_cam);
        if (!uv.has_value() || (*uv)(0) < 0 || (*uv)(0) >= kWidth ||
            (*uv)(1) < 0 || (*uv)(1) >= kHeight)
          continue;
        const colmap::point3D_t point_id = next_point_id + j;
        obs_per_image[cam_id].emplace_back(point_id, points2D.size());
        colmap::Point2D p2d;
        p2d.xy = *uv + 0.3 * Eigen::Vector2d(noise(gen), noise(gen));
        points2D.push_back(p2d);
      }
      frame.AddDataId(colmap::data_t(
          colmap::sensor_t(colmap::SensorType::CAMERA, cam_id), image_id));
      colmap::Image image;
      image.SetImageId(image_id);
      image.SetCameraId(cam_id);
      image.SetFrameId(frame_id);
      image.SetPoints2D(points2D);
      images.push_back(std::move(image));
    }
    recon.AddFrame(frame);
    for (auto &image : images) {
      recon.AddImage(std::move(image));
    }

    for (size_t cam_id = 0; cam_id < 2; cam_id++) {
      const colmap::image_t image_id =
          next_image_id - 2 + static_cast<colmap::image_t>(cam_id);
      for (const auto &[point_id, idx] : obs_per_image[cam_id]) {
        if (!recon.ExistsPoint3D(point_id)) {
          colmap::Point3D point;
          point.xyz =
              gt_points[point_id - next_point_id] +
              0.1 * Eigen::Vector3d(noise(gen), noise(gen), noise(gen));
          recon.AddPoint3D(point_id, point);
        }
        recon.Image(image_id).SetPoint3DForPoint2D(idx, point_id);
        recon.Point3D(point_id).track.AddElement(image_id, idx);
      }
    }
  }
  next_point_id += gt_points.size();
  printf("[INFO] synthetic VIO map: %d images, %zu points, %d observations\n",
         static_cast<int>(recon.NumImages()), recon.NumPoints3D(),
         static_cast<int>(recon.ComputeNumObservations()));

  // --- IMU data + constraints ----------------------------------------------
  const auto imu_data = make_imu(motion, timestamps.front(),
                                 timestamps.back(), 200.0, bg_true, ba_true);
  auto imu = BackendSystem::build_imu_constraints(keyframes, imu_data, 0.0,
                                                  kNoises, 9.81, true);
  CHECK_TRUE(imu != nullptr, "failed to build IMU constraints");
  CHECK_TRUE(imu->factors.size() == kNumFrames - 1, "wrong factor count");

  BackendOptions opts;
  opts.loss_scale = 0;           // Gaussian observation noise
  opts.max_reproj_error_px = -1; // no pruning in the smoke test
  opts.print_summary = false;
  BackendSystem backend(opts, kNoises, 9.81);
  BackendSummary summary = backend.solve_and_export(recon, timestamps, "",
                                                    imu.get());
  printf("[INFO] VIO-BA reproj error: before=%.4f px, after=%.6f px\n",
         summary.mean_reproj_error_before, summary.mean_reproj_error_after);
  CHECK_TRUE(summary.solved, "VIO-BA solve failed");
  CHECK_TRUE(summary.mean_reproj_error_after < 0.5,
             "VIO-BA did not re-converge");

  // poses: max error vs ground truth
  double max_trans_err = 0, max_init_err = 0;
  for (size_t i = 0; i < kNumFrames; i++) {
    const auto &refined = recon.Frame(i + 1).RigFromWorld();
    const Eigen::Vector3d p_ref =
        -(refined.rotation().toRotationMatrix().transpose() *
          refined.translation());
    const Eigen::Vector3d p_gt = motion.p(timestamps[i]);
    max_trans_err = std::max(max_trans_err, (p_gt - p_ref).norm());
    max_init_err = std::max(max_init_err, (p_gt - init_pos[i]).norm());
  }
  printf("[INFO] translation error vs GT: init max=%.4f m, refined max=%.6f "
         "m\n",
         max_init_err, max_trans_err);
  CHECK_TRUE(max_trans_err < max_init_err, "BA did not improve the poses");
  CHECK_TRUE(max_trans_err < 0.01, "refined poses far from ground truth");

  // velocities and biases (blocks updated in place by the solver)
  double max_vel_err = 0, max_bg_err = 0, max_ba_err = 0;
  for (size_t i = 0; i < kNumFrames; i++) {
    const auto &sb = imu->sb[i];
    max_vel_err =
        std::max(max_vel_err,
                 (Eigen::Map<const Eigen::Vector3d>(sb.data()) - gt_vel[i])
                     .norm());
    max_bg_err = std::max(
        max_bg_err, (Eigen::Map<const Eigen::Vector3d>(sb.data() + 3) - bg_true)
                        .norm());
    max_ba_err = std::max(
        max_ba_err, (Eigen::Map<const Eigen::Vector3d>(sb.data() + 6) - ba_true)
                        .norm());
  }
  printf("[INFO] refined state errors: vel=%.2e m/s, bg=%.2e rad/s, ba=%.2e "
         "m/s^2\n",
         max_vel_err, max_bg_err, max_ba_err);
  CHECK_TRUE(max_vel_err < 0.02, "velocities far from ground truth");
  CHECK_TRUE(max_bg_err < 2e-3, "gyro bias far from ground truth");
  CHECK_TRUE(max_ba_err < 0.05, "accel bias far from ground truth");

  printf("[PASS] synthetic visual-inertial BA\n");
  return EXIT_SUCCESS;
}

int main() {
  if (test_exp_log() != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (test_preintegration_accuracy() != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (test_bias_jacobians() != EXIT_SUCCESS)
    return EXIT_FAILURE;
  if (test_synthetic_vio_ba() != EXIT_SUCCESS)
    return EXIT_FAILURE;
  printf("[PASS] all backend IMU tests\n");
  return EXIT_SUCCESS;
}
