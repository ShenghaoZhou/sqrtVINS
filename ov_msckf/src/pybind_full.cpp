/*
 * OpenVINS: An Open Platform for Visual-Inertial Research
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2023 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2019 Kevin Eckenhoff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 */

/**
 * @brief Python bindings for the original OpenVINS (full-covariance EKF)
 * system.
 *
 * Kept in its own translation unit (with module-pinned include dirs) because
 * the sqrt and full-covariance modules expose colliding header paths. It is
 * registered into the same Python module by ov_srvins/src/pybind.cpp.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "utils/sensor_data.h"
#include "utils/yaml_parse.h"

namespace py = pybind11;

void bind_full_system(pybind11::module_ &m) {
  // Options for the full-covariance system
  py::class_<ov_msckf::VioManagerOptions>(m, "FullVioManagerOptions")
      .def(py::init<>())
      .def("print_and_load", [](ov_msckf::VioManagerOptions &self,
                                std::shared_ptr<ov_core::YamlParser> parser) {
        self.print_and_load(parser);
      }, py::arg("parser") = nullptr)
      .def_readwrite("cv_backend", &ov_msckf::VioManagerOptions::cv_backend)
      .def_readwrite("num_opencv_threads",
                     &ov_msckf::VioManagerOptions::num_opencv_threads)
      .def_readwrite("use_stereo", &ov_msckf::VioManagerOptions::use_stereo);

  // The full-covariance VioManager (self-contained: tracks, propagates and
  // updates internally, like the original OpenVINS)
  py::class_<ov_msckf::VioManager, std::shared_ptr<ov_msckf::VioManager>>(
      m, "FullVioManager")
      .def(py::init<ov_msckf::VioManagerOptions &>())
      .def("feed_measurement_imu", &ov_msckf::VioManager::feed_measurement_imu)
      .def("feed_measurement_camera",
           &ov_msckf::VioManager::feed_measurement_camera)
      .def("initialized", &ov_msckf::VioManager::initialized)
      .def("initialized_time", &ov_msckf::VioManager::initialized_time)
      .def("get_state", &ov_msckf::VioManager::get_state)
      .def("get_params", &ov_msckf::VioManager::get_params);

  // Full-covariance state (subset of accessors useful from Python)
  py::class_<ov_msckf::State, std::shared_ptr<ov_msckf::State>>(m, "FullState")
      .def("timestamp",
           [](const ov_msckf::State &s) { return s._timestamp; })
      .def("pos_inG",
           [](const ov_msckf::State &s) {
             return s._imu->pos().cast<double>();
           })
      .def("vel_inG",
           [](const ov_msckf::State &s) {
             return s._imu->vel().cast<double>();
           })
      .def("quat_GtoI",
           [](const ov_msckf::State &s) {
             // JPL q_GtoI stored as R_GtoI in the state; return q_GtoI
             Eigen::Matrix3d R_GtoI = s._imu->Rot().cast<double>();
             Eigen::Quaterniond q_ItoG(R_GtoI.transpose());
             Eigen::Vector4d q;
             // match OpenVINS JPL convention: [x,y,z,w]
             q << q_ItoG.x(), q_ItoG.y(), q_ItoG.z(), q_ItoG.w();
             return q;
           })
      .def("bias_gyro",
           [](const ov_msckf::State &s) {
             return s._imu->bias_g().cast<double>();
           })
      .def("bias_accel",
           [](const ov_msckf::State &s) {
             return s._imu->bias_a().cast<double>();
           });
}
