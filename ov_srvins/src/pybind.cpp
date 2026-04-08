#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>

#include "core/SqrtEstimator.h"
#include "core/Frontend.h"
#include "state/Propagator.h"
#include "state/State.h"
#include "utils/sensor_data.h"
#include "core/VioManagerOptions.h"

namespace py = pybind11;
using namespace ov_srvins;
using namespace ov_core;

PYBIND11_MODULE(ov_srvins_py, m) {
    m.doc() = "Sqrt-VINS Python Bindings";

    // Bind ov_core types
    py::class_<ImuData>(m, "ImuData")
        .def(py::init<>())
        .def_readwrite("timestamp", &ImuData::timestamp)
        .def_readwrite("wm", &ImuData::wm)
        .def_readwrite("am", &ImuData::am);

    py::class_<CameraData>(m, "CameraData")
        .def(py::init<>())
        .def_readwrite("timestamp", &CameraData::timestamp)
        .def_readwrite("sensor_ids", &CameraData::sensor_ids)
        // Images and masks are cv::Mat, which require special handling or a cv::Mat binder
        // For now, we expose them but they might need a custom binder for numpy conversion
        .def_readwrite("images", &CameraData::images)
        .def_readwrite("masks", &CameraData::masks);

    // Bind VioManagerOptions
    py::class_<VioManagerOptions>(m, "VioManagerOptions")
        .def(py::init<>())
        .def("print_and_load", &VioManagerOptions::print_and_load, py::arg("parser") = nullptr)
        .def_readwrite("try_zupt", &VioManagerOptions::try_zupt)
        .def_readwrite("zupt_max_velocity", &VioManagerOptions::zupt_max_velocity)
        .def_readwrite("num_pts", &VioManagerOptions::num_pts);

    // Bind NoiseManager
    py::class_<NoiseManager>(m, "NoiseManager")
        .def(py::init<>())
        .def_readwrite("sigma_w", &NoiseManager::sigma_w)
        .def_readwrite("sigma_wb", &NoiseManager::sigma_wb)
        .def_readwrite("sigma_a", &NoiseManager::sigma_a)
        .def_readwrite("sigma_ab", &NoiseManager::sigma_ab);

    // Bind State
    py::class_<State, std::shared_ptr<State>>(m, "State")
        .def_readonly("timestamp", &State::timestamp)
        .def("clear", &State::clear, py::arg("fully") = false);

    // Bind SqrtEstimator
    py::class_<SqrtEstimator, std::shared_ptr<SqrtEstimator>>(m, "SqrtEstimator")
        .def(py::init<VioManagerOptions&>())
        .def("feed_imu", &SqrtEstimator::feed_imu)
        .def("try_zupt", &SqrtEstimator::try_zupt)
        .def("propagate", &SqrtEstimator::propagate)
        .def("get_state", &SqrtEstimator::get_state)
        .def("get_propagator", &SqrtEstimator::get_propagator);

    // Bind Propagator
    py::class_<Propagator, std::shared_ptr<Propagator>>(m, "Propagator")
        .def(py::init<NoiseManager, DataType>())
        .def("feed_imu", &Propagator::feed_imu)
        .def("clean_old_imu_measurements", &Propagator::clean_old_imu_measurements)
        .def("propagate", &Propagator::propagate);

    // Bind Frontend
    py::class_<Frontend, std::shared_ptr<Frontend>>(m, "Frontend")
        .def(py::init<VioManagerOptions&, std::shared_ptr<State>>())
        .def("feed_camera", &Frontend::feed_camera)
        .def("get_historical_viz_image", &Frontend::get_historical_viz_image)
        .def("set_startup_time", &Frontend::set_startup_time);
}
