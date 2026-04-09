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
#include "initializer/InertialInitializer.h"
#include "initializer/InertialInitializerOptions.h"
#include "feat/FeatureDatabase.h"
#include "track/TrackBase.h"
#include "utils/ndarray_converter.h"
#include "feat/Feature.h"
#include "types/IMU.h"

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
        .def_readwrite("images", &CameraData::images)
        .def_readwrite("masks", &CameraData::masks);

    // Bind YamlParser
    py::class_<YamlParser, std::shared_ptr<YamlParser>>(m, "YamlParser")
        .def(py::init<const std::string &, bool>(), py::arg("config_path"), py::arg("fail_if_not_found") = true);

    // Bind Option structures
    py::class_<StateOptions>(m, "StateOptions")
        .def(py::init<>())
        .def_readwrite("num_cameras", &StateOptions::num_cameras)
        .def_readwrite("max_clone_size", &StateOptions::max_clone_size);

    py::class_<UpdaterOptions>(m, "UpdaterOptions")
        .def(py::init<>())
        .def_readwrite("chi2_multipler", &UpdaterOptions::chi2_multipler)
        .def_readwrite("sigma_pix", &UpdaterOptions::sigma_pix);

    py::class_<FeatureInitializerOptions>(m, "FeatureInitializerOptions")
        .def(py::init<>());

    py::class_<InertialInitializerOptions>(m, "InertialInitializerOptions")
        .def(py::init<>())
        .def_readwrite("init_window_time", &InertialInitializerOptions::init_window_time)
        .def_readwrite("init_max_features", &InertialInitializerOptions::init_max_features)
        .def_readwrite("init_dyn_use", &InertialInitializerOptions::init_dyn_use);

    py::class_<VioManagerOptions>(m, "VioManagerOptions")
        .def(py::init<>())
        .def("print_and_load", &VioManagerOptions::print_and_load, py::arg("parser") = nullptr)
        .def_readwrite("state_options", &VioManagerOptions::state_options)
        .def_readwrite("init_options", &VioManagerOptions::init_options)
        .def_readwrite("imu_noises", &VioManagerOptions::imu_noises)
        .def_readwrite("msckf_options", &VioManagerOptions::msckf_options)
        .def_readwrite("slam_options", &VioManagerOptions::slam_options)
        .def_readwrite("featinit_options", &VioManagerOptions::featinit_options)
        .def_readwrite("try_zupt", &VioManagerOptions::try_zupt)
        .def_readwrite("zupt_max_velocity", &VioManagerOptions::zupt_max_velocity)
        .def_readwrite("num_pts", &VioManagerOptions::num_pts)
        .def_readwrite("use_mask", &VioManagerOptions::use_mask)
        .def_readwrite("num_opencv_threads", &VioManagerOptions::num_opencv_threads);

    py::class_<NoiseManager>(m, "NoiseManager")
        .def(py::init<>())
        .def_readwrite("sigma_w", &NoiseManager::sigma_w)
        .def_readwrite("sigma_wb", &NoiseManager::sigma_wb)
        .def_readwrite("sigma_a", &NoiseManager::sigma_a)
        .def_readwrite("sigma_ab", &NoiseManager::sigma_ab);

    // Bind IMU and State
    py::class_<ov_type::IMU, std::shared_ptr<ov_type::IMU>>(m, "IMU")
        .def(py::init<>())
        .def("pos", &ov_type::IMU::pos)
        .def("Rot", &ov_type::IMU::Rot)
        .def("vel", &ov_type::IMU::vel)
        .def("bias_g", &ov_type::IMU::bias_g)
        .def("bias_a", &ov_type::IMU::bias_a);

    py::class_<State, std::shared_ptr<State>>(m, "State")
        .def_readonly("timestamp", &State::timestamp)
        .def_readonly("imu", &State::imu)
        .def_readonly("is_initialized", &State::is_initialized)
        .def("clear", &State::clear, py::arg("fully") = false);

    py::class_<Feature, std::shared_ptr<Feature>>(m, "Feature")
        .def_readonly("featid", &Feature::featid);

    py::class_<FeatureDatabase, std::shared_ptr<FeatureDatabase>>(m, "FeatureDatabase")
        .def(py::init<>());

    // Bind TrackBase
    py::class_<TrackBase, std::shared_ptr<TrackBase>>(m, "TrackBase")
        .def("get_feature_database", &TrackBase::get_feature_database)
        .def("set_num_features", &TrackBase::set_num_features);

    // Bind SqrtEstimator
    py::class_<SqrtEstimator, std::shared_ptr<SqrtEstimator>>(m, "SqrtEstimator")
        .def(py::init<VioManagerOptions&>())
        .def("feed_imu", &SqrtEstimator::feed_imu)
        .def("try_zupt", &SqrtEstimator::try_zupt)
        .def("propagate", &SqrtEstimator::propagate)
        .def("update", &SqrtEstimator::update)
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
        .def("get_trackFEATS", &Frontend::get_trackFEATS)
        .def("process_measurements_rules", [](Frontend &self, double timestamp, const std::vector<int> &sensor_ids) {
            std::vector<std::shared_ptr<ov_core::Feature>> featsup_MSCKF;
            std::vector<std::shared_ptr<ov_core::Feature>> feats_slam_UPDATE;
            std::vector<std::shared_ptr<ov_core::Feature>> feats_slam_DELAYED;
            self.process_measurements_rules(timestamp, sensor_ids, featsup_MSCKF, feats_slam_UPDATE, feats_slam_DELAYED);
            return std::make_tuple(featsup_MSCKF, feats_slam_UPDATE, feats_slam_DELAYED);
        })
        .def("set_startup_time", &Frontend::set_startup_time);

    // Bind InertialInitializer
    py::class_<InertialInitializer, std::shared_ptr<InertialInitializer>>(m, "InertialInitializer")
        .def(py::init<const InertialInitializerOptions &, std::shared_ptr<ov_core::FeatureDatabase>, 
                      std::shared_ptr<ov_srvins::Propagator>, const UpdaterOptions &, const UpdaterOptions &, 
                      const ov_core::FeatureInitializerOptions &>())
        .def("initialize", &InertialInitializer::initialize);
}
