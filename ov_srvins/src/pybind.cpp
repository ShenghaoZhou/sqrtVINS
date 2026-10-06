#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/chrono.h>
#include <pybind11/numpy.h>

#include "core/SqrtEstimator.h"
#include "core/Frontend.h"
#include "core/InitRunner.h"
#include "core/Pipeline.h"
#include "core/System.h"
#ifdef SQRTVINS_BACKEND
#include "backend/BackendSystem.h"
#endif
#include "state/Propagator.h"
#include "state/State.h"
#include "utils/sensor_data.h"
#include "core/VinsOptions.h"
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

// Defined in ov_msckf/src/pybind_full.cpp (full-covariance system bindings,
// compiled in a separate TU with its module's include dirs pinned)
void bind_full_system(pybind11::module_ &m);

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
        .def_readwrite("init_dyn_use", &InertialInitializerOptions::init_dyn_use)
        .def_readwrite("init_async", &InertialInitializerOptions::init_async);

#ifdef SQRTVINS_BACKEND
    py::class_<BackendOptions>(m, "BackendOptions")
        .def(py::init<>())
        .def_readwrite("enabled", &BackendOptions::enabled)
        .def_readwrite("keyframe_stride", &BackendOptions::keyframe_stride)
        .def_readwrite("min_track_length", &BackendOptions::min_track_length)
        .def_readwrite("max_triang_error_px",
                       &BackendOptions::max_triang_error_px)
        .def_readwrite("max_reproj_error_px",
                       &BackendOptions::max_reproj_error_px)
        .def_readwrite("refine_after_pruning",
                       &BackendOptions::refine_after_pruning)
        .def_readwrite("use_imu_factors", &BackendOptions::use_imu_factors)
        .def_readwrite("online_enabled", &BackendOptions::online_enabled)
        .def_readwrite("window_size", &BackendOptions::window_size)
        .def_readwrite("window_solve_stride",
                       &BackendOptions::window_solve_stride)
        .def_readwrite("window_max_iterations",
                       &BackendOptions::window_max_iterations)
        .def_readwrite("window_max_solver_time",
                       &BackendOptions::window_max_solver_time)
        .def_readwrite("max_num_iterations",
                       &BackendOptions::max_num_iterations)
        .def_readwrite("num_threads", &BackendOptions::num_threads)
        .def_readwrite("loss_scale", &BackendOptions::loss_scale)
        .def_readwrite("print_summary", &BackendOptions::print_summary);

#endif
    py::class_<VinsOptions>(m, "VinsOptions")
        .def(py::init<>())
        .def("print_and_load", &VinsOptions::print_and_load, py::arg("parser") = nullptr)
        .def_readwrite("state_options", &VinsOptions::state_options)
        .def_readwrite("init_options", &VinsOptions::init_options)
#ifdef SQRTVINS_BACKEND
        .def_readwrite("backend_options", &VinsOptions::backend_options)
#endif
        .def_readwrite("imu_noises", &VinsOptions::imu_noises)
        .def_readwrite("msckf_options", &VinsOptions::msckf_options)
        .def_readwrite("slam_options", &VinsOptions::slam_options)
        .def_readwrite("featinit_options", &VinsOptions::featinit_options)
        .def_readwrite("try_zupt", &VinsOptions::try_zupt)
        .def_readwrite("zupt_max_velocity", &VinsOptions::zupt_max_velocity)
        .def_readwrite("num_pts", &VinsOptions::num_pts)
        .def_readwrite("use_mask", &VinsOptions::use_mask)
        .def_readwrite("num_opencv_threads", &VinsOptions::num_opencv_threads)
        .def_readwrite("cv_backend", &VinsOptions::cv_backend);

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
        .def("cam_imu_timeoffset",
             [](const State &s) { return (double)s.calib_dt_CAMtoIMU->value()(0); })
        .def_property_readonly("is_initialized",
                               [](const State &s) { return s.is_initialized.load(); })
        .def("margtimestep", &State::margtimestep)
        .def("num_clones", [](const State &s) { return (int)s.clones_IMU.size(); })
        .def("clear", &State::clear, py::arg("fully") = false);

    py::class_<Feature, std::shared_ptr<Feature>>(m, "Feature")
        .def_readonly("featid", &Feature::featid);

    py::class_<FeatureDatabase, std::shared_ptr<FeatureDatabase>>(m, "FeatureDatabase")
        .def(py::init<>())
        .def("cleanup", &FeatureDatabase::cleanup)
        .def("cleanup_measurements", &FeatureDatabase::cleanup_measurements);

    // Bind TrackBase
    py::class_<TrackBase, std::shared_ptr<TrackBase>>(m, "TrackBase")
        .def("get_feature_database", &TrackBase::get_feature_database)
        .def("set_num_features", &TrackBase::set_num_features);

    // Bind SqrtEstimator
    py::class_<SqrtEstimator, std::shared_ptr<SqrtEstimator>>(m, "SqrtEstimator")
        .def(py::init<const VinsOptions&>())
        .def("feed_imu", &SqrtEstimator::feed_imu)
        .def("feed_measurement_imu", &SqrtEstimator::feed_measurement_imu)
        .def("feed_imu_batch",
             [](SqrtEstimator &self,
                py::array_t<double, py::array::c_style | py::array::forcecast> timestamps,
                py::array_t<double, py::array::c_style | py::array::forcecast> wm,
                py::array_t<double, py::array::c_style | py::array::forcecast> am) {
               auto ts_buf = timestamps.unchecked<1>();
               auto wm_buf = wm.unchecked<2>();
               auto am_buf = am.unchecked<2>();
               if (ts_buf.shape(0) != wm_buf.shape(0) || ts_buf.shape(0) != am_buf.shape(0))
                 throw std::runtime_error("feed_imu_batch: timestamps/wm/am must have the same length");
               size_t n = (size_t)ts_buf.shape(0);
               std::vector<ov_core::ImuData> msgs(n);
               for (size_t i = 0; i < n; i++) {
                 msgs[i].timestamp = ts_buf(i);
                 for (int j = 0; j < 3; j++) {
                   msgs[i].wm(j) = wm_buf(i, j);
                   msgs[i].am(j) = am_buf(i, j);
                 }
               }
               self.feed_imu_batch(msgs);
             }, py::arg("timestamps"), py::arg("wm"), py::arg("am"))
        .def("try_zupt", &SqrtEstimator::try_zupt, py::arg("timestamp"))
        .def("notify_moved", &SqrtEstimator::notify_moved)
        .def("has_moved_since_zupt", &SqrtEstimator::has_moved_since_zupt)
        .def("propagate", &SqrtEstimator::propagate)
        .def("update", &SqrtEstimator::update)
        .def("process_frame",
             [](SqrtEstimator &self, Frontend &frontend, double timestamp,
                const std::vector<int> &sensor_ids) {
               // Canonical per-frame filter step (core/Pipeline.h): propagate,
               // select features against the post-propagation state, update,
               // then clean the tracker databases - all inside a single
               // Python->C++ call with no feature list round-trip.
               ov_core::CameraData message;
               message.timestamp = timestamp;
               message.sensor_ids = sensor_ids;
               return ov_srvins::process_frame(self, frontend, message);
             }, py::arg("frontend"), py::arg("timestamp"), py::arg("sensor_ids"))
        .def("set_zupt_database", &SqrtEstimator::set_zupt_database)
        .def("get_state", &SqrtEstimator::get_state)
        .def("get_propagator", &SqrtEstimator::get_propagator);

    // IMU batch feeding policy shared by the C++ and Python drivers
    // (core/Pipeline.h): end index of the IMU batch for a camera frame.
    m.def("imu_batch_end",
          [](py::array_t<double, py::array::c_style | py::array::forcecast> imu_times,
             size_t start, double cam_time_imu) {
            auto ts = imu_times.unchecked<1>();
            size_t n = (size_t)ts.shape(0);
            size_t k = start;
            while (k < n && ts(k) <= cam_time_imu)
              k++;
            if (k < n)
              k++;
            return k;
          }, py::arg("imu_times"), py::arg("start"), py::arg("cam_time_imu"));

    // Bind Propagator
    py::class_<Propagator, std::shared_ptr<Propagator>>(m, "Propagator")
        .def(py::init<NoiseManager, DataType>())
        .def("feed_imu", &Propagator::feed_imu, py::arg("message"),
             py::arg("oldest_time") = -1)
        .def("clean_old_imu_measurements", &Propagator::clean_old_imu_measurements)
        .def("propagate", &Propagator::propagate);

    // Bind Frontend
    py::class_<Frontend, std::shared_ptr<Frontend>>(m, "Frontend")
        .def(py::init<const VinsOptions&, std::shared_ptr<State>>())
        .def("feed_camera", &Frontend::feed_camera)
        .def("get_historical_viz_image", &Frontend::get_historical_viz_image)
        .def("get_trackFEATS", &Frontend::get_trackFEATS)
        .def("process_measurements_rules", [](Frontend &self, std::shared_ptr<State> state, double timestamp, const std::vector<int> &sensor_ids) {
            std::vector<std::shared_ptr<ov_core::Feature>> featsup_MSCKF;
            std::vector<std::shared_ptr<ov_core::Feature>> feats_slam;
            self.process_measurements_rules(state, timestamp, sensor_ids, featsup_MSCKF, feats_slam);
            return std::make_tuple(featsup_MSCKF, feats_slam);
        })
        .def("set_startup_time", &Frontend::set_startup_time);

    // Fully wired pipeline bundle (canonical construction sequence)
    py::class_<System>(m, "System")
        .def_readonly("estimator", &System::estimator)
        .def_readonly("frontend", &System::frontend)
        .def_readonly("initializer", &System::initializer)
        .def_readonly("init_runner", &System::init_runner)
#ifdef SQRTVINS_BACKEND
        .def_readonly("backend", &System::backend)
#endif
        .def_static("create", &System::create, py::arg("params"));

#ifdef SQRTVINS_BACKEND
    // Bundle-adjustment backend (Phase 1: offline BA recorder + solver)
    py::class_<BackendSummary>(m, "BackendSummary")
        .def_readonly("solved", &BackendSummary::solved)
        .def_readonly("num_keyframes", &BackendSummary::num_keyframes)
        .def_readonly("num_images", &BackendSummary::num_images)
        .def_readonly("num_points", &BackendSummary::num_points)
        .def_readonly("num_observations", &BackendSummary::num_observations)
        .def_readonly("mean_reproj_error_before",
                      &BackendSummary::mean_reproj_error_before)
        .def_readonly("mean_reproj_error_after",
                      &BackendSummary::mean_reproj_error_after)
        .def_readonly("mean_reproj_error_final",
                      &BackendSummary::mean_reproj_error_final)
        .def_readonly("num_pruned_points", &BackendSummary::num_pruned_points);

    py::class_<BackendSystem, std::shared_ptr<BackendSystem>>(m,
                                                              "BackendSystem")
        .def("feed_imu", &BackendSystem::feed_imu)
        .def("record_observations", &BackendSystem::record_observations)
        .def("record_pose", &BackendSystem::record_pose)
        .def("num_keyframes", &BackendSystem::num_keyframes)
        .def("online_enabled", &BackendSystem::online_enabled)
        .def("get_refined_poses", &BackendSystem::get_refined_poses)
        .def("export_online_trajectory",
             &BackendSystem::export_online_trajectory)
        .def("run_offline_ba", &BackendSystem::run_offline_ba);
#endif

    // Bind InertialInitializer
    py::class_<InertialInitializer, std::shared_ptr<InertialInitializer>>(m, "InertialInitializer")
        .def(py::init<const InertialInitializerOptions &, std::shared_ptr<ov_core::FeatureDatabase>,
                      std::shared_ptr<ov_srvins::Propagator>, const UpdaterOptions &, const UpdaterOptions &,
                      const ov_core::FeatureInitializerOptions &>())
        .def("initialize", &InertialInitializer::initialize);

    // Bind InitRunner (drives init per frame; sync by default, async shadow
    // solve when init_async is enabled)
    py::class_<InitRunner, std::shared_ptr<InitRunner>>(m, "InitRunner")
        .def(py::init<const VinsOptions &, std::shared_ptr<SqrtEstimator>,
                      std::shared_ptr<Frontend>,
                      std::shared_ptr<InertialInitializer>>())
        .def("try_initialize", &InitRunner::try_initialize,
             py::arg("cam_time"), py::arg("wait_for_jerk"));

    // Bind the original OpenVINS (full-covariance) system
    bind_full_system(m);
}
