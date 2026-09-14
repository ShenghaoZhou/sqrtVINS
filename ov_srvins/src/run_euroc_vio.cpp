/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Isolation runner: drives VioManager (the original ROS1-serial orchestration)
 * directly on a EuRoC ASL folder dataset, bypassing the Frontend/SqrtEstimator
 * path used by run_euroc. Used to compare the old vs new update orchestration.
 *
 * Usage: run_euroc_vio <dataset_path> <config.yaml> <bag_start> <out.txt>
 */
#include <Eigen/Eigen>
#include <algorithm>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/colors.h"
#include "utils/print.h"

using namespace ov_srvins;

struct CamEvent {
  double timestamp;
  std::string filename0, filename1;
};

static std::vector<std::vector<std::string>> load_csv(const std::string &path) {
  std::ifstream file(path);
  if (!file) {
    PRINT_ERROR(RED "Unable to open %s\n" RESET, path.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::vector<std::vector<std::string>> rows;
  std::string line;
  std::getline(file, line); // header
  while (std::getline(file, line)) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
      field.erase(std::remove_if(field.begin(), field.end(), ::isspace),
                  field.end());
      fields.push_back(field);
    }
    if (!fields.empty())
      rows.push_back(fields);
  }
  return rows;
}

int main(int argc, char **argv) {
  if (argc < 5) {
    PRINT_ERROR(RED "Usage: run_euroc_vio <dataset> <config> <bag_start> "
                    "<out.txt> [num_cameras]\n" RESET);
    return EXIT_FAILURE;
  }
  std::string dataset_path = argv[1];
  std::string config_path = argv[2];
  double bag_start = std::atof(argv[3]);
  std::string path_est = argv[4];
  int num_cameras = (argc > 5) ? std::atoi(argv[5]) : 2;

  auto parser = std::make_shared<ov_core::YamlParser>(config_path);
  std::string verbosity = "INFO";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  VioManagerOptions params;
  params.print_and_load(parser);
  params.num_opencv_threads = 0;
  params.use_multi_threading_pubs = 0;
  params.use_multi_threading_subs = false;
  // serial.launch overrides (ROS params override YAML in the original system)
  params.state_options.do_fej = true;
  params.init_options.init_window_time = 2.0;
  params.init_options.use_bg_estimator = false;
  params.init_options.init_dyn_num_pose = 5;
  params.init_options.init_dyn_mle_max_iter = 10;
  params.init_options.init_max_features = 200;
  params.init_options.init_max_slam = 50;
  params.init_options.init_max_feat = 50;

  if (!parser->successful()) {
    PRINT_ERROR(RED "[SERIAL]: unable to parse all parameters\n" RESET);
    std::exit(EXIT_FAILURE);
  }
  auto sys = std::make_shared<VioManager>(params);

  //===================================================================================
  // Load EuRoC ASL folder data
  //===================================================================================
  std::vector<ov_core::ImuData> imu_msgs;
  for (const auto &row : load_csv(dataset_path + "/mav0/imu0/data.csv")) {
    ov_core::ImuData msg;
    msg.timestamp = std::atoll(row[0].c_str()) * 1e-9;
    msg.wm << std::atof(row[1].c_str()), std::atof(row[2].c_str()),
        std::atof(row[3].c_str());
    msg.am << std::atof(row[4].c_str()), std::atof(row[5].c_str()),
        std::atof(row[6].c_str());
    imu_msgs.push_back(msg);
  }
  auto load_cam = [&](const std::string &cam)
      -> std::vector<std::pair<double, std::string>> {
    std::vector<std::pair<double, std::string>> out;
    for (const auto &row : load_csv(dataset_path + "/mav0/" + cam + "/data.csv"))
      out.push_back({std::atoll(row[0].c_str()) * 1e-9, row[1]});
    return out;
  };
  auto cam0 = load_cam("cam0");
  auto cam1 = load_cam("cam1");
  std::map<double, size_t> cam1_lookup;
  for (size_t j = 0; j < cam1.size(); j++)
    cam1_lookup[cam1[j].first] = j;

  double t_begin = std::min(imu_msgs.front().timestamp, cam0.front().first);
  t_begin = std::min(t_begin, cam1.front().first);
  double time_init = t_begin + bag_start;
  double max_camera_time = -1;

  // rosbag view starts at time_init: drop earlier messages
  imu_msgs.erase(std::remove_if(imu_msgs.begin(), imu_msgs.end(),
                                [&](const ov_core::ImuData &m) {
                                  return m.timestamp < time_init;
                                }),
                 imu_msgs.end());

  std::vector<CamEvent> cam_events;
  for (const auto &c0 : cam0) {
    if (c0.first < time_init)
      continue;
    if (num_cameras == 1) {
      cam_events.push_back({c0.first, c0.second, ""});
      max_camera_time = std::max(max_camera_time, c0.first);
      continue;
    }
    auto it = cam1_lookup.find(c0.first);
    if (it == cam1_lookup.end()) {
      for (const auto &c1 : cam1) {
        if (c1.first < c0.first)
          continue;
        if (std::abs(c1.first - c0.first) < 0.02)
          it = cam1_lookup.find(c1.first);
        break;
      }
    }
    if (it == cam1_lookup.end())
      continue;
    cam_events.push_back({c0.first, c0.second, cam1[it->second].second});
    max_camera_time = std::max(max_camera_time, c0.first);
  }
  PRINT_INFO("[SERIAL]: total of %zu imu and %zu camera messages!\n",
             imu_msgs.size(), cam_events.size());

  //===================================================================================
  // Output file (ov_eval Recorder format)
  //===================================================================================
  std::filesystem::path out_dir = std::filesystem::path(path_est).parent_path();
  if (!out_dir.empty())
    std::filesystem::create_directories(out_dir);
  std::ofstream outfile(path_est);
  if (outfile.fail()) {
    PRINT_ERROR(RED "Unable to open output file %s\n" RESET, path_est.c_str());
    return EXIT_FAILURE;
  }
  outfile << "# timestamp(s) tx ty tz qx qy qz qw Pr11 Pr12 Pr13 Pr22 Pr23 "
             "Pr33 Pt11 Pt12 Pt13 Pt22 Pt23 Pt33"
          << std::endl;

  //===================================================================================
  // Processing loop (mirrors the original ROS1Visualizer serial semantics)
  //===================================================================================
  std::deque<ov_core::CameraData> camera_queue;
  double last_vis_timestamp = -1;
  std::map<int, double> camera_last_timestamp;
  double time_delta = 1.0 / params.track_frequency;
  size_t m = 0;
  size_t n_pose_saved = 0;

  for (const auto &ev : cam_events) {
    while (m < imu_msgs.size() && imu_msgs.at(m).timestamp < ev.timestamp &&
           imu_msgs.at(m).timestamp <= max_camera_time) {
      const auto &msg = imu_msgs.at(m++);
      sys->feed_measurement_imu(msg);
      double timestamp_imu_inC =
          msg.timestamp - sys->get_state()->calib_dt_CAMtoIMU->value()(0);
      while (!camera_queue.empty() &&
             camera_queue.at(0).timestamp < timestamp_imu_inC) {
        sys->feed_measurement_camera(camera_queue.at(0));

        // visualize() -> publish_state(): save pose
        std::shared_ptr<State> state = sys->get_state();
        if (!(last_vis_timestamp == state->timestamp && sys->initialized())) {
          last_vis_timestamp = state->timestamp;
          if (sys->initialized()) {
            double timestamp_inI =
                state->timestamp + state->calib_dt_CAMtoIMU->value()(0);
            Eigen::MatrixXd cov = StateHelper::get_marginal_covariance(
                                      state, {state->imu->pose()->p(),
                                              state->imu->pose()->q()})
                                      .cast<double>();
            Eigen::Matrix3d cov_pos = cov.block<3, 3>(0, 0);
            Eigen::Matrix3d cov_rot = cov.block<3, 3>(3, 3);
            Eigen::Vector4d quat = state->imu->quat().cast<double>();
            Eigen::Vector3d pos = state->imu->pos().cast<double>();
            outfile.precision(5);
            outfile.setf(std::ios::fixed, std::ios::floatfield);
            outfile << timestamp_inI << " ";
            outfile.precision(6);
            outfile << pos(0) << " " << pos(1) << " " << pos(2) << " "
                    << quat(0) << " " << quat(1) << " " << quat(2) << " "
                    << quat(3);
            outfile.precision(10);
            outfile << " " << cov_rot(0, 0) << " " << cov_rot(0, 1) << " "
                    << cov_rot(0, 2) << " " << cov_rot(1, 1) << " "
                    << cov_rot(1, 2) << " " << cov_rot(2, 2) << " "
                    << cov_pos(0, 0) << " " << cov_pos(0, 1) << " "
                    << cov_pos(0, 2) << " " << cov_pos(1, 1) << " "
                    << cov_pos(1, 2) << " " << cov_pos(2, 2) << std::endl;
            n_pose_saved++;
          }
        }
        camera_queue.pop_front();
      }
    }
    if (ev.timestamp > max_camera_time)
      break;
    if (ev.timestamp < time_init)
      continue;

    // callback_stereo: frame drop check
    if (camera_last_timestamp.find(0) != camera_last_timestamp.end() &&
        ev.timestamp < camera_last_timestamp.at(0) + time_delta)
      continue;
    camera_last_timestamp[0] = ev.timestamp;

    cv::Mat img0 = cv::imread(dataset_path + "/mav0/cam0/data/" + ev.filename0,
                              cv::IMREAD_GRAYSCALE);
    cv::Mat img1;
    if (num_cameras == 2)
      img1 = cv::imread(dataset_path + "/mav0/cam1/data/" + ev.filename1,
                        cv::IMREAD_GRAYSCALE);
    if (img0.empty() || (num_cameras == 2 && img1.empty())) {
      PRINT_ERROR(RED "[SERIAL]: unable to load image pair at %.6f\n" RESET,
                  ev.timestamp);
      continue;
    }

    ov_core::CameraData message;
    message.timestamp = ev.timestamp;
    message.images.push_back(img0.clone());
    message.masks.push_back(cv::Mat::zeros(img0.rows, img0.cols, CV_8UC1));
    if (num_cameras == 2) {
      message.sensor_ids = {0, 1};
      message.images.push_back(img1.clone());
      message.masks.push_back(cv::Mat::zeros(img1.rows, img1.cols, CV_8UC1));
    } else {
      message.sensor_ids = {0};
    }
    camera_queue.push_back(message);
    std::sort(camera_queue.begin(), camera_queue.end());
  }

  outfile.close();
  PRINT_INFO("[SERIAL]: done, saved %zu poses to %s\n", n_pose_saved,
             path_est.c_str());
  return EXIT_SUCCESS;
}
