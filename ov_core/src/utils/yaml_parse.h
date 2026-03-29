/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
 * Copyright (C) 2018-2026 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2019 Kevin Eckenhoff
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

#ifndef YAML_PARSER_H
#define YAML_PARSER_H

#include <Eigen/Eigen>
#include <filesystem>
#include <memory>
#include <yaml-cpp/yaml.h>

#if ROS_AVAILABLE == 1
#include <ros/ros.h>
#elif ROS_AVAILABLE == 2
#include <rclcpp/rclcpp.hpp>
#endif

#include "colors.h"
#include "print.h"
#include "quat_ops.h"

namespace ov_core {

/**
 * @brief Helper class to do YAML parsing using yaml-cpp from both file and ROS.
 */
class YamlParser {
public:
  /**
   * @brief Constructor that loads the configuration file
   * @param config_path Path to the YAML file we will parse
   * @param fail_if_not_found If we should terminate the program if we can't open the config file
   */
  explicit YamlParser(const std::string &config_path, bool fail_if_not_found = true)
      : config_path_(config_path) {

    // Check if file exists
    if (!std::filesystem::exists(config_path)) {
      if (fail_if_not_found) {
        PRINT_ERROR(RED "unable to open the configuration file!\n%s\n" RESET, config_path.c_str());
        std::exit(EXIT_FAILURE);
      }
      return;
    }

    // Open the file
    try {
      config = YAML::LoadFile(config_path);
    } catch (const YAML::Exception &e) {
      if (fail_if_not_found) {
        PRINT_ERROR(RED "unable to parse the configuration file!\n%s\nError: %s\n" RESET, config_path.c_str(), e.what());
        std::exit(EXIT_FAILURE);
      }
    }
  }

#if ROS_AVAILABLE == 1
  /// Allows setting of the node handler if we have ROS to override parameters
  void set_node_handler(std::shared_ptr<ros::NodeHandle> nh_) { this->nh = nh_; }
#endif

#if ROS_AVAILABLE == 2
  /// Allows setting of the node if we have ROS to override parameters
  void set_node(std::shared_ptr<rclcpp::Node> &node_) { this->node = node_; }
#endif

  /**
   * @brief Will get the folder this config file is in
   * @return Config folder
   */
  std::string get_config_folder() {
    return std::filesystem::path(config_path_).parent_path().string() + "/";
  }

  /**
   * @brief Check to see if all parameters were read succesfully
   * @return True if we found all parameters
   */
  bool successful() const { return all_params_found_successfully; }

  /**
   * @brief Custom parser for the ESTIMATOR parameters.
   */
  template <class T>
  void parse_config(const std::string &node_name, T &node_result, bool required = true) {
#if ROS_AVAILABLE == 1
    if (nh != nullptr && nh->getParam(node_name, node_result)) {
      PRINT_INFO(GREEN "overriding node " BOLDGREEN "%s" RESET GREEN " with value from ROS!\n" RESET, node_name.c_str());
      return;
    }
#elif ROS_AVAILABLE == 2
    if (node != nullptr && node->has_parameter(node_name)) {
      PRINT_INFO(GREEN "overriding node " BOLDGREEN "%s" RESET GREEN " with value from ROS!\n" RESET, node_name.c_str());
      node->get_parameter<T>(node_name, node_result);
      return;
    }
#endif

    // Else we just parse from the YAML file!
    parse_yaml(config, node_name, node_result, required);
  }

  /**
   * @brief Custom parser for the external parameter files with levels.
   */
  template <class T>
  void parse_external(const std::string &external_node_name, const std::string &sensor_name, const std::string &node_name,
                      T &node_result, bool required = true) {

#if ROS_AVAILABLE == 1
    std::string rosnode = sensor_name + "_" + node_name;
    if (nh != nullptr && nh->getParam(rosnode, node_result)) {
      PRINT_INFO(GREEN "overriding node " BOLDGREEN "%s" RESET GREEN " with value from ROS!\n" RESET, rosnode.c_str());
      return;
    }
#elif ROS_AVAILABLE == 2
    std::string rosnode = sensor_name + "_" + node_name;
    if (node != nullptr && node->has_parameter(rosnode)) {
      PRINT_INFO(GREEN "overriding node " BOLDGREEN "%s" RESET GREEN " with value from ROS!\n" RESET, rosnode.c_str());
      node->get_parameter<T>(rosnode, node_result);
      return;
    }
#endif

    // Else we just parse from the YAML file!
    parse_external_yaml(external_node_name, sensor_name, node_name, node_result, required);
  }

  /**
   * @brief Custom parser for Matrix3d in the external parameter files with levels.
   */
  void parse_external(const std::string &external_node_name, const std::string &sensor_name, const std::string &node_name,
                      Eigen::Matrix3d &node_result, bool required = true) {

#if ROS_AVAILABLE == 1
    std::string rosnode = sensor_name + "_" + node_name;
    std::vector<double> matrix_flat;
    if (nh != nullptr && nh->getParam(rosnode, matrix_flat)) {
      PRINT_INFO(GREEN "overriding node " BOLDGREEN "%s" RESET GREEN " with value from ROS!\n" RESET, rosnode.c_str());
      if (matrix_flat.size() == 9) {
        node_result = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(matrix_flat.data());
      }
      return;
    }
#elif ROS_AVAILABLE == 2
    std::string rosnode = sensor_name + "_" + node_name;
    std::vector<double> matrix_flat;
    if (node != nullptr && node->has_parameter(rosnode)) {
      PRINT_INFO(GREEN "overriding node " BOLDGREEN "%s" RESET GREEN " with value from ROS!\n" RESET, rosnode.c_str());
      node->get_parameter<std::vector<double>>(rosnode, matrix_flat);
      if (matrix_flat.size() == 9) {
        node_result = Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(matrix_flat.data());
      }
      return;
    }
#endif

    // Else we just parse from the YAML file!
    parse_external_yaml(external_node_name, sensor_name, node_name, node_result, required);
  }

  /**
   * @brief Custom parser for Matrix4d in the external parameter files with levels.
   */
  void parse_external(const std::string &external_node_name, const std::string &sensor_name, const std::string &node_name,
                      Eigen::Matrix4d &node_result, bool required = true) {

#if ROS_AVAILABLE == 1
    std::string rosnode = sensor_name + "_" + node_name;
    std::vector<double> matrix_flat;
    if (nh != nullptr && nh->getParam(rosnode, matrix_flat)) {
      PRINT_INFO(GREEN "overriding node " BOLDGREEN "%s" RESET GREEN " with value from ROS!\n" RESET, rosnode.c_str());
      if (matrix_flat.size() == 16) {
        node_result = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(matrix_flat.data());
      }
      return;
    }
#elif ROS_AVAILABLE == 2
    std::string rosnode = sensor_name + "_" + node_name;
    std::vector<double> matrix_flat;
    if (node != nullptr && node->has_parameter(rosnode)) {
      PRINT_INFO(GREEN "overriding node " BOLDGREEN "%s" RESET GREEN " with value from ROS!\n" RESET, rosnode.c_str());
      node->get_parameter<std::vector<double>>(rosnode, matrix_flat);
      if (matrix_flat.size() == 16) {
        node_result = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(matrix_flat.data());
      }
      return;
    }
#endif

    // Else we just parse from the YAML file!
    parse_external_yaml(external_node_name, sensor_name, node_name, node_result, required);
  }

#if ROS_AVAILABLE == 2
  void parse_config(const std::string &node_name, int &node_result, bool required = true) {
    int64_t val = node_result;
    if (node != nullptr && node->has_parameter(node_name)) {
      PRINT_INFO(GREEN "overriding node " BOLDGREEN "%s" RESET GREEN " with value from ROS!\n" RESET, node_name.c_str());
      node->get_parameter<int64_t>(node_name, val);
      node_result = (int)val;
      return;
    }
    parse_yaml(config, node_name, node_result, required);
  }

  void parse_external(const std::string &external_node_name, const std::string &sensor_name, const std::string &node_name,
                      std::vector<int> &node_result, bool required = true) {
    std::vector<int64_t> val;
    for (auto tmp : node_result)
      val.push_back(tmp);
    std::string rosnode = sensor_name + "_" + node_name;
    if (node != nullptr && node->has_parameter(rosnode)) {
      PRINT_INFO(GREEN "overriding node " BOLDGREEN "%s" RESET GREEN " with value from ROS!\n" RESET, rosnode.c_str());
      node->get_parameter<std::vector<int64_t>>(rosnode, val);
      node_result.clear();
      for (auto tmp : val)
        node_result.push_back((int)tmp);
      return;
    }
    parse_external_yaml(external_node_name, sensor_name, node_name, node_result, required);
  }
#endif

private:
  /// Path to the config file
  std::string config_path_;

  /// Our config file with the data in it
  YAML::Node config;

  /// Record if all parameters were found
  bool all_params_found_successfully = true;

#if ROS_AVAILABLE == 1
  /// ROS1 node handler that will override values
  std::shared_ptr<ros::NodeHandle> nh;
#endif

#if ROS_AVAILABLE == 2
  /// Our ROS2 rclcpp node pointer
  std::shared_ptr<rclcpp::Node> node = nullptr;
#endif

  /**
   * @brief Helper to parse a value from a YAML node
   */
  template <class T>
  void parse_yaml(const YAML::Node &node, const std::string &node_name, T &node_result, bool required = true) {
    if (!node[node_name]) {
      if (required) {
        PRINT_WARNING(YELLOW "the node %s of type [%s] was not found...\n" RESET, node_name.c_str(), typeid(node_result).name());
        all_params_found_successfully = false;
      }
      return;
    }

    try {
      node_result = node[node_name].as<T>();
    } catch (const std::exception &e) {
      if (required) {
        PRINT_WARNING(YELLOW "unable to parse %s node in the config file! Error: %s\n" RESET, node_name.c_str(), e.what());
        all_params_found_successfully = false;
      }
    }
  }

  /**
   * @brief Specialization for bool to handle strings like "true", "True", "TRUE", "1"
   */
  void parse_yaml(const YAML::Node &node, const std::string &node_name, bool &node_result, bool required = true) {
    if (!node[node_name]) {
      if (required) {
        PRINT_WARNING(YELLOW "the node %s of type [bool] was not found...\n" RESET, node_name.c_str());
        all_params_found_successfully = false;
      }
      return;
    }

    try {
      node_result = node[node_name].as<bool>();
    } catch (...) {
      // Fallback for custom bool strings if needed, though yaml-cpp handles most standard ones.
      try {
        std::string val = node[node_name].as<std::string>();
        if (val == "1" || val == "true" || val == "True" || val == "TRUE") node_result = true;
        else if (val == "0" || val == "false" || val == "False" || val == "FALSE") node_result = false;
        else throw std::runtime_error("Invalid bool");
      } catch (...) {
        if (required) {
          PRINT_WARNING(YELLOW "unable to parse %s node as bool!\n" RESET, node_name.c_str());
          all_params_found_successfully = false;
        }
      }
    }
  }

  /**
   * @brief Specialization for Eigen matrices
   */
  void parse_yaml(const YAML::Node &node, const std::string &node_name, Eigen::Matrix3d &node_result, bool required = true) {
    if (!node[node_name]) {
      if (required) {
        PRINT_WARNING(YELLOW "the node %s was not found...\n" RESET, node_name.c_str());
        all_params_found_successfully = false;
      }
      return;
    }

    try {
      YAML::Node m = node[node_name];
      if (m.IsSequence() && m.size() == 3) {
        for (int i = 0; i < 3; ++i) {
          if (m[i].IsSequence() && m[i].size() == 3) {
            for (int j = 0; j < 3; ++j) {
              node_result(i, j) = m[i][j].as<double>();
            }
          }
        }
      }
    } catch (const std::exception &e) {
      if (required) {
        PRINT_WARNING(YELLOW "unable to parse %s matrix! Error: %s\n" RESET, node_name.c_str(), e.what());
        all_params_found_successfully = false;
      }
    }
  }

  void parse_yaml(const YAML::Node &node, const std::string &node_name, Eigen::Matrix4d &node_result, bool required = true) {
    std::string node_name_local = node_name;
    if (node_name == "T_cam_imu" && !node[node_name]) {
      node_name_local = "T_imu_cam";
    } else if (node_name == "T_imu_cam" && !node[node_name]) {
      node_name_local = "T_cam_imu";
    }

    if (!node[node_name_local]) {
      if (required) {
        PRINT_WARNING(YELLOW "the node %s was not found...\n" RESET, node_name_local.c_str());
        all_params_found_successfully = false;
      }
      return;
    }

    try {
      YAML::Node m = node[node_name_local];
      if (m.IsSequence() && m.size() == 4) {
        for (int i = 0; i < 4; ++i) {
          if (m[i].IsSequence() && m[i].size() == 4) {
            for (int j = 0; j < 4; ++j) {
              node_result(i, j) = m[i][j].as<double>();
            }
          }
        }
      }
      if (node_name_local != node_name) {
        node_result = ov_core::Inv_se3(node_result);
      }
    } catch (const std::exception &e) {
      if (required) {
        PRINT_WARNING(YELLOW "unable to parse %s matrix! Error: %s\n" RESET, node_name_local.c_str(), e.what());
        all_params_found_successfully = false;
      }
    }
  }

  /**
   * @brief Helper for external YAML files
   */
  template <class T>
  void parse_external_yaml(const std::string &external_node_name, const std::string &sensor_name, const std::string &node_name,
                           T &node_result, bool required = true) {
    if (!config[external_node_name]) {
      if (required) {
        PRINT_ERROR(RED "the external node %s could not be found!\n" RESET, external_node_name.c_str());
        std::exit(EXIT_FAILURE);
      }
      return;
    }

    std::string path = config[external_node_name].as<std::string>();
    std::string relative_folder = std::filesystem::path(config_path_).parent_path().string() + "/";
    std::string full_path = relative_folder + path;

    try {
      YAML::Node external_config = YAML::LoadFile(full_path);
      if (!external_config[sensor_name]) {
        if (required) {
          PRINT_WARNING(YELLOW "the sensor %s was not found in %s...\n" RESET, sensor_name.c_str(), full_path.c_str());
          all_params_found_successfully = false;
        }
        return;
      }
      parse_yaml(external_config[sensor_name], node_name, node_result, required);
    } catch (const std::exception &e) {
      PRINT_ERROR(RED "unable to open or parse external configuration file!\n%s\nError: %s\n" RESET, full_path.c_str(), e.what());
      std::exit(EXIT_FAILURE);
    }
  }
};

} /* namespace ov_core */

#endif /* YAML_PARSER_H */
