/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
 * Copyright (C) 2018-2026 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2023 Patrick Geneva
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

#include "Frontend.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>

#include "feat/Feature.h"
#include "feat/FeatureDatabase.h"
#include "track/TrackAruco.h"
#include "track/TrackDescriptor.h"
#include "track/TrackKLT.h"

#include "state/State.h"
#include "utils/print.h"

using namespace ov_core;
using namespace ov_srvins;

Frontend::Frontend(VioManagerOptions &params_, std::shared_ptr<State> state_)
    : params(params_), state(state_) {

  // Let's make a feature extractor
  int init_max_features =
      std::floor((float)params.init_options.init_max_features /
                 (float)params.state_options.num_cameras);
  if (params.use_klt) {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackKLT(
        state->cam_intrinsics_cameras, init_max_features,
        state->options.max_aruco_features, params.use_stereo,
        params.histogram_method, params.fast_threshold, params.grid_x,
        params.grid_y, params.min_px_dist, params.ransac_th));
  } else {
    trackFEATS = std::shared_ptr<TrackBase>(new TrackDescriptor(
        state->cam_intrinsics_cameras, init_max_features,
        state->options.max_aruco_features, params.use_stereo,
        params.histogram_method, params.fast_threshold, params.grid_x,
        params.grid_y, params.min_px_dist, params.knn_ratio));
  }

  // Initialize our aruco tag extractor
  if (params.use_aruco) {
    trackARUCO = std::shared_ptr<TrackBase>(new TrackAruco(
        state->cam_intrinsics_cameras, state->options.max_aruco_features,
        params.use_stereo, params.histogram_method, params.downsize_aruco));
  }
}

void Frontend::feed_camera(ov_core::CameraData &message) {
  // Downsample if needed
  for (size_t i = 0; i < message.sensor_ids.size() && params.downsample_cameras;
       i++) {
    cv::Mat img = message.images.at(i);
    cv::Mat mask = message.masks.at(i);
    cv::Mat img_temp, mask_temp;
    cv::pyrDown(img, img_temp, cv::Size(img.cols / 2.0, img.rows / 2.0));
    message.images.at(i) = img_temp;
    cv::pyrDown(mask, mask_temp, cv::Size(mask.cols / 2.0, mask.rows / 2.0));
    message.masks.at(i) = mask_temp;
  }

  // Perform our feature tracking!
  trackFEATS->feed_new_camera(message);
}

void Frontend::process_measurements_rules(
    double timestamp, const std::vector<int> &sensor_ids,
    std::vector<std::shared_ptr<ov_core::Feature>> &featsup_MSCKF,
    std::vector<std::shared_ptr<ov_core::Feature>> &feats_slam_UPDATE,
    std::vector<std::shared_ptr<ov_core::Feature>> &feats_slam_DELAYED) {

  std::vector<std::shared_ptr<Feature>> feats_lost, feats_marg, feats_maxtracks;
  std::vector<std::shared_ptr<Feature>> feats_slam;
  feats_lost =
      trackFEATS->get_feature_database()->features_not_containing_newer(
          state->timestamp, false, true);

  if ((int)state->clones_IMU.size() == state->options.max_clone_size + 1 ||
      (int)state->clones_IMU.size() > 5) {
    feats_marg = trackFEATS->get_feature_database()->features_containing(
        state->margtimestep(), false, true);
    if (trackARUCO != nullptr &&
        timestamp - startup_time >= params.dt_slam_delay) {
      feats_slam = trackARUCO->get_feature_database()->features_containing(
          state->margtimestep(), false, true);
    }
  }

  auto it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    bool found_cam = false;
    for (const auto &pair : (*it1)->uvs) {
      int cam_id = (int)pair.first;
      if (std::find(sensor_ids.begin(), sensor_ids.end(), cam_id) != sensor_ids.end()) {
        found_cam = true;
        break;
      }
    }
    if (found_cam)
      it1++;
    else
      it1 = feats_lost.erase(it1);
  }

  it1 = feats_lost.begin();
  while (it1 != feats_lost.end()) {
    if (std::find(feats_marg.begin(), feats_marg.end(), (*it1)) !=
        feats_marg.end())
      it1 = feats_lost.erase(it1);
    else
      it1++;
  }

  auto it2 = feats_marg.begin();
  while (it2 != feats_marg.end()) {
    bool reached_max = false;
    for (const auto &cams : (*it2)->timestamps)
      if ((int)cams.second.size() > state->options.max_clone_size) {
        reached_max = true;
        break;
      }
    if (reached_max) {
      feats_maxtracks.push_back(*it2);
      it2 = feats_marg.erase(it2);
    } else
      it2++;
  }

  int curr_aruco_tags = 0;
  for (auto &f : state->features_SLAM)
    if ((int)f.second->featid <= 4 * state->options.max_aruco_features)
      curr_aruco_tags++;

  if (state->options.max_slam_features > 0 &&
      timestamp - startup_time >= params.dt_slam_delay &&
      (int)state->features_SLAM.size() <
          state->options.max_slam_features + curr_aruco_tags) {
    int amount_to_add = (state->options.max_slam_features + curr_aruco_tags) -
                        (int)state->features_SLAM.size();
    int valid_amount = (amount_to_add > (int)feats_maxtracks.size())
                           ? (int)feats_maxtracks.size()
                           : amount_to_add;
    if (valid_amount > 0) {
      feats_slam.insert(feats_slam.end(), feats_maxtracks.end() - valid_amount,
                        feats_maxtracks.end());
      feats_maxtracks.erase(feats_maxtracks.end() - valid_amount,
                            feats_maxtracks.end());
    }
  }

  for (auto &landmark : state->features_SLAM) {
    if (trackARUCO != nullptr) {
      std::shared_ptr<Feature> feat =
          trackARUCO->get_feature_database()->get_feature(
              landmark.second->featid);
      if (feat != nullptr)
        feats_slam.push_back(feat);
    }
    std::shared_ptr<Feature> feat =
        trackFEATS->get_feature_database()->get_feature(
            landmark.second->featid);
    if (feat != nullptr)
      feats_slam.push_back(feat);
    bool current_unique_cam =
        std::find(sensor_ids.begin(), sensor_ids.end(),
                  landmark.second->unique_camera_id) !=
        sensor_ids.end();
    if (feat == nullptr && current_unique_cam)
      landmark.second->should_marg = true;
    if (landmark.second->update_fail_count > 1)
      landmark.second->should_marg = true;
  }

  for (auto const &f : feats_slam) {
    if (state->features_SLAM.find(f->featid) != state->features_SLAM.end())
      feats_slam_UPDATE.push_back(f);
    else
      feats_slam_DELAYED.push_back(f);
  }

  featsup_MSCKF = feats_lost;
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_marg.begin(),
                       feats_marg.end());
  featsup_MSCKF.insert(featsup_MSCKF.end(), feats_maxtracks.begin(),
                       feats_maxtracks.end());

  auto compare_feat = [](const std::shared_ptr<Feature> &a,
                         const std::shared_ptr<Feature> &b) -> bool {
    size_t asize = 0, bsize = 0;
    for (const auto &pair : a->timestamps)
      asize += pair.second.size();
    for (const auto &pair : b->timestamps)
      bsize += pair.second.size();
    return asize < bsize;
  };
  std::sort(featsup_MSCKF.begin(), featsup_MSCKF.end(), compare_feat);

  if ((int)featsup_MSCKF.size() > state->options.max_msckf_in_update)
    featsup_MSCKF.erase(featsup_MSCKF.begin(),
                        featsup_MSCKF.end() -
                            state->options.max_msckf_in_update);
}

cv::Mat Frontend::get_historical_viz_image(bool did_zupt, bool is_init) {
  if (state == nullptr || trackFEATS == nullptr)
    return cv::Mat();
  std::vector<size_t> highlighted_ids;
  for (const auto &feat : state->features_SLAM)
    highlighted_ids.push_back(feat.first);
  std::string overlay = (did_zupt) ? "zvupt" : "";
  overlay = (!is_init) ? "init" : overlay;
  cv::Mat img_history;
  trackFEATS->display_history(img_history, 255, 255, 0, 255, 255, 255,
                               highlighted_ids, overlay);
  if (trackARUCO != nullptr)
    trackARUCO->display_history(img_history, 0, 255, 255, 255, 255, 255,
                                 highlighted_ids, overlay);
  return img_history;
}
