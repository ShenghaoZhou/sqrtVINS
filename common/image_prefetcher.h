/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 */

/**
 * @brief Background stereo image prefetcher for the EuRoC runners.
 *
 * PNG decoding is the single largest cost of dataset replay (~5.5 ms/frame,
 * more than tracking and the estimator update combined). This helper decodes
 * upcoming frames on a worker thread so the processing loop never blocks on
 * disk/decode. Depends only on OpenCV and the STL (like euroc_common.h, it
 * must stay independent of the estimator modules).
 */

#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "euroc_common.h"

/// Decodes stereo pairs ahead of time on a single worker thread.
class StereoImagePrefetcher {
public:
  struct Frame {
    cv::Mat img0, img1;
  };

  StereoImagePrefetcher(const std::string &dataset_path,
                        const std::vector<CamReading> &cam_data,
                        size_t max_ahead = 16)
      : cam0_dir_(dataset_path + "/mav0/cam0/data/"),
        cam1_dir_(dataset_path + "/mav0/cam1/data/"), cam_data_(cam_data),
        max_ahead_(max_ahead) {
    thread_ = std::thread(&StereoImagePrefetcher::worker, this);
  }

  /// Stops the worker and joins the thread.
  ~StereoImagePrefetcher() { stop(); }

  StereoImagePrefetcher(const StereoImagePrefetcher &) = delete;
  StereoImagePrefetcher &operator=(const StereoImagePrefetcher &) = delete;

  /// Returns the next decoded pair (mats may be empty for missing files,
  /// or after stop()).
  Frame next() {
    std::unique_lock<std::mutex> lck(mtx_);
    cv_not_empty_.wait(lck, [&] { return !queue_.empty() || done_; });
    if (queue_.empty()) {
      return Frame{};
    }
    Frame frame = std::move(queue_.front());
    queue_.pop_front();
    lck.unlock();
    cv_not_full_.notify_one();
    return frame;
  }

  void stop() {
    {
      std::lock_guard<std::mutex> lck(mtx_);
      done_ = true;
    }
    cv_not_full_.notify_one();
    cv_not_empty_.notify_all();
    if (thread_.joinable()) {
      thread_.join();
    }
  }

private:
  void worker() {
    for (size_t i = 0; i < cam_data_.size(); i++) {
      {
        std::unique_lock<std::mutex> lck(mtx_);
        cv_not_full_.wait(lck, [&] { return queue_.size() < max_ahead_ || done_; });
        if (done_) {
          return;
        }
      }
      Frame frame;
      frame.img0 = cv::imread(cam0_dir_ + cam_data_.at(i).filename_cam0,
                              cv::IMREAD_GRAYSCALE);
      frame.img1 = cv::imread(cam1_dir_ + cam_data_.at(i).filename_cam1,
                              cv::IMREAD_GRAYSCALE);
      {
        std::lock_guard<std::mutex> lck(mtx_);
        queue_.push_back(std::move(frame));
      }
      cv_not_empty_.notify_one();
    }
  }

  std::string cam0_dir_, cam1_dir_;
  const std::vector<CamReading> &cam_data_;
  size_t max_ahead_;
  std::thread thread_;
  std::mutex mtx_;
  std::condition_variable cv_not_full_, cv_not_empty_;
  std::deque<Frame> queue_;
  bool done_ = false;
};
