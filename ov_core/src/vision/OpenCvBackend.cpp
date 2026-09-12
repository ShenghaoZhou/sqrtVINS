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

#include "vision/OpenCvBackend.h"

#include <stdexcept>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

using namespace ov_core;
using namespace ov_core::vision;

namespace {

/// Copy an OpenCV matrix into an owned buffer and return a view of it
OwnedImage own(const cv::Mat &mat) {
  OwnedImage owned;
  owned.storage = std::make_shared<std::vector<uint8_t>>(
      mat.data, mat.data + mat.total() * mat.elemSize());
  owned.view = Image();
  owned.view.data = owned.storage->data();
  owned.view.width = mat.cols;
  owned.view.height = mat.rows;
  owned.view.stride = mat.cols * mat.elemSize();
  owned.view.channels = mat.channels();
  return owned;
}

} // namespace

bool OpenCvBackend::supports(Op op) const {
  switch (op) {
  case Op::HistogramEqualization:
  case Op::CLAHE:
  case Op::ImagePyramid:
  case Op::FeatureDetection:
  case Op::SubpixRefinement:
  case Op::SparsePointTracking:
  case Op::FundamentalRansac:
    return true;
  }
  return false;
}

OwnedImage OpenCvBackend::equalizeHistogram(const Image &src) {
  cv::Mat out;
  cv::equalizeHist(src.toCv(), out);
  return own(out);
}

OwnedImage OpenCvBackend::applyCLAHE(const Image &src, double clip_limit,
                                     int tile_size) {
  auto clahe = cv::createCLAHE(clip_limit, cv::Size(tile_size, tile_size));
  cv::Mat out;
  clahe->apply(src.toCv(), out);
  return own(out);
}

Pyramid OpenCvBackend::buildPyramid(const Image &src, int levels,
                                    int win_size) {
  // cv::buildOpticalFlowPyramid returns the image levels interleaved with
  // their Scharr derivative levels (CV_16S); the derivatives are opaque to
  // the abstraction, so the full result is kept as the native handle while
  // the generic levels only expose the 8-bit image levels
  std::vector<cv::Mat> mats;
  cv::buildOpticalFlowPyramid(src.toCv(), mats, cv::Size(win_size, win_size),
                              levels);
  auto native = std::make_shared<std::vector<cv::Mat>>(std::move(mats));

  Pyramid pyr;
  pyr.native = native;
  pyr.native_backend = name();
  for (const auto &mat : *native) {
    if (mat.depth() == CV_8U)
      pyr.levels.push_back(Image::fromCv(mat));
  }
  return pyr;
}

std::vector<Keypoint> OpenCvBackend::detect(const Image &src, const Image &mask,
                                            int threshold,
                                            bool nonmax_suppression) {
  std::vector<cv::KeyPoint> pts;
  cv::FAST(src.toCv(), pts, threshold, nonmax_suppression);
  std::vector<Keypoint> out;
  out.reserve(pts.size());
  for (const auto &pt : pts) {
    if (mask.valid()) {
      int x = (int)pt.pt.x;
      int y = (int)pt.pt.y;
      if (x < 0 || x >= mask.width || y < 0 || y >= mask.height)
        continue;
      if (mask.data[(size_t)y * mask.stride + (size_t)x] > 127)
        continue;
    }
    out.push_back(Keypoint());
    out.back().x = pt.pt.x;
    out.back().y = pt.pt.y;
    out.back().response = pt.response;
    out.back().size = pt.size;
    out.back().octave = pt.octave;
  }
  return out;
}

bool OpenCvBackend::refineSubpix(const Image &src, std::vector<cv::Point2f> &pts,
                                 int win_size, int max_iters, double eps) {
  if (pts.empty())
    return true;
  cv::cornerSubPix(src.toCv(), pts, cv::Size(win_size, win_size),
                   cv::Size(-1, -1),
                   cv::TermCriteria(cv::TermCriteria::COUNT +
                                        cv::TermCriteria::EPS,
                                    max_iters, eps));
  return true;
}

void OpenCvBackend::track(const Pyramid &prev, const Pyramid &curr,
                          const std::vector<cv::Point2f> &pts_prev,
                          std::vector<cv::Point2f> &pts_next,
                          std::vector<uchar> &status, int win_size,
                          int max_levels, int max_iters, double eps) {
  // Prefer the native pyramid (which includes the derivative levels needed by
  // calcOpticalFlowPyrLK); if the pyramid comes from another backend, rebuild
  // one from its full-resolution image level
  std::vector<cv::Mat> prev_mats;
  std::vector<cv::Mat> curr_mats;
  if (prev.native_backend == name() && prev.native) {
    prev_mats = *std::static_pointer_cast<std::vector<cv::Mat>>(prev.native);
  } else if (!prev.levels.empty()) {
    cv::buildOpticalFlowPyramid(prev.levels.at(0).toCv(), prev_mats,
                                cv::Size(win_size, win_size), max_levels);
  }
  if (curr.native_backend == name() && curr.native) {
    curr_mats = *std::static_pointer_cast<std::vector<cv::Mat>>(curr.native);
  } else if (!curr.levels.empty()) {
    cv::buildOpticalFlowPyramid(curr.levels.at(0).toCv(), curr_mats,
                                cv::Size(win_size, win_size), max_levels);
  }
  if (prev_mats.empty() || curr_mats.empty())
    throw std::runtime_error("vision: point tracking requires a valid pyramid");

  std::vector<float> error;
  cv::TermCriteria term_crit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
                             max_iters, eps);
  cv::calcOpticalFlowPyrLK(prev_mats, curr_mats, pts_prev, pts_next, status,
                           error, cv::Size(win_size, win_size), max_levels,
                           term_crit, cv::OPTFLOW_USE_INITIAL_FLOW);
}

void OpenCvBackend::reject(const std::vector<cv::Point2f> &pts0_n,
                           const std::vector<cv::Point2f> &pts1_n,
                           double focal, double threshold, double confidence,
                           std::vector<uchar> &inliers) {
  cv::findFundamentalMat(pts0_n, pts1_n, cv::FM_RANSAC, threshold / focal,
                         confidence, inliers);
}
