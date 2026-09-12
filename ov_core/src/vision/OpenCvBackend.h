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

#ifndef OV_CORE_OPENCV_BACKEND_H
#define OV_CORE_OPENCV_BACKEND_H

#include "vision/CVBackend.h"

namespace ov_core {
namespace vision {

/**
 * @brief Traditional OpenCV implementation of the vision operations.
 *
 * This backend wraps the historical OpenCV calls (equalizeHist / CLAHE,
 * buildOpticalFlowPyramid + calcOpticalFlowPyrLK, FAST, cornerSubPix and
 * findFundamentalMat) and preserves the exact numerical behavior of the
 * pre-refactoring front-end.
 */
class OpenCvBackend : public CVBackend, public ImageOps, public FeatureDetector,
                      public PointTracker, public TwoViewRansac {
public:
  std::string name() const override { return "opencv"; }

  ImageOps &imageOps() override { return *this; }
  FeatureDetector &featureDetector() override { return *this; }
  PointTracker &pointTracker() override { return *this; }
  TwoViewRansac &twoViewRansac() override { return *this; }

  bool supports(Op op) const override;

  OwnedImage equalizeHistogram(const Image &src) override;
  OwnedImage applyCLAHE(const Image &src, double clip_limit,
                        int tile_size) override;
  Pyramid buildPyramid(const Image &src, int levels, int win_size) override;

  std::vector<Keypoint> detect(const Image &src, const Image &mask,
                               int threshold,
                               bool nonmax_suppression) override;
  bool refineSubpix(const Image &src, std::vector<cv::Point2f> &pts,
                    int win_size, int max_iters, double eps) override;

  void track(const Pyramid &prev, const Pyramid &curr,
             const std::vector<cv::Point2f> &pts_prev,
             std::vector<cv::Point2f> &pts_next, std::vector<uchar> &status,
             int win_size, int max_levels, int max_iters,
             double eps) override;

  void reject(const std::vector<cv::Point2f> &pts0_n,
              const std::vector<cv::Point2f> &pts1_n, double focal,
              double threshold, double confidence,
              std::vector<uchar> &inliers) override;
};

} // namespace vision
} // namespace ov_core

#endif // OV_CORE_OPENCV_BACKEND_H
