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

#ifndef OV_CORE_OCEAN_BACKEND_H
#define OV_CORE_OCEAN_BACKEND_H

// This backend is only compiled when the build system found the Ocean
// framework (see ENABLE_OCEAN_BACKEND in ov_core/CMakeLists.txt)
#ifdef OV_HAVE_OCEAN

#include <memory>

#include <ocean/base/Frame.h>
#include <ocean/cv/FramePyramid.h>

#include "vision/CVBackend.h"

namespace ov_core {
namespace vision {

/**
 * @brief Implementation of the vision operations based on the Ocean framework
 *        (https://github.com/facebookresearch/ocean).
 *
 * Supported operations and their mapping:
 *  - HistogramEqualization: LUT-based equalization on the image data
 *  - CLAHE: not available in Ocean, use supports(Op::CLAHE) and fall back
 *  - ImagePyramid: ocean::CV::FramePyramid (attached as native handle)
 *  - FeatureDetection: ocean::CV::Detector::FASTFeatureDetector
 *  - SubpixRefinement: not available in Ocean
 *  - SparsePointTracking: pyramidal patch tracker built on Ocean's
 *    zero-mean sum of squared differences (an alternative to the gradient
 *    based Lucas-Kanade implementation of OpenCV)
 *  - FundamentalRansac: ocean::Geometry::RANSAC::fundamentalMatrix followed
 *    by a Sampson-distance inlier test with the same error metric as the
 *    OpenCV backend
 */
class OceanBackend : public CVBackend, public ImageOps, public FeatureDetector,
                     public PointTracker, public TwoViewRansac {
public:
  OceanBackend();
  ~OceanBackend() override;

  std::string name() const override { return "ocean"; }

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

private:
  /// Returns the native pyramid of a generic pyramid. If the pyramid does
  /// not carry an Ocean native handle (e.g. it was created by another
  /// backend), a new FramePyramid is built from its first level; the returned
  /// shared pointer keeps the underlying data alive in either case.
  static std::shared_ptr<const Ocean::CV::FramePyramid>
  nativePyramid(const Pyramid &pyr);
};

} // namespace vision
} // namespace ov_core

#endif // OV_HAVE_OCEAN

#endif // OV_CORE_OCEAN_BACKEND_H
