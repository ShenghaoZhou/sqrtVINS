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

#ifndef OV_CORE_CV_BACKEND_H
#define OV_CORE_CV_BACKEND_H

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "vision/Types.h"

namespace ov_core {
namespace vision {

/// Identifiers for the individual computer vision operations. Backends may
/// implement a subset; use CVBackend::supports() to query availability.
enum class Op {
  HistogramEqualization,  ///< Global grayscale histogram equalization
  CLAHE,                  ///< Contrast limited adaptive histogram equalization
  ImagePyramid,           ///< Construction of an image pyramid
  FeatureDetection,       ///< FAST-like corner detection on a subimage
  SubpixRefinement,       ///< Corner sub-pixel refinement of point positions
  SparsePointTracking,    ///< Sparse point tracking between two pyramids (KLT)
  FundamentalRansac,      ///< Two-view fundamental matrix RANSAC outlier rejection
};

/**
 * @brief High-level image pre-processing operations.
 *
 * All implementations must be callable from multiple threads concurrently on
 * distinct inputs (the front-end parallelizes per-camera work).
 */
class ImageOps {
public:
  virtual ~ImageOps() = default;

  /// Global histogram equalization of a grayscale image.
  virtual OwnedImage equalizeHistogram(const Image &src) = 0;

  /// Contrast limited adaptive histogram equalization.
  /// @param clip_limit threshold for contrast limiting
  /// @param tile_size size of the grid tiles for local equalization
  virtual OwnedImage applyCLAHE(const Image &src, double clip_limit = 10.0,
                                int tile_size = 8) = 0;

  /// Build a pyramid suitable for sparse point tracking with the given
  /// search window size. The returned pyramid owns its memory.
  virtual Pyramid buildPyramid(const Image &src, int levels, int win_size) = 0;
};

/**
 * @brief Feature detection and point refinement operations.
 */
class FeatureDetector {
public:
  virtual ~FeatureDetector() = default;

  /// Detect FAST-like features inside an image.
  /// @param src image to detect in (only the subimage the caller passes)
  /// @param mask optional mask with the same size as src; values > 127 mark
  ///        regions where no features should be returned (may be empty)
  /// @param threshold FAST intensity threshold
  /// @param nonmax_suppression whether to apply non-maximum suppression
  virtual std::vector<Keypoint>
  detect(const Image &src, const Image &mask, int threshold,
         bool nonmax_suppression) = 0;

  /// Refine integer point positions to sub-pixel accuracy (cornerSubPix-like).
  /// Returns false if the backend does not implement this operation; callers
  /// must be able to cope with unrefined points in that case.
  virtual bool refineSubpix(const Image &src, std::vector<cv::Point2f> &pts,
                            int win_size = 5, int max_iters = 20,
                            double eps = 0.001) = 0;

  /// Optional whole-image detection on the full-resolution image. Backends
  /// whose detector is expensive to invoke (e.g. learned detectors wrapped
  /// through the Python bindings) can implement this so the grider calls the
  /// detector once per frame on the full image and distributes the returned
  /// keypoints over the grid cells itself, instead of invoking detect() once
  /// per cell. @param src full-resolution grayscale image
  /// @param mask mask with the same size as src; values > 127 mark regions
  ///        where no features should be returned (always valid here)
  /// @return true if this operation is implemented and `out` was filled with
  ///         keypoints in level-0 coordinates; false to fall back to the
  ///         default per-cell detect() calls
  virtual bool detectWholeImage(const Image &src, const cv::Mat &mask,
                                int threshold, bool nonmax_suppression,
                                std::vector<Keypoint> &out) {
    (void)src;
    (void)mask;
    (void)threshold;
    (void)nonmax_suppression;
    (void)out;
    return false;
  }
};

/**
 * @brief Sparse point tracking between two image pyramids (KLT-like).
 *
 * Semantics mirror OpenCV's calcOpticalFlowPyrLK with
 * OPTFLOW_USE_INITIAL_FLOW: pts_next on input contains initial guesses for
 * the point locations in the second image and is overwritten with the
 * tracked locations. Points out of bounds of the target image are reported
 * with status = 0.
 */
class PointTracker {
public:
  virtual ~PointTracker() = default;

  /// @param prev pyramid of the first image
  /// @param curr pyramid of the second image
  /// @param pts_prev points in the first image (level-0 coordinates)
  /// @param pts_next initial guesses / tracked results (level-0 coordinates)
  /// @param status per-point success flag (1 = tracked, 0 = failed)
  /// @param win_size search window at each pyramid level
  /// @param max_levels maximal number of pyramid levels to use
  /// @param max_iters maximal number of iterations per level
  /// @param eps termination epsilon per level
  virtual void track(const Pyramid &prev, const Pyramid &curr,
                     const std::vector<cv::Point2f> &pts_prev,
                     std::vector<cv::Point2f> &pts_next,
                     std::vector<uchar> &status, int win_size, int max_levels,
                     int max_iters = 30, double eps = 0.01) = 0;
};

/**
 * @brief Two-view outlier rejection based on a fundamental matrix RANSAC.
 *
 * Points must already be normalized (undistorted) camera observations. The
 * error metric is the Sampson distance, thresholded in pixel units scaled by
 * the focal length, matching the historical OpenCV usage.
 */
class TwoViewRansac {
public:
  virtual ~TwoViewRansac() = default;

  /// @param pts0_n normalized points in the first image
  /// @param pts1_n normalized points in the second image (same size)
  /// @param focal max focal length of the two cameras (error scaling)
  /// @param threshold RANSAC pixel error threshold
  /// @param confidence RANSAC success probability
  /// @param inliers per-point inlier flag (1 = inlier)
  virtual void reject(const std::vector<cv::Point2f> &pts0_n,
                      const std::vector<cv::Point2f> &pts1_n, double focal,
                      double threshold, double confidence,
                      std::vector<uchar> &inliers) = 0;
};

/**
 * @brief A complete computer vision backend.
 *
 * A backend bundles implementations of the individual operations above. The
 * front-end only talks to this interface, so the actual implementation of the
 * core computer vision operations is selected at runtime (e.g. the
 * traditional OpenCV one, or one built on top of the Ocean framework).
 *
 * Implementations must support concurrent use from multiple threads on
 * distinct inputs.
 */
class CVBackend {
public:
  virtual ~CVBackend() = default;

  /// Human readable name of this backend ("opencv", "ocean", ...)
  virtual std::string name() const = 0;

  virtual ImageOps &imageOps() = 0;
  virtual FeatureDetector &featureDetector() = 0;
  virtual PointTracker &pointTracker() = 0;
  virtual TwoViewRansac &twoViewRansac() = 0;

  /// Whether this backend implements the given operation. Calling an
  /// unsupported operation throws std::runtime_error.
  virtual bool supports(Op op) const = 0;

  /// Create a backend by name.
  ///  - "opencv": the traditional OpenCV implementation (always available)
  ///  - "ocean": implementation based on the Ocean framework (available when
  ///    the library was compiled with ENABLE_OCEAN_BACKEND)
  /// Throws std::runtime_error for unknown names or backends that were not
  /// compiled into this build.
  static std::shared_ptr<CVBackend> create(const std::string &name);
};

} // namespace vision
} // namespace ov_core

#endif // OV_CORE_CV_BACKEND_H
