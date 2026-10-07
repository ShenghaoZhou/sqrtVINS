/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
 * Copyright (C) 2018-2026 Guoquan Huang
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

#ifndef PYSQRTVINS_PY_BACKEND_H
#define PYSQRTVINS_PY_BACKEND_H

#include <pybind11/pybind11.h>

#include "vision/CVBackend.h"

namespace py = pybind11;

/**
 * @brief CV backend that dispatches the vision operations to Python.
 *
 * Wraps a duck-typed Python object so that the Sqrt-VINS visual frontend can
 * be extended from Python (e.g. replacing FAST detection with a learned
 * detector such as XFeat). The Python object may implement any subset of the
 * following methods; operations it does not implement are delegated to a
 * fallback backend (by default the built-in OpenCV backend):
 *
 *  - detect(image, mask, threshold, nonmax_suppression) -> (N,3) or (N,5)
 *      float array of (x, y, response[, size, octave]). Called once per
 *      frame on the full-resolution image (never per grid cell).
 *  - refine_subpix(image, pts, win_size, max_iters, eps) -> (N,2) float
 *      array with the refined points, or None to skip refinement.
 *  - equalize_histogram(image) -> uint8 image
 *  - apply_clahe(image, clip_limit, tile_size) -> uint8 image
 *  - build_pyramid(image, levels, win_size) -> list of uint8 images
 *  - track(prev_pyramid, curr_pyramid, pts_prev, pts_next, win_size,
 *      max_levels, max_iters, eps) -> (pts_next (N,2), status (N,))
 *  - reject(pts0_n, pts1_n, focal, threshold, confidence) -> inliers (N,)
 *
 * Images are passed as HxW (grayscale) uint8 numpy arrays; pyramids as lists
 * of such arrays (level 0 = full resolution). All arrays handed to Python are
 * copies, so the implementation may retain them.
 *
 * If a Python callback raises, the traceback is printed and the operation is
 * permanently disabled (delegated to the fallback backend from then on) so a
 * buggy callback cannot crash worker threads of the frontend.
 */
class PyBackend : public ov_core::vision::CVBackend,
                  public ov_core::vision::ImageOps,
                  public ov_core::vision::FeatureDetector,
                  public ov_core::vision::PointTracker,
                  public ov_core::vision::TwoViewRansac {

public:
  /**
   * @brief Wrap a Python implementation object.
   * @param impl duck-typed Python object (see class docs)
   * @param name human readable backend name
   * @param fallback_name name of the built-in backend handling the operations
   * the Python object does not implement ("opencv" or "ocean")
   */
  PyBackend(py::object impl, std::string name, std::string fallback_name);
  ~PyBackend() override;

  std::string name() const override { return name_; }

  ov_core::vision::ImageOps &imageOps() override { return *this; }
  ov_core::vision::FeatureDetector &featureDetector() override {
    return *this;
  }
  ov_core::vision::PointTracker &pointTracker() override { return *this; }
  ov_core::vision::TwoViewRansac &twoViewRansac() override { return *this; }

  bool supports(ov_core::vision::Op op) const override;

  // ImageOps
  ov_core::vision::OwnedImage
  equalizeHistogram(const ov_core::vision::Image &src) override;
  ov_core::vision::OwnedImage applyCLAHE(const ov_core::vision::Image &src,
                                         double clip_limit,
                                         int tile_size) override;
  ov_core::vision::Pyramid buildPyramid(const ov_core::vision::Image &src,
                                        int levels, int win_size) override;

  // FeatureDetector
  std::vector<ov_core::vision::Keypoint>
  detect(const ov_core::vision::Image &src, const ov_core::vision::Image &mask,
         int threshold, bool nonmax_suppression) override;
  bool refineSubpix(const ov_core::vision::Image &src,
                    std::vector<cv::Point2f> &pts, int win_size, int max_iters,
                    double eps) override;
  bool detectWholeImage(const ov_core::vision::Image &src, const cv::Mat &mask,
                        int threshold, bool nonmax_suppression,
                        std::vector<ov_core::vision::Keypoint> &out) override;

  // PointTracker
  void track(const ov_core::vision::Pyramid &prev,
             const ov_core::vision::Pyramid &curr,
             const std::vector<cv::Point2f> &pts_prev,
             std::vector<cv::Point2f> &pts_next, std::vector<uchar> &status,
             int win_size, int max_levels, int max_iters,
             double eps) override;

  // TwoViewRansac
  void reject(const std::vector<cv::Point2f> &pts0_n,
              const std::vector<cv::Point2f> &pts1_n, double focal,
              double threshold, double confidence,
              std::vector<uchar> &inliers) override;

private:
  /// Whether the Python object implements the given method (and it has not
  /// been disabled after raising). Acquires the GIL itself.
  bool hasOp(const char *method, bool &broken_flag) const;

  /// Print the traceback of a callback failure and permanently delegate this
  /// operation to the fallback backend. The GIL must be held by the caller.
  void reportError(const char *method, bool &broken_flag,
                   py::error_already_set &e);

  /// Run a Python callback body, converting any failure (Python exception or
  /// return-value validation error) into a permanent delegation to the
  /// fallback backend. The caller must hold the GIL.
  template <typename F>
  bool tryPy(const char *method, bool &broken_flag, F &&body) {
    try {
      body();
      return true;
    } catch (py::error_already_set &e) {
      reportError(method, broken_flag, e);
    } catch (const std::exception &e) {
      broken_flag = true;
      fprintf(stderr,
              "[pysqrtvins] Python callback '%s' returned invalid data (%s); "
              "permanently delegating this operation to the '%s' fallback "
              "backend\n",
              method, e.what(), fallback_->name().c_str());
    }
    return false;
  }

  /// The duck-typed Python implementation object
  py::object impl_;

  /// Backend name
  std::string name_;

  /// Built-in backend handling the operations not implemented in Python
  std::shared_ptr<ov_core::vision::CVBackend> fallback_;

  /// Operations that raised and have been disabled
  mutable bool broken_detect_ = false;
  mutable bool broken_refine_subpix_ = false;
  mutable bool broken_equalize_ = false;
  mutable bool broken_clahe_ = false;
  mutable bool broken_pyramid_ = false;
  mutable bool broken_track_ = false;
  mutable bool broken_reject_ = false;
};

#endif // PYSQRTVINS_PY_BACKEND_H
