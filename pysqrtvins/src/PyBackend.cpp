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

#include "PyBackend.h"

#include <cstring>

#include <pybind11/numpy.h>

using namespace ov_core;
using namespace ov_core::vision;

namespace {

/// Copy a vision image into a freshly allocated numpy array (HxW or HxWxC).
py::array_t<uint8_t> imageToNumpy(const Image &img) {
  std::vector<py::ssize_t> shape =
      img.channels == 1
          ? std::vector<py::ssize_t>{img.height, img.width}
          : std::vector<py::ssize_t>{img.height, img.width, img.channels};
  py::array_t<uint8_t> out(shape);
  size_t row_bytes = (size_t)img.width * (size_t)img.channels;
  for (int y = 0; y < img.height; y++) {
    std::memcpy(out.mutable_data() + (size_t)y * row_bytes,
                img.data + (size_t)y * img.stride, row_bytes);
  }
  return out;
}

/// Copy a grayscale mask into a numpy array (None if empty).
py::object maskToNumpy(const cv::Mat &mask) {
  if (mask.empty())
    return py::none();
  return imageToNumpy(Image::fromCv(mask));
}

/// Copy a pyramid into a python list of numpy arrays (level 0 first).
py::list pyramidToNumpy(const Pyramid &pyr) {
  py::list out;
  for (const Image &level : pyr.levels)
    out.append(imageToNumpy(level));
  return out;
}

/// Validate and copy a uint8 numpy image (HxW or HxWxC) into an OwnedImage.
OwnedImage numpyToOwnedImage(const py::array &arr, const char *what) {
  py::array_t<uint8_t, py::array::forcecast> u8(arr);
  if (u8.ndim() != 2 && u8.ndim() != 3)
    throw std::runtime_error(std::string(what) +
                             ": expected a 2D or 3D uint8 image");
  int height = (int)u8.shape(0);
  int width = (int)u8.shape(1);
  int channels = u8.ndim() == 3 ? (int)u8.shape(2) : 1;
  auto storage = std::make_shared<std::vector<uint8_t>>(
      (size_t)height * width * channels);
  auto buf = u8.unchecked();
  size_t row_bytes = (size_t)width * channels;
  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      for (int c = 0; c < channels; c++) {
        uint8_t v = u8.ndim() == 3 ? buf(y, x, c) : buf(y, x);
        storage->at((size_t)y * row_bytes + (size_t)x * channels + c) = v;
      }
    }
  }
  OwnedImage out;
  out.storage = storage;
  out.view.data = storage->data();
  out.view.width = width;
  out.view.height = height;
  out.view.stride = row_bytes;
  out.view.channels = channels;
  return out;
}

/// Convert an (N,2) float numpy array to points.
std::vector<cv::Point2f> numpyToPoints(const py::array &arr, const char *what) {
  py::array_t<float, py::array::c_style | py::array::forcecast> pts(arr);
  if (pts.ndim() != 2 || pts.shape(1) != 2)
    throw std::runtime_error(std::string(what) + ": expected an (N,2) array");
  auto buf = pts.unchecked<2>();
  std::vector<cv::Point2f> out;
  out.reserve((size_t)pts.shape(0));
  for (py::ssize_t i = 0; i < pts.shape(0); i++)
    out.emplace_back(buf(i, 0), buf(i, 1));
  return out;
}

/// Convert points to an (N,2) float32 numpy array.
py::array_t<float> pointsToNumpy(const std::vector<cv::Point2f> &pts) {
  py::array_t<float> out({(py::ssize_t)pts.size(), (py::ssize_t)2});
  auto buf = out.mutable_unchecked<2>();
  for (size_t i = 0; i < pts.size(); i++) {
    buf(i, 0) = pts.at(i).x;
    buf(i, 1) = pts.at(i).y;
  }
  return out;
}

/// Convert an (N,3|4|5) float numpy array (x, y, response[, size[, octave]])
/// to keypoints.
std::vector<Keypoint> numpyToKeypoints(const py::array &arr, const char *what) {
  py::array_t<float, py::array::c_style | py::array::forcecast> kps(arr);
  if (kps.ndim() != 2 || kps.shape(1) < 3 || kps.shape(1) > 5)
    throw std::runtime_error(std::string(what) +
                             ": expected an (N,3), (N,4) or (N,5) array of "
                             "(x, y, response[, size[, octave]])");
  auto buf = kps.unchecked<2>();
  std::vector<Keypoint> out;
  out.reserve((size_t)kps.shape(0));
  for (py::ssize_t i = 0; i < kps.shape(0); i++) {
    Keypoint kp;
    kp.x = buf(i, 0);
    kp.y = buf(i, 1);
    kp.response = buf(i, 2);
    if (kps.shape(1) > 3)
      kp.size = buf(i, 3);
    if (kps.shape(1) > 4)
      kp.octave = (int)buf(i, 4);
    out.push_back(kp);
  }
  return out;
}

/// Convert a (N,) uint8/bool numpy array to a flag vector.
std::vector<uchar> numpyToFlags(const py::array &arr, const char *what) {
  py::array_t<uint8_t, py::array::c_style | py::array::forcecast> flags(arr);
  if (flags.ndim() != 1)
    throw std::runtime_error(std::string(what) + ": expected an (N,) array");
  auto buf = flags.unchecked<1>();
  std::vector<uchar> out;
  out.reserve((size_t)flags.shape(0));
  for (py::ssize_t i = 0; i < flags.shape(0); i++)
    out.push_back((uchar)(buf(i) ? 1 : 0));
  return out;
}

} // namespace

PyBackend::PyBackend(py::object impl, std::string name,
                     std::string fallback_name)
    : impl_(std::move(impl)), name_(std::move(name)),
      fallback_(CVBackend::create(fallback_name)) {}

PyBackend::~PyBackend() {
  if (Py_IsInitialized()) {
    py::gil_scoped_acquire gil;
    impl_ = py::object();
  } else {
    // The interpreter is gone; intentionally leak the reference
    new py::object(std::move(impl_));
  }
}

bool PyBackend::hasOp(const char *method, bool &broken_flag) const {
  if (broken_flag)
    return false;
  py::gil_scoped_acquire gil;
  return py::hasattr(impl_, method);
}

void PyBackend::reportError(const char *method, bool &broken_flag,
                            py::error_already_set &e) {
  // The GIL is held by the caller
  broken_flag = true;
  fprintf(stderr,
          "[pysqrtvins] Python callback '%s' raised; printing the traceback "
          "and permanently delegating this operation to the '%s' fallback "
          "backend:\n",
          method, fallback_->name().c_str());
  e.restore();
  PyErr_Print();
}

bool PyBackend::supports(Op op) const {
  switch (op) {
  case Op::HistogramEqualization:
    return hasOp("equalize_histogram", broken_equalize_) ||
           fallback_->supports(op);
  case Op::CLAHE:
    return hasOp("apply_clahe", broken_clahe_) || fallback_->supports(op);
  case Op::ImagePyramid:
    return hasOp("build_pyramid", broken_pyramid_) ||
           fallback_->supports(op);
  case Op::FeatureDetection:
    return hasOp("detect", broken_detect_) || fallback_->supports(op);
  case Op::SubpixRefinement:
    return hasOp("refine_subpix", broken_refine_subpix_) ||
           fallback_->supports(op);
  case Op::SparsePointTracking:
    return hasOp("track", broken_track_) || fallback_->supports(op);
  case Op::FundamentalRansac:
    return hasOp("reject", broken_reject_) || fallback_->supports(op);
  }
  return false;
}

OwnedImage PyBackend::equalizeHistogram(const Image &src) {
  if (hasOp("equalize_histogram", broken_equalize_)) {
    OwnedImage out;
    py::gil_scoped_acquire gil;
    bool ok = tryPy("equalize_histogram", broken_equalize_, [&] {
                      py::object result =
                          impl_.attr("equalize_histogram")(imageToNumpy(src));
                      out = numpyToOwnedImage(
                          py::reinterpret_borrow<py::array>(result),
                          "equalize_histogram");
                    });
    if (ok)
      return out;
  }
  return fallback_->imageOps().equalizeHistogram(src);
}

OwnedImage PyBackend::applyCLAHE(const Image &src, double clip_limit,
                                 int tile_size) {
  if (hasOp("apply_clahe", broken_clahe_)) {
    OwnedImage out;
    py::gil_scoped_acquire gil;
    bool ok = tryPy("apply_clahe", broken_clahe_, [&] {
                      py::object result = impl_.attr("apply_clahe")(
                          imageToNumpy(src), clip_limit, tile_size);
                      out = numpyToOwnedImage(
                          py::reinterpret_borrow<py::array>(result),
                          "apply_clahe");
                    });
    if (ok)
      return out;
  }
  return fallback_->imageOps().applyCLAHE(src, clip_limit, tile_size);
}

Pyramid PyBackend::buildPyramid(const Image &src, int levels, int win_size) {
  if (hasOp("build_pyramid", broken_pyramid_)) {
    Pyramid pyr;
    py::gil_scoped_acquire gil;
    bool ok = tryPy("build_pyramid", broken_pyramid_, [&] {
                      py::sequence result = impl_.attr("build_pyramid")(
                          imageToNumpy(src), levels, win_size);
                      if (py::len(result) == 0)
                        throw std::runtime_error(
                            "build_pyramid: returned an empty list");
                      for (py::handle item : result) {
                        OwnedImage owned = numpyToOwnedImage(
                            py::reinterpret_borrow<py::array>(item),
                            "build_pyramid");
                        pyr.levels.push_back(owned.view);
                        pyr.storage.push_back(owned.storage);
                      }
                    });
    if (ok)
      return pyr;
  }
  return fallback_->imageOps().buildPyramid(src, levels, win_size);
}

std::vector<Keypoint> PyBackend::detect(const Image &src, const Image &mask,
                                        int threshold,
                                        bool nonmax_suppression) {
  if (hasOp("detect", broken_detect_)) {
    std::vector<Keypoint> out;
    py::gil_scoped_acquire gil;
    bool ok = tryPy("detect", broken_detect_, [&] {
                      py::object np_mask = mask.valid()
                                               ? py::object(imageToNumpy(mask))
                                               : py::object(py::none());
                      py::object result =
                          impl_.attr("detect")(imageToNumpy(src), np_mask,
                                               threshold, nonmax_suppression);
                      out = numpyToKeypoints(
                          py::reinterpret_borrow<py::array>(result), "detect");
                    });
    if (ok)
      return out;
  }
  return fallback_->featureDetector().detect(src, mask, threshold,
                                             nonmax_suppression);
}

bool PyBackend::refineSubpix(const Image &src, std::vector<cv::Point2f> &pts,
                             int win_size, int max_iters, double eps) {
  if (hasOp("refine_subpix", broken_refine_subpix_)) {
    bool refined = false;
    py::gil_scoped_acquire gil;
    bool ok = tryPy("refine_subpix", broken_refine_subpix_, [&] {
                      py::object result = impl_.attr("refine_subpix")(
                          imageToNumpy(src), pointsToNumpy(pts), win_size,
                          max_iters, eps);
                      if (result.is_none())
                        return; // no refinement requested, keep input points
                      pts = numpyToPoints(
                          py::reinterpret_borrow<py::array>(result),
                          "refine_subpix");
                      refined = true;
                    });
    if (ok)
      return refined;
  }
  return fallback_->featureDetector().refineSubpix(src, pts, win_size,
                                                   max_iters, eps);
}

bool PyBackend::detectWholeImage(const Image &src, const cv::Mat &mask,
                                 int threshold, bool nonmax_suppression,
                                 std::vector<Keypoint> &out) {
  if (!hasOp("detect", broken_detect_))
    return false;
  py::gil_scoped_acquire gil;
  return tryPy("detect", broken_detect_, [&] {
    py::object result = impl_.attr("detect")(imageToNumpy(src),
                                             maskToNumpy(mask), threshold,
                                             nonmax_suppression);
    out = numpyToKeypoints(py::reinterpret_borrow<py::array>(result),
                           "detect");
  });
}

void PyBackend::track(const Pyramid &prev, const Pyramid &curr,
                      const std::vector<cv::Point2f> &pts_prev,
                      std::vector<cv::Point2f> &pts_next,
                      std::vector<uchar> &status, int win_size, int max_levels,
                      int max_iters, double eps) {
  if (hasOp("track", broken_track_)) {
    py::gil_scoped_acquire gil;
    bool ok = tryPy("track", broken_track_, [&] {
                      py::sequence result = impl_.attr("track")(
                          pyramidToNumpy(prev), pyramidToNumpy(curr),
                          pointsToNumpy(pts_prev), pointsToNumpy(pts_next),
                          win_size, max_levels, max_iters, eps);
                      if (py::len(result) != 2)
                        throw std::runtime_error(
                            "track: expected a (pts_next, status) tuple");
                      std::vector<cv::Point2f> tracked = numpyToPoints(
                          py::reinterpret_borrow<py::array>(result[0]),
                          "track");
                      std::vector<uchar> flags = numpyToFlags(
                          py::reinterpret_borrow<py::array>(result[1]),
                          "track");
                      if (tracked.size() != pts_prev.size() ||
                          flags.size() != pts_prev.size())
                        throw std::runtime_error(
                            "track: output sizes do not match the number of "
                            "input points");
                      pts_next = std::move(tracked);
                      status = std::move(flags);
                    });
    if (ok)
      return;
  }
  fallback_->pointTracker().track(prev, curr, pts_prev, pts_next, status,
                                  win_size, max_levels, max_iters, eps);
}

void PyBackend::reject(const std::vector<cv::Point2f> &pts0_n,
                       const std::vector<cv::Point2f> &pts1_n, double focal,
                       double threshold, double confidence,
                       std::vector<uchar> &inliers) {
  if (hasOp("reject", broken_reject_)) {
    py::gil_scoped_acquire gil;
    bool ok = tryPy("reject", broken_reject_, [&] {
                      py::object result =
                          impl_.attr("reject")(pointsToNumpy(pts0_n),
                                               pointsToNumpy(pts1_n), focal,
                                               threshold, confidence);
                      std::vector<uchar> flags = numpyToFlags(
                          py::reinterpret_borrow<py::array>(result), "reject");
                      if (flags.size() != pts0_n.size())
                        throw std::runtime_error(
                            "reject: output size does not match the number "
                            "of input points");
                      inliers = std::move(flags);
                    });
    if (ok)
      return;
  }
  fallback_->twoViewRansac().reject(pts0_n, pts1_n, focal, threshold,
                                    confidence, inliers);
}
