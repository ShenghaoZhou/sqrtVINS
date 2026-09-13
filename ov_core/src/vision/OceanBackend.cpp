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

#ifdef OV_HAVE_OCEAN

#include "vision/OceanBackend.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <ocean/base/RandomGenerator.h>
#include <ocean/cv/ZeroMeanSumSquareDifferences.h>
#include <ocean/cv/detector/FASTFeatureDetector.h>
#include <ocean/geometry/RANSAC.h>
#include <ocean/math/SquareMatrix3.h>

#include "utils/print.h"

using namespace ov_core;
using namespace ov_core::vision;

using Ocean::CV::FramePyramid;
using Ocean::Frame;
using Ocean::FrameType;

namespace {

/// Zero-mean SSD between two patchSize x patchSize patches centered at the
/// given integer positions (compile-time patch size for SIMD instantiation)
template <unsigned int tPatchSize>
inline uint32_t ssdPatch(const Frame &f0, const Frame &f1, float cx0, float cy0,
                         float cx1, float cy1) {
  return Ocean::CV::ZeroMeanSumSquareDifferences::patch8BitPerChannel<
      1u, tPatchSize>(f0.constdata<uint8_t>(), f1.constdata<uint8_t>(),
                      f0.width(), f1.width(), (unsigned int)cx0,
                      (unsigned int)cy0, (unsigned int)cx1, (unsigned int)cy1,
                      f0.paddingElements(), f1.paddingElements());
}

/// Runtime dispatch over the supported (odd) patch sizes
inline uint32_t ssdPatchDispatch(const Frame &f0, const Frame &f1, int patch,
                                 float cx0, float cy0, float cx1, float cy1) {
  switch (patch) {
  case 7:
    return ssdPatch<7u>(f0, f1, cx0, cy0, cx1, cy1);
  case 9:
    return ssdPatch<9u>(f0, f1, cx0, cy0, cx1, cy1);
  case 11:
    return ssdPatch<11u>(f0, f1, cx0, cy0, cx1, cy1);
  case 17:
    return ssdPatch<17u>(f0, f1, cx0, cy0, cx1, cy1);
  case 21:
    return ssdPatch<21u>(f0, f1, cx0, cy0, cx1, cy1);
  case 15:
  default:
    return ssdPatch<15u>(f0, f1, cx0, cy0, cx1, cy1);
  }
}

} // namespace

OceanBackend::OceanBackend() = default;
OceanBackend::~OceanBackend() = default;

bool OceanBackend::supports(Op op) const {
  switch (op) {
  case Op::HistogramEqualization:
  case Op::ImagePyramid:
  case Op::FeatureDetection:
  case Op::SparsePointTracking:
  case Op::FundamentalRansac:
    return true;
  case Op::CLAHE:
  case Op::SubpixRefinement:
    return false;
  }
  return false;
}

OwnedImage OceanBackend::equalizeHistogram(const Image &src) {
  OwnedImage owned;
  const size_t count = (size_t)src.width * (size_t)src.height;
  owned.storage = std::make_shared<std::vector<uint8_t>>(count);

  // Determine the 256-bin histogram of the grayscale source
  uint32_t histogram[256] = {0};
  for (int y = 0; y < src.height; ++y) {
    const uint8_t *row = src.data + (size_t)y * src.stride;
    for (int x = 0; x < src.width; ++x)
      histogram[row[x]]++;
  }

  // Equalization lookup table from the cumulative distribution
  uint8_t lut[256];
  uint32_t sum = 0;
  for (int i = 0; i < 256; ++i) {
    sum += histogram[i];
    lut[i] = (uint8_t)((sum * 255u + count / 2u) / count);
  }

  // Apply the lookup table
  uint8_t *dst = owned.storage->data();
  for (int y = 0; y < src.height; ++y) {
    const uint8_t *row = src.data + (size_t)y * src.stride;
    for (int x = 0; x < src.width; ++x)
      dst[(size_t)y * src.width + x] = lut[row[x]];
  }

  owned.view = Image();
  owned.view.data = owned.storage->data();
  owned.view.width = src.width;
  owned.view.height = src.height;
  owned.view.stride = (size_t)src.width;
  owned.view.channels = 1;
  return owned;
}

OwnedImage OceanBackend::applyCLAHE(const Image &, double, int) {
  // Ocean does not provide CLAHE; front-ends must check supports(Op::CLAHE)
  // and fall back to global histogram equalization
  throw std::runtime_error("vision: backend 'ocean' does not implement CLAHE");
}

Pyramid OceanBackend::buildPyramid(const Image &src, int levels, int win_size) {
  (void)win_size;
  if (!src.valid() || src.channels != 1)
    throw std::runtime_error("vision: ocean backend requires a valid "
                             "single-channel input image");

  // Wrap the image data in an Ocean frame (we copy so the pyramid owns
  // tightly packed memory with known padding)
  FrameType frame_type((unsigned int)src.width, (unsigned int)src.height,
                       FrameType::FORMAT_Y8, FrameType::ORIGIN_UPPER_LEFT);
  Frame frame(frame_type, src.data, Frame::CM_COPY_REMOVE_PADDING_LAYOUT);

  // The 5x5 Gaussian downsampling places the filter center on even pixel
  // locations, so layer L index i corresponds to full-resolution position
  // 2^L * i -- the same alignment convention as OpenCV's buildOpticalFlowPyramid
  // (and the one our tracker assumes). The default 2x2 box filter would be
  // offset by half a pixel per level.
  auto pyramid = std::make_shared<FramePyramid>(
      frame, FramePyramid::DM_FILTER_14641, (unsigned int)levels,
      true /*copyFirstLayer*/, nullptr);

  Pyramid pyr;
  pyr.native = pyramid;
  pyr.native_backend = name();
  pyr.levels.reserve(pyramid->layers());
  for (unsigned int layer_index = 0; layer_index < pyramid->layers();
       ++layer_index) {
    const Frame &layer = pyramid->layer(layer_index);
    Image image;
    image.data = layer.constdata<uint8_t>();
    image.width = (int)layer.width();
    image.height = (int)layer.height();
    image.stride = (size_t)layer.strideElements();
    image.channels = 1;
    pyr.levels.push_back(image);
  }
  return pyr;
}

std::vector<Keypoint> OceanBackend::detect(const Image &src, const Image &mask,
                                           int threshold,
                                           bool nonmax_suppression) {
  (void)nonmax_suppression; // Ocean FAST always applies non-maximum suppression
  std::vector<Keypoint> out;
  if (!src.valid() || src.width < 9 || src.height < 9)
    return out;

  FrameType frame_type((unsigned int)src.width, (unsigned int)src.height,
                       FrameType::FORMAT_Y8, FrameType::ORIGIN_UPPER_LEFT);
  Frame frame(frame_type, src.data, Frame::CM_COPY_REMOVE_PADDING_LAYOUT);

  // Note: the subregion variant of the detector is used so that callers can
  // run grid-based detection; Ocean scores features internally
  Ocean::CV::Detector::FASTFeatures features;
  if (!Ocean::CV::Detector::FASTFeatureDetector::Comfort::detectFeatures(
          frame, 0u, 0u, (unsigned int)src.width, (unsigned int)src.height,
          (unsigned int)std::max(1, threshold), false /*frameIsUndistorted*/,
          true /*preciseScoring*/, features))
    return out;

  out.reserve(features.size());
  for (const auto &feature : features) {
    const float x = feature.observation().x();
    const float y = feature.observation().y();
    if (mask.valid()) {
      int mx = (int)x;
      int my = (int)y;
      if (mx < 0 || mx >= mask.width || my < 0 || my >= mask.height)
        continue;
      if (mask.data[(size_t)my * mask.stride + (size_t)mx] > 127)
        continue;
    }
    out.push_back(Keypoint());
    out.back().x = x;
    out.back().y = y;
    out.back().response = feature.strength();
  }
  return out;
}

bool OceanBackend::refineSubpix(const Image &, std::vector<cv::Point2f> &, int,
                                int, double) {
  // Ocean does not provide a cornerSubPix equivalent
  return false;
}

std::shared_ptr<const FramePyramid>
OceanBackend::nativePyramid(const Pyramid &pyr) {
  if (pyr.native_backend == "ocean" && pyr.native)
    return std::static_pointer_cast<const FramePyramid>(pyr.native);

  // Mixed-backend use: build a native pyramid from the first level; this
  // copies the base layer and downsamples, so it is valid but slower
  if (pyr.levels.empty() || !pyr.levels.front().valid())
    return nullptr;
  const Image &base = pyr.levels.front();
  if (base.channels != 1)
    return nullptr;
  FrameType frame_type((unsigned int)base.width, (unsigned int)base.height,
                       FrameType::FORMAT_Y8, FrameType::ORIGIN_UPPER_LEFT);
  Frame frame(frame_type, base.data, Frame::CM_COPY_REMOVE_PADDING_LAYOUT);
  return std::make_shared<FramePyramid>(frame, 5u, true);
}

void OceanBackend::track(const Pyramid &prev, const Pyramid &curr,
                         const std::vector<cv::Point2f> &pts_prev,
                         std::vector<cv::Point2f> &pts_next,
                         std::vector<uchar> &status, int win_size,
                         int max_levels, int max_iters, double eps) {
  (void)max_iters; // the SSD tracker is single-pass coarse-to-fine
  (void)eps;

  status.assign(pts_prev.size(), 0);
  // The caller may pre-fill pts_next with initial guesses (mirroring
  // OPTFLOW_USE_INITIAL_FLOW semantics); capture them before clearing
  std::vector<cv::Point2f> init_guess = pts_next;
  pts_next.assign(pts_prev.size(), cv::Point2f());
  if (pts_prev.empty())
    return;
  if (init_guess.size() != pts_prev.size())
    init_guess = pts_prev;

  auto prev_pyr = nativePyramid(prev);
  auto curr_pyr = nativePyramid(curr);
  if (!prev_pyr || !curr_pyr || prev_pyr->layers() == 0 ||
      curr_pyr->layers() == 0)
    return;

  // Odd patch size for the SSD comparison window
  int patch = std::max(7, win_size);
  if (patch % 2 == 0)
    patch += 1;
  const int half = patch / 2;
  const int radius = std::max(2, patch / 3);

  const unsigned int layers =
      std::min((unsigned int)std::max(1, max_levels + 1),
               std::min(prev_pyr->layers(), curr_pyr->layers()));

  for (size_t i = 0; i < pts_prev.size(); ++i) {
    const double px = pts_prev[i].x;
    const double py = pts_prev[i].y;
    // The caller pre-fills pts_next with initial guesses (mirroring
    // OPTFLOW_USE_INITIAL_FLOW semantics)
    double disp_x = init_guess[i].x - px;
    double disp_y = init_guess[i].y - py;

    bool ok = true;
    for (int level = (int)layers - 1; level >= 0 && ok; --level) {
      const Frame &f0 = prev_pyr->layer((unsigned int)level);
      const Frame &f1 = curr_pyr->layer((unsigned int)level);
      const double scale = (double)(1 << level);
      const int Tx = (int)std::floor(px / scale);
      const int Ty = (int)std::floor(py / scale);

      // The template patch must lie fully inside the previous frame; if it
      // does not fit at this (coarse) level, simply skip the level instead of
      // dropping the point entirely
      if (Tx < (int)half || Ty < (int)half || Tx >= (int)f0.width() - (int)half ||
          Ty >= (int)f0.height() - (int)half)
        continue;

      // Search window around the predicted position in the current frame
      const double pred_x = (px + disp_x) / scale;
      const double pred_y = (py + disp_y) / scale;
      const int x0 = std::max((int)half, (int)std::floor(pred_x) - radius);
      const int x1 = std::min((int)f1.width() - 1 - (int)half,
                              (int)std::floor(pred_x) + radius);
      const int y0 = std::max((int)half, (int)std::floor(pred_y) - radius);
      const int y1 = std::min((int)f1.height() - 1 - (int)half,
                              (int)std::floor(pred_y) + radius);
      if (x0 > x1 || y0 > y1) {
        ok = false;
        break;
      }

      // Two-pass search: find the best SSD value, then take the candidate
      // position closest to the prediction among all candidates within a
      // small tolerance of the minimum. A plain first/last-minimum scan is
      // not usable here: on low-texture patches the SSD landscape is nearly
      // flat, so any fixed tie-breaking order latches onto the corner of the
      // search window, and the kick compounds across pyramid levels.
      uint32_t best = 0xFFFFFFFFu;
      for (int yy = y0; yy <= y1; ++yy) {
        for (int xx = x0; xx <= x1; ++xx) {
          const uint32_t value = ssdPatchDispatch(f0, f1, patch, (float)Tx,
                                                  (float)Ty, (float)xx, (float)yy);
          if (value < best)
            best = value;
        }
      }
      const uint32_t tolerance = best / 16u + 4u; // ~6% + quantization slack
      const int pred_ix = std::min(x1, std::max(x0, (int)std::floor(pred_x)));
      const int pred_iy = std::min(y1, std::max(y0, (int)std::floor(pred_y)));
      double pred_dist = std::numeric_limits<double>::max();
      int bx = pred_ix;
      int by = pred_iy;
      for (int yy = y0; yy <= y1; ++yy) {
        for (int xx = x0; xx <= x1; ++xx) {
          const uint32_t value = ssdPatchDispatch(f0, f1, patch, (float)Tx,
                                                  (float)Ty, (float)xx, (float)yy);
          if (value <= best + tolerance) {
            const double dist = std::pow(xx - pred_x, 2) + std::pow(yy - pred_y, 2);
            if (dist < pred_dist) {
              pred_dist = dist;
              bx = xx;
              by = yy;
            }
          }
        }
      }

      // Sub-pixel refinement on the finest level via parabola fitting
      double offset_x = 0.0;
      double offset_y = 0.0;
      if (level == 0) {
        const uint32_t center = ssdPatchDispatch(f0, f1, patch, (float)Tx,
                                                 (float)Ty, (float)bx, (float)by);
        if (bx - 1 >= (int)half && bx + 1 < (int)f1.width() - (int)half) {
          const uint32_t left = ssdPatchDispatch(f0, f1, patch, (float)Tx,
                                                 (float)Ty, (float)(bx - 1), (float)by);
          const uint32_t right = ssdPatchDispatch(f0, f1, patch, (float)Tx,
                                                  (float)Ty, (float)(bx + 1), (float)by);
          const double denom = (double)left - 2.0 * (double)center +
                               (double)right;
          if (std::abs(denom) > 1e-9)
            offset_x = std::max(
                -0.5, std::min(0.5, 0.5 * ((double)left - (double)right) / denom));
        }
        if (by - 1 >= (int)half && by + 1 < (int)f1.height() - (int)half) {
          const uint32_t top = ssdPatchDispatch(f0, f1, patch, (float)Tx,
                                                (float)Ty, (float)bx, (float)(by - 1));
          const uint32_t bottom = ssdPatchDispatch(f0, f1, patch, (float)Tx,
                                                   (float)Ty, (float)bx, (float)(by + 1));
          const double denom = (double)top - 2.0 * (double)center +
                               (double)bottom;
          if (std::abs(denom) > 1e-9)
            offset_y = std::max(
                -0.5, std::min(0.5, 0.5 * ((double)top - (double)bottom) / denom));
        }
      }

      // Measure the displacement relative to the integer template anchor
      // (which is unbiased) rather than reporting the absolute matched
      // position: reporting the absolute position would silently drop the
      // fractional part of the tracked point every frame (the template is
      // sampled at floor(p)), biasing tracks by ~-0.5 px per frame
      disp_x = ((double)bx + offset_x - (double)Tx) * scale;
      disp_y = ((double)by + offset_y - (double)Ty) * scale;
    }

    status[i] = ok ? 1 : 0;
    pts_next[i] = cv::Point2f((float)(px + disp_x), (float)(py + disp_y));
  }
}

void OceanBackend::reject(const std::vector<cv::Point2f> &pts0_n,
                          const std::vector<cv::Point2f> &pts1_n,
                          double focal, double threshold, double confidence,
                          std::vector<uchar> &inliers) {
  (void)confidence; // Ocean's RANSAC uses its own iteration scheme
  const size_t size = std::min(pts0_n.size(), pts1_n.size());
  inliers.assign(size, 0);
  if (size < 8)
    return;

  std::vector<Ocean::Vector2> left(size);
  std::vector<Ocean::Vector2> right(size);
  for (size_t i = 0; i < size; ++i) {
    left[i] = Ocean::Vector2(pts0_n[i].x, pts0_n[i].y);
    right[i] = Ocean::Vector2(pts1_n[i].x, pts1_n[i].y);
  }

  Ocean::RandomGenerator random_generator;
  Ocean::SquareMatrix3 right_F_left(false /*setInvalid*/);
  Ocean::Indices32 used_indices;
  if (!Ocean::Geometry::RANSAC::fundamentalMatrix(
          left.data(), right.data(), size, random_generator, right_F_left,
          8u /*testCandidates*/, 200u /*iterations*/, Ocean::Scalar(0.001),
          &used_indices)) {
    return;
  }

  // Inlier test using the Sampson distance in normalized coordinates, scaled
  // to pixels via the focal length (same metric as the OpenCV backend)
  const double normalized_threshold = threshold / focal;
  const double f11 = right_F_left(0u, 0u), f12 = right_F_left(0u, 1u),
               f13 = right_F_left(0u, 2u);
  const double f21 = right_F_left(1u, 0u), f22 = right_F_left(1u, 1u),
               f23 = right_F_left(1u, 2u);
  const double f31 = right_F_left(2u, 0u), f32 = right_F_left(2u, 1u),
               f33 = right_F_left(2u, 2u);
  for (size_t i = 0; i < size; ++i) {
    const double x0 = left[i].x(), y0 = left[i].y();
    const double x1 = right[i].x(), y1 = right[i].y();
    const double fx0x1 = x1 * (f11 * x0 + f12 * y0 + f13) +
                         y1 * (f21 * x0 + f22 * y0 + f23) +
                         (f31 * x0 + f32 * y0 + f33);
    const double fx0 = f11 * x0 + f12 * y0 + f13;
    const double fy0 = f21 * x0 + f22 * y0 + f23;
    const double fx1 = f11 * x1 + f21 * y1 + f31;
    const double fy1 = f12 * x1 + f22 * y1 + f32;
    const double denom = fx0 * fx0 + fy0 * fy0 + fx1 * fx1 + fy1 * fy1;
    const double distance = denom > 1e-12
                                ? std::abs(fx0x1) / std::sqrt(denom)
                                : std::abs(fx0x1);
    inliers[i] = distance <= normalized_threshold ? 1 : 0;
  }
}

#endif // OV_HAVE_OCEAN
