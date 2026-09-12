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

#ifndef OV_CORE_VISION_TYPES_H
#define OV_CORE_VISION_TYPES_H

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace ov_core {
namespace vision {

/**
 * @brief Non-owning view of an 8-bit image.
 *
 * This is the currency type of the vision abstraction layer. It decouples the
 * tracking algorithms from the concrete image representation of the backend
 * (e.g. cv::Mat for OpenCV, ocean::Frame for Ocean). Constructing a view from
 * either representation is zero-copy.
 */
struct Image {
  /// Pointer to the first pixel element (row-major, interleaved channels)
  const uint8_t *data = nullptr;
  /// Width in pixels
  int width = 0;
  /// Height in pixels
  int height = 0;
  /// Number of bytes between two consecutive rows
  size_t stride = 0;
  /// Number of channels per pixel (1 for grayscale, which is what all current
  /// front-end operations expect)
  int channels = 1;

  bool valid() const { return data != nullptr && width > 0 && height > 0; }

  /// Zero-copy view of an OpenCV matrix. The matrix must outlive the view.
  static Image fromCv(const cv::Mat &mat) {
    Image image;
    image.data = mat.ptr<uint8_t>();
    image.width = mat.cols;
    image.height = mat.rows;
    image.stride = mat.step[0];
    image.channels = mat.channels();
    return image;
  }

  /// Zero-copy OpenCV header for this view. The underlying memory is shared,
  /// the returned matrix must not outlive the viewed buffer.
  cv::Mat toCv() const {
    // cv::Mat does not have a const-aware interface, the view is treated as
    // immutable by convention
    return cv::Mat(height, width, CV_MAKETYPE(CV_8U, channels),
                   const_cast<uint8_t *>(data),
                   static_cast<size_t>(stride));
  }
};

/// Owning 8-bit image buffer together with a view into it. Operations that
/// produce images return this so that the caller controls the lifetime.
struct OwnedImage {
  std::shared_ptr<std::vector<uint8_t>> storage;
  Image view;
};

/// A single detected feature location.
struct Keypoint {
  float x = 0.0f;
  float y = 0.0f;
  /// Detector strength / response (used for ranking, larger is better)
  float response = 0.0f;
  /// Diameter of the meaningful neighborhood around the keypoint
  float size = 7.0f;
  /// Pyramid level the keypoint was detected at
  int octave = 0;
};

/**
 * @brief Multi-level image pyramid (level 0 = full resolution).
 *
 * The pyramid owns its memory through the storage pointers so that front-ends
 * can keep it alive between frames. Additionally a backend may attach an
 * opaque native handle (e.g. derivative planes needed for tracking) which its
 * own operations can consume directly and without conversions. Consumers
 * other than the creating backend can rely on:
 *  - levels[0] being the full-resolution 8-bit (grayscale) image, and
 *  - levels containing the halved pyramid images in increasing level order.
 * Backends that need additional per-level data (e.g. derivative images)
 * attach it to the native handle.
 */
struct Pyramid {
  /// Views of each level (level 0 = full resolution). Views are backed by
  /// storage (for generic pyramids) or by the native handle (for backend
  /// pyramids).
  std::vector<Image> levels;

  /// Owned memory backing the levels (may be empty when native is used)
  std::vector<std::shared_ptr<std::vector<uint8_t>>> storage;

  /// Opaque native pyramid object of the creating backend (may be null)
  std::shared_ptr<void> native;

  /// Name of the backend that created the native handle
  std::string native_backend;

  bool empty() const { return levels.empty(); }
};

} // namespace vision
} // namespace ov_core

#endif // OV_CORE_VISION_TYPES_H
