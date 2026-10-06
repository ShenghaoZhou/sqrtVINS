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

#ifndef OV_SRVINS_BACKENDOPTIONS_H
#define OV_SRVINS_BACKENDOPTIONS_H

#include <cstddef>
#include <string>

namespace ov_srvins {

/**
 * @brief Options for the colmap/Ceres bundle-adjustment backend.
 *
 * Phase 1 scope: offline (post-run) vision-only BA. The recorder snapshots
 * keyframe poses and feature observations during the filter run; after the
 * run, a colmap::Reconstruction is assembled and refined with Ceres.
 */
struct BackendOptions {

  /// Master switch (parsed as "backend_enabled")
  bool enabled = false;

  /// Record a keyframe every Nth visually-updated frame (1 = every frame).
  /// Decimates the offline BA problem size; observations are still taken
  /// from all tracks alive at the keyframe time.
  int keyframe_stride = 5;

  /// Minimum number of keyframe observations for a track to enter BA
  size_t min_track_length = 3;

  /// Triangulated points whose mean reprojection error under the filter
  /// poses exceeds this [px] are rejected before BA (kills degenerate
  /// triangulations that would poison the problem and the error metrics)
  double max_triang_error_px = 10.0;

  /// Points with a post-BA reprojection error above this threshold [px] are
  /// pruned before the (optional) second solve
  double max_reproj_error_px = 5.0;

  /// Run a second BA solve after outlier pruning
  bool refine_after_pruning = true;

  /// Ceres solver settings
  int max_num_iterations = 100;
  int num_threads = 4;

  /// Cauchy robust loss scale [px] on reprojection residuals (0 = disabled)
  double loss_scale = 1.0;

  /// Print the Ceres/colmap solver summary to stdout
  bool print_summary = true;
};

} // namespace ov_srvins

#endif // OV_SRVINS_BACKENDOPTIONS_H
