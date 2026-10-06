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

#ifndef OV_SRVINS_SYSTEM_H
#define OV_SRVINS_SYSTEM_H

#include <memory>

namespace ov_srvins {

class SqrtEstimator;
class Frontend;
class InertialInitializer;
class InitRunner;
class VinsOptions;
class BackendSystem;

/**
 * @brief Bundle of the fully wired pipeline components.
 *
 * System::create() performs the canonical construction/wiring sequence
 * (estimator, frontend on the estimator's state, ZUPT updater on the
 * tracker's feature database, initializer on the same database and
 * propagator, init runner over all three) so every driver — C++, Python,
 * ROS — gets identical, correct-by-construction wiring.
 */
struct System {
  std::shared_ptr<SqrtEstimator> estimator;
  std::shared_ptr<Frontend> frontend;
  std::shared_ptr<InertialInitializer> initializer;
  std::shared_ptr<InitRunner> init_runner;

  /// Bundle-adjustment backend (null unless
  /// VinsOptions::backend_options.enabled). Phase 1: offline BA recorder.
  std::shared_ptr<BackendSystem> backend;

  /// Construct and wire all components from the system parameters
  static System create(const VinsOptions &params);

private:
  /// Only create() can build a System: the components are useless (null)
  /// unwired, so a default-constructed System would be a bug
  System() = default;
};

} // namespace ov_srvins

#endif // OV_SRVINS_SYSTEM_H
