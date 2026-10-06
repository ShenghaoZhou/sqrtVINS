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

#include "System.h"

#include "Frontend.h"
#include "InitRunner.h"
#include "SqrtEstimator.h"
#include "VinsOptions.h"
#include "initializer/InertialInitializer.h"
#include "track/TrackBase.h"
#ifdef SQRTVINS_BACKEND
#include "backend/BackendSystem.h"
#endif

using namespace ov_srvins;

System System::create(const VinsOptions &params) {
  System sys;
  sys.estimator = std::make_shared<SqrtEstimator>(params);
  sys.frontend = std::make_shared<Frontend>(params, sys.estimator->get_state());

  // Wire the tracker database into the ZUPT updater (disparity check)
  auto db = sys.frontend->get_trackFEATS()->get_feature_database();
  sys.estimator->set_zupt_database(db);

  sys.initializer = std::make_shared<InertialInitializer>(
      params.init_options, db, sys.estimator->get_propagator(),
      params.msckf_options, params.slam_options, params.featinit_options);
  sys.init_runner = std::make_shared<InitRunner>(params, sys.estimator,
                                                 sys.frontend, sys.initializer);
#ifdef SQRTVINS_BACKEND
  if (params.backend_options.enabled) {
    sys.backend = std::make_shared<BackendSystem>(
          params.backend_options, params.imu_noises, params.gravity_mag);
  }
#endif
  return sys;
}
