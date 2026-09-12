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

#include "vision/CVBackend.h"

#include <stdexcept>

#include "vision/OpenCvBackend.h"
#include "utils/print.h"

#ifdef OV_HAVE_OCEAN
#include "vision/OceanBackend.h"
#endif

using namespace ov_core;
using namespace ov_core::vision;

std::shared_ptr<CVBackend> CVBackend::create(const std::string &name) {
  if (name == "opencv")
    return std::make_shared<OpenCvBackend>();
#ifdef OV_HAVE_OCEAN
  if (name == "ocean")
    return std::make_shared<OceanBackend>();
#endif
  std::string message = "vision: unknown or unavailable CV backend '" + name +
                        "' (available: opencv"
#ifdef OV_HAVE_OCEAN
                        ", ocean"
#endif
                        ")";
  throw std::runtime_error(message);
}
