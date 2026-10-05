/*
 * Sqrt-VINS: A Sqrt-filter-based Visual-Inertial Navigation System
 * Copyright (C) 2025-2026 Yuxiang Peng
 * Copyright (C) 2025-2026 Chuchu Chen
 * Copyright (C) 2025-2026 Kejian Wu
 * Copyright (C) 2018-2026 Guoquan Huang
 * Copyright (C) 2018-2023 OpenVINS Contributors
 * Copyright (C) 2018-2023 Patrick Geneva
 * Copyright (C) 2018-2019 Kevin Eckenhoff
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

#ifndef OV_SRVINS_SQRTESTIMATOR_H
#define OV_SRVINS_SQRTESTIMATOR_H

#include <memory>
#include <vector>
#include "VinsOptions.h"
#include "utils/sensor_data.h"

namespace ov_core {
class Feature;
}

namespace ov_srvins {

class State;
class Propagator;
class UpdaterZeroVelocity;

/**
 * @brief Core class that handles the Square-Root EKF estimation logic.
 *
 * This class encapsulates the state estimation process, including IMU propagation,
 * zero-velocity updates, and visual feature updates (MSCKF and SLAM).
 */
class SqrtEstimator {
public:
  /**
   * @brief Constructor
   * @param params_ System parameters
   */
  SqrtEstimator(const VinsOptions &params_);

  /**
   * @brief Feed IMU data to the estimator (propagator and ZUPT)
   * @param message IMU data
   * @param oldest_time Oldest time to keep in buffers
   */
  void feed_imu(const ov_core::ImuData &message, double oldest_time);

  /**
   * @brief Feed a single IMU measurement, computing the buffer trim time
   * internally (mirrors VioManager::feed_measurement_imu)
   * @param message IMU data
   */
  void feed_measurement_imu(const ov_core::ImuData &message);

  /**
   * @brief Feed a batch of IMU measurements in timestamp order
   *
   * Equivalent to calling feed_measurement_imu() on each message (the
   * pre-init rolling-window trim must be computed per message), but crosses
   * the language/API boundary once for the whole batch.
   * @param messages IMU data sorted by timestamp
   */
  void feed_imu_batch(const std::vector<ov_core::ImuData> &messages);

  /**
   * @brief Try to perform a zero-velocity update
   *
   * Motion bookkeeping is internal: when zupt_only_at_beginning is enabled,
   * ZUPT is refused once motion has been observed (see notify_moved()).
   * Callers that drive propagate()/update() manually (instead of
   * process_frame()) must call notify_moved() after each visual update.
   * @param timestamp Target timestamp
   * @return True if a ZUPT update was performed
   */
  bool try_zupt(double timestamp);

  /// Notify the estimator that motion has occurred since the last ZUPT
  /// (called by the pipeline after initialization and after each visual
  /// update, and by manual pipeline drivers after their own updates)
  void notify_moved() { has_moved_since_zupt_ = true; }

  /// Whether motion has been observed since startup
  bool has_moved_since_zupt() const { return has_moved_since_zupt_; }

  /**
   * @brief Propagate the state forward and add a new clone
   * @param timestamp Target timestamp
   * @return True if propagation was successful
   */
  bool propagate(double timestamp);

  /**
   * @brief Perform the full state update with visual features
   * @param message Camera data
   * @param featsup_MSCKF MSCKF features
   * @param feats_slam SLAM features (separated into update and delayed-init
   * sets internally, after marginalize_slam - matching VioManager ordering)
   */
  void update(std::vector<std::shared_ptr<ov_core::Feature>> &featsup_MSCKF,
              std::vector<std::shared_ptr<ov_core::Feature>> &feats_slam);

  /**
   * @brief Set the feature database for ZUPT updates
   * @param db Feature database from trackers
   */
  void set_zupt_database(std::shared_ptr<ov_core::FeatureDatabase> db);

  /// Accessor for the state
  std::shared_ptr<State> get_state() { return state; }

  /// Accessor for the propagator
  std::shared_ptr<Propagator> get_propagator() { return propagator; }

  /// Accessor for the ZUPT updater
  std::shared_ptr<UpdaterZeroVelocity> get_updater_zupt() { return updaterZUPT; }

private:
  /// Async initialization commit machinery (InitRunner only)
  friend class InitRunner;

  /**
   * @brief Swap in a different state object (used to commit an asynchronously
   * initialized shadow state). Callers holding a state pointer must re-fetch
   * via get_state(); main thread only.
   */
  void swap_state(std::shared_ptr<State> new_state) { state = new_state; }

  /**
   * @brief Pin the pre-init IMU trim floor: while pinned, the propagator's
   * buffer retains IMU data back to oldest_time even if the rolling pre-init
   * window would discard it. Used while an async initialization solve is in
   * flight so the post-commit catch-up propagation never starves.
   */
  void pin_imu_trim(double oldest_time) { trim_floor_pin_ = oldest_time; }

  /// Release a previously pinned trim floor
  void unpin_imu_trim() { trim_floor_pin_ = -1; }

  /// Perform marginalization of old states and features
  void handle_marginalization();

  /**
   * @brief Compute the buffer trim time (oldest IMU to keep) for a feeding
   * message at the given timestamp.
   *
   * Pre-init: keep only a rolling init_window_time + 0.1 s window (the static
   * initializer averages IMU over [buffer_oldest, last_static_timestamp], so
   * an untrimmed buffer would pull pre-static motion into the bias solution),
   * clamped by the pinned trim floor while an async init solve is in flight.
   * Post-init: keep everything back to the oldest clone (margtimestep).
   */
  double compute_oldest_imu_time(double timestamp) const;

  /// Manager parameters
  VinsOptions params;

  /// Our master state object
  std::shared_ptr<State> state;

  /// Propagator of our state
  std::shared_ptr<Propagator> propagator;

  /// Our zero velocity tracker
  std::shared_ptr<UpdaterZeroVelocity> updaterZUPT;

  /// ZUPT bookkeeping: set once motion is observed (init with velocity, or a
  /// visual update); used when zupt_only_at_beginning is enabled
  bool has_moved_since_zupt_ = false;

  /// Pinned pre-init IMU trim floor (-1 when not pinned); see pin_imu_trim
  double trim_floor_pin_ = -1;
};

} // namespace ov_srvins

#endif // OV_SRVINS_SQRTESTIMATOR_H
