# Bundle-Adjustment Backend

A Ceres-based bundle-adjustment backend for the SqrtVINS frontend. The
sqrt-filter remains the real-time estimator; this backend builds an
independent colmap `Reconstruction` and solves it with Ceres, then returns
results to the filter as soft measurements.

Enabled by the `SQRTVINS_BACKEND` build define (`cmake/NO_ROS.cmake`), which
also adds the sources listed in `ov_srvins/src/backend/`.

## Model mapping

| colmap concept | VIO meaning |
|---|---|
| `Rig` | the IMU reference sensor (sensor `0`) |
| `Sensor` | one camera, `sensor_from_rig` = calibrated `T_cam_imu` |
| `Frame` | one keyframe = one filter clone |
| `Image` | that keyframe's observations in one camera |
| `Point3D` | one feature track |

Poses use colmap's 7-param block `[qx qy qz qw tx ty tz]` = `R_GtoI` directly,
with `t = -R_GtoI * p_IinG`, so `p_IinG = -Rᵀt`. Gravity is `(0,0,+9.81)`.

`ColmapMapAdapter` builds the reconstruction from the filter's
`FeatureDatabase` and pose snapshots; `BackendSystem` owns recording,
solving, and exporting.

## Phases

| Phase | Commit | What it adds |
|---|---|---|
| 1 — offline vision-only BA | `379fd7e` | vendored colmap subset + triangulation + reproj BA |
| 2 — offline VIO-BA | `19a3588` | on-manifold IMU preintegration + 15-dim whitened `ImuFactor` |
| 2b — online windowed BA | `7fd5fe9` | fixed-lag sliding-window solve thread |
| 3a — pose feedback | `97f4357` | soft pose measurements back into the filter |
| 3c — loop hooks | `97f4357` | relative-pose factor + constraint injection API |
| 3b — landmark feedback | — | **decided: not implemented** (see below) |

### Phase 2 — IMU factors

`ImuPreintegration.{h,cpp}` does Forster on-manifold mid-point preintegration;
the error state is `[θ, p, v, bg, ba]`. `ImuFactor.h` is a 15-dim whitened
autodiff residual on the colmap 7-blocks plus a per-keyframe
`[v, bg, ba]` 9-block; `ExpSO3`/`LogSO3` are Jet-templated so Ceres can
differentiate through them.

The Ceres `Problem` is assembled in the `CeresBundleAdjuster` constructor, so
factors must be added **before** `Solve()` — and re-added after the adjuster is
recreated on the prune-and-re-solve path.

### Phase 2b — online window

`BackendSystem::online_worker` runs a fixed-lag window: `backend_window_size`
keyframes (default 15), one solve every `backend_window_solve_stride` keyframes
(default 2), oldest window keyframe held constant as the gauge, and
`backend_window_max_iterations` / `backend_window_max_solver_time` budgets
(15 iterations / 50 ms). Refined poses are published through
`get_refined_poses()` keyed by keyframe camera timestamps.

### Phase 3a — pose feedback

`update/UpdaterBackend.{h,cpp}` writes soft 6-dof pose measurements onto the
clones matched to the refined window poses. The backend's global frame is the
filter's global frame (the window is anchored to the oldest window keyframe,
held constant), so no alignment step is needed. The newest clone is always
skipped — its pose is a copy of the current IMU pose, which this update does
not touch.

Feedback sigmas are deliberately conservative: the window's information
overlaps the filter's own visual measurements, so a tight BA covariance would
double count.

### Phase 3c — loop-closure hooks

`backend/RelativePoseFactor.h` is a 6-dof whitened relative-pose factor on two
colmap 7-blocks (`r_θ = Log(R_measᵀ · R_j · R_iᵀ)`,
`r_p = R_j (p_j − p_i) − p_meas`), matching the `ImuFactor` convention.
`BackendSystem::add_loop_constraint(ts_i, ts_j, R_ItoJ, p_JinI, cov6)` records
a constraint and `inject_loop_factors` adds it to every solve in which both
endpoints are present in the current frame set.

**Retrieval is intentionally out of scope.** The vendored colmap excludes
feature-extraction and retrieval modules, and the frontend tracks no
descriptors. An external loop module is expected to supply the
`(i, j, R, p, cov)` tuples. `add_loop_constraint` is thread-safe.

## Phase 3b — landmark feedback: decision not to implement

Recorded 2026-08-06 as a deliberate skip, not an unfinished item.

**Why it looked needed.** The backend triangulates better 3D points than the
filter, so feeding them back should refine the map. The naive version is a copy
of 3a: measure the backend's `Point3D` against the matching SLAM feature.

**Why the naive version is ill-posed.** `feat_rep_slam` is
`ANCHORED_MSCKF_INVERSE_DEPTH` (see `config/euroc_mav/estimator_config_srvins_backend.yaml`),
so the SLAM state variable is **not** a global position. It is the 3-vector
`[u, v, ρ] = [x/z, y/z, 1/z]` in the **anchor camera frame**
(`ov_core/src/types/Landmark.cpp`). The global position is only ever derived,
never stored (`UpdaterHelper.cpp`):

```cpp
p_FinG = cam_clone.R_GtoC.transpose() * feature.p_FinA + cam_clone.p_CinG;
```

Writing an absolute world position onto that variable with an identity
Jacobian is wrong three ways at once: the variable is not in the global frame;
the true Jacobian is a 3×9 block over `(p_FinA, θ_anchor, p_anchor)` plus
calibration; and the map is rational in `ρ`, which diverges as `z → 0`. It can
be made well-posed by first transforming the backend point into the anchor
frame and forming the residual directly on `[u, v, ρ]` — but that only fixes
the bookkeeping.

**The real reason to skip.** The anchored representation makes pose and landmark
state **co-move by construction**. 3a feedback writes onto anchor pose variables
only, and `p_FinG` is a function of the anchor pose — so every landmark is
re-localized implicitly when its anchor moves. There is nothing to
desynchronize.

Re-anchoring preserves this. When a landmark's anchor is about to leave the
window, `UpdaterSLAM::perform_anchor_change` recomputes the value from **both**
current anchor poses (`UpdaterSLAM.cpp`):

```cpp
Mat3 R_OLDtoNEW = R_GtoNEW * R_GtoOLD.transpose();
Vec3 p_OLDinNEW = R_GtoNEW * (p_OLDinG - p_NEWinG);
new_feat.p_FinA = R_OLDtoNEW * landmark->get_xyz(false) + p_OLDinNEW;
```

`get_buffer_unsafe` reads the feedback-updated estimate, so a 3a correction
carries through the re-anchor without introducing drift. The filter's own
SLAM/MSCKF updates also exchange landmark↔anchor information directly: the
anchor contributes a 3×6 Jacobian `H_anc` (`UpdaterHelper.cpp`), so the filter
is a legitimate closed-loop landmark estimator on its own. 3b would have been a
redundant refinement source, not a missing link.

There is also a gauge argument. The windowed solve is gauged to the oldest
window keyframe, so refined poses and refined points co-move in one frame.
Landmark information residual on the anchor pose is ≈ 0 for trajectory
purposes — the same redundancy 3a already suppresses with inflated sigmas.
Adding 3b would double count the same information twice.

**Association was never the blocker.** `BackendSystem::record_observations`
stores `feat->featid` from the same `FeatureDatabase` that keys
`state->features_SLAM`, so a 1:1 mapping already exists. What would remain is
lifecycle handling: MSCKF-only features have no state variable,
`max_slam` evicts and re-adds features, and anchors are transient.

**What is given up.** All of it is marginal:

1. *Landmark covariance never sees the backend's information.* U factors are
   built from the filter's own reprojections only. The filter carries excess
   uncertainty on landmarks, which under-weights later measurements — a safe
   direction. Additionally `dx` is zero on landmark variables during 3a, so
   the landmark↔anchor cross-covariance is not refreshed when the anchor
   tightens. This is the same partial-update approximation MSCKF and SLAM
   already rely on, so it is not a new inconsistency.
2. *No absorption of the backend's map, only of its trajectory.* ATE is
   unaffected (the pose correction already encodes the points' contribution),
   and exported maps are unaffected (the backend's `Point3D`s are exported
   directly). Only the filter-internal map stays at filter quality, and
   nothing consumes it as a product.
3. *A window-bounded gap.* `inject_loop_factors` only applies a constraint
   whose both endpoints are in the current frame set. Landmarks whose anchors
   are already marginalized into `U` do not move. Consistent, but uncorrected.

**When this should be revisited.** Only with 3c, and scoped to landmarks
straddling a loop seam — the anchors involved are already in `U`, so a value
correction on the landmark itself is what would actually be needed.

**Falsifiers worth running.**

1. Offline VIO-BA with points excluded from the objective vs. included.
   Identical ATE confirms the redundancy empirically rather than by gauge
   argument.
2. On a long sequence, log the `UpdaterSLAM` chi² gate rejection rate with and
   without 3a feedback. If feedback makes the filter's own landmark
   measurements systematically worse-gated, the stale covariance is biting and
   3b becomes worth the cost.

## Configuration

`config/euroc_mav/estimator_config_srvins_backend.yaml` (parsed in
`core/VinsOptions.cpp`):

| Key | Default | Meaning |
|---|---|---|
| `backend_enabled` | `true` | build and run the backend |
| `backend_keyframe_stride` | `5` | record one keyframe every N camera frames |
| `backend_min_track_length` | `3` | minimum observations to triangulate |
| `backend_max_reproj_error` | `5.0` | reprojection error threshold (px) |
| `backend_refine_after_pruning` | `true` | re-solve after degenerate-point pruning |
| `backend_use_imu_factors` | `true` | inject IMU preintegration factors |
| `backend_online_enabled` | `false` | run the fixed-lag window thread |
| `backend_window_size` | `15` | keyframes kept in the window |
| `backend_window_solve_stride` | `2` | solve every N keyframes |
| `backend_window_max_iterations` | `15` | per-solve iteration budget |
| `backend_window_max_solver_time` | `0.05` | per-solve wall-clock budget (s) |
| `backend_feedback_enabled` | `false` | feed refined poses back into the filter |
| `backend_feedback_sigma_pos` | `0.05` | feedback position sigma (m) |
| `backend_feedback_sigma_ori` | `0.02` | feedback orientation sigma (rad) |
| `backend_feedback_gate_chi2` | `50.0` | per-clone chi² gate |
| `backend_max_num_iterations` | `100` | offline solver iterations |
| `backend_num_threads` | `8` | Ceres threads |
| `backend_loss_scale` | `1.0` | loss function scale |
| `backend_print_summary` | `true` | print the `BackendSummary` |

Note the two deliberately-off switches: `backend_online_enabled` and
`backend_feedback_enabled`. Each is validated but disabled by default.

## Results

EuRoC ATE RMSE, Sim(3)-aligned (`eval_backend_ate.py`,
`pixi run python eval_backend_ate.py <gt.csv> <filter_traj> <ba_traj>`):

| Sequence | Filter | + 3a feedback | Offline VIO-BA |
|---|---|---|---|
| V1_01_easy | 0.0933 | **0.0909** | 0.0935 |
| V2_01_easy | 0.1013 | **0.0938** | 0.0886 |

3a improves both sequences. Online windowed BA on V1_01 reaches 0.0937 m
(250 solves, mean 58 ms, max 63 ms against the 2-keyframe cadence). Final
reprojection error is ~0.7–0.9 px.

## Build and test

```bash
cmake --build build -j
./build/ov_srvins/run_euroc            # runner
./build/ov_srvins/test_backend_adapter # synthetic stereo-rig BA
./build/ov_srvins/test_backend_imu     # preintegration + VIO-BA + loop factor
```

`test_backend_imu` covers SO(3) Exp/Log, preintegration accuracy against
analytic integration, bias Jacobians, a synthetic visual-inertial BA recovering
ground truth to <1 mm, and the relative-pose factor's residual conventions.

## Filter-machinery gotchas

Hard-won, and they will bite the next person adding an updater:

1. Call `State::setup_matrix_buffer()` **before** `store_update_factor`.
   `setup_matrix_buffer` sizes the buffer to the full state width but does not
   zero it, and `EigenMatrixBuffer::append_left_rows` aborts if the incoming
   matrix is wider than the current `cols_`. This shipped unnoticed because
   `try_zupt` — the only other caller — is disabled in every config.
2. `get_marginal_U_block` requires `Hx_order` in **ascending variable id**
   order, else it prints and aborts. Clone ids **decrease with timestamp**:
   `StateHelper::clone` inserts a new clone at `kCloneStartId` and shifts
   everything older right, so the layout is
   `[IMU + calib, newest clone … oldest clone, SLAM features]`.
3. The orientation error convention is `R_true = (I − [δθ]×) R̂` — JPL
   left-multiplicative with a **minus** sign — while position is additive
   (`p_true = p_est + δp`). The correct residual/H pairing is therefore
   `res_θ = w · log_so3(R_est · R_baᵀ)` and `res_p = w · (p_ba − p_est)`,
   both with `H = +w·I`. Getting `res_p`'s sign wrong pushes position *away*
   from the measurement (V2_01 went 0.101 → 0.125 m).
4. `get_marginal_U` + `efficient_QR` (the ZUPT non-explicit branch) is only
   valid when the measured variables are a **state prefix**. Use
   `get_marginal_U_block` with `U_dense` / `U_tri` instead.
