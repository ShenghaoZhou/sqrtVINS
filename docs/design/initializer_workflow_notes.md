=== ad8c20a66d2e509f4 ===
All files read. Here is the full analysis.

---

# 1. End-to-end initialization pipeline

Entry: `InertialInitializer::initialize(state, wait_for_jerk)` — `ov_srvins/src/initializer/InertialInitializer.cpp:63`. `state` is a fresh `State` created in `SqrtEstimator` ctor (`ov_srvins/src/core/SqrtEstimator.cpp:45`), whose `State` ctor already allocated `imu`, `calib_dt_CAMtoIMU`, `calib_IMUtoCAM`, `cam_intrinsics` and set `U_ = 1e-3·I` (`ov_srvins/src/state/State.cpp:36-145`).

## Stage A — window & motion decision (InertialInitializer.cpp)
1. **Window gate** (InertialInitializer.cpp:67-81): compute `lastest_cam_time`/`oldest_cam_time` over all feature timestamps in `db_`. Require `lastest_cam_time - oldest_cam_time >= init_window_time` (1.0 s). Output: failure or proceed.
2. `oldest_win_time = lastest_cam_time - init_window_time - init_window_offset` (InertialInitializer.cpp:83-87; `init_window_offset=0.001`); require non-negative.
3. `db_->cleanup_measurements(oldest_win_time)` (InertialInitializer.cpp:91) — drop all measurements older than the window.
4. `db_->update_disp_info()` (InertialInitializer.cpp:97) builds `db_->map_disp[time0][time1] = avg pixel disparity` over all ordered timestamp pairs (`ov_core/src/feat/FeatureDatabase.cpp:470-511`). Then `disparity_check(oldest_win_time, lastest_cam_time, db_, init_max_disparity, disparity_detected_moving)` (`ov_srvins/src/utils/Helper.cpp:38-92`): reads `map_disp[lastest]`, `is_move = avg_disp > init_max_disparity` (1.0 px). **Camera-only (optical flow), no IMU.** Gate: must find at least one valid disparity entry.
5. **Decision** (InertialInitializer.cpp:108-153). `is_still = !disparity_detected_moving`; `has_jerk = is_static_prev_ && !is_still`. With `wait_for_jerk=false` (default driver path, ZUPT on):
   - `is_still` → **StaticInitializer::initialize(state, prev_static_timestamp_)** (case 1)
   - `!is_still && init_dyn_use` → **DynamicInitializer::initialize(state)** (case 2)
   With `wait_for_jerk=true` the two extra cases (3: jerk→static; 4: no-jerk moving→dynamic) and the "do nothing" case 5 apply. Afterwards update `prev_static_timestamp_ = lastest_cam_time` if still else −1, `is_static_prev_ = is_still` (lines 149-153). On success set `state->is_initialized = true` (line 156).

## Stage B — static initialization (`static/StaticInitializer.cpp:53-123`)
Inputs: `imu_data_` (shared propagator buffer), `last_static_timestamp`.
1. Window = IMU readings in `[imu_data[0].timestamp, last_static_timestamp]` (lines 60-71); gate ≥2 (line 74).
2. `a_avg = mean(a)`, `w_avg = mean(w)` over window (lines 81-89).
3. `z_axis = a_avg/|a_avg|`; `gram_schmidt(z_axis, Ro)` builds `R_GtoI = [x y z]` with `z=z_axis` and x,y from cross products with `e_1=(1,0,0)`/`e_2=(0,1,0)` (whichever is least aligned) — `ov_srvins/src/utils/Helper.cpp:164-192`. `q_GtoI = rot_2_quat(Ro)` (line 95).
4. Biases: `bg = w_avg`; `ba = a_avg − R(q_GtoI)·[0,0,gravity_mag]` (lines 101-102).
5. Set `state->timestamp = window.back().timestamp`; IMU 16-vector = `[q_GtoI; p=0; v=0; bg; ba]` (indices 0-3/4-6/7-9/10-12/13-15); set value and fej (lines 105-111). **No clones created.**
6. **Covariance**: `StateHelper::set_initial_imu_square_root_covariance(state, prior_diag)` (line 120) → `state->U_.topLeftCorner(15,15).diagonal() = prior_diag`, where `prior_diag = Ones(15)` scaled `[q:0.017, p:0.05, v:0.01, bg:0.02, ba:0.02]` (lines 114-119; def `ov_srvins/src/state/StateHelper.cpp:43-47`). Rest of `U_` retains constructor value (1e-3·I, all cross terms 0).

## Stage C — dynamic initialization (`dynamic/DynamicInitializer.cpp:64-472`)
Inputs: propagator IMU buffer, full feature db copy, camera extrinsics/intrinsics from options.

0. Gates: `window_length + init_window_offset >= init_window_time`; number of distinct camera timestamps ≥ `init_dyn_num_pose` (5) (lines 93-104).
1. **Keyframing** (lines 474-529): always first+last timestamps; then greedy pick of remaining `init_dyn_num_pose−2` keyframes maximizing *minimal* disparity per candidate, using `db_->map_disp` (line 507).
2. **Rotation-motion gate** (lines 129-154): over IMU readings in `[oldest_cam+calib_camimu_dt, latest_cam+calib_camimu_dt]`, accumulate `theta += |−(wm−bg)·dt|` (mean of `|am−ba|` too). Require `180/π·theta ≥ init_dyn_min_deg` (45°).
3. **Continuous preintegration** `preintegrate(...)` (lines 531-600): per keyframe time (except the oldest, which gets a dummy `CpiV1` with `DT=0, R=I, alpha=0, H_a=I`, lines 543-551), create `CpiV1(sigma_w,sigma_wb,sigma_a,sigma_ab)`, `setLinearizationPoints(bg, ba)` (with `bg=init_dyn_bias_g`, `ba=init_dyn_bias_a`), feed IMU between the two camera+offset times via `select_imu_readings`. If `use_bg_estimator` call `feed_IMU_Jq` else `feed_IMU_meanonly` (lines 587-593). Only **means** (`DT`, `R_k2tau`, `alpha_tau`) and (if bg enabled) **`J_q,H_a,H_b`** are produced — the preintegration *covariance* `P_meas` (full `feed_IMU`, `ov_core/src/cpi/CpiV1.cpp:33-356`) is **never** used in the initializer. Gates: ≥2 IMU readings per keyframe, dt match within 0.01 s.
4. **Gyro-bias estimator** (only if `use_bg_estimator`): build `feat_norms[feat][cam][time]=uv_norm` (lines 173-188), then `solve_bg(...)` (lines 951-1045) — LM on the rotation-only smallest-eigenvalue cost, then re-run preintegrate with the new bg (lines 194-199). See §2 for the math.
5. **Relative camera rotations to I0**: `R_CktoI0[cam][time] = cpi.R_k2tau^T · R_ItoC(cam)^T` (lines 208-221). Feature bearing in I0: `bearings_in_I0[time][cam][feat] = R_CktoI0 · [uv_norm;1]` (lines 239-254).
6. **Translation directions** `find_eigen_vectors` per ordered pair `(time0<time1)` × all cam pairs (lines 602-696): accumulate `H_plane += pn·pn^T/|pn|²` with `pn = f0×f1` (normalized to stop outliers dominating, line 631), `SelfAdjointEigenSolver`, smallest eigenvector `t̂`. Outlier rejection: keep `|n_plane^T·t̂ − median| < 3σ`; gate ≥ `kMinMeas=5` inliers. Output: full 3×3 eigenvector matrix + outlier feat-ids.
7. **Velocity/gravity linear system** `build_linear_system` (lines 698-761): for each pair id, take `eig_vec = eigenvectors.rightCols(2)` (the two directions spanning the null-space of the translation constraint) and append 2 rows:
   - `A[row,0:3] = −(t1−t0)·eig_vec^T`; `A[row,3:6] = ½(t1²−t0²)·eig_vec^T`; `b = eig_vec^T·(alpha_tau1 − alpha_tau0 + R_I0toIk1^T·p_C1inI − R_I0toIk0^T·p_C0inI)` with `p_CinI = −R_ItoC^T·p_IinC` (lines 714-721).
   - `efficient_QR(A,b)`, keep top 6 rows, `x_init = upper-triangular solve` (lines 757-760). 6-vector `[v_I0inI0; g_inI0]`.
8. **Gravity refinement** `refine_gravity` (lines 763-849): 2-DOF gravity `g(α,β)=mag·[cosα·sinβ, sinα·sinβ, cosβ]` (Helper.cpp:556-561), `α=atan2(g_y,g_x)`, `β=acos(g_z/|g|)`. LM up to `init_grav_opt_max_iter` (20), init λ=1.0: `A_new=[A1, A2·J]`, `J=∂g/∂(α,β)` (Helper.cpp:545-554), `r=−(A1 v + A2 g − b)`, `H=A_new^T A_new` with diagonal `·(λ+1)`, `dx=H.llt()·A_new^T r`; LM ρ update, λ decay 0.5, convergence `dx.norm()/x.norm()<1e-6`. Output `[v_I0inI0; g_inI0]`.
9. **Gravity-aligned frame**: `gram_schmidt(g_inI0, R_GtoI0)` (line 334); build `imu_init` 16-vector: `q_GtoI0`, `v_inG = R_GtoI0^T·v_I0inI0`, `bg`, `ba` (lines 337-342). Drop features in `outlier_ids_total` (lines 349-353).
10. **`Solver srf_solver(state, imu_init, features, keyframe_times, ...)`**; its ctor runs `StateHelper::initialize_state(state, x_init, *map_camera_times.begin())` (`Solver.cpp:53`). `solve()` (below). On failure the state is reset: `state = make_shared<State>(...); StateHelper::initialize_state(state, VecX::Zero(15), -1)` (DynamicInitializer.cpp:360-365).
11. **Post-solve cleanup** (lines 420-441): keep only the clone at `state->timestamp`; every other clone is pushed to `state_to_marg_`, `remove_timestamp`, erased, then `StateHelper::marginalize(state)`; if `do_fej` set imu and remaining-clone fej = value.

### `Solver::solve()` (`dynamic/Solver.cpp:56-194`) — the batch SRIEKF optimizer
1. Force `state->options.do_fej = false` and `feat_rep_msckf = ANCHORED_MSCKF_INVERSE_DEPTH` during init; restore after (lines 60-68, 179-180). Rank features by measurement count at keyframe times, keep top `init_max_feat` (50) (lines 70-98).
2. **Propagate & clone** (lines 102-132): for each keyframe time `> state->timestamp`: `propagator_->propagate(state, time)` → `StateHelper::clone(state, imu->pose()->clone())` stored in `clones_IMU[time]`; timeoffset propagation if `do_calib_camera_timeoffset`; then `state->resize_U_to_square()`.
3. **Iterative loop** ≤ `init_dyn_mle_max_iter` (20) (lines 134-157), each iteration:
   - restore full ranked `features_vec_` (retriangulate), `state->clear()`, `setup_matrix_buffer()`, `calculate_clone_poses()`;
   - `UpdaterMSCKF::update(state, feats, msckf_options, feat_init_options, /*is_iterative=*/true, /*require_HUT=*/false)` — triangulates **only features not already in `state->features_MSCKF`** (UpdaterMSCKF.cpp:153-157), Huber-weights res/H (init_ba_huber_th=2.0, UpdaterMSCKF.cpp:258-268), nullspace-projects, **stores whitened Jacobians** into `H_update_/res_update_` and per-feature `LandmarkMsckf::store_msckf_jacobians` (UpdaterMSCKF.cpp:334-371);
   - `StateHelper::iterative_update_llt(state)` (StateHelper.cpp:355-…): SRIEKF batch step using `H^T r + H^T H·x_{k}−x_0` form, `S = UH^T(UH^T)^T + I`, downdate-all → set `xk_minus_x0_` → update-all (StateHelper.cpp:381-403);
   - `UpdaterMSCKF::update_features(state)` propagates `dx` into each MSCKF landmark.
   - Break when `convergence_check()` (dx-norm ratio < `init_ba_dx_converge_thres` OR res ratio < `init_ba_res_converge_thres`, Solver.cpp:196-226) and `features_MSCKF.size() >= init_min_feat` (30).
4. **Final covariance step** (only if `init_dyn_mle_max_iter != 0` and enough features, lines 159-176):
   - `clear()`, `calculate_clone_poses()`, fej if `do_fej`;
   - `select_slam_features(feat_slam_init)` (lines 228-356): drop features lost at latest frame; project each MSCKF feature into the latest frame, build a 4×4 spatial grid, pick up to `init_max_slam` (50) spatially-dispersed features; for anchored reps **re-anchor** them to the latest clone (`anchor_clone_timestamp = state->timestamp`, `set_from_xyz(p_FinC)`);
   - `UpdaterSLAM::delayed_init_from_MSCKF(...)` (UpdaterSLAM.cpp:256-325) → `StateHelper::initialize(state, landmark, Hx_order, H_x, H_f, res, chi2_mult=-1, sigma_pix_inv)` (UpdaterSLAM.cpp:320);
   - `UpdaterMSCKF::update(..., is_iterative=true, require_HUT=true)` — this time builds `HUT = H_x2·[U_dense^T | U_tri^T]`, chi2 gate **skipped** (`if(!is_iterative)`, UpdaterMSCKF.cpp:298), stores the factored sqrt-form `R_sqrt_inv_H_UT_` and `HT_R_inv_res_` (UpdaterMSCKF.cpp:287-332);
   - `StateHelper::initialize_slam_in_U(state)` (StateHelper.cpp:604-653) appends SLAM landmarks to `U_` using the stored init factors;
   - `StateHelper::update_llt(state, /*is_iterative=*/true)` — full sqrt-Kalman batch update + downdate (Solver.cpp:175).
5. Final gate (lines 181-193): `features_MSCKF.size() + features_SLAM.size() >= init_min_feat`, then `state->clear(true)` (drops MSCKF features).

---

# 2. Dependency table

| Step | Dependency | Call site |
|---|---|---|
| Disparity / window logic | pure Eigen + FeatureDatabase stats (no OpenCV, no IMU) | `Helper.cpp:38`, `FeatureDatabase.cpp:470` |
| Static init (mean, gram_schmidt, biases) | pure Eigen closed-form | `StaticInitializer.cpp:81-120`, `Helper.cpp:164` |
| CpiV1 preintegration means / Jq | pure Eigen closed-form (exp-map series + J_r) | `CpiV1.cpp:358` (`feed_IMU_meanonly`), `CpiV1.cpp:449` (`feed_IMU_Jq`); called at `DynamicInitializer.cpp:588-592` |
| Keyframing | pure Eigen (greedy on `map_disp`) | `DynamicInitializer.cpp:474` |
| Translation direction (essential-matrix-style) | Eigen `SelfAdjointEigenSolver` | `DynamicInitializer.cpp:643,681` |
| Rotation-only solver (Kneip smallest-eigenvalue w/ Jacobian) | **reimplemented** in `OpengvHelper.cpp:34-525` (pure Eigen closed-form cubic via `acos`); **no actual opengv lib, no ceres** | `solve_bg`→`get_bg_jacobians` `DynamicInitializer.cpp:1047-1080`; `solve_rotation` `DynamicInitializer.cpp:851-949` (**dead code — never called from `initialize()`**) |
| v/g linear system + solve | pure Eigen + `efficient_QR` (`Helper.h:91,98,104`) | `DynamicInitializer.cpp:698-761` |
| Gravity refinement | iterative LM (Gauss-Newton + damping), hand-written | `DynamicInitializer.cpp:763-849`; helpers `Helper.cpp:545-561` |
| Gyro-bias refinement | iterative LM with **numerical** Jacobian (perturb 1e-6) + analytic smallest-eigenvalue gradient | `DynamicInitializer.cpp:951-1045` |
| Batch optimization | SRIEKF iterative update (Gauss-Newton-style, downdate/update), hand-written; **no ceres** | `StateHelper::iterative_update_llt` `StateHelper.cpp:355`; `StateHelper::update_llt` |
| Feature triangulation (linear, 1-DOF, LM refine) | pure Eigen (colPivHouseholderQr, SVD, HouseholderQR) | `FeatureInitializer.cpp:38,166,313`; called `UpdaterMSCKF.cpp:160-171` |
| Feature Jacobians / Huber / nullspace QR | pure Eigen + `efficient_QR` | `UpdaterMSCKF.cpp:183-372` |
| SLAM delayed init factors | pure Eigen (triangular solve, QR) | `StateHelper::initialize` `StateHelper.cpp:415`, `initialize_invertible` `StateHelper.cpp:532` |
| OpenCV | **none** in the entire initializer path | — (tracking stays Python `cv2` per port scope) |

---

# 3. InertialInitializerOptions — fields, defaults, roles

(`ov_srvins/src/initializer/InertialInitializerOptions.h:59-204`)

| Field | Default | Role |
|---|---|---|
| `init_window_time` | 1.0 | Window length (s) for static/dynamic init |
| `init_max_disparity` | 1.0 | px disparity threshold: below ⇒ "still" |
| `init_max_features` | 50 | requested tracked features (validated ≥15) |
| `init_dyn_use` | false | enable dynamic initializer |
| `init_dyn_mle_max_iter` | 20 | Solver batch iterations (also reuse as iter cap for bg/gravity LM) |
| `init_dyn_num_pose` | 5 | number of keyframes |
| `init_dyn_min_deg` | 45.0 | min rotation (deg) before dynamic init |
| `init_dyn_min_rec_cond` | 1e-15 | (unused in this code path; min reciprocal cond. for covariance recovery) |
| `init_dyn_bias_g` | 0 | initial gyro bias (bg), optimized by `solve_bg` if `use_bg_estimator` |
| `init_dyn_bias_a` | 0 | initial accel bias (ba), **not optimized** |
| `init_window_offset` | 0.001 | shifts cleanup boundary |
| `init_grav_opt_max_iter` | 20 | LM iterations for gravity refinement (also reused for bg LM and solve_rotation) |
| `init_grav_opt_init_lambda` | 1.0 | LM initial λ |
| `init_grav_opt_converge_thres` | 1e-6 | LM relative-`dx` convergence |
| `init_grav_opt_lambda_decay` | 0.5 | LM λ multiply/divide factor |
| `init_prior_q` | 0.1 | sqrt-cov diag for orientation in `initialize_state` (rows 0,1) |
| `init_prior_p` | 0.0 | position prior (0 ⇒ uninformative/rank-deficient U) |
| `init_prior_v` | 0.5 | velocity prior |
| `init_prior_bg` | 0.1 | gyro-bias prior |
| `init_prior_ba` | 0.1 | accel-bias prior |
| `init_prior_t` | 0.001 | timeoffset prior (if calibrating) |
| `init_prior_qc` | 0.02 | extrinsic rotation prior |
| `init_prior_pc` | 0.01 | extrinsic translation prior |
| `init_prior_fc` | 1.0 | intrinsic focal/center prior (4) |
| `init_prior_dc1` | 0.01 | distortion k1,k2 prior (2) |
| `init_prior_dc2` | 1e-5 | distortion p1,p2 prior (2) |
| `init_max_reproj` | 1.0 | px gate for new features in first iterative pass |
| `init_ba_dx_converge_thres` | 1e-6 | Solver dx/x convergence |
| `init_ba_res_converge_thres` | 1e-6 | Solver res ratio convergence |
| `init_min_feat` | 30 | min MSCKF features to accept init |
| `init_max_feat` | 50 | features kept for Solver |
| `init_ba_huber_th` | 2.0 | Huber threshold px (applied in iterative MSCKF update) |
| `init_max_slam` | 50 | SLAM features to promote after batch |
| `record_init_pose` / `record_init_timing` | true / true | log windows (files at `/tmp/...`) |
| `sigma_w, sigma_wb, sigma_a, sigma_ab` (+squares) | 1.6968e-4, 1.9393e-5, 2e-3, 3e-3 | IMU noise for CpiV1 |
| `sigma_pix` | 1 | pixel noise (used via `sigma_pix_inv`) |
| `gravity_mag` | 9.81 | global gravity magnitude |
| `num_cameras` | 1 | camera count |
| `use_stereo` | true | stereo flag |
| `downsample_cameras` | false | halves calibration/resolution |
| `calib_camimu_dt` | 0.0 | cam–imu time offset used in init |
| `use_bg_estimator` | false | enable gyro-bias LM |
| `optimizer_debug`, `eigenvector_debug`, `residual_debug`, `ba_debug` | false | debug prints |
| `camera_intrinsics`, `camera_extrinsics` | — | loaded camera models / 7-vec q_ItoC,p_IinC |

---

# 4. FeatureInitializer (`ov_core/src/feat/FeatureInitializer.h/.cpp`)

`ClonePose` = `{Mat3 _Rot (=R_GtoC), Vec3 _pos (=p_CinG)}` (FeatureInitializer.h:61-92).

**`single_triangulation`** (FeatureInitializer.cpp:38-164) — linear 3-DOF in the **anchor frame**:
- Anchor: camera with most valid measurements, `anchor_clone_timestamp` = *latest* time of that camera (lines 49-77).
- Per measurement (lines 100-135): `R_AtoCi = R_GtoCi·R_GtoA^T`, `p_CiinA = R_GtoA·(p_CiinG−p_AinG)`, `b_i = (R_AtoCi^T·[uv_norm;1]) normalized`, `Bperp = skew(b_i)`; `A += Bperp^T Bperp`, `b += Bperp^T Bperp·p_CiinA`.
- Solve `p_f = A.colPivHouseholderQr().solve(b)` (line 138); **gate** `cond(A)=σ_max/σ_min > max_cond_number(10000)` OR `p_f(2) ∉ [min_dist=0.10, max_dist=60]` OR NaN (lines 141-158).
- Outputs `feat->p_FinA = p_f`, `feat->p_FinG = R_GtoA^T·p_FinA + p_AinG` (lines 161-162).

**`single_triangulation_1d`** (FeatureInitializer.cpp:166-311) — 1-DOF: feature constrained to the **anchor bearing ray**:
- Anchor = camera with most measurements; `anchor_clone_timestamp` = *last* timestamp of that camera; `idx_anchor_bearing = 0` (**first** bearing; commented-out variant used the last) (lines 174-188).
- `bearing_inA = [uv_norm(anchor, idx0); 1] normalized` (lines 201-205).
- Note `do_stereo_only_tri` is computed then **hardcoded `= false`** (lines 207-217), so the stereo-only branch (lines 220-253) is **dead code**; the general loop (lines 255-293) runs: for each measurement except the anchor bearing, `BperpBanchor = Bperp·bearing_inA`, `A += |BperpBanchor|²`, `b += BperpBanchor·(Bperp·p_CiinA)`.
- `depth = b/A`, `p_f = depth·bearing_inA` (lines 296-297); gate only `p_f(2)` bounds + NaN (lines 299-305). Outputs `p_FinA/p_FinG` as above.

**`single_gaussnewton`** (FeatureInitializer.cpp:313-531) — LM refinement in **anchored inverse-depth (α,β,ρ)=(x/z, y/z, 1/z)**:
- init from `p_FinA`: `ρ=1/z, α=x/z, β=y/z` (lines 319-321). LM params: `init_lamda=1e-3`, `max_runs=5`, `max_lamda=1e10`, `min_dx=1e-6`, `min_dcost=1e-6`, `lam_mult=10` (options). Loop condition `runs<max_runs && lam<max_lamda && eps>min_dx` (lines 346-347).
- Per measurement (lines 391-425): `hi = R_AtoCi·(α,β,1) + ρ·p_AinCi` with `p_AinCi=−R_AtoCi·p_CiinA`; 2×3 Jacobian of `(hi1/hi3, hi2/hi3)` w.r.t `(α,β,ρ)` (lines 398-412); `res = uv_norm − (hi1/hi3, hi2/hi3)`; accumulate `grad+=H^T res`, `Hess+=H^T H`, `err+=|res|²`.
- LM step: `Hess_l` with diagonal scaled `(1+λ)`; `dx = Hess_l.colPivHouseholderQr().solve(grad)` (lines 431-436). Accept if cost decreases (recompute, `λ/=10`, `eps=|dx|`); reject ⇒ `λ*=10`; converge if `(cost_old−cost)/cost_old < min_dcost` (lines 450-474).
- Result `p_FinA = (α/ρ, β/ρ, 1/ρ)` (lines 478-480). **Baseline gate**: `HouseholderQR(p_FinA)` → tangent plane `Q.block(0,1,3,2)`; `base_line = |Q·p_CiinA|`, take max over all poses; reject if `p_FinA(2)∉[min_dist,max_dist]` OR `|p_FinA|/base_line_max > max_baseline(40)` OR NaN (lines 483-526). Then `p_FinG` (line 529).

**"first estimate" vs "estimate"**: there is no FEJ inside FeatureInitializer itself — "first estimate" = the linear triangulation output (`p_FinA`/`p_FinG`, from `single_triangulation(_1d)`); "estimate" = the LM-refined value. The FEJ notion enters at the state boundary: in iterative mode `LandmarkMsckf::set_from_xyz(p_FinA, false)` stores the *value* and `set_from_xyz(p_FinA_fej, true)` stores the *FEJ linearization point*, and `p_FinA_fej` is initialized equal to `p_FinA` (UpdaterMSCKF.cpp:204-205, 354-355). During the init Solver `do_fej` is forced **false** for the camera poses, so the feature Jacobians are recomputed at the current estimate each iteration; only `LandmarkMsckf` maintains a (frozen) fej copy. Triangulation is re-run each Solver iteration only for features **not yet** in `state->features_MSCKF` (UpdaterMSCKF.cpp:153-157); existing landmarks keep their stored value.

`FeatureInitializerOptions` (FeatureInitializerOptions.h:42-112): `triangulate_1d=false`, `refine_features=true`, `max_runs=5`, `init_lamda=1e-3`, `max_lamda=1e10`, `min_dx=1e-6`, `min_dcost=1e-6`, `lam_mult=10`, `min_dist=0.10`, `max_dist=60`, `max_baseline=40`, `max_cond_number=10000`, `tri_debug=false`. (`max_cond_number` is used only by the 3-DOF triangulation; the 1-DOF variant does no conditioning check.)

---

# 5. Covariance handed back to the filter (what the JAX port must reproduce)

The initializer produces a fully-formed sqrt-filter state (value + `U_`), **not** a mean/covariance pair. `U_` is an **upper-triangular square-root covariance** (`P = U^T U`, rows≥cols during cloning).

**Static path** — `StateHelper::set_initial_imu_square_root_covariance` (StateHelper.cpp:43-47): `U_.topLeftCorner(15,15).diagonal() = prior_diag` where `prior_diag`= `[q:0.017³, p:0.05³, v:0.01³, bg:0.02³, ba:0.02³]`. All off-diagonals 0 (U_ starts as `1e-3·I` from the State ctor). No clones, no SLAM.

**Dynamic path** — `StateHelper::initialize_state(state, x_init=[q_GtoI0; v_inG; bg; ba], t0)` (Solver.cpp:53; StateHelper.cpp:655-805):
- Rebuilds `variables_` (imu + optional timeoffset + per-camera calib pose/intrinsics).
- `U_ = Zero(current_id,current_id)`; then `U_imu_init = I(15×15)`:
  - orientation block `(0:3,0:3) = diag(init_prior_q=0.1, 0.1, 0.001)·R(q_GtoI)^T` — yaw gets a 0.001 prior, whole block right-multiplied by `R_GtoI^T` (lines 719-723);
  - `p`: `init_prior_p=0.0·I` ⇒ **zero rows** (rank-deficient, uninformative position); `v`: 0.5·I; `bg`,`ba`: 0.1·I (lines 725-732);
  - `efficient_QR(U_imu_init)`, store the upper triangle (lines 734-735);
  - calibration priors on the diagonal: `init_prior_t=0.001` (timeoffset), `init_prior_qc/pc` (extrinsics), `init_prior_fc/dc1/dc2` (intrinsics) (lines 738-769).
- The Solver then (a) propagates & clones each keyframe (`StateHelper::clone` inserts at `kCloneStartId`; `State` ctor line 40), (b) runs the iterative SRIEKF loop, and (c) runs the final factored step. The final `U_` (upper triangular, size = `current_id + 6·(#clones)` + 3·(#SLAM)) is what the filter continues with.

**Three distinct factor channels the port must reproduce:**

1. **Factored measurement (sqrt-Kalman) update**: `State::store_update_factor(R_sqrt_inv_H_UT_, HT_R_inv_res_)` (State.cpp:253-257). From the final `UpdaterMSCKF::update(..., require_HUT=true)` (UpdaterMSCKF.cpp:287-332): per feature, after nullspace QR, `HUT = H_x2·[U_dense^T | U_tri^T]` (the `get_marginal_U_block` split: `U_dense` = rows above the small-variable block, `U_tri` = upper-triangular small block, `HUT` right block uses `.triangularView<Lower>()`, StateHelper.cpp:91-…); `RHTr = H_x2^T·res2·sigma_pix_sq_inv` expanded from `Hx_order` to full state id (UpdaterMSCKF.cpp:321-328); both multiplied by `sigma_pix_inv` (line 329). These rows accumulate into `R_sqrt_inv_H_UT_` and `HT_R_inv_res_`; `StateHelper::update_llt(state, is_iterative=true)` (StateHelper.h:150) then performs `F=chol(I+F^T F)` on the stacked `R^{-1/2}H U^T` blocks and applies `dx` per-variable (per lead's verified facts). In the init loop `is_iterative=true` ⇒ chi2 gate skipped and `H^T H·dx` term included with downdate.

2. **Iterative-mode Jacobian store**: `State::store_update_jacobians(Hx, res, x_order)` (State.cpp:259-268) — whitened `H_x2·sigma_pix_inv`, `res2·sigma_pix_inv`, used by `iterative_update_llt` (StateHelper.cpp:355-…), which forms `S = U·H^T(U·H^T)^T + I` and solves `dx = U^T(U·H^T·S^{-1}·(r + H·x_{k−}x_0))` then **downdates all variables then updates** (lines 381-403). Also per-feature `LandmarkMsckf::store_msckf_jacobians(H_f1,H_x1,res1)` from the QR-split top `feat_size` rows (UpdaterMSCKF.cpp:364-370).

3. **SLAM initialization factors**: `StateHelper::initialize` (StateHelper.cpp:415-530) then `initialize_invertible` (StateHelper.cpp:532-602). `H_L` (feature Jacobian) is made lower-triangular via `reverse_mat`+`efficient_QR(H_R, res, H_L)`+reversal; `Hx_init/Hf_init/res_init` = top 3 rows. `chi_2_mult=-1` skips the chi2 gate (updaterSLAM.cpp:320). In `initialize_invertible`: `U_HRT = Σ_i U_.block(var,meas_var)·H_R^T` (i.e. `U·H_R^T`, lines 569-578); `H_Linv` = lower-triangular solve of `Hf`; `new_variable->update(H_Linv·(res + H_R·dx))` with `dx = xk_minus_x0_[H_order]` (init only, lines 588-598); then **`store_init_factor(new_var, (1/sigma_pix_inv)·H_Linv^T, −U_HRT·H_Linv^T)`** (lines 599-601; State.cpp:245-251). These become `factor_init_tri_` (per-landmark 3×3 lower-triangular sqrt blocks = `(1/σ_pix)H_Linv^T`) and `factor_init_dense_` (cross `U` blocks to existing state = `−U_HRT·H_Linv^T`). `StateHelper::initialize_slam_in_U` (StateHelper.cpp:604-653) appends them at the **end** of `U_` (`U(0,curr_id)=dense`, `U(curr_id,curr_id)=tri`), bumps ids, and grows `xk_minus_x0_/xk_minus_xk1_`.

**Hand-off summary for the port**: after init the filter owns a square-root-covariance `U_` (upper triangular, with the SLAM landmarks appended at the tail and, for anchored reps, `anchor_clone_timestamp` re-pointed at the latest clone in `select_slam_features`), one surviving IMU clone at `state->timestamp`, `features_SLAM` (values set via `initialize_invertible`, fej=value if `do_fej`), empty `features_MSCKF` (`state->clear(true)`), and `is_initialized=true`. The static path instead leaves `U_` as a pure 15×15 diagonal sqrt-cov with no clones and no SLAM; the first propagation then runs the normal clone/QR pipeline.

**Numerical traps for the JAX port** (all verified in code): (1) `U_` is rank-deficient at init (`init_prior_p=0.0`, and orientation block `diag(0.1,0.1,0.001)·R^T`), so `efficient_QR` and the batch `update_llt`/`iterative_update_llt` must tolerate zero rows; (2) all SLAM-landmark cross-covariance is carried **implicitly** in `factor_init_dense_` = `−U_HRT·H_Linv^T` and `H_Linv^T` must be scaled by `1/sigma_pix_inv` — reproducing these factor blocks exactly is what makes the post-init filter numerically match; (3) `StateHelper::initialize` reorders rows/columns of the QR-split `Hx_init/Hf_init/res_init` with three separate `reverse_mat`/`reverse_vec` calls (StateHelper.cpp:443-461) — the block structure (top 3 = init, rest = update) after `reverse_mat`+`efficient_QR` must be reproduced in the same order; (4) the preintegration covariance `P_meas` and `H_a/H_b` are computed but **unused** by the initializer (only `DT`, `R_k2tau`, `alpha_tau`, `J_q` matter) — they can be dropped from the init port; (5) `solve_rotation` (DynamicInitializer.cpp:851-949) and the `do_stereo_only_tri` branch (FeatureInitializer.cpp:220-253) are dead code in the current pipeline and need not be ported.

=== aa66d65b5ef719d7c ===
I have confirmed all the load-bearing details against the source (efficient_QR internals, `iterative_update_llt`, `update_llt`, `initialize`/`initialize_invertible`, `State::clear` default, `Type::_fej`, `gram_schmidt`, `feed_IMU_meanonly`/`feed_IMU_Jq`, `solve_bg`, `find_eigen_vectors`, `build_linear_system`, `refine_gravity`, `static_initialize`). Here is the design.

---

# JAX initializer design for the sqrtVINS port

## 0. Conventions and state object (grounding for everything below)

The port's filter state is Hamilton (scalar-first) quaternions, error-state θ=so(3)+dp with **left (global-frame) perturbation** (this matches the C++ JPL convention so that `U_` has the same *meaning*: `P = UᵀU`, `U` upper-triangular).

```python
import jax, jax.numpy as jnp, jax.scipy as jsp
jax.config.update("jax_enable_x64", True)   # mandatory, see risks

@dataclass
class VarBlock:                  # replaces shared_ptr<Type> id indirection + std::map<double,...>
    name: str                    # 'imu' | f'clone_{ts}' | f'calib_ext_{cam}' | f'calib_intr_{cam}'
    start: int; size: int        # (start,start+size) in the flat error-state vector
    kind: str                    # 'quat3' (θ only, 3) | 'vec' (additive)
    # pose blocks: kind='pose', size=6 -> (θ,dp)

@dataclass
class InitState:
    ts: float
    vars: dict[str, VarBlock]            # ordered, sorted by start; std::map sortedness -> sort by key
    value: jnp.ndarray                   # (n,)  one float per ERROR-state dof? NO -- see below
    fej: jnp.ndarray | None              # (n,) parallel frozen points, None == fej:=value (sec.4)
    U: jnp.ndarray                       # (n,n) upper-tri sqrt covariance (P=UᵀU)
    xk_minus_x0: jnp.ndarray             # (n,)
```

Rotations are never stored as error-state dofs; `value` holds **pose values** (`(n_p + 3·n_clone) 7-vectors` is messy) — the practical layout: a flat **mean vector** `x` of type-consistent entries is avoided; instead keep
`poses: dict[str, jnp.ndarray]` (clone ts → `(7,)` `[w,x,y,z,p]`), `imu: (16,)` `[q,p,v,bg,ba]`, and a **covariance-order flat array** `xcov (n,)` that only stores the linear dofs (θ,dp,v,bg,ba). `U` is indexed by `xcov`. This is exactly the C++ split (value+fej stored per-`Type`, `U_` indexed by `id()`). All JAX autodiff differentiates a *reconstruction* function `x → poses`, i.e. `exp_so3(θ)·R̄`.

The initializer runs once; every size is fixed by budgets:
`n = 15 + 6·(init_dyn_num_pose−1) + calib_dof`, `n_features ≤ init_max_feat=50`, `n_meas/feat ≤ num_cameras·(N_KF−1)`. **All dynamic-size C++ structures (EigenMatrixBuffer `append_*`, `std::map`) become preallocated padded `jnp` arrays with integer masks** (sentinel `-1`). `state->clear()` in the Solver loop = re-derive the per-feature masks.

---

## 1. Step-by-step tractability, with a recommendation per step

| # | Step | C++ | Recommended env | Why / reasoning |
|---|---|---|---|---|
| A1 | Window gate (timestamp span ≥ 1.0 s) | FeatureDatabase | numpy | Pure range check on floats. No autodiff, no arrays to speak of. |
| A2–A3 | `cleanup_measurements`, `update_disp_info` → avg per-pair pixel disparity, `disparity_check` | FeatureDatabase stats | numpy | Aggregation over `dict[ts]→dict[ts]→avg\|uv−uv\|`. Sorting/median-free; no autodiff value. Camera-only optical flow already in Python cv2. |
| A4 | Decision static/dynamic + `is_static_prev_`/`has_jerk` bookkeeping | InertialInitializer | Python | 5-state boolean machine, zero math. |
| B1–B6 | Static init: window mean, `gram_schmidt`, biases, 15×15 diag sqrt-cov | StaticInitializer | numpy (math) + jnp (state assembly) | Closed form. Keep `gram_schmidt` and prior-diag in numpy; the returned `InitState` is jnp so the filter can consume it. |
| C1 | Rotation-motion gate (`Σ\|(wm−bg)dt\| ≥ 45°`) | DynamicInitializer | numpy | Running sum; no autodiff. |
| C2 | Keyframing (greedy max-min disparity) | DynamicInitializer | numpy | Sequential greedy with min selection; needs no GPU, and argmin over changing sets is awkward under jit. |
| C3 | **CpiV1 preintegration means** (`DT,R_k2tau,alpha_tau` + `J_q,H_a,H_b`) | CpiV1 closed-form exp-series | **JAX `lax.scan`** | Closed-form recurrence with the `small_w` branch (threshold `0.008726646`); perfect scan body. Autodiff *not* needed (J_q is analytic) but scan+`where` gives branchless, jittable, variable-length (`dt=0` is the no-op sentinel). Drop `P_meas` (unused by init). |
| C4 | **Gyro-bias solver (opengv-style rotation-only)** | OpengvHelper (Cayley + cubic smallest-eigenvalue + numeric Jacobians) | **JAX LM** | Nonlinear; autodiff replaces the Cayley closed-form + `GetNumericalJacobians`. See §2.1. |
| C5 | Relative rotations `R_CktoI0`, bearings in I0 | DynamicInitializer | JAX | Pure `Rᵀ`/`×`/normalize; trivially batched. |
| C6 | Translation directions (`find_eigen_vectors`, essential-matrix style) | Eigen `SelfAdjointEigenSolver` + median/std rejection | **numpy** | `np.linalg.eigh` + `np.median`/`np.std` + boolean mask. No autodiff; 3×3 eigensolver is not the reason JAX is here. |
| C7 | Velocity/gravity linear system + QR compress | `efficient_QR(A,b)` + triangular solve | **JAX** | `jnp.linalg.qr` + `triangular_solve`. Exact, see §2.3. |
| C8 | Gravity refinement (2-DOF LM) | hand LM | **JAX or numpy** (JAX preferred for code reuse) | 5-dof dense LM, ≤20 iters; the analytic `J=∂g/∂(α,β)` in `Helper.cpp:545-554` is 5 lines — port it directly, no autodiff needed. |
| C9 | Feature triangulation (3-dof linear, 1-dof, LM refine) | FeatureInitializer | **JAX `vmap`** | `jnp.linalg.solve` + `svd` cond gate (3-dof), closed form (1-dof), ≤5-iter LM (refine). Batches over 50 features. |
| C10 | **Batch SRIEKF Solver** (propagate+clone, iterative `HᵀH` update, final factored step, SLAM init) | Solver + StateHelper | **JAX (core value)** | Autodiff Jacobians via `jacfwd` on SO(3) perturbed residual; `vmap` over features; QR nullspace; `lax.fori_loop`/python loop for ≤20 iters. This is where JAX pays for itself. |
| — | SLAM selection + `delayed_init_from_MSCKF` + init factors | StateHelper::initialize | **JAX** | Triangular/QR algebra + grid dispersion; factor blocks are exactly the handoff (§3). |
| — | JPL→Hamilton bridge | — | numpy | One-time config/validation layer; **not** in the init math (see §0/risks). |

**Recommendation summary.** JAX: C3, C4, C5, C7, C8, C9, C10, SLAM-init. numpy: A1–A3, B, C1, C2, C6. Python/cv2: feature tracking (already cv2), decision state machine, feature-DB bookkeeping. Nothing needs an external solver (no ceres, no opengv): the two nonlinear pieces (C4, C8, C9-refine) are all small LM loops that port directly and are *more* robust under autodiff.

---

## 2. Concrete replacements for every opengv call site

### 2.1 `solve_bg` (rotation-only smallest-eigenvalue LM) — the only true opengv use

C++ path (`DynamicInitializer.cpp:951-1080`): for each ordered (kf₀,kf₁) × (cam₀,cam₁) pair, precompute six 3×3 sums `xxF..zxF = Σᵢ f1[k]·f1[l]·f2f2ᵀ`; then LM over `Δbg` where the residual of a pair is the **smallest eigenvalue of** `M(R(Δbg))` with `R = R_I0toC0·exp(Jq0·Δbg)·R_k2tau0·R_k2tau1ᵀ·exp(−Jq1·Δbg)ᵀ·R_I1toC1ᵀ`, and the gradient is chained through the Cayley cubic closed form plus **numerical** `dcayley/dθ` (perturb 1e-5) plus `dR/dΔbg` (analytic). Dead code `solve_rotation` (851–949) is **not** ported.

**JAX replacement** — discard Cayley entirely; keep the *objective*, minimize `λ_min(M(R(θ)))` over a 3-vector θ (so(3) left-perturbation, `R = exp_so3(θ)R̄`), summed over pairs:

```python
# precomputed once (numpy):  per pair p: f1 (K,3), f2 (K,3) normalized bearings (from feat_norms)
def pair_cost_M(R, f1, f2):
    # M = Σᵢ (f1_i × Rᵀ f2_i)(f1_i × Rᵀ f2_i)ᵀ ;  exact equivalent of ComposeMwithJacobians
    return jnp.einsum('ki,kj->ij', jnp.cross(f1, f2 @ R), jnp.cross(f1, f2 @ R))
def rot_only_cost(theta, Rbar, Jq0, Jq1, f1, f2):
    R = exp_so3(theta) @ Rbar
    e = jnp.linalg.eigvalsh(pair_cost_M(R, f1, f2))   # ascending
    return e[0]                                       # λ_min
```
Gradient: **do not** differentiate through `jnp.linalg.eigvalsh` (repeated-eigenvalue adjoint → NaN). Use the *frozen-eigenvector trick* (stable, still autodiff):
```python
w, V = jnp.linalg.eigh(M(R))          # forward pass only, V col 0 = smallest eigenvector
def scaled_cost(R): return jnp.einsum('i,ij,j->', V[:,0], pair_cost_M(R,f1,f2), V[:,0])  # vᵀM(R)v, v frozen
dM = jax.jacfwd(scaled_cost)(R)        # scalar → (3,3) — differentiate only through M construction
```
Fallback if `eigvalsh` ever reports a (nearly) repeated pair (`w[1]-w[0] < 1e-12`): central finite-difference on the scalar `rot_only_cost`, perturb `1e-6` (same order as C++'s `1e-5`/`1e-6`).

LM loop exactly replicates C++ tolerances: `λ₀=1.0`, `max_iter=20` (`init_grav_opt_max_iter`), `ρ` acceptance with decay `0.5` (`init_grav_opt_lambda_decay`), converge when `‖Δθ‖ < 1e-6` (`init_grav_opt_converge_thres`). `bg += θ` on exit. Because the objective is identical and the fixed point is the minimizer, output matches the C++ bg to ~1e-6 even though the iterates differ.

### 2.2 Translation directions (`find_eigen_vectors`)

Not opengv but the same class. numpy, byte-for-byte the same algorithm: `H = Σ pn pnᵀ/‖pn‖²` (`pn=f0×f1`, normalized so outliers don't dominate) → `eigh` smallest vector `t̂` → residual `|n_planeᵀ t̂|` → keep `|res−median| < 3σ` (population σ, `sqrt(Σ(res−mean)²/(N−1))`) → rebuild `H` → `eigh`. Gates: `N_meas ≥ 5` (`kMinMeas`), inliers ≥ 5. Zero autodiff; keep in numpy (`np.median` needs sort). Output: `(eigvecs (3,3), outliers: set[feat_id], ok)`.

### 2.3 Velocity/gravity linear system (`build_linear_system`)

`A (2N,6)`, `b (2N,)` built from `eig_vec = eigvecs[:, :2]` rows; then `efficient_QR(A,b)` keeps top 6 rows; `x = triu_solve(A_top,b_top)`.

`efficient_QR(A,b)` (Helper.cpp:342-377) = row-permutation by first-nonzero column + Householder; **row permutations and unitary left-multiplication cannot change the least-squares solution**, so the exact replacement is:
```python
Rtop, _, btop = jax.scipy.linalg.qr_updated  # not needed:
Q, R = jnp.linalg.qr(A, mode='reduced')       # (2N,6),(6,6)  same R as Eigen (unique R up to signs)
b_rot = Q.T @ b
x = jsp.linalg.triangular_solve(R, b_rot, lower=False)
```
`R` is uniquely the QR factor (up to per-row sign) of the same `A`, so `x` matches C++ in exact arithmetic; fp64 keeps it ~1e-15. Then `refine_gravity` is the same LM as 2.1 with the **closed-form** `J = ∂g(α,β)/∂(α,β)` from `Helper.cpp:545-554` (do not autodiff a 5-line formula): `A_new=[A₁, A₂·J]`, `r=−(A₁v+A₂g−b)`, `H=A_newᵀA_new`, `H.diag()·=(λ+1)`, `dx=cholesky_solve(A_newᵀr)`, ρ with `λ·0.5`, converge `‖dx‖/‖x‖<1e-6`, ≤20 iters. numpy or JAX, identical.

### 2.4 MSCKF nullspace projection (`efficient_QR(Hx,r,Hf)`, 3-arg)

C++ = column-reverse `Hx`, Givens-upper-triangularize `Hf` applying rotations to `Hx,r` (row-permutation by first-nonzero column of `Hx`), column-reverse back, split top 3 rows (feature) / rest (state).

**Equivalence proof that makes the replacement exact:** (i) the column reversal of `Hx` commutes with the Givens (rotations are chosen purely from `Hf`), so it cancels; (ii) any two QR decompositions of `Hf` differ by a unitary that maps `col(Hf)ᵀ` into itself, so the projected state block `H_botᵀH_bot`, `H_botᵀr_bot`, `r_botᵀr_bot` are **invariant**; (iii) the stored feature block satisfies `R_top·Q_topᵀ = Hfᵀ`... precisely `Q_top = Hf·R⁻¹` ⇒ `R⁻¹Q_topᵀ = (HfᵀHf)⁻¹Hfᵀ = Hf⁺`, so `update_features`' `δ = R⁻¹(res_msckf−Hx_msckf·dx)` is `Hf⁺(r−Hx·dx)`, invariant. Hence:

```python
def nullspace_project(Hf, Hx, res):          # Hf (2m,3) full col-rank (triangulation gates guarantee)
    Q, R = jnp.linalg.qr(Hf, mode='complete')    # (2m,2m),(2m,3); R rows>2 are ~0
    T = Q.T @ jnp.concatenate([Hx, res[:,None]], -1)   # (2m, n+1)
    Hf_msckf, Hx_msckf, res_msckf = R[:3], T[:3,:n], T[:3,n]
    H_bot, res_bot               = T[3:,:n], T[3:,n]      # nullspace-projected state blocks
```
**Must** use `mode='complete'` (reduced QR would drop the nullspace rows). Different QR than C++ → different iterates, same fixed point.

### 2.5 `iterative_update_llt` and `update_llt` (SRIEKF mean/covariance) — exact JAX, skipping the reversal dance

C++ (StateHelper.cpp:355-403, 273-…) permutes/reverses columns and rows purely so it can use triangular multipliers; the net math is:

```python
def iterative_update_llt(U, H, r, xk, offset=0):
    # H (m,n) whitened jacobians, r (m,); C++ reverses cols, QRs the (m, n-15) block,
    # keeps top n rows -> lower-triangular Hn_low == here obtained directly:
    Q, R = jnp.linalg.qr(H[:, :n-15], mode='complete')        # only non-IMU cols
    Hn_low = jnp.tril((jnp.concatenate([H[:, n-15:], R], -1)[:n])[::-1, ::-1])
    rn = (Q.T @ r)[:n][::-1] + Hn_low @ xk                    # r + H(xk−x0)
    UHt = U @ Hn_low.T
    S   = UHt @ UHt.T + jnp.eye(n)
    dx  = U.T @ (U @ (Hn_low.T @ jsp.linalg.solve(S, rn)))    # = P⁺(Hᵀr + HᵀH·dxₖ)
    return dx                                                 # mean-only; U untouched (see Solver)

def update_llt(U, W, g, offset=0):
    # W (k,n) stacked R^{-1/2}H Uᵀ factor rows (R_sqrt_inv_H_UT_), g (n,) = HT_R_inv_res_
    n = U.shape[-1]; m = n - offset
    F  = jnp.linalg.cholesky(jnp.eye(m) + W[:, :m] @ W[:, :m].T).T     # FᵀF = I + WWᵀ
    Un = U.at[:m].set(jsp.linalg.triangular_solve(F.T, U[:m], lower=True))
    Un = jnp.triu(Un)                                                 # U⁺ = F^{-T}U
    dx = Un.T @ (Un @ g)                                              # mean: U⁺ᵀU⁺·HᵀR⁻¹r
    return Un, dx
```
`is_iterative` adds `H_updateᵀ(H_update·xk_minus_x0)` to `g` and does the downdate-then-update on variable values (i.e. `retract(-xk)` then `retract(dx)` — **not** a covariance downdate; `U_` is untouched during the iteration loop). The `offset = #x_init·3` columns are the freshly-initialized SLAM landmarks handled by `initialize_slam_in_U`. This reproduces C++ bit-for-bit (the reversal is exactly the double-reversal `P F P` shown above to satisfy `F′ᵀF′ = I + AᵀA`).

---

## 3. Module API (numpy in, InitState + init-factors out)

```
sqrtvins_jax/
  initializer/
    window.py        static.py      dynamic.py      preint.py
    solver.py        triangulate.py lm.py           qrtools.py
  math_ham.py        jpl_bridge.py
```

```python
# math_ham.py — Hamilton core (also used by the filter)
def rot_to_quat_ham(R: jnp.ndarray) -> jnp.ndarray        # (4,) wxyz
def quat_multiply_ham(a, b) -> jnp.ndarray                # (4,)  a⊗b (left-mult convention)
def exp_so3_ham(theta) -> jnp.ndarray; def log_so3_ham(R); def skew_x(v) -> jnp.ndarray
def retract_pose(q7, dqdp: jnp.ndarray) -> jnp.ndarray    # q7:(7,) [wxyz,p]; left-mult exp_so3_ham
def jpl_to_hamilton(q_jpl: np.ndarray) -> np.ndarray      # (4,) q_jpl[[3,0,1,2]]  + unit test
# jpl_bridge.py — only for config load + trajectory comparison vs C++ (rot2rpy/so3)

# window.py
def build_disparity_map(db, ordered_times) -> dict[float, dict[float, float]]
def disparity_check(disp, oldest_allowed, cur, max_disparity=1.0) -> tuple[bool, bool]  # (ok, is_move)
def pick_keyframes(times, disp, num_pose=5, num_cameras=1) -> list[float]

# static.py
def static_initialize(imu: np.ndarray,             # (T,7) [ts,gx,gy,gz,ax,ay,az]
                      last_static_ts: float, gravity_mag: float,
                      prior_diag: np.ndarray | None = None) -> InitState
# prior_diag default = [0.017]*3+[0.05]*3+[0.01]*3+[0.02]*3+[0.02]*3 ; U = diag(prior), no clones

# preint.py  (CpiV1 means; P_meas deliberately dropped)
@jax.jit
def preintegrate(imu: jnp.ndarray,            # (K,7) zero-padded, dt<=0 rows are no-ops
                 t0: float, t1: float, dt_camimu: float,
                 bg: jnp.ndarray, ba: jnp.ndarray,
                 sigma_w: float, sigma_wb: float, sigma_a: float, sigma_ab: float,
                 with_jq: bool) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray,
                                         jnp.ndarray, jnp.ndarray, jnp.ndarray]
# -> (DT, R_k2tau (3,3), alpha (3,), J_q (3,3), H_a (3,3), H_b (3,3)); lax.scan, small_w branch

# dynamic.py
def dynamic_initialize(imu, db, extrinsics, intrinsics, opts) -> InitState | None
def solve_bg(cpi: dict[float, Preint], feat_norms, extrinsics, opts) -> tuple[np.ndarray, bool]
def find_eigen_vectors(b0, b1) -> tuple[np.ndarray, set[int], bool]   # (3,3), outliers
def build_linear_system(cpi, id_to_eigvecs, id_to_timepair, id_to_campair,
                        extrinsics) -> tuple[np.ndarray, np.ndarray, np.ndarray]  # A(6,6),b(6),x(6)
def refine_gravity(A, b, vg_init, opts) -> tuple[np.ndarray, bool]     # LM, ≤20 iter, λ0=1.0
def gram_schmidt(gravity_inI) -> np.ndarray                            # R_GtoI (3,3) [x y z] cols
def select_slam_features(feats, latest_q7, opts) -> list[int]          # 4×4 spatial grid, ≤50

# triangulate.py  (vmap'd; all shapes fixed by budgets)
@jax.jit
def triangulate_batch(q7s: jnp.ndarray,             # (N_FEAT, M_MAX, 7) clone poses
                      uv: jnp.ndarray,              # (N_FEAT, M_MAX, 2) normalized uv
                      mask: jnp.ndarray,            # (N_FEAT, M_MAX) bool
                      one_d: bool, opts) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
# -> p_FinA (N_FEAT,3), p_FinG (N_FEAT,3), ok (N_FEAT,)
# 3-dof: A=Σ BperpᵀBperp, b=Σ BperpᵀBperp·p_CiinA, solve; gates cond=σmax/σmin>1e4 (svd),
#       0.1≤p_z≤60, NaN -> ok=False
# 1-dof: depth=b/A along anchor bearing; gates p_z bounds only
@jax.jit
def gaussnewton_refine(q7s, uv, mask, p_FinA0) -> tuple[jnp.ndarray, jnp.ndarray, bool]
# inverse-depth (α,β,ρ); LM λ0=1e-3, ≤5 runs (init_lamda/max_runs), λ·10, conv dcost<1e-6,
# gates p_z∈[0.1,60], ‖p‖/max_baseline≤40, NaN

# solver.py  (the batch SRIEKF)
def run_batch_solver(imu0: np.ndarray, kf_times: np.ndarray,
                     imu_data: np.ndarray, feats, masks,
                     extrinsics, intrinsics, opts) -> tuple[InitState, bool]
@jax.jit
def solver_jacobians(params, feat_pad, mask_pad) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
#   residual(x, f) over SO(3)-perturbed clones (anchored inv-depth (a,b,ρ));
#   Hx=jacfwd(λx: res(x,f),0)(x) (N_FEAT,M,2,n);  Hf=jacfwd(λf: res(x,f),1)(f) (N_FEAT,M,2,3)
@jax.jit
def nullspace_batch(Hf, Hx, res) -> ...          # sec 2.4, vmap'd
@jax.jit
def huber_scale(H, r, thresh=2.0) -> tuple[jnp.ndarray, jnp.ndarray]
def iterative_update_llt(...)                    # sec 2.5
def update_llt(...)                              # sec 2.5
@jax.jit
def update_features(Hf_msckf, Hx_msckf, res_msckf, dx) -> jnp.ndarray
#   δ = solve_triangular(Hf_msckf, res_msckf − Hx_msckf·dx, lower=False); value += δ (a,b,ρ additive)

# Handoff factors (what StateHelper::initialize-equivalent must return)
@dataclass
class InitFactor:
    tri: jnp.ndarray      # (3,3) LOWER-tri per landmark = (1/σ_pix)·H_Linvᵀ  (State.cpp:245-251)
    dense: jnp.ndarray    # (n_existing,3) = −U_HRT·H_Linvᵀ   (the implicit cross-covariance)
def initialize_slam_factor(U, H_R, H_L, res, sigma_pix_inv, xk):
    # 1. reverse-col H_L (→ lower), reverse-col H_R; QR(H_R,res,H_L) via jnp.linalg.qr on
    #    H_L with Q applied to [H_R|res]  (exact, see 2.4 invariance);
    # 2. Hx_init=top3, Hf_init=top3, res_init=top3, then reverse rows/cols/vec as C++;
    # 3. H_update=bottom rows; HUT = H_update@[U_denseᵀ|U_triᵀ] (U_dense=U[:k,sm:], U_tri=U[sm:,sm:]);
    # 4. U_HRT = U[:, small_vars] @ H_Rᵀ;  H_Linv = solve_triangular(Hf_init, I, lower=True);
    #    new_val = H_Linv@(res_init + H_R_init@xk[small]);  factor = ((1/σ_pix)·H_Linvᵀ, −U_HRT·H_Linvᵀ)
    # 5. store update factor (HUT·σ_pix_inv, RHTr expanded to full width) for update_llt
def initialize_slam_in_U(U, factors: list[InitFactor], xcov_size) -> jnp.ndarray
    # append: U[:k,new]=Σ denseᵀ, U[new,new]=tri; grow xk_minus_x0_/xk_minus_xk1_ by zeros
```

`qrtools.py` is just the thin wrappers of §2.3–2.5 (`qr_tall`, `nullspace_project`, `qr_tall_triangular_solve`).

---

## 4. FEJ / "first estimate" double-state mechanism

**What the C++ keeps during init** (verified): `initialize_state` sets `imu fej = value`; the Solver forces `do_fej=false` (Solver.cpp:60-68) so every **camera/clone** Jacobian is recomputed at the current estimate each iteration; `LandmarkMsckf` keeps a *frozen* `p_FinA_fej` copy (initialized `=p_FinA`), used only by the *post-init filter*; post-solve cleanup sets `fej = value` for IMU and the surviving clone when `do_fej`; SLAM landmarks get `fej = value` on initialization. `Type::_fej` is a plain stored array that `update()` never touches.

**JAX representation:** carry `fej: jnp.ndarray | None` per variable as a parallel array `fej (n,)`.

**Simplification — provably safe:** at handoff **every** quantity satisfies `fej == value` (shown above: IMU, surviving clone, SLAM landmarks, and MSCKF features are dropped by `state->clear(true)`). Therefore the initializer can set `fej = None` everywhere and the filter **materializes `fej = value` lazily on first read** (a `@property` on the VarBlock). This cannot change any result because fej is only *read* to fix Jacobian linearization points, and at the instant of materialization `fej == value`, identical to explicit storage. Inside the initializer Solver, `do_fej=false` means FEJ is **never referenced at all** — the jacfwd Jacobians are by construction recomputed at the current `x` each iteration, which is exactly the `do_fej=false` semantics.

One caveat to encode as a test, not a simplification: the **perturbation convention** used by jacfwd (left-perturbation `exp_so3(θ)·R̄`, matching C++) must be consistent between the initializer's final `U_` and the filter's first update, or the post-init covariance is rotated. Add a handoff test: re-run the final `update_llt` with a unit measurement and compare `U` against C++.

---

## 5. Phased plan

**Phase 0 — foundation (no estimator yet).** `math_ham.py`, `jpl_bridge.py` + unit test (`rot_2_quat_jpl == jpl_of(rot_to_quat_ham)` on random R, and a traj-vs-C++ sanity check against `run_euroc` output, since `trajectory_comparison_euroc.png` already exists), `preint.preintegrate` (`lax.scan`, `small_w` branch, drop `P_meas`), `lm.py`, `qrtools.py`, `gram_schmidt`. Validate preint `(DT,R_k2tau,alpha,J_q)` against C++ debug dumps to `/tmp`. **Env risk gate here:** confirm a `jaxlib` wheel exists for cp314; if not, drop the JAX env to python 3.12 under pixi (the driver and cv2 tracking stay on the user's 3.14 numpy env, exchanging arrays via files/`pyarrow`).

**Phase 1 — static MVP (runnable filter from a static start).** `window.py` (A1–A3), decision machine (A4) with **dynamic branch stubbed to `None`**, `static_initialize`. Handoff: `InitState` with 15×15 diag `U` (sqrt-cov: diag `[0.017³,0.05³,0.01³,0.02³,0.02³]` — note the C++ prior vector is `[0.017,…]`, not cubed), `fej=None`, no clones, `is_initialized=True`. Stub `dynamic_initialize` to return the "moving, no dynamic support" failure so the decision path degrades to "keep accumulating until static".

**Phase 2 — dynamic full (the big block).** Since C4–C10 are tightly coupled, port them together but land each sub-piece with a unit test against C++ dumps: C1 rotation-motion gate, C2 keyframing, C5 relative rotations/bearings, C6 translation dirs, C7 v/g system, C8 gravity refinement, then `triangulate.py`, then `run_batch_solver` (propagate+clone, `iterative_update_llt` mean loop with `jacfwd`+`vmap`, `convergence_check`, final factored `update_llt` + SLAM `select_slam_features` + `initialize_slam_factor`/`initialize_slam_in_U`), then post-solve marginalization (`U[survivors]` → `qr` → upper-tri) and clone fej. Validation: hand the produced `U_` + values to the filter and compare trajectory RMSE vs `run_euroc` over the init→first-100-frames window; also compare final `U_` Frobenius norm. Keep calib options **forced off**.

**Phase 3 — timeoffset.** `do_calib_camera_timeoffset`: add `calib_dt_CAMtoIMU` VarBlock (size 1) in `initialize_state`, `init_prior_t=0.001` on the `U` diagonal, `StateHelper::propagate_timeoffset` (last-w/vel-shaped `dnc_dt`) in the Solver clone loop, and the `+dt_camimu` shift in preintegrate/C5/C9. Stub remains: extrinsics/intrinsics fixed.

**Phase 4 — extrinsics.** `do_calib_camera_pose`: add per-cam PoseJPL VarBlock (6) with `init_prior_qc/pc`, and the calib blocks in `solver_jacobians` (the residual depends on `q_ItoC,p_IinC`), matching `get_feature_jacobian_full`'s `x_order`. Prior to this phase the extrinsics enter only through the fixed transforms `R_CktoI0` and `p_CinI`.

**Phase 5 — intrinsics.** `do_calib_camera_intrinsics`: per-cam `Vec(8)` VarBlock with `init_prior_fc/dc1/dc2`, `CamBase::distort` (Port: cam model from Python cv2 `calib` — the port keeps cv2 for distortion, so this is a `jnp.where`-free fixed distort + its jacfwd derivative), and the intrinsics columns in `Hx`.

At each phase, unimplemented branches return the C++ *failure* sentinels (`false`/`None`/`ok=False`) so the decision state machine and downstream gating behave identically while the phase is stubbed.

---

## 6. Numerical risks (JAX API → failure mode → mitigation)

1. **`jnp.linalg.eigh` autodiff through repeated eigenvalues → NaN gradient** in the bg/rotation cost. → Frozen-eigenvector trick (§2.1); fallback finite-difference `1e-6` on NaN.
2. **fp32 default**: sqrt-cov and LM need 1e-12; consumer Ada (RTX 2000) fp64 is ~1/32 rate. → `jax_enable_x64=True`; run the initializer on the CPU backend (`jax.default_backend()`/`jax.devices("cpu")`) — it runs once, sizes ≤ 50 features; reserve GPU for the filter loop. Keep fp64 end-to-end.
3. **`jnp.linalg.qr` on rank-deficient `Hf`** (collinear feature rays) → non-unique `R`, silently wrong nullspace. → Replicate the C++ triangulation gates (cond>1e4, `p_z∈[0.1,60]`, NaN) *before* QR so `Hf` is full-rank; mask `ok=False` features out of the vmap batches; NaN-trap and drop.
4. **`mode='complete'` required** for the nullspace projector; `mode='reduced'` silently drops the nullspace rows. → Assert `Q.shape == (2m,2m)` in `nullspace_batch`.
5. **Data-dependent loops under jit**: convergence break / NaN checks. → Run ≤20 LM/iteration loops as python `for`/`while` over **jit-compiled kernels** (triangulate, jacobians, nullspace, update_llt); the 20×50 graph is small. `lax.while_loop` only if the whole Solver is fused.
6. **Variable-length IMU/preintegration** under `lax.scan` → pad with `dt=0`; verify the closed-form is the identity at `dt=0` (the `small_w` branch forces `w_dt=0`, no 0/0). Mask any residual risk with `jnp.where(dt==0, carry, updated)`.
7. **Hamilton/JPL convention mismatch** (conjugation or scalar-position) silently rotates the whole estimate. → Bridge layer + the §0 handoff test + trajectory RMSE vs `run_euroc`; do not trust `rot_to_quat_ham` until the R→q round-trip and a real traj match.
8. **Python 3.14 wheel gap for JAX** → Phase-0 env gate; pixi env at 3.12/3.13 if needed, driver on 3.14.
9. **Row-permutation sign conventions** in the reversal/QR tricks differ between Eigen and LAPACK — provably irrelevant for `HᵀH`, `Hᵀr`, `rᵀr`, and the feature update (§2.4 proof), so no mitigation beyond using those invariants (never compare raw `H`/`Q` against C++, compare `U`, `dx`, trajectories).

**Files read for this design:** `ov_srvins/src/initializer/static/StaticInitializer.cpp`, `initializer/dynamic/{DynamicInitializer,OpengvHelper,Solver}.cpp`, `state/StateHelper.cpp`, `state/State.cpp`, `state/State.h`, `utils/Helper.{h,cpp}`, `update/UpdaterMSCKF.cpp`, `ov_core/src/cpi/CpiV1.cpp`, `ov_core/src/types/{Type,PoseJPL}.h`.

