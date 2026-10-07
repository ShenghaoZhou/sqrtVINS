# OKVIS2-X VI-BA Backend vs. the sqrtVINS Backend

Analysis of `third_party/OKVIS2-X` (okvis_ceres) and comparison with the
backend in this directory. Recorded 2026-10-07.

## 1. How OKVIS2-X's backend works

### Architecture: three layers, two graphs

Layers (all in `okvis_ceres`):

- `ViGraph` — owns the single `ceres::Problem`, the state map (pose 7-block +
  speed&bias 9-block per frame), homogeneous-point landmarks, and all factor
  bookkeeping. `optimise()` is a thin ceres `Solve()`.
- `ViGraphEstimator` — adds window-management surgery:
  `eliminateStateByImuMerge`, `convertToPoseGraphMst` /
  `convertToObservations`, `freezePosesUntil` / `unfreezePosesFrom`, landmark
  merging.
- `ViSlamBackend` — orchestration: keyframe bookkeeping, loop closure, GPS,
  and **two parallel graphs**: `realtimeGraph_` (bounded sliding window,
  optimized every frame) and `fullGraph_` (the whole trajectory, optimized
  asynchronously after loop closures), kept consistent via backlogs
  (`addStatesBacklog_`, `touchedStates_`, `touchedLandmarks_`).

Threading (`okvis_multisensor_processing/src/ThreadedSlam.cpp`): frontend
thread does detect/match/`addStates`; a second thread runs
`optimisePublishMarginalise` = `optimiseRealtimeGraph` → `applyStrategy`; a
third publishes; loop closures spawn a one-shot `optimiseFullGraph` thread.
There is **no filter at all** — the smoother *is* the estimator, and the
newest optimized state is what gets published.

### Factor graph contents

- **Every camera frame is a state** (pose + speed/bias), not just keyframes.
  IMU preintegration factors connect *consecutive frames*
  (`ViGraph.cpp:450-459`).
- Landmarks are homogeneous 4-vector parameter blocks, shared across both
  graphs; reprojection errors with per-observation `CauchyLoss(1.0)`.
- IMU error (`okvis/ceres/ImuError.hpp`): Forster-style preintegration with
  first-order bias-Jacobian correction, **plus `redoPreintegration()`** — it
  keeps the raw IMU measurements and re-integrates from scratch whenever the
  bias estimate moves beyond a threshold (with a `redoPropagationAlways`
  escape hatch). It also `append()`s measurements when a frame is eliminated.

### Boundedness without marginalization

The most distinctive design choice. OKVIS2 **removed Schur-complement
marginalization** entirely (`MarginalizationError` survives only in tests).
`applyStrategy()` (`ViSlamBackend.cpp:555`) keeps the realtime window bounded
by three mechanisms instead:

1. **`eliminateImuFrames`** — non-keyframe frames are deleted and their two
   adjacent preintegrations merged into one (`eliminateStateByImuMerge`,
   `ViGraphEstimator.cpp:38`). Lossless up to preintegration linearization.
2. **`convertToPoseGraphMst`** — the oldest keyframes' reprojection factors
   are *condensed* into binary relative-pose edges over a maximum spanning
   tree (information is progressively halved when observations are shared
   between edges). Crucially this is **reversible**: `convertToObservations`
   re-expands a pose-graph frame back into landmark observations when it
   re-enters the active area ("expand frontier").
3. **`freezePosesUntil`** — everything older than
   `numRealtimePoseGraphFrames` is simply `SetParameterBlockConstant`. No
   prior replaces the frozen part; the pose-graph edges carry its information.

### Loop closure

`attemptLoopClosure` (`ViSlamBackend.cpp:2361`) does **not** optimize first.
It (a) verifies the loop with a drift-percentage + 3-sigma heuristic, (b)
immediately distributes the correction geometrically along the trajectory
(rotating velocities and transforming landmarks by the same frame change),
then (c) the full graph is optimized **in the background** with the
relative-pose constraint added, and (d) `synchroniseRealtimeAndFullGraph`
merges the result back, replaying the backlog of states/landmarks that
arrived during optimization.

Solver config: realtime uses `DENSE_SCHUR`; the default/full graph uses
`SPARSE_NORMAL_CHOLESKY` + `DOGLEG`, with a `CeresIterationCallback`
enforcing wall-clock time limits.

## 2. Comparison

| Aspect | OKVIS2-X | sqrtVINS backend |
|---|---|---|
| **Role of smoother** | *Is* the estimator (no filter) | Auxiliary to the sqrt filter; results fed back as soft pose measurements (3a) |
| **Data model** | Custom `ViGraph` states/landmarks | colmap-lite `Reconstruction` (rig = IMU, frame = keyframe, Point3D = track) |
| **States in graph** | Every camera frame + per-frame speed/bias | Keyframes only (stride 5), per-keyframe `[v,bg,ba]` 9-block |
| **IMU factor** | Preintegration per frame pair, **raw measurements kept**, `redoPreintegration` on bias drift, `append()` on frame elimination | Preintegration per keyframe pair (spans ~5 frames), frozen at filter biases, first-order `J_*_bg/ba` correction only |
| **Boundedness** | IMU-merge + MST pose-graph condensation (reversible) + freezing | Fixed-lag window (15 kf) + **soft boundary prior** (`PosePriorFactor`/`SbPriorFactor`) + warm start |
| **Out-of-window info** | Preserved as pose-graph edges spanning the whole trajectory | Dropped; only the boundary prior survives |
| **Loop closure** | Immediate geometric alignment + async **full** BA + sync backlog | Relative-pose factors injected into windowed solves; correction is window-bounded |
| **Robust loss** | Cauchy per observation | colmap loss + prune-and-re-solve |
| **Solver** | DENSE_SCHUR (realtime), SPARSE_NORMAL_CHOLESKY+DOGLEG (full), time-limit callback | colmap's Ceres adjuster (SPARSE_SCHUR), iteration/time budgets |
| **Extrinsics/intrinsics** | Online-calibratable parameter blocks | Held constant |

## 3. Assessment

**Fundamental difference.** OKVIS2-X is a pure keyframe smoother where the
backend must answer at camera rate (hence the `onlyNewestState`
quick-optimize mode and time-limit callbacks); the sqrtVINS backend
deliberately rides on a strong sqrt-filter and only needs to *refine*, so it
can afford keyframe-stride sparsity and full solves on a 15-keyframe window.
Given that premise, the sqrtVINS design is appropriately much simpler —
OKVIS's two-graph sync protocol (`touchedStates_`, backlogs, `isSynched`
debug checks) exists precisely because its realtime graph *is* the published
estimate and can never block on the full BA. The filter provides the same
decoupling for free.

**Where OKVIS2-X is genuinely stronger, and worth borrowing from:**

1. **Out-of-window information.** The biggest gap. OKVIS keeps the entire
   history alive as a pose graph (MST condensation + freeze), so loop
   closures correct the *whole* trajectory; the sqrtVINS window drops
   everything outside 15 keyframes except the boundary prior, which is why
   the 3b decision note in `README.md` says "anchors already marginalized
   into `U` do not move." The MST condensation trick (reversible,
   information-splitting) is a concrete, marginalization-free design that
   fits the stated no-Schur-prior preference — it could be the long-term
   graph for Phase 3c+ without touching the windowed solve.
2. **`redoPreintegration` on bias drift.** The sqrtVINS `ImuFactor` is
   linearized once at filter biases. After loop-closure corrections or long
   runs, biases can move enough that first-order correction degrades. OKVIS's
   cheap heuristic — re-integrate only when `|Δb|` exceeds a threshold — is
   directly portable since the raw IMU stream is already recorded
   (`imu_data_`).
3. **Loop-verification heuristics** before accepting a constraint
   (drift-percentage budget + 3-sigma check, `ViSlamBackend.cpp:2461-2497`):
   cheap insurance for an external loop module's output.
4. **Immediate geometric loop alignment** (distribute the correction along
   the trajectory, rotate velocities, transform landmarks consistently) —
   gives a good warm start for the subsequent optimization and an instant
   corrected trajectory to publish, rather than waiting for the next windowed
   solve to absorb the loop factor.

**Where sqrtVINS is arguably cleaner:** the soft-boundary-prior + warm-start
seam is simpler and better-behaved than OKVIS's freeze-and-hope (frozen poses
with no prior make the window gauge sensitive to whatever the frozen frontier
happened to be); and the filter feedback loop (3a) with chi² gating is a more
principled estimator coupling than OKVIS's "publish the smoothed state
directly."

One caveat when reading OKVIS2-X as reference: it carries GPS, submap-ICP
(LiDAR/depth), and online extrinsics calibration machinery, and its comments
admit some parts are disabled or hacked (lost-component handling "currently
disabled", mutable preintegration state is a self-described "TERRIBLE HACK").
The core windowed-VIO + pose-graph-condensation design (items 1–2 above) is
the solid, paper-backed part.
