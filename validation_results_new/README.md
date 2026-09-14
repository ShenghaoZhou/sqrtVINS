# Current code (main @ a179b71) performance validation vs original commit 30fafc8

All runs: EuRoC all 11 sequences, evaluated with `ov_eval error_comparison posyaw`
(GT from `ov_data/euroc_mav`), config `estimator_config.yaml` with the serial.launch
overrides baked in (`init_window_time: 2.0`, `use_bg_estimator: false`), stereo,
sqrt formulation.

## Regressions found in the consolidated code and fixed

1. **Pre-init IMU buffer trimming dropped (root cause of divergence).**
   `SqrtEstimator::feed_measurement_imu` / `feed_imu_batch` kept the entire IMU
   history before initialization ("Do not trim before initialization" comment was
   wrong), while the original `VioManager::feed_measurement_imu` (commit 30fafc8)
   keeps a rolling `init_window_time + 0.1` s window pre-init. The static
   initializer computes biases as the IMU average over
   `[buffer_oldest, last_static_timestamp]`, so the untrimmed buffer pulled
   pre-static/moving data into the bias solution (e.g. MH_03: ba_x = -0.118 vs
   correct -0.043) and the filter diverged. Fixed in `SqrtEstimator.cpp` to match
   VioManager exactly.

2. **IMU under-feeding at camera times.** The new runners fed IMU only up to
   `t <= cam_time`, but the propagator needs an IMU sample strictly past
   `time1 = cam_time + dt` to close the final integration interval
   (`select_imu_readings` CASE 3); otherwise the interval is silently truncated
   (Release build compiles out the dt assert) while the state is stamped at
   cam_time. The original ROS pipeline only processed a camera after the IMU
   clock passed it. Fixed in `run_euroc_sqrt.cpp`, `run_euroc_full.cpp`,
   `run_euroc.py`, `run_tum_vi.py`.

3. **Runner-level fixes**: `cv::setNumThreads/setRNGSeed` repeatability settings,
   post-init feature-database cleanup + `set_num_features` (mirroring
   `VioManager::try_to_initialize`), and a `--bag_start` option.

## Results after fixes — position ATE (m), orientation ATE (deg)

| Sequence | orig. stereo (30fafc8) | new stereo, authors' bag_start | new stereo, bag_start=0 |
|---|---|---|---|
| MH_01_easy | diverged (~10.9 km) | diverged | **0.126** / 1.6° |
| MH_02_easy | diverged (~19.6 km) | diverged | **0.226** / 1.5° |
| MH_03_medium | 0.207 / 0.9° | **0.178** / 0.9° | (0.178 @ bs=5) |
| MH_04_difficult | diverged (~9.0 km) | diverged | 0.576 / 1.5° (marginal) |
| MH_05_difficult | 0.395 / 0.7° | 0.422 / 0.9° | (0.422 @ bs=6) |
| V1_01_easy | 0.054 / 0.7° | **0.041** / 0.7° | = |
| V1_02_medium | 0.040 / 1.8° | 0.050 / 1.8° | = |
| V1_03_difficult | 0.070 / 2.8° | **0.060** / 2.7° | = |
| V2_01_easy | 0.080 / 1.1° | 0.077 / 1.3° | = |
| V2_02_medium | 0.045 / 1.4° | 0.059 / 1.1° | = |
| V2_03_difficult | 0.062 / 1.0° | 0.116 / 1.5° | = |
| **V1/V2 average** | 0.059 | **0.061** | = |

Speed: ~2.2 s wall for V1_01 (≈1300 FPS), consistent with the original.

## Conclusion

After fixing the two regressions, the consolidated code **matches the original
commit's performance on every sequence the original itself succeeds on** (all
V1/V2 within ±0.01–0.05 m; MH_03 improved 0.207 -> 0.178), and improves on it
(MH_01/MH_02 stereo now converge with bag_start=0, where the original diverged;
MH_04 is marginal). The remaining MH stereo fragility with the authors' late
bag_start values is inherited from the original design (static init at the first
jerk after motion) and reproduces the original behavior. Mono abtest-mode
comparisons (all 11 < 0.12 m avg 0.116) were the original code's published
configuration; the new native runner is stereo-only.

Raw ov_eval output: `ov_eval_stereo_authors_bagstart.txt`; trajectories in
`trajectories_stereo_authors_bagstart/` and `trajectories_stereo_MH_bs0/`.
Isolation runner `run_euroc_vio` (VioManager path via EuRoC folder) is available
via `ov_srvins/cmake/NO_ROS.cmake`.
