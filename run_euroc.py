import os
import sys
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Add build directory to path for ov_srvins_py
sys.path.append(os.path.join(os.getcwd(), 'build/ov_srvins'))
import ov_srvins_py as vins


def run_vio(dataset_path, config_path, max_frames=None):
    # 1. Setup options
    options = vins.VinsOptions()
    parser = vins.YamlParser(config_path)
    options.print_and_load(parser)

    # Repeatability settings used by the C++ runner (run_euroc)
    cv2.setNumThreads(options.num_opencv_threads)
    cv2.setRNGSeed(0)

    # 2. Construct the fully wired pipeline (estimator, frontend, initializer,
    # init runner; sync init by default, async shadow solve when init_async
    # is enabled in the config)
    system = vins.System.create(options)
    estimator, frontend = system.estimator, system.frontend
    init_runner = system.init_runner

    # 4. Load Data
    imu_df = pd.read_csv(os.path.join(dataset_path, 'mav0/imu0/data.csv'))
    cam0_df = pd.read_csv(os.path.join(dataset_path, 'mav0/cam0/data.csv'))
    cam1_df = pd.read_csv(os.path.join(dataset_path, 'mav0/cam1/data.csv'))

    # Rename columns to remove spaces and units
    imu_df.columns = [c.strip().split(' ')[0] for c in imu_df.columns]
    cam0_df.columns = [c.strip().split(' ')[0] for c in cam0_df.columns]
    cam1_df.columns = [c.strip().split(' ')[0] for c in cam1_df.columns]

    # Sort by timestamp
    imu_df = imu_df.sort_values('#timestamp')
    cam0_df = cam0_df.sort_values('#timestamp')
    cam1_df = cam1_df.sort_values('#timestamp')

    cam0_times = cam0_df['#timestamp'].values * 1e-9
    cam1_times = cam1_df['#timestamp'].values * 1e-9
    imu_times = imu_df['#timestamp'].values * 1e-9
    imu_wm = imu_df[['w_RS_S_x', 'w_RS_S_y', 'w_RS_S_z']].to_numpy()
    imu_am = imu_df[['a_RS_S_x', 'a_RS_S_y', 'a_RS_S_z']].to_numpy()

    # Match cam1 to each cam0 frame by timestamp (EuRoC stereo pairs share stamps)
    cam1_lookup = {t: i for i, t in enumerate(cam1_times)}

    # Skip camera frames before the first IMU
    start_cam = 0
    while start_cam < len(cam0_times) and cam0_times[start_cam] < imu_times[0]:
        start_cam += 1

    n_frames = len(cam0_times) - start_cam
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    print(f"Starting VIO processing for {n_frames} stereo images...")

    # Initial startup time
    frontend.set_startup_time(cam0_times[start_cam])

    # Processing Loop
    imu_idx = 0
    trajectory = []
    timestamps = []
    zero_mask = None
    init_frame = None

    # Timing accumulators (wall time in each pipeline stage)
    t_imu = t_read = t_track = t_fused = 0.0
    n_zupt = n_upd = 0

    t_run0 = time.perf_counter()
    for i in range(start_cam, start_cam + n_frames):
        curr_cam_time = cam0_times[i]

        # Feed IMU measurements up to this camera time in one batched call.
        # Feeding policy is shared with the C++ runner (vins.imu_batch_end).
        # Use the CURRENT estimated camera-IMU time offset like the C++ runner
        # does (it drifts when online timeoffset calibration is enabled).
        t0 = time.perf_counter()
        t_off = estimator.get_state().cam_imu_timeoffset()
        k = vins.imu_batch_end(imu_times, imu_idx, curr_cam_time + t_off)
        if k > imu_idx:
            estimator.feed_imu_batch(imu_times[imu_idx:k], imu_wm[imu_idx:k], imu_am[imu_idx:k])
            imu_idx = k
        t_imu += time.perf_counter() - t0

        # Read images
        t0 = time.perf_counter()
        img0 = cv2.imread(os.path.join(dataset_path, 'mav0/cam0/data', cam0_df.iloc[i]['filename']), cv2.IMREAD_GRAYSCALE)
        j = cam1_lookup.get(curr_cam_time, -1)
        img1 = cv2.imread(os.path.join(dataset_path, 'mav0/cam1/data', cam1_df.iloc[j]['filename']), cv2.IMREAD_GRAYSCALE) if j >= 0 else None
        t_read += time.perf_counter() - t0
        if img0 is None or img1 is None:
            continue

        # Feed Camera measurement
        cam_msg = vins.CameraData()
        cam_msg.timestamp = curr_cam_time
        cam_msg.sensor_ids = [0, 1]
        cam_msg.images = [img0, img1]
        if options.use_mask:
            cam_msg.masks = [zero_mask, zero_mask]
        else:
            if zero_mask is None:
                zero_mask = np.zeros(img0.shape, dtype=np.uint8)
            cam_msg.masks = [zero_mask, zero_mask]

        try:
            t0 = time.perf_counter()
            frontend.feed_camera(cam_msg)
            t_track += time.perf_counter() - t0

            state = estimator.get_state()

            # Check for initialization (post-init bookkeeping included)
            if not state.is_initialized:
                if init_runner.try_initialize(curr_cam_time, not options.try_zupt):
                    print(f"VIO Initialized at {curr_cam_time}!")
                    init_frame = i - start_cam
                continue

            # Try a zero-velocity update (ZUPT motion bookkeeping is internal
            # to the estimator)
            t0 = time.perf_counter()
            did_zupt = estimator.try_zupt(curr_cam_time)
            t_zupt_local = time.perf_counter() - t0
            if did_zupt:
                n_zupt += 1
                continue

            # Propagation, feature selection, update, and database cleanup
            # in a single C++ call (no feature list round-trip)
            t0 = time.perf_counter()
            did_update = estimator.process_frame(frontend, curr_cam_time, [0, 1])
            t_fused += time.perf_counter() - t0
            if did_update:
                n_upd += 1
        except Exception as e:
            print(f"Error at frame {i - start_cam}: {e}")
            continue

        # Store State
        state = estimator.get_state()
        if state.is_initialized:
            pos = state.imu.pos()
            trajectory.append(pos.copy())
            timestamps.append(state.timestamp)

        if (i - start_cam) % 200 == 0:
            print(f"Processed {i - start_cam}/{n_frames} frames...")

    t_run = time.perf_counter() - t_run0
    n_done = max(len(timestamps), 1)
    print("\n==== Timing Summary ====")
    print(f"total wall time:        {t_run:8.2f} s  ({t_run / n_done * 1000:6.2f} ms/frame, {n_done / t_run:5.1f} FPS)")
    print(f"IMU batch feeding:      {t_imu:8.2f} s")
    print(f"image reading (cv2):    {t_read:8.2f} s")
    print(f"frontend tracking:      {t_track:8.2f} s")
    print(f"propagate+update (fused): {t_fused:6.2f} s  ({t_fused / max(n_upd, 1) * 1000:.2f} ms/update, n={n_upd})")
    print(f"ZUPT updates: {n_zupt}")

    return np.array(timestamps), np.array(trajectory)


def umeyama_align(model, data):
    """Least-squares rigid transform (R, t) with data ~= R @ model + t."""
    mu_m = model.mean(axis=0)
    mu_d = data.mean(axis=0)
    H = (model - mu_m).T @ (data - mu_d)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = mu_d - R @ mu_m
    return R, t


def evaluate_ate(est_stamps, est_traj, gt_stamps, gt_traj):
    # Interpolate GT to the estimate timestamps
    if len(est_traj) == 0:
        return 0, None, None

    f_x = interp1d(gt_stamps, gt_traj[:, 0], bounds_error=False, fill_value="extrapolate")
    f_y = interp1d(gt_stamps, gt_traj[:, 1], bounds_error=False, fill_value="extrapolate")
    f_z = interp1d(gt_stamps, gt_traj[:, 2], bounds_error=False, fill_value="extrapolate")

    gt_interp = np.zeros((len(est_stamps), 3))
    gt_interp[:, 0] = f_x(est_stamps)
    gt_interp[:, 1] = f_y(est_stamps)
    gt_interp[:, 2] = f_z(est_stamps)

    # ATE (RMSE) after Umeyama SE(3) alignment (no scale: system is metric)
    R, t = umeyama_align(est_traj, gt_interp)
    est_traj_aligned = est_traj @ R.T + t

    errors = np.linalg.norm(est_traj_aligned - gt_interp, axis=1)
    ate = np.sqrt(np.mean(errors**2)) # RMSE ATE

    return ate, est_traj_aligned, gt_interp


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else "data/euroc/MH_01_easy"
    config_path = sys.argv[2] if len(sys.argv) > 2 else "config/euroc_mav/estimator_config.yaml"
    max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else None

    # Run VIO
    est_stamps, est_traj = run_vio(dataset_path, config_path, max_frames)

    if len(est_traj) == 0:
        print("VIO failed to produce any trajectory.")
        sys.exit(1)

    # Save trajectory for offline re-evaluation
    tag = os.path.basename(os.path.normpath(dataset_path))
    np.savez(f"trajectory_{tag}.npz", stamps=est_stamps, traj=est_traj)
    print(f"Trajectory saved to trajectory_{tag}.npz")

    # Load GT (optional: some dataset copies ship without ground truth)
    gt_path = os.path.join(dataset_path, 'mav0/state_groundtruth_estimate0/data.csv')
    if not os.path.exists(gt_path):
        print(f"No ground truth found at {gt_path}; skipping ATE evaluation.")
        sys.exit(0)

    gt_df = pd.read_csv(gt_path)
    gt_df.columns = [c.strip().split(' ')[0] for c in gt_df.columns]
    gt_stamps = gt_df['#timestamp'].values * 1e-9
    gt_traj = gt_df[['p_RS_R_x', 'p_RS_R_y', 'p_RS_R_z']].values

    # Evaluate
    ate, est_aligned, gt_interp = evaluate_ate(est_stamps, est_traj, gt_stamps, gt_traj)
    print(f"Final ATE: {ate:.4f} meters")

    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'g--', label='GT Trajectory')
    plt.plot(est_aligned[:, 0], est_aligned[:, 1], 'b-', label='Estimated (Aligned)')

    plt.scatter(est_aligned[0, 0], est_aligned[0, 1], c='r', marker='o', s=100, label='Start')
    plt.scatter(est_aligned[-1, 0], est_aligned[-1, 1], c='k', marker='x', s=100, label='End')

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title(f'Sqrt-VINS Trajectory on EuRoC MH_01_easy\nATE: {ate:.4f}m')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    plt.savefig('trajectory_comparison_euroc.png')
    print("Plot saved as trajectory_comparison_euroc.png")
