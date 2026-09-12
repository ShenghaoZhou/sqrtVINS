import os
import sys
import numpy as np
import pandas as pd
import cv2

# Add build directory to path for ov_srvins_py
sys.path.append(os.path.join(os.getcwd(), 'build/ov_srvins'))
import ov_srvins_py as vins


def run_vio(dataset_path, config_path, max_frames=1000):
    # 1. Setup options
    options = vins.VioManagerOptions()
    parser = vins.YamlParser(config_path)
    options.print_and_load(parser)

    # 2. Initialize Estimator and Frontend
    estimator = vins.SqrtEstimator(options)
    frontend = vins.Frontend(options, estimator.get_state())

    # 3. Setup Initializer
    initializer = vins.InertialInitializer(
        options.init_options,
        frontend.get_trackFEATS().get_feature_database(),
        estimator.get_propagator(),
        options.msckf_options,
        options.slam_options,
        options.featinit_options
    )

    # 4. Load Data
    imu_df = pd.read_csv(os.path.join(dataset_path, 'mav0/imu0/data.csv'))
    cam0_df = pd.read_csv(os.path.join(dataset_path, 'mav0/cam0/data.csv'))
    cam1_df = pd.read_csv(os.path.join(dataset_path, 'mav0/cam1/data.csv'))

    imu_df.columns = [c.strip().split(' ')[0] for c in imu_df.columns]
    cam0_df.columns = [c.strip().split(' ')[0] for c in cam0_df.columns]
    cam1_df.columns = [c.strip().split(' ')[0] for c in cam1_df.columns]

    imu_df = imu_df.sort_values('#timestamp')
    cam0_df = cam0_df.sort_values('#timestamp')
    cam1_df = cam1_df.sort_values('#timestamp')

    # Intersect cam0/cam1 timestamps for stereo processing (keep ns integers
    # for exact file lookups, seconds for the processing loop)
    cam_times_ns = np.intersect1d(cam0_df['#timestamp'].values,
                                  cam1_df['#timestamp'].values)
    cam_times = cam_times_ns / 1e9
    cam0_lookup = cam0_df.set_index('#timestamp')['filename']
    cam1_lookup = cam1_df.set_index('#timestamp')['filename']

    imu_times = imu_df['#timestamp'].values / 1e9
    imu_wm = imu_df[['w_RS_S_x', 'w_RS_S_y', 'w_RS_S_z']].to_numpy()
    imu_am = imu_df[['a_RS_S_x', 'a_RS_S_y', 'a_RS_S_z']].to_numpy()

    # Preload zero masks
    zero_mask = None

    imu_idx = 0
    cam_idx = 0
    trajectory = []
    timestamps = []
    num_feats = []
    has_moved_since_zupt = False

    # Skip camera frames before the first IMU
    while cam_idx < len(cam_times) and cam_times[cam_idx] < imu_times[0]:
        cam_idx += 1

    print(f"Starting VIO processing for {min(max_frames, len(cam_times) - cam_idx)} stereo frames...")

    frontend.set_startup_time(cam_times[cam_idx])

    processed = 0
    while cam_idx < len(cam_times) and processed < max_frames:
        curr_cam_time = cam_times[cam_idx]

        # Feed IMU measurements up to this camera time in one batched call
        k = np.searchsorted(imu_times, curr_cam_time, side='right')
        if k > imu_idx:
            estimator.feed_imu_batch(imu_times[imu_idx:k], imu_wm[imu_idx:k], imu_am[imu_idx:k])
            imu_idx = k

        # Load stereo images
        img0 = cv2.imread(os.path.join(dataset_path, 'mav0/cam0/data', cam0_lookup.loc[cam_times_ns[cam_idx]]), cv2.IMREAD_GRAYSCALE)
        img1 = cv2.imread(os.path.join(dataset_path, 'mav0/cam1/data', cam1_lookup.loc[cam_times_ns[cam_idx]]), cv2.IMREAD_GRAYSCALE)
        if img0 is None or img1 is None:
            cam_idx += 1
            continue

        cam_msg = vins.CameraData()
        cam_msg.timestamp = curr_cam_time
        cam_msg.sensor_ids = [0, 1]
        cam_msg.images = [img0, img1]
        if zero_mask is None:
            zero_mask = np.zeros(img0.shape, dtype=np.uint8)
        cam_msg.masks = [zero_mask, zero_mask.copy()]

        try:
            frontend.feed_camera(cam_msg)

            state = estimator.get_state()

            # Check for initialization
            if not state.is_initialized:
                if initializer.initialize(state, False):
                    print(f"VIO Initialized at {curr_cam_time}!")
                    frontend.set_startup_time(curr_cam_time)
                    frontend.get_trackFEATS().get_feature_database().cleanup_measurements(state.timestamp)
                    if np.linalg.norm(state.imu.vel()) > options.zupt_max_velocity:
                        has_moved_since_zupt = True
                cam_idx += 1
                processed += 1
                continue

            # Try a zero-velocity update
            if estimator.try_zupt(curr_cam_time, has_moved_since_zupt):
                cam_idx += 1
                processed += 1
                continue

            # Propagation, feature selection, update, and database cleanup
            if estimator.propagate_and_update(frontend, curr_cam_time, [0, 1]):
                has_moved_since_zupt = True
        except Exception as e:
            print(f"Error at frame {cam_idx}: {e}")

        # Store State
        state = estimator.get_state()
        if state.is_initialized:
            pos = state.imu.pos()
            trajectory.append(pos.copy())
            timestamps.append(state.timestamp)

        processed += 1
        if processed % 200 == 0:
            print(f"Processed {processed}/{max_frames} frames...")

        cam_idx += 1

    return np.array(timestamps), np.array(trajectory)


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else \
        "/media/shzhou/T7_2/asl_dataset/euroc_mav/MH_01_easy"
    config_path = sys.argv[2] if len(sys.argv) > 2 else \
        "config/euroc_mav/estimator_config.yaml"
    out_csv = sys.argv[3] if len(sys.argv) > 3 else "/tmp/euroc_traj.csv"
    max_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

    est_stamps, est_traj = run_vio(dataset_path, config_path, max_frames)

    if len(est_traj) == 0:
        print("VIO failed to produce any trajectory.")
        sys.exit(1)

    pd.DataFrame({'timestamp': est_stamps,
                  'p0': est_traj[:, 0], 'p1': est_traj[:, 1],
                  'p2': est_traj[:, 2]}).to_csv(out_csv, index=False)
    print(f"Wrote {len(est_traj)} poses to {out_csv}")
