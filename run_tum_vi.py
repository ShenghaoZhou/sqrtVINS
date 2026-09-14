import os
import sys
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

# Add build directory to path for ov_srvins_py
sys.path.append(os.path.join(os.getcwd(), 'build/ov_srvins'))
import ov_srvins_py as vins

def run_vio(dataset_path, config_path):
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
    
    # Rename columns to remove spaces and units
    imu_df.columns = [c.strip().split(' ')[0] for c in imu_df.columns]
    cam0_df.columns = [c.strip().split(' ')[0] for c in cam0_df.columns]
    
    # Sort by timestamp
    imu_df = imu_df.sort_values('#timestamp')
    cam0_df = cam0_df.sort_values('#timestamp')
    
    cam_times = cam0_df['#timestamp'].values / 1e9
    imu_times = imu_df['#timestamp'].values / 1e9
    imu_wm = imu_df[['w_RS_S_x', 'w_RS_S_y', 'w_RS_S_z']].to_numpy()
    imu_am = imu_df[['a_RS_S_x', 'a_RS_S_y', 'a_RS_S_z']].to_numpy()
    
    # Load Masks if enabled
    masks = []
    if options.use_mask:
        mask0 = cv2.imread(os.path.join(os.getcwd(), 'config/tum_vi/mask_tumvi0.png'), cv2.IMREAD_GRAYSCALE)
        if mask0 is None:
             print("Warning: Could not load mask. Disabling masking.")
             options.use_mask = False
        else:
            masks = [mask0]

    # 4. Processing Loop
    imu_idx = 0
    cam_idx = 0
    trajectory = []
    timestamps = []
    has_moved_since_zupt = False
    zero_mask = None
    
    # Skip camera frames before the first IMU
    while cam_idx < len(cam_times) and cam_times[cam_idx] < imu_times[0]:
        cam_idx += 1

    print(f"Starting VIO processing for {len(cam_times) - cam_idx} images...")
    
    # Initial startup time
    frontend.set_startup_time(cam_times[cam_idx])
    
    try:
        while cam_idx < len(cam_times):
            curr_cam_time = cam_times[cam_idx]

            # Feed IMU measurements up to this camera time in one batched call.
            # Include one sample past the camera time: the propagator needs an
            # IMU reading after the camera time to close the final integration
            # interval (select_imu_readings).
            k = np.searchsorted(imu_times, curr_cam_time, side='right')
            if k < len(imu_times):
                k += 1
            if k > imu_idx:
                estimator.feed_imu_batch(imu_times[imu_idx:k], imu_wm[imu_idx:k], imu_am[imu_idx:k])
                imu_idx = k

            # Feed Camera measurement
            cam_msg = vins.CameraData()
            cam_msg.timestamp = curr_cam_time
            cam_msg.sensor_ids = [0]

            img0_path = os.path.join(dataset_path, f"mav0/cam0/data/{cam0_df.iloc[cam_idx]['filename']}")

            img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)

            if img0 is None:
                cam_idx += 1
                continue

            cam_msg.images = [img0]
            if options.use_mask:
                cam_msg.masks = masks
            else:
                if zero_mask is None:
                    zero_mask = np.zeros(img0.shape, dtype=np.uint8)
                cam_msg.masks = [zero_mask]

            try:
                frontend.feed_camera(cam_msg)

                state = estimator.get_state()

                # Check for initialization
                if not state.is_initialized:
                    if initializer.initialize(state, not options.try_zupt):
                        print(f"VIO Initialized at {curr_cam_time}!")
                        frontend.set_startup_time(curr_cam_time)
                        frontend.get_trackFEATS().get_feature_database().cleanup_measurements(state.timestamp)
                        if np.linalg.norm(state.imu.vel()) > options.zupt_max_velocity:
                            has_moved_since_zupt = True
                    cam_idx += 1
                    continue

                # Try a zero-velocity update (mirrors track_image_and_update)
                if estimator.try_zupt(curr_cam_time, has_moved_since_zupt):
                    cam_idx += 1
                    continue

                # Propagation, feature selection, update, and database cleanup
                # in a single C++ call (no feature list round-trip)
                if estimator.propagate_and_update(frontend, curr_cam_time, [0]):
                    has_moved_since_zupt = True
            except Exception as e:
                # print(f"Error at frame {cam_idx}: {e}")
                pass
            
            # Store State
            state = estimator.get_state()
            if state.is_initialized:
                pos = state.imu.pos()
                trajectory.append(pos.copy())
                timestamps.append(state.timestamp)
            if cam_idx % 100 == 0:
                print(f"Processed {cam_idx}/{len(cam_times)} frames...")
            
            cam_idx += 1
            # Run more frames for better ATE evaluation
            if cam_idx > 5000:
                break
            
    except Exception as e:
        print(f"Error during processing: {e}")

    return np.array(timestamps), np.array(trajectory)

def evaluate_ate(est_stamps, est_traj, gt_stamps, gt_traj):
    # Align timestamps
    # Interpolate GT to estimate timestamps
    if len(est_traj) == 0:
        return 0, None, None
        
    f_x = interp1d(gt_stamps, gt_traj[:, 0], bounds_error=False, fill_value="extrapolate")
    f_y = interp1d(gt_stamps, gt_traj[:, 1], bounds_error=False, fill_value="extrapolate")
    f_z = interp1d(gt_stamps, gt_traj[:, 2], bounds_error=False, fill_value="extrapolate")
    
    gt_interp = np.zeros((len(est_stamps), 3))
    gt_interp[:, 0] = f_x(est_stamps)
    gt_interp[:, 1] = f_y(est_stamps)
    gt_interp[:, 2] = f_z(est_stamps)
    
    # Calculate ATE (Absolute Trajectory Error) after Umeyama alignment
    # Since we only have XYZ, we'll do a simple alignment of the first point
    offset = gt_interp[0] - est_traj[0]
    est_traj_aligned = est_traj + offset
    
    errors = np.linalg.norm(est_traj_aligned - gt_interp, axis=1)
    ate = np.sqrt(np.mean(errors**2)) # RMSE ATE
    
    return ate, est_traj_aligned, gt_interp

if __name__ == "__main__":
    dataset_path = "data/tum-vi/dataset-room4_512_16"
    config_path = "config/tum_vi/estimator_config_mono.yaml"
    
    # Run VIO
    est_stamps, est_traj = run_vio(dataset_path, config_path)
    
    if len(est_traj) == 0:
        print("VIO failed to produce any trajectory.")
        sys.exit(1)
        
    # Load GT
    gt_df = pd.read_csv(os.path.join(dataset_path, 'mav0/mocap0/data.csv'))
    gt_df.columns = [c.strip().split(' ')[0] for c in gt_df.columns]
    gt_stamps = gt_df['#timestamp'].values / 1e9
    gt_traj = gt_df[['p_RS_R_x', 'p_RS_R_y', 'p_RS_R_z']].values
    
    # Evaluate
    ate, est_aligned, gt_interp = evaluate_ate(est_stamps, est_traj, gt_stamps, gt_traj)
    print(f"Final ATE (XY): {ate:.4f} meters")
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(gt_traj[:, 0], gt_traj[:, 1], 'g--', label='GT Trajectory')
    plt.plot(est_aligned[:, 0], est_aligned[:, 1], 'b-', label='Estimated (Aligned)')
    
    plt.scatter(est_aligned[0, 0], est_aligned[0, 1], c='r', marker='o', s=100, label='Start')
    plt.scatter(est_aligned[-1, 0], est_aligned[-1, 1], c='k', marker='x', s=100, label='End')
    
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title(f'Sqrt-VINS Trajectory on TUM-VI Room4\nATE: {ate:.4f}m')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    plt.savefig('trajectory_comparison.png')
    print("Plot saved as trajectory_comparison.png")
    # plt.show() # Headless
