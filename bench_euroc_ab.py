"""A/B benchmark: old vs new Python binding orchestration on EuRoC.

--old : per-message ImuData construction + feed_imu(-1) + split
        process_measurements_rules / propagate / update, no database cleanup
        (the pre-fix orchestration).
default: batched IMU feed + fused propagate_and_update + ZUPT + cleanup
        (the new orchestration).

Image reading is excluded from timing in both modes.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import cv2

sys.path.append(os.path.join(os.getcwd(), 'build/ov_srvins'))
import ov_srvins_py as vins

OLD = '--old' in sys.argv
dataset_path = sys.argv[sys.argv.index('--dataset') + 1] if '--dataset' in sys.argv else '/media/shzhou/T7_2/asl_dataset/euroc_mav/MH_01_easy/'
config_path = 'config/euroc_mav/estimator_config.yaml'

options = vins.VioManagerOptions()
options.print_and_load(vins.YamlParser(config_path))
estimator = vins.SqrtEstimator(options)
frontend = vins.Frontend(options, estimator.get_state())
initializer = vins.InertialInitializer(
    options.init_options,
    frontend.get_trackFEATS().get_feature_database(),
    estimator.get_propagator(),
    options.msckf_options, options.slam_options, options.featinit_options)

imu_df = pd.read_csv(os.path.join(dataset_path, 'mav0/imu0/data.csv'))
cam0_df = pd.read_csv(os.path.join(dataset_path, 'mav0/cam0/data.csv'))
cam1_df = pd.read_csv(os.path.join(dataset_path, 'mav0/cam1/data.csv'))
for df in (imu_df, cam0_df, cam1_df):
    df.columns = [c.strip().split(' ')[0] for c in df.columns]
imu_df = imu_df.sort_values('#timestamp')
cam0_df = cam0_df.sort_values('#timestamp')
cam1_df = cam1_df.sort_values('#timestamp')

imu_times = imu_df['#timestamp'].values / 1e9
imu_wm = imu_df[['w_RS_S_x', 'w_RS_S_y', 'w_RS_S_z']].to_numpy()
imu_am = imu_df[['a_RS_S_x', 'a_RS_S_y', 'a_RS_S_z']].to_numpy()
cam0_times = cam0_df['#timestamp'].values / 1e9
cam1_lookup = {t: i for i, t in enumerate((cam1_df['#timestamp'].values / 1e9))}

frontend.set_startup_time(cam0_times[0])
zero_mask = np.zeros((480, 752), dtype=np.uint8)

imu_idx = 0
traj_t, traj_p = [], []
t_estimator = 0.0        # binding/estimator time only (no imread)
t_run0 = time.perf_counter()
n_frames = len(cam0_times)
SEG = 500
seg_t0, seg_frames = time.perf_counter(), 0
n_err = 0

for i in range(n_frames):
    curr_cam_time = cam0_times[i]
    img0 = cv2.imread(os.path.join(dataset_path, 'mav0/cam0/data', cam0_df.iloc[i]['filename']), cv2.IMREAD_GRAYSCALE)
    j = cam1_lookup.get(curr_cam_time, -1)
    img1 = cv2.imread(os.path.join(dataset_path, 'mav0/cam1/data', cam1_df.iloc[j]['filename']), cv2.IMREAD_GRAYSCALE) if j >= 0 else None
    if img0 is None or img1 is None:
        continue

    t0 = time.perf_counter()
    if OLD:
        while imu_idx < len(imu_times) and imu_times[imu_idx] <= curr_cam_time:
            imu_msg = vins.ImuData()
            imu_msg.timestamp = imu_times[imu_idx]
            imu_msg.wm = imu_wm[imu_idx]
            imu_msg.am = imu_am[imu_idx]
            estimator.feed_imu(imu_msg, -1.0)
            imu_idx += 1
    else:
        k = np.searchsorted(imu_times, curr_cam_time, side='right')
        if k > imu_idx:
            estimator.feed_imu_batch(imu_times[imu_idx:k], imu_wm[imu_idx:k], imu_am[imu_idx:k])
            imu_idx = k
    t_estimator += time.perf_counter() - t0

    cam_msg = vins.CameraData()
    cam_msg.timestamp = curr_cam_time
    cam_msg.sensor_ids = [0, 1]
    cam_msg.images = [img0, img1]
    cam_msg.masks = [zero_mask, zero_mask]

    try:
        t0 = time.perf_counter()
        frontend.feed_camera(cam_msg)
        t_estimator += time.perf_counter() - t0

        state = estimator.get_state()
        if not state.is_initialized:
            if initializer.initialize(state, not options.try_zupt):
                print(f"VIO Initialized at frame {i}")
            continue

        if not OLD:
            if estimator.try_zupt(curr_cam_time, True):
                continue

        t0 = time.perf_counter()
        if OLD:
            feats_msckf, feats_up, feats_delayed = frontend.process_measurements_rules(curr_cam_time, [0, 1])
            if estimator.propagate(curr_cam_time):
                estimator.update(feats_msckf, feats_up, feats_delayed)
        else:
            estimator.propagate_and_update(frontend, curr_cam_time, [0, 1])
        t_estimator += time.perf_counter() - t0
        state = estimator.get_state()
        if state.is_initialized:
            traj_t.append(state.timestamp)
            traj_p.append(state.imu.pos().copy())
    except Exception as e:
        n_err += 1
        if n_err <= 3:
            print(f"frame {i}: {e}")
        continue

    seg_frames += 1
    if seg_frames == SEG:
        now = time.perf_counter()
        print(f"frames {i - seg_frames + 1:5d}-{i:5d}: {now - seg_t0 - 0:6.2f} s wall | estimator-only {t_estimator:6.2f} s cum")
        seg_t0, seg_frames = now, 0

t_run = time.perf_counter() - t_run0
tag = os.path.basename(os.path.normpath(dataset_path))
np.savez("benchtraj_%s_%s.npz" % ("old" if OLD else "new", tag), stamps=np.array(traj_t), traj=np.array(traj_p))
print(f"\nmode={'OLD' if OLD else 'NEW'}  total wall: {t_run:.2f} s  estimator+binding only: {t_estimator:.2f} s  "
      f"({t_estimator / n_frames * 1000:.2f} ms/frame)  errors: {n_err}")
