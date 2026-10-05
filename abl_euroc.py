"""Ablation of the new binding features to isolate an accuracy regression on EuRoC V1_01.

Toggles (all default OFF = original orchestration):
  --batch    feed IMU via feed_imu_batch (proper oldest_time) instead of per-msg feed_imu(-1)
  --cleanup  call FeatureDatabase cleanup_measurements/cleanup like VioManager does
  --fused    use process_frame (implies cleanup) instead of split calls
  --zupt     call estimator.try_zupt each frame
Saves trajectory npz and prints timing.
"""
import os, sys, time
import numpy as np
import pandas as pd
import cv2

sys.path.append(os.path.join(os.getcwd(), 'build/ov_srvins'))
import ov_srvins_py as vins

flags = {f: f in sys.argv for f in ('--batch', '--cleanup', '--fused', '--zupt', '--trimmsg', '--proponlytrim', '--triminit', '--cleanuppre', '--propfirst')}
dataset_path = sys.argv[sys.argv.index('--dataset') + 1] if '--dataset' in sys.argv else \
    '/media/shzhou/T7/Dataset/euroc/vicon_room1/V1_01_easy/'
config_path = 'config/euroc_mav/estimator_config.yaml'
tag = os.path.basename(os.path.normpath(dataset_path))
name = ('abl_' + '_'.join(k[2:] for k in flags if flags[k]) + f'_{tag}') or 'abl_base_' + tag
if not any(flags.values()):
    name = f'abl_base_{tag}'

options = vins.VinsOptions()
options.print_and_load(vins.YamlParser(config_path))
estimator = vins.SqrtEstimator(options)
frontend = vins.Frontend(options, estimator.get_state())
initializer = vins.InertialInitializer(
    options.init_options, frontend.get_trackFEATS().get_feature_database(),
    estimator.get_propagator(), options.msckf_options, options.slam_options,
    options.featinit_options)
db = frontend.get_trackFEATS().get_feature_database()

imu_df = pd.read_csv(os.path.join(dataset_path, 'mav0/imu0/data.csv'))
cam0_df = pd.read_csv(os.path.join(dataset_path, 'mav0/cam0/data.csv'))
cam1_df = pd.read_csv(os.path.join(dataset_path, 'mav0/cam1/data.csv'))
for df in (imu_df, cam0_df, cam1_df):
    df.columns = [c.strip().split(' ')[0] for c in df.columns]
imu_df, cam0_df, cam1_df = imu_df.sort_values('#timestamp'), cam0_df.sort_values('#timestamp'), cam1_df.sort_values('#timestamp')
imu_times = imu_df['#timestamp'].values * 1e-9
imu_wm = imu_df[['w_RS_S_x', 'w_RS_S_y', 'w_RS_S_z']].to_numpy()
imu_am = imu_df[['a_RS_S_x', 'a_RS_S_y', 'a_RS_S_z']].to_numpy()
cam0_times = cam0_df['#timestamp'].values * 1e-9
cam1_lookup = {t: i for i, t in enumerate(cam1_df['#timestamp'].values * 1e-9)}

frontend.set_startup_time(cam0_times[0])
zero_mask = np.zeros((480, 752), dtype=np.uint8)

imu_idx = 0
traj_t, traj_p = [], []
t_est = 0.0
t0_all = time.perf_counter()
n_err = 0

for i in range(len(cam0_times)):
    curr_cam_time = cam0_times[i]
    img0 = cv2.imread(os.path.join(dataset_path, 'mav0/cam0/data', cam0_df.iloc[i]['filename']), cv2.IMREAD_GRAYSCALE)
    j = cam1_lookup.get(curr_cam_time, -1)
    img1 = cv2.imread(os.path.join(dataset_path, 'mav0/cam1/data', cam1_df.iloc[j]['filename']), cv2.IMREAD_GRAYSCALE) if j >= 0 else None
    if img0 is None or img1 is None:
        continue

    t0 = time.perf_counter()
    if flags['--batch'] or flags['--fused']:
        k = np.searchsorted(imu_times, curr_cam_time, side='right')
        if k > imu_idx:
            estimator.feed_imu_batch(imu_times[imu_idx:k], imu_wm[imu_idx:k], imu_am[imu_idx:k])
            imu_idx = k
    else:
        while imu_idx < len(imu_times) and imu_times[imu_idx] <= curr_cam_time:
            m = vins.ImuData()
            m.timestamp = imu_times[imu_idx]
            m.wm = imu_wm[imu_idx]
            m.am = imu_am[imu_idx]
            if flags['--triminit']:
                st = estimator.get_state()
                if st.is_initialized:
                    estimator.feed_measurement_imu(m)
                else:
                    estimator.feed_imu(m, -1.0)
            elif flags['--trimmsg']:
                estimator.feed_measurement_imu(m)
            else:
                estimator.feed_imu(m, -1.0)
            imu_idx += 1
        if flags['--proponlytrim']:
            st = estimator.get_state()
            ot = st.margtimestep()
            if st.is_initialized and ot <= st.timestamp:
                estimator.get_propagator().clean_old_imu_measurements(ot)
    t_est += time.perf_counter() - t0

    cam_msg = vins.CameraData()
    cam_msg.timestamp = curr_cam_time
    cam_msg.sensor_ids = [0, 1]
    cam_msg.images = [img0, img1]
    cam_msg.masks = [zero_mask, zero_mask]

    try:
        t0 = time.perf_counter()
        frontend.feed_camera(cam_msg)
        t_est += time.perf_counter() - t0

        state = estimator.get_state()
        if not state.is_initialized:
            if initializer.initialize(state, not options.try_zupt):
                print(f"VIO Initialized at frame {i}")
            continue

        if flags['--zupt'] and estimator.try_zupt(curr_cam_time):
            continue

        t0 = time.perf_counter()
        if flags['--fused']:
            estimator.process_frame(frontend, curr_cam_time, [0, 1])
        elif flags['--propfirst']:
            # VioManager order: propagate, optional pre-rules cleanup, rules, update
            ok = estimator.propagate(curr_cam_time)
            if ok:
                state = estimator.get_state()
                if flags['--cleanuppre'] and state.num_clones() > options.state_options.max_clone_size + 1:
                    db.cleanup_measurements(state.margtimestep())
            feats_msckf, feats_slam = frontend.process_measurements_rules(state, curr_cam_time, [0, 1])
            if ok:
                estimator.update(feats_msckf, feats_slam)
                estimator.notify_moved()
            if flags['--cleanup']:
                state = estimator.get_state()
                if state.num_clones() > options.state_options.max_clone_size + 1:
                    db.cleanup_measurements(state.margtimestep())
                db.cleanup()
        else:
            feats_msckf, feats_slam = frontend.process_measurements_rules(state, curr_cam_time, [0, 1])
            if estimator.propagate(curr_cam_time):
                estimator.update(feats_msckf, feats_slam)
                estimator.notify_moved()
            if flags['--cleanup']:
                state = estimator.get_state()
                if state.num_clones() > options.state_options.max_clone_size + 1:
                    db.cleanup_measurements(state.margtimestep())
                db.cleanup()
        t_est += time.perf_counter() - t0
    except Exception as e:
        n_err += 1
        if n_err <= 3:
            print(f"frame {i}: {e}")
        continue

    state = estimator.get_state()
    if state.is_initialized:
        traj_t.append(state.timestamp)
        traj_p.append(state.imu.pos().copy())

t_run = time.perf_counter() - t0_all
np.savez(f'{name}.npz', stamps=np.array(traj_t), traj=np.array(traj_p))
print(f"{name}: wall {t_run:.1f}s, estimator {t_est:.1f}s ({t_est/len(cam0_times)*1000:.2f} ms/frame), errors {n_err}")
