"""ATE comparison on EuRoC: default FAST+KLT frontend vs XFeat detection.

Runs the full Sqrt-VINS pipeline twice on the same sequence with identical
configuration, changing only the feature detector:

  1. default  - built-in OpenCV backend (FAST grid detection + KLT)
  2. xfeat    - XFeat detection (vismatch accelerated_features) injected via
                frontend.set_cv_backend(pysqrtvins.vision.Backend(...));
                pyramids / KLT tracking / RANSAC stay with OpenCV.

Both runs are evaluated against the ground truth with the same ATE metric
used by run_euroc.py (Umeyama SE(3) alignment, no scale, RMSE).

Usage:
    pixi run python pysqrtvins/examples/compare_xfeat_euroc.py \
        --dataset /path/to/MH_03_medium

The dataset directory may either be the EuRoC root (containing mav0/) or the
mav0 directory itself (cam0/, cam1/, imu0/ at its top level).
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "build" / "pysqrtvins"))
sys.path.insert(0, str(REPO / "build" / "ov_srvins"))
sys.path.insert(0, str(REPO))  # run_euroc.py (ATE evaluation)
sys.path.insert(
    0,
    str(
        REPO / "third_party" / "vismatch" / "vismatch" / "third_party"
        / "accelerated_features"
    ),
)

import pysqrtvins as sv  # noqa: E402
import ov_srvins_py as vins  # noqa: E402
from run_euroc import evaluate_ate  # noqa: E402


def resolve_mav0(dataset_path):
    """EuRoC root or the mav0 directory itself."""
    mav0 = os.path.join(dataset_path, "mav0")
    return mav0 if os.path.isdir(mav0) else dataset_path


def run_vio(dataset_path, config_path, backend=None, max_frames=None):
    """Same pipeline as run_euroc.run_vio, with an optional injected CV
    backend (must be set before the first feed_camera)."""
    options = vins.VinsOptions()
    parser = vins.YamlParser(config_path)
    options.print_and_load(parser)

    cv2.setNumThreads(options.num_opencv_threads)
    cv2.setRNGSeed(0)

    system = vins.System.create(options)
    estimator, frontend = system.estimator, system.frontend
    init_runner = system.init_runner

    if backend is not None:
        frontend.set_cv_backend(backend)

    root = resolve_mav0(dataset_path)
    imu_df = pd.read_csv(os.path.join(root, "imu0/data.csv"))
    cam0_df = pd.read_csv(os.path.join(root, "cam0/data.csv"))
    cam1_df = pd.read_csv(os.path.join(root, "cam1/data.csv"))
    for df in (imu_df, cam0_df, cam1_df):
        df.columns = [c.strip().split(" ")[0] for c in df.columns]
    imu_df = imu_df.sort_values("#timestamp")
    cam0_df = cam0_df.sort_values("#timestamp")
    cam1_df = cam1_df.sort_values("#timestamp")

    cam0_times = cam0_df["#timestamp"].values * 1e-9
    cam1_times = cam1_df["#timestamp"].values * 1e-9
    imu_times = imu_df["#timestamp"].values * 1e-9
    imu_wm = imu_df[["w_RS_S_x", "w_RS_S_y", "w_RS_S_z"]].to_numpy()
    imu_am = imu_df[["a_RS_S_x", "a_RS_S_y", "a_RS_S_z"]].to_numpy()

    cam1_lookup = {t: i for i, t in enumerate(cam1_times)}

    start_cam = 0
    while start_cam < len(cam0_times) and cam0_times[start_cam] < imu_times[0]:
        start_cam += 1

    n_frames = len(cam0_times) - start_cam
    if max_frames is not None:
        n_frames = min(n_frames, max_frames)
    print(f"  processing {n_frames} stereo frames...")

    frontend.set_startup_time(cam0_times[start_cam])

    imu_idx = 0
    trajectory, timestamps = [], []
    zero_mask = None
    t_track = 0.0

    t_run0 = time.perf_counter()
    for i in range(start_cam, start_cam + n_frames):
        curr_cam_time = cam0_times[i]

        t_off = estimator.get_state().cam_imu_timeoffset()
        k = vins.imu_batch_end(imu_times, imu_idx, curr_cam_time + t_off)
        if k > imu_idx:
            estimator.feed_imu_batch(
                imu_times[imu_idx:k], imu_wm[imu_idx:k], imu_am[imu_idx:k]
            )
            imu_idx = k

        img0 = cv2.imread(
            os.path.join(root, "cam0/data", cam0_df.iloc[i]["filename"]),
            cv2.IMREAD_GRAYSCALE,
        )
        j = cam1_lookup.get(curr_cam_time, -1)
        img1 = (
            cv2.imread(
                os.path.join(root, "cam1/data", cam1_df.iloc[j]["filename"]),
                cv2.IMREAD_GRAYSCALE,
            )
            if j >= 0
            else None
        )
        if img0 is None or img1 is None:
            continue

        cam_msg = vins.CameraData()
        cam_msg.timestamp = curr_cam_time
        cam_msg.sensor_ids = [0, 1]
        cam_msg.images = [img0, img1]
        if zero_mask is None:
            zero_mask = np.zeros(img0.shape, dtype=np.uint8)
        cam_msg.masks = [zero_mask, zero_mask]

        try:
            t0 = time.perf_counter()
            frontend.feed_camera(cam_msg)
            t_track += time.perf_counter() - t0

            if not estimator.get_state().is_initialized:
                if init_runner.try_initialize(curr_cam_time, not options.try_zupt):
                    print(f"  VIO initialized at {curr_cam_time:.3f}")
                continue

            if estimator.try_zupt(curr_cam_time):
                continue

            estimator.process_frame(frontend, curr_cam_time, [0, 1])
        except Exception as e:
            print(f"  error at frame {i - start_cam}: {e}")
            continue

        state = estimator.get_state()
        if state.is_initialized:
            trajectory.append(state.imu.pos().copy())
            timestamps.append(state.timestamp)

        if (i - start_cam) % 500 == 0:
            print(f"  frame {i - start_cam}/{n_frames}")

    t_run = time.perf_counter() - t_run0
    n_done = max(len(timestamps), 1)
    print(
        f"  wall {t_run:.1f}s ({1e3 * t_run / n_done:.1f} ms/frame), "
        f"frontend tracking {t_track:.1f}s"
    )
    return np.array(timestamps), np.array(trajectory)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, help="EuRoC sequence directory")
    ap.add_argument(
        "--config", default=str(REPO / "config/euroc_mav/estimator_config.yaml")
    )
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--device", default="cpu", help="torch device for XFeat")
    ap.add_argument("--max-keypoints", type=int, default=1024)
    ap.add_argument(
        "--skip-default", action="store_true", help="only run the XFeat config"
    )
    args = ap.parse_args()

    root = resolve_mav0(args.dataset)
    gt_path = os.path.join(root, "state_groundtruth_estimate0/data.csv")
    gt_stamps = gt_traj = None
    if os.path.exists(gt_path):
        gt_df = pd.read_csv(gt_path)
        gt_df.columns = [c.strip().split(" ")[0] for c in gt_df.columns]
        gt_stamps = gt_df["#timestamp"].values * 1e-9
        gt_traj = gt_df[["p_RS_R_x", "p_RS_R_y", "p_RS_R_z"]].values
    else:
        print(f"warning: no ground truth at {gt_path}")

    tag = os.path.basename(os.path.normpath(args.dataset))
    results = {}

    if not args.skip_default:
        print("\n=== run 1/2: default frontend (FAST grid + KLT) ===")
        stamps, traj = run_vio(args.dataset, args.config, max_frames=args.max_frames)
        np.savez(f"trajectory_{tag}_default.npz", stamps=stamps, traj=traj)
        results["default"] = (stamps, traj)

    print("\n=== run 2/2: XFeat detection injected from Python ===")
    from xfeat_frontend_example import XFeatDetector

    detector = XFeatDetector(device=args.device, max_keypoints=args.max_keypoints)
    backend = sv.vision.Backend(detector, name="xfeat")
    stamps, traj = run_vio(
        args.dataset, args.config, backend=backend, max_frames=args.max_frames
    )
    np.savez(f"trajectory_{tag}_xfeat.npz", stamps=stamps, traj=traj)
    results["xfeat"] = (stamps, traj)

    print("\n==== ATE (RMSE, meters, SE(3)-aligned) ====")
    for name, (stamps, traj) in results.items():
        if gt_stamps is None or len(traj) == 0:
            print(f"  {name:8s}: n={len(traj)} poses (no evaluation)")
            continue
        ate, _, _ = evaluate_ate(stamps, traj, gt_stamps, gt_traj)
        drift = 100.0 * ate / np.linalg.norm(gt_traj[-1] - gt_traj[0])
        print(f"  {name:8s}: ATE {ate:.4f} m  (n={len(traj)} poses)")
        results[name] = ate


if __name__ == "__main__":
    main()
