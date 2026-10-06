"""Compare filter vs BA trajectories against EuRoC ground truth (ATE RMSE).

Both trajectories get identical treatment: nearest-neighbor GT association
(50 ms gate), Umeyama Sim(3) alignment, RMSE of position error.
"""
import sys
import numpy as np
import pandas as pd


def load_est(path):
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1:4]


def load_gt(csv_path):
    # both GT flavors are t(ns), px, py, pz, ... in the first four columns
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 0] * 1e-9, data[:, 1:4]


def associate(t_est, p_est, t_gt, p_gt, max_dt=0.05):
    idx = np.searchsorted(t_gt, t_est)
    idx = np.clip(idx, 1, len(t_gt) - 1)
    left = np.abs(t_gt[idx - 1] - t_est)
    right = np.abs(t_gt[idx] - t_est)
    choose_right = right < left
    best = np.where(choose_right, idx, idx - 1)
    dt = np.where(choose_right, right, left)
    ok = dt < max_dt
    return p_est[ok], p_gt[best[ok]]


def umeyama(src, dst, with_scale=True):
    """Return s, R, t aligning src -> dst (dst ~ s R src + t)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(xs)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / (xs**2).sum() * len(xs)) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate(est_path, gt_csv, with_scale=True):
    t_est, p_est = load_est(est_path)
    t_gt, p_gt = load_gt(gt_csv)
    p_e, p_g = associate(t_est, p_est, t_gt, p_gt)
    s, R, t = umeyama(p_e, p_g, with_scale)
    p_aligned = (s * (R @ p_e.T).T) + t
    err = np.linalg.norm(p_aligned - p_g, axis=1)
    return dict(n=len(err), rmse=float(np.sqrt((err**2).mean())),
                mean=float(err.mean()), median=float(np.median(err)),
                max=float(err.max()), scale=s)


if __name__ == "__main__":
    gt_csv = sys.argv[1]
    for est, name in [(sys.argv[2], "filter"), (sys.argv[3], "backend-BA")]:
        r = ate(est, gt_csv, with_scale=("--noscale" not in sys.argv))
        print(f"{name:12s} n={r['n']:5d} rmse={r['rmse']:.4f} m  "
              f"mean={r['mean']:.4f}  median={r['median']:.4f}  "
              f"max={r['max']:.4f}  (align scale={r['scale']:.4f})")
