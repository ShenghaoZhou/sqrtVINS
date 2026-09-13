"""
so(3)/SO(3) primitives — JAX port of ov_core/src/utils/quat_ops.h.

Convention: JPL quaternion stored as [x, y, z, w] (scalar LAST).

All functions are pure JAX ops, differentiable, and safe under jit/vmap.
Branches use jnp.where with safe denominators so the traced graph stays
straight-line and jacrev/jacfwd see a continuous gradient path.

Reference: Trawny, N. & Roumeliotis, S. (2005). "Indirect Kalman filter for
3D attitude estimation." Tech. Rep 2, U. Minnesota.
"""

import jax
import jax.numpy as jnp

# Ensure float64 everywhere
jax.config.update("jax_enable_x64", True)


def skew_x(w: jnp.ndarray) -> jnp.ndarray:
    """Skew-symmetric matrix from a 3-vector. [w×] = [[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]]"""
    return jnp.array(
        [[0.0, -w[2], w[1]],
         [w[2], 0.0, -w[0]],
         [-w[1], w[0], 0.0]],
        dtype=w.dtype,
    )


def vee(w_x: jnp.ndarray) -> jnp.ndarray:
    """Inverse of skew_x: extract the 3-vector from a skew-symmetric matrix."""
    return jnp.array([w_x[2, 1], w_x[0, 2], w_x[1, 0]], dtype=w_x.dtype)


def rot_2_quat(R: jnp.ndarray) -> jnp.ndarray:
    """
    Rotation matrix → JPL quaternion [x,y,z,w].

    Port of rot_2_quat (quat_ops.h:98-134). Uses the max-diagonal trick
    to avoid division by zero; flips sign so w >= 0; normalizes.
    """
    T = jnp.trace(R)
    # Four branches, each divides by the largest diagonal element
    # C++ divides by `4*q_k` where `q_k = sqrt((1+2*R_kk-T)/4)`, i.e. the
    # divisor is `2*sqrt(1+2*R_kk-T)` — not 4. Getting this wrong returns the
    # quaternion for *half* the true rotation angle.
    s0 = 2.0 * jnp.sqrt(jnp.maximum(1.0 + 2.0 * R[0, 0] - T, 0.0))
    s1 = 2.0 * jnp.sqrt(jnp.maximum(1.0 + 2.0 * R[1, 1] - T, 0.0))
    s2 = 2.0 * jnp.sqrt(jnp.maximum(1.0 + 2.0 * R[2, 2] - T, 0.0))
    s3 = 2.0 * jnp.sqrt(jnp.maximum(1.0 + T, 0.0))

    # Each branch: pick the sqrt term, compute the other three
    q0_from_diag0 = (R[0, 1] + R[1, 0]) / jnp.where(s0 > 1e-12, s0, 1.0)
    q1_from_diag0 = (R[0, 2] + R[2, 0]) / jnp.where(s0 > 1e-12, s0, 1.0)
    q2_from_diag0 = (R[1, 2] - R[2, 1]) / jnp.where(s0 > 1e-12, s0, 1.0)
    w_from_diag0 = jnp.sqrt(jnp.maximum(1.0 + 2.0 * R[0, 0] - T, 0.0) / 4.0)

    q1_from_diag1 = (R[0, 1] + R[1, 0]) / jnp.where(s1 > 1e-12, s1, 1.0)
    q2_from_diag1 = (R[1, 2] + R[2, 1]) / jnp.where(s1 > 1e-12, s1, 1.0)
    q3_from_diag1 = (R[2, 0] - R[0, 2]) / jnp.where(s1 > 1e-12, s1, 1.0)
    w_from_diag1 = jnp.sqrt(jnp.maximum(1.0 + 2.0 * R[1, 1] - T, 0.0) / 4.0)

    q0_from_diag2 = (R[0, 2] + R[2, 0]) / jnp.where(s2 > 1e-12, s2, 1.0)
    q1_from_diag2 = (R[1, 2] + R[2, 1]) / jnp.where(s2 > 1e-12, s2, 1.0)
    q2_from_diag2 = (R[0, 1] - R[1, 0]) / jnp.where(s2 > 1e-12, s2, 1.0)
    w_from_diag2 = jnp.sqrt(jnp.maximum(1.0 + 2.0 * R[2, 2] - T, 0.0) / 4.0)

    q0_from_diag3 = (R[1, 2] - R[2, 1]) / jnp.where(s3 > 1e-12, s3, 1.0)
    q1_from_diag3 = (R[2, 0] - R[0, 2]) / jnp.where(s3 > 1e-12, s3, 1.0)
    q2_from_diag3 = (R[0, 1] - R[1, 0]) / jnp.where(s3 > 1e-12, s3, 1.0)
    w_from_diag3 = jnp.sqrt(jnp.maximum(1.0 + T, 0.0) / 4.0)

    q0 = jnp.where(R[0, 0] >= jnp.maximum(jnp.maximum(R[1, 1], R[2, 2]), T),
                   w_from_diag0,
                   jnp.where(R[1, 1] >= jnp.maximum(jnp.maximum(R[0, 0], R[2, 2]), T),
                             q1_from_diag1,
                             jnp.where(R[2, 2] >= jnp.maximum(R[0, 0], R[1, 1]),
                                       q0_from_diag2,
                                       q0_from_diag3)))
    q1 = jnp.where(R[0, 0] >= jnp.maximum(jnp.maximum(R[1, 1], R[2, 2]), T),
                   q0_from_diag0,
                   jnp.where(R[1, 1] >= jnp.maximum(jnp.maximum(R[0, 0], R[2, 2]), T),
                             w_from_diag1,
                             jnp.where(R[2, 2] >= jnp.maximum(R[0, 0], R[1, 1]),
                                       q1_from_diag2,
                                       q1_from_diag3)))
    q2 = jnp.where(R[0, 0] >= jnp.maximum(jnp.maximum(R[1, 1], R[2, 2]), T),
                   q1_from_diag0,
                   jnp.where(R[1, 1] >= jnp.maximum(jnp.maximum(R[0, 0], R[2, 2]), T),
                             q2_from_diag1,
                             jnp.where(R[2, 2] >= jnp.maximum(R[0, 0], R[1, 1]),
                                       w_from_diag2,
                                       q2_from_diag3)))
    w = jnp.where(R[0, 0] >= jnp.maximum(jnp.maximum(R[1, 1], R[2, 2]), T),
                  q2_from_diag0,
                  jnp.where(R[1, 1] >= jnp.maximum(jnp.maximum(R[0, 0], R[2, 2]), T),
                            q3_from_diag1,
                            jnp.where(R[2, 2] >= jnp.maximum(R[0, 0], R[1, 1]),
                                      q2_from_diag2,
                                      w_from_diag3)))

    q = jnp.array([q0, q1, q2, w], dtype=R.dtype)
    # Force w >= 0
    q = jnp.where(w[None] < 0, -q, q)
    return q / jnp.linalg.norm(q)


def quat_2_Rot(q: jnp.ndarray) -> jnp.ndarray:
    """
    JPL quaternion [x,y,z,w] → SO(3) rotation matrix.

    Port of quat_2_Rot (quat_ops.h:170-181):
      R = (2w²-1)I - 2w·[q×] + 2·q·qᵀ
    """
    qv = q[:3]
    w = q[3]
    return ((2.0 * w * w - 1.0) * jnp.eye(3, dtype=q.dtype)
            - 2.0 * w * skew_x(qv)
            + 2.0 * jnp.outer(qv, qv))


def quat_multiply(q: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """
    Multiply two JPL quaternions [x,y,z,w].

    Port of quat_multiply (quat_ops.h:206-224). Uses the L-matrix form:
      q⊗p = L(q)·p = [[w·I - [q×],  q], [-qᵀ,  w]] · [p; pw]
    Result is sign-flipped to w >= 0 and normalized.
    """
    qv, w = q[:3], q[3]
    # Build the 4x4 L-matrix
    L = jnp.zeros((4, 4), dtype=q.dtype)
    L = L.at[:3, :3].set(w * jnp.eye(3, dtype=q.dtype) - skew_x(qv))
    L = L.at[:3, 3].set(qv)
    L = L.at[3, :3].set(-qv)
    L = L.at[3, 3].set(w)

    q_t = L @ p
    # Force w >= 0
    q_t = jnp.where(q_t[3] < 0, -q_t, q_t)
    return q_t / jnp.linalg.norm(q_t)


def exp_so3(w: jnp.ndarray) -> jnp.ndarray:
    """
    SO(3) matrix exponential (Rodrigues formula).

    Port of exp_so3 (quat_ops.h:265-289). Branches at theta < 1e-7 with
    A=1, B=0.5 (Taylor). Uses safe denominators so the gradient is
    continuous at the branch.
    """
    w_x = skew_x(w)
    theta = jnp.linalg.norm(w)
    theta_safe = jnp.where(theta > 1e-7, theta, 1.0)
    A = jnp.where(theta > 1e-7, jnp.sin(theta_safe) / theta_safe, 1.0)
    B = jnp.where(theta > 1e-7, (1.0 - jnp.cos(theta_safe)) / (theta_safe ** 2), 0.5)
    return jnp.eye(3, dtype=w.dtype) + A * w_x + B * w_x @ w_x


def log_so3(R: jnp.ndarray) -> jnp.ndarray:
    """
    SO(3) matrix logarithm → axis-angle [omega_x, omega_y, omega_z].

    Port of log_so3 (quat_ops.h:312-356). Three branches:
      1. trace(R) → -1 (theta = pi): special formula
      2. trace(R) - 3 < -1e-7 (normal case): theta = acos((tr-1)/2)
      3. trace(R) → 3 (theta → 0): Taylor expansion

    Uses jnp.where with safe denominators for gradient continuity.
    """
    tr = jnp.trace(R)

    # Branch 1: trace ≈ -1 (theta = pi)
    # omega = (pi / sqrt(2 + 2*Rkk)) * v_k where v is the k-th column of R
    # pick k so that Rkk is bounded away from -1 (i.e. 2+2*Rkk is bounded away from 0).
    # C++ order: try R33 first, then R22, then R11.
    diag = jnp.array([R[0, 0], R[1, 1], R[2, 2]], dtype=R.dtype)
    # Column vectors (row-major R indexing; R[1,k], R[2,k] are the non-diagonal
    # entries of column k, and (1+R[k,k]) is its diagonal entry).
    v0 = jnp.array([1.0 + R[0, 0], R[1, 0], R[2, 0]], dtype=R.dtype)  # uses R11
    v1 = jnp.array([R[0, 1], 1.0 + R[1, 1], R[2, 1]], dtype=R.dtype)  # uses R22
    v2 = jnp.array([R[0, 2], R[1, 2], 1.0 + R[2, 2]], dtype=R.dtype)  # uses R33

    def pi_branch(v: jnp.ndarray, Rkk: jnp.ndarray) -> jnp.ndarray:
        denom = jnp.sqrt(jnp.maximum(2.0 + 2.0 * Rkk, 1e-10))
        return (jnp.pi / denom) * v

    # Match the C++ cascade: R33 first, then R22, then R11.
    omega_pi = jnp.where(
        jnp.abs(diag[2] + 1.0) > 1e-5,
        pi_branch(v2, diag[2]),
        jnp.where(
            jnp.abs(diag[1] + 1.0) > 1e-5,
            pi_branch(v1, diag[1]),
            pi_branch(v0, diag[0]),
        ),
    )

    # Branch 2 & 3: normal case and near-identity
    tr_3 = tr - 3.0  # always <= 0
    theta = jnp.arccos(jnp.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    sin_theta = jnp.sin(theta)
    # magnitude = theta / (2 * sin_theta), with Taylor for theta → 0
    mag_normal = jnp.where(
        sin_theta > 1e-7,
        theta / (2.0 * jnp.where(sin_theta > 1e-7, sin_theta, 1.0)),
        0.5 - tr_3 / 12.0,
    )
    # antisymmetric part of R
    anti = jnp.array([R[2, 1] - R[1, 2],
                      R[0, 2] - R[2, 0],
                      R[1, 0] - R[0, 1]], dtype=R.dtype)
    omega_normal = mag_normal * anti

    # Select: pi branch when tr + 1 < 1e-10, else normal
    is_pi = tr + 1.0 < 1e-10
    return jnp.where(is_pi, omega_pi, omega_normal)


def exp_se3(vec: jnp.ndarray) -> jnp.ndarray:
    """
    SE(3) matrix exponential: [omega, u] → 4x4 homogeneous matrix.

    Port of exp_se3 (quat_ops.h:379-413).
    """
    w, u = vec[:3], vec[3:]
    theta = jnp.linalg.norm(w)
    w_x = skew_x(w)
    theta_safe = jnp.where(theta > 1e-7, theta, 1.0)
    A = jnp.where(theta > 1e-7, jnp.sin(theta_safe) / theta_safe, 1.0)
    B = jnp.where(theta > 1e-7, (1.0 - jnp.cos(theta_safe)) / (theta_safe ** 2), 0.5)
    C = jnp.where(theta > 1e-7, (1.0 - A) / (theta_safe ** 2), 1.0 / 6.0)

    I3 = jnp.eye(3, dtype=vec.dtype)
    V = I3 + B * w_x + C * w_x @ w_x
    R = I3 + A * w_x + B * w_x @ w_x

    T = jnp.zeros((4, 4), dtype=vec.dtype)
    T = T.at[:3, :3].set(R)
    T = T.at[:3, 3].set(V @ u)
    T = T.at[3, 3].set(1.0)
    return T


def log_se3(T: jnp.ndarray) -> jnp.ndarray:
    """
    SE(3) matrix logarithm: 4x4 → [omega, u].

    Port of log_se3 (quat_ops.h:440-462). The C++ special-cases `t < 1e-10`
    by returning `[w, T]` directly. The general formula with a safe denominator
    collapses to the same thing in that limit, so we use one expression.
    """
    R = T[:3, :3]
    tvec = T[:3, 3]
    w = log_so3(R)
    theta = jnp.linalg.norm(w)

    # The C++ has a hard branch here: `if (t < 1e-10) return [w, T]`.
    # At theta -> 0 the unit axis w/theta is ill-defined, so the safe
    # denominator is needed ONLY for that division. The three coefficient
    # terms must use the real theta: with theta = 0, W = 0 makes WT = 0
    # and W@WT = 0, so u = tvec exactly, matching the C++ branch. Using
    # theta_safe there would substitute theta = 1's coefficients and leave
    # `u` off by O(||tvec||).
    theta_safe = jnp.where(theta > 1e-10, theta, 1.0)
    W = skew_x(w / theta_safe)
    tan_half = jnp.tan(0.5 * theta)
    # Avoid division by zero in (1 - theta/(2*tan_half)); tan_half -> 0
    # as theta -> 0, and theta/tan_half -> 1 there.
    denom = jnp.where(jnp.abs(tan_half) > 1e-12, tan_half, 1.0)
    WT = W @ tvec
    u = tvec - (0.5 * theta) * WT + (1.0 - theta / (2.0 * denom)) * (W @ WT)

    return jnp.concatenate([w, u], axis=0)


def hat_se3(vec: jnp.ndarray) -> jnp.ndarray:
    """Hat operator: [omega, u] → 4x4 se(3) matrix. Port of hat_se3 (quat_ops.h:476-482)."""
    w, u = vec[:3], vec[3:]
    T = jnp.zeros((4, 4), dtype=vec.dtype)
    T = T.at[:3, :3].set(skew_x(w))
    T = T.at[:3, 3].set(u)
    return T


def Inv_se3(T: jnp.ndarray) -> jnp.ndarray:
    """
    Analytical SE(3) inverse: [[R, t], [0, 1]] → [[Rᵀ, -Rᵀt], [0, 1]].

    Port of Inv_se3 (quat_ops.h:498-504).
    """
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = jnp.eye(4, dtype=T.dtype)
    Tinv = Tinv.at[:3, :3].set(R.T)
    Tinv = Tinv.at[:3, 3].set(-R.T @ t)
    return Tinv


def Inv(q: jnp.ndarray) -> jnp.ndarray:
    """JPL quaternion inverse: [x,y,z,w] → [-x,-y,-z,w]. Port of Inv (quat_ops.h:519-524)."""
    qinv = q.copy()
    qinv = qinv.at[:3].set(-q[:3])
    return qinv


def Omega(w: jnp.ndarray) -> jnp.ndarray:
    """
    4x4 skew matrix for quaternion integration: [[-w×, -wᵀ], [w, 0]].

    Port of Omega (quat_ops.h:535-542).
    """
    mat = jnp.zeros((4, 4), dtype=w.dtype)
    mat = mat.at[:3, :3].set(-skew_x(w))
    mat = mat.at[3, :3].set(-w)
    mat = mat.at[:3, 3].set(w)
    return mat


def quatnorm(q: jnp.ndarray) -> jnp.ndarray:
    """Normalize + ensure w >= 0. Port of quatnorm (quat_ops.h:551-557)."""
    q = jnp.where(q[3] < 0, -q, q)
    return q / jnp.linalg.norm(q)


def Jl_so3(w: jnp.ndarray) -> jnp.ndarray:
    """
    Left Jacobian of SO(3).

    Port of Jl_so3 (quat_ops.h:575-590). Branches at theta < 1e-6 → I.
    """
    theta = jnp.linalg.norm(w)
    theta_safe = jnp.where(theta > 1e-6, theta, 1.0)
    a = w / theta_safe
    sin_over_theta = jnp.where(theta > 1e-6, jnp.sin(theta_safe) / theta_safe, 1.0)
    one_minus_sin = jnp.where(theta > 1e-6, 1.0 - jnp.sin(theta_safe) / theta_safe, 0.0)
    one_minus_cos = jnp.where(theta > 1e-6, (1.0 - jnp.cos(theta_safe)) / theta_safe, 0.0)

    return (sin_over_theta * jnp.eye(3, dtype=w.dtype)
            + one_minus_sin * jnp.outer(a, a)
            + one_minus_cos * skew_x(a))


def Jr_so3(w: jnp.ndarray) -> jnp.ndarray:
    """Right Jacobian of SO(3) = Jl(-w). Port of Jr_so3 (quat_ops.h:606-608)."""
    return Jl_so3(-w)


def rot2rpy(R: jnp.ndarray) -> jnp.ndarray:
    """
    Rotation matrix → [roll, pitch, yaw] (R = Rz(yaw)·Ry(pitch)·Rx(roll)).

    Port of rot2rpy (quat_ops.h:622-634).
    """
    rpy = jnp.zeros(3, dtype=R.dtype)
    # pitch
    rpy = rpy.at[1].set(
        jnp.arctan2(-R[2, 0], jnp.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    )
    cos_pitch = jnp.cos(rpy[1])
    # yaw and roll (branch on |cos(pitch)|)
    is_sing = jnp.abs(cos_pitch) <= 1e-12
    rpy = rpy.at[2].set(jnp.where(
        is_sing, 0.0,
        jnp.arctan2(R[1, 0] / jnp.where(cos_pitch > 1e-12, cos_pitch, 1.0),
                    R[0, 0] / jnp.where(cos_pitch > 1e-12, cos_pitch, 1.0)),
    ))
    rpy = rpy.at[0].set(jnp.where(
        is_sing,
        jnp.arctan2(R[0, 1], R[1, 1]),
        jnp.arctan2(R[2, 1] / jnp.where(cos_pitch > 1e-12, cos_pitch, 1.0),
                    R[2, 2] / jnp.where(cos_pitch > 1e-12, cos_pitch, 1.0)),
    ))
    return rpy


def rot_x(t: float) -> jnp.ndarray:
    """Rotation about X axis by angle t."""
    ct, st = jnp.cos(t), jnp.sin(t)
    return jnp.array([[1.0, 0.0, 0.0],
                      [0.0, ct, -st],
                      [0.0, st, ct]])


def rot_y(t: float) -> jnp.ndarray:
    """Rotation about Y axis by angle t."""
    ct, st = jnp.cos(t), jnp.sin(t)
    return jnp.array([[ct, 0.0, st],
                      [0.0, 1.0, 0.0],
                      [-st, 0.0, ct]])


def rot_z(t: float) -> jnp.ndarray:
    """Rotation about Z axis by angle t."""
    ct, st = jnp.cos(t), jnp.sin(t)
    return jnp.array([[ct, -st, 0.0],
                      [st, ct, 0.0],
                      [0.0, 0.0, 1.0]])


# =============================================================================
# Hamilton ↔ JPL bridge
# =============================================================================
#
# Hamilton convention: [w, x, y, z] (scalar FIRST) — used by cv2, numpy, and
# most external systems.
# JPL convention: [x, y, z, w] (scalar LAST) — used internally by OpenVINS.
#

def ham_to_jpl(q_ham: jnp.ndarray) -> jnp.ndarray:
    """Hamilton [w,x,y,z] → JPL [x,y,z,w]."""
    return jnp.array([q_ham[1], q_ham[2], q_ham[3], q_ham[0]], dtype=q_ham.dtype)


def jpl_to_ham(q_jpl: jnp.ndarray) -> jnp.ndarray:
    """JPL [x,y,z,w] → Hamilton [w,x,y,z]."""
    return jnp.array([q_jpl[3], q_jpl[0], q_jpl[1], q_jpl[2]], dtype=q_jpl.dtype)


def R_from_ham(q_ham: jnp.ndarray) -> jnp.ndarray:
    """
    Rotation matrix from Hamilton quaternion [w,x,y,z].

    R = (1 - 2*|v|²)·I + 2·v·vᵀ - 2·w·[v×]
    where v = [x,y,z] and w is the scalar part.
    """
    w, v = q_ham[0], q_ham[1:]
    v_sq = jnp.dot(v, v)
    return ((1.0 - 2.0 * v_sq) * jnp.eye(3, dtype=q_ham.dtype)
            + 2.0 * jnp.outer(v, v)
            - 2.0 * w * skew_x(v))


def ham_multiply(q_ham: jnp.ndarray, p_ham: jnp.ndarray) -> jnp.ndarray:
    """Multiply two Hamilton quaternions [w,x,y,z], matching C++ `quat_multiply`.

    NOTE the cross term is `cross(pv, qv)`, NOT the textbook `cross(qv, pv)`.
    C++ `quat_multiply(q1, q2)` computes the standard product `q2 * q1` (reversed
    argument order) — OpenVINS stores the conjugate of the rotation quaternion,
    and conjugation reverses product order. `ham_multiply` must reproduce that so
    `jpl_to_ham(quat_multiply(a, b)) == ham_multiply(jpl_to_ham(a), jpl_to_ham(b))`.
    Verified numerically: the textbook order mismatches by up to 1.915.
    """
    qw, qv = q_ham[0], q_ham[1:]
    pw, pv = p_ham[0], p_ham[1:]
    w = qw * pw - jnp.dot(qv, pv)
    v = qw * pv + pw * qv + jnp.cross(pv, qv)
    # `jnp.array([w, ...])` (not `[[w], ...]`): a 0-d `w` in a nested list
    # becomes shape (1, 1), which cannot concat with the (3,) vector.
    result = jnp.array([w, v[0], v[1], v[2]], dtype=q_ham.dtype)
    return jnp.where(result[0] < 0, -result, result)
