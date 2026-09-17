import numpy as np

def slerp(q0, q1, t):
    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)

    # Normalize in case inputs aren't exactly unit length
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = np.dot(q0, q1)

    # If the dot product is negative, negate q1 so we take the
    # shortest rotational path (q and -q represent the same rotation).
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # Clamp for numerical safety before acos
    dot = np.clip(dot, -1.0, 1.0)

    # If the quaternions are nearly identical, fall back to linear
    # interpolation (with renormalization) to avoid division by ~0.
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        result = result / np.linalg.norm(result)
        return result.tolist()

    theta_0 = np.arccos(dot)          # angle between the inputs
    theta = theta_0 * t               # angle at the interpolation point
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)

    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    result = s0 * q0 + s1 * q1
    return result.tolist()