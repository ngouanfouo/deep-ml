import numpy as np

def novel_view_synthesis(source_image: np.ndarray, depth_map: np.ndarray,
                        K_src: np.ndarray, K_tgt: np.ndarray,
                        R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Synthesize a novel view by reprojecting source pixels using depth.

    Args:
        source_image: float array of shape (H, W, C)
        depth_map: float array of shape (H, W), per-pixel depth (>0 valid)
        K_src: 3x3 source camera intrinsic matrix
        K_tgt: 3x3 target camera intrinsic matrix
        R: 3x3 rotation matrix (source to target)
        t: length-3 translation vector (source to target)

    Returns:
        target_image: float array of shape (H, W, C)
    """
    H, W, C = source_image.shape
    target = np.zeros((H, W, C), dtype=float)
    zbuf = np.full((H, W), np.inf, dtype=float)   # z-buffer for occlusion handling

    K_src_inv = np.linalg.inv(K_src)
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float)
    K_tgt = np.asarray(K_tgt, dtype=float)

    for v in range(H):
        for u in range(W):
            d = depth_map[v, u]
            if d <= 0:
                continue

            # 1. Back-project source pixel to 3D in source camera frame
            p_src = d * (K_src_inv @ np.array([u, v, 1.0]))

            # 2. Transform to target camera frame
            p_tgt = R @ p_src + t
            z_tgt = p_tgt[2]
            if z_tgt <= 0:
                continue

            # 3. Project onto target image plane
            proj = K_tgt @ p_tgt
            u_t = proj[0] / proj[2]
            v_t = proj[1] / proj[2]

            u_i = int(round(u_t))
            v_i = int(round(v_t))

            # 4. Check bounds and depth buffer
            if 0 <= u_i < W and 0 <= v_i < H:
                if z_tgt < zbuf[v_i, u_i]:
                    zbuf[v_i, u_i] = z_tgt
                    target[v_i, u_i] = source_image[v, u]

    return target