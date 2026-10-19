import numpy as np

def simplified_gptq(w: np.ndarray, H_diag: np.ndarray, bits: int = 4, blocksize: int = 128):
    """
    Simplified GPTQ: sequential symmetric quant with within-block
    diagonal-Hessian error compensation.
    """
    w = np.asarray(w, dtype=np.float64)
    H_diag = np.asarray(H_diag, dtype=np.float64)

    n = w.shape[0]
    qmax = 2 ** (bits - 1) - 1

    max_abs = np.max(np.abs(w)) if n > 0 else 0.0
    scale = float(max_abs / qmax) if max_abs != 0.0 else 1.0

    w_b = w.copy()
    q = np.zeros(n, dtype=np.int64)
    w_hat = np.zeros(n, dtype=np.float64)

    for start in range(0, n, blocksize):
        end = min(start + blocksize, n)

        for i in range(start, end):
            q_i = int(np.clip(np.round(w_b[i] / scale), -qmax, qmax))
            q[i] = q_i
            w_hat[i] = q_i * scale

            e = w_b[i] - w_hat[i]

            for j in range(i + 1, end):
                w_b[j] -= e * H_diag[i] / (H_diag[j] + 1e-8)

    return q, scale, w_hat