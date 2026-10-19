import numpy as np

def pack_int4(q: np.ndarray) -> np.ndarray:
    """Pack even-length nibble array (0..15) into uint8 bytes: lo | (hi<<4)."""
    q = np.asarray(q, dtype=np.uint8).reshape(-1)
    lo = q[0::2] & 0xF
    hi = q[1::2] & 0xF
    return (lo | (hi << 4)).astype(np.uint8)

def unpack_int4(packed: np.ndarray, n: int) -> np.ndarray:
    """Unpack bytes to length-n int nibbles."""
    packed = np.asarray(packed, dtype=np.uint8).reshape(-1)
    out = np.empty(packed.size * 2, dtype=np.int64)
    out[0::2] = packed & 0xF
    out[1::2] = (packed >> 4) & 0xF
    return out[:n]

def w4a16_matmul(x, packed_w, scales, in_features: int):
    """
    x: (batch, in_features) float
    packed_w: (out_features, in_features//2) uint8
    scales: (out_features,)
    Dequant: W = scale * (q - 8), y = x @ W.T
    """
    x = np.asarray(x, dtype=np.float64)
    packed_w = np.asarray(packed_w, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float64)

    out_features = packed_w.shape[0]
    q = unpack_int4(packed_w.reshape(-1), out_features * in_features)
    q = q.reshape(out_features, in_features)

    W = scales[:, None] * (q - 8)
    return x @ W.T