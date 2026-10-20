import numpy as np


def sae_forward(X, W_enc, b_enc, W_dec, b_dec):
    pre = X @ W_enc + b_enc
    Z = np.maximum(pre, 0.0)
    Xhat = Z @ W_dec + b_dec
    return pre, Z, Xhat


def sae_loss_and_grads(X, W_enc, b_enc, W_dec, b_dec, lam):
    """(loss, {"W_enc", "b_enc", "W_dec", "b_dec"}) with exact gradients."""
    X = np.asarray(X, dtype=float)
    W_enc = np.asarray(W_enc, dtype=float)
    b_enc = np.asarray(b_enc, dtype=float)
    W_dec = np.asarray(W_dec, dtype=float)
    b_dec = np.asarray(b_dec, dtype=float)

    B = X.shape[0]

    pre, Z, Xhat = sae_forward(X, W_enc, b_enc, W_dec, b_dec)

    R = Xhat - X  # (B, d)

    loss_recon = np.sum(R * R) / B
    loss_l1 = lam * np.sum(Z) / B
    loss = loss_recon + loss_l1

    # dL/dXhat = (2/B) R
    dXhat = (2.0 / B) * R

    # Gradient w.r.t. decoder params.
    dW_dec = Z.T @ dXhat            # (n, d)
    db_dec = np.sum(dXhat, axis=0)  # (d,)

    # Gradient w.r.t. pre (through Z and the ReLU mask).
    dZ = dXhat @ W_dec.T            # (2/B) R W_dec^T  -> (B, n)
    dZ += (lam / B) * np.ones_like(Z)  # dL1/dZ = lam/B
    mask = (pre > 0).astype(float)
    dpre = dZ * mask                # (B, n)

    dW_enc = X.T @ dpre             # (d, n)
    db_enc = np.sum(dpre, axis=0)   # (n,)

    grads = {
        "W_enc": dW_enc,
        "b_enc": db_enc,
        "W_dec": dW_dec,
        "b_dec": db_dec,
    }
    return float(loss), grads


def sae_step(params, X, lam, lr):
    """One gradient step, then unit-normalise decoder rows; returns a new dict."""
    X = np.asarray(X, dtype=float)

    loss, grads = sae_loss_and_grads(
        X,
        params["W_enc"], params["b_enc"], params["W_dec"], params["b_dec"],
        lam,
    )

    new_params = {}
    for key in ("W_enc", "b_enc", "W_dec", "b_dec"):
        new_params[key] = np.asarray(params[key], dtype=float) - lr * grads[key]

    # Renormalise decoder rows to unit norm (guard against zero rows).
    W_dec = new_params["W_dec"]
    norms = np.linalg.norm(W_dec, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    new_params["W_dec"] = W_dec / norms

    return new_params


def sae_diagnostics(Z, X, Xhat, dense_thresh=0.5, decimals=4):
    Z = np.asarray(Z, dtype=float)
    X = np.asarray(X, dtype=float)
    Xhat = np.asarray(Xhat, dtype=float)

    B, n = Z.shape
    active = Z > 0  # (B, n)

    # Mean number of active latents per row.
    l0 = float(np.mean(np.sum(active, axis=1)))

    # Explained variance.
    x_mean = X.mean(axis=0, keepdims=True)
    ss_res = float(np.sum((X - Xhat) ** 2))
    ss_tot = float(np.sum((X - x_mean) ** 2))
    if ss_tot > 0:
        explained_variance = 1.0 - ss_res / ss_tot
    else:
        explained_variance = 0.0

    # Per-latent activity across rows.
    active_per_latent = active.sum(axis=0)  # (n,)
    dead_frac = float(np.mean(active_per_latent == 0))
    dense_frac = float(np.mean(active_per_latent > dense_thresh * B))

    return {
        "l0": round(l0, decimals),
        "explained_variance": round(float(explained_variance), decimals),
        "dead_frac": round(dead_frac, decimals),
        "dense_frac": round(dense_frac, decimals),
    }