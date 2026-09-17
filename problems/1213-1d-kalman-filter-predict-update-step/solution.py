def kalman_1d(mu, sigma2, u, Q, z, R):
    # Prediction step
    mu_pred = mu + u
    sigma2_pred = sigma2 + Q

    # Measurement update step
    K = sigma2_pred / (sigma2_pred + R)
    posterior_mean = mu_pred + K * (z - mu_pred)
    posterior_variance = (1.0 - K) * sigma2_pred

    return (posterior_mean, posterior_variance, K)