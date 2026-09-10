import numpy as np

def kinematic_vehicle_model(initial_state: np.ndarray, controls: np.ndarray, wheelbase: float, dt: float) -> np.ndarray:
    """
    Simulate vehicle motion using the kinematic bicycle model.
    
    Args:
        initial_state: numpy array of shape (4,) with [x, y, theta, v]
        controls: numpy array of shape (N, 2) with [acceleration, steering_angle] per step
        wheelbase: distance between front and rear axles in meters
        dt: time step in seconds
    
    Returns:
        numpy array of shape (N+1, 4) with [x, y, theta, v] at each time step
    """
    initial_state = np.asarray(initial_state, dtype=float)
    controls = np.asarray(controls, dtype=float).reshape(-1, 2)
    
    N = controls.shape[0]
    trajectory = np.zeros((N + 1, 4), dtype=float)
    trajectory[0] = initial_state
    
    x, y, theta, v = initial_state
    
    for i in range(N):
        a = controls[i, 0]
        delta = controls[i, 1]
        
        # Forward Euler: all derivatives use OLD state values
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + (v / wheelbase) * np.tan(delta) * dt
        v_new = v + a * dt
        
        # Commit the new state
        x, y, theta, v = x_new, y_new, theta_new, v_new
        trajectory[i + 1] = [x, y, theta, v]
    
    return trajectory