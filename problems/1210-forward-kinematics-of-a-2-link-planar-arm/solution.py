import numpy as np

def forward_kinematics(l1, l2, theta1, theta2):
    # End-effector position
    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)
    position = np.array([x, y], dtype=float)

    # Jacobian entries
    dx_dtheta1 = -l1 * np.sin(theta1) - l2 * np.sin(theta1 + theta2)
    dx_dtheta2 = -l2 * np.sin(theta1 + theta2)
    dy_dtheta1 = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
    dy_dtheta2 = l2 * np.cos(theta1 + theta2)

    jacobian = np.array([
        [dx_dtheta1, dx_dtheta2],
        [dy_dtheta1, dy_dtheta2]
    ], dtype=float)

    return position, jacobian