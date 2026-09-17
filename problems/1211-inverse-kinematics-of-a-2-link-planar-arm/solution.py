import math

def inverse_kinematics(x, y, l1, l2, elbow='up'):
    # Distance from base to target
    r2 = x * x + y * y
    r = math.sqrt(r2)

    # Reachability check
    if r > l1 + l2 or r < abs(l1 - l2):
        return None

    # Law of cosines for the elbow angle
    cos_theta2 = (r2 - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)
    # Clamp for numerical safety
    cos_theta2 = max(-1.0, min(1.0, cos_theta2))
    theta2 = math.acos(cos_theta2)

    # Select configuration: 'up' uses negative theta2, 'down' uses positive
    if elbow == 'up':
        theta2 = -theta2
    elif elbow == 'down':
        theta2 = theta2
    else:
        raise ValueError("elbow must be 'up' or 'down'")

    # Compute the base angle
    theta1 = math.atan2(y, x) - math.atan2(
        l2 * math.sin(theta2),
        l1 + l2 * math.cos(theta2)
    )

    return [theta1, theta2]