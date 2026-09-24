import math

def rotation_layer(X, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    
    rotated = []
    for x, y in X:
        x_new = x * c - y * s
        y_new = x * s + y * c
        rotated.append([x_new, y_new])
    
    return rotated