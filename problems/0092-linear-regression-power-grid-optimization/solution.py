import math
import numpy as np

PI = 3.14159

def power_grid_forecast(consumption_data):
    # 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
    n = len(consumption_data)
    detrended = []
    for i in range(1, n + 1):
        fluctuation = 10 * math.sin(2 * PI * i / 10)
        detrended.append(consumption_data[i-1] - fluctuation)
    
    # 2) Perform linear regression on the detrended data.
    x = np.array(range(1, n + 1))
    y = np.array(detrended)
    
    # Calculate slope (m) and intercept (b) using least squares
    # m = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
    intercept = (sum_y - slope * sum_x) / n
    
    # 3) Predict day 15's base consumption.
    day_15 = 15
    base_prediction = slope * day_15 + intercept
    
    # 4) Add the day 15 fluctuation back.
    fluctuation_day_15 = 10 * math.sin(2 * PI * day_15 / 10)
    final_prediction = base_prediction + fluctuation_day_15
    
    # 5) Round, then add a 5% safety margin (rounded up).
    rounded = round(final_prediction)
    margin = math.ceil(rounded * 0.05)
    final_result = rounded + margin
    
    # 6) Return the final integer.
    return final_result