import numpy as np

def accuracy_score(y_true, y_pred):
    # Your code here
    # Count the number of correct predictions
    correct = np.sum(y_true == y_pred)
    # Calculate accuracy as correct / total
    accuracy = correct / len(y_true)
    return accuracy