import numpy as np

def distillation_loss(
    student_logits: np.ndarray,
    teacher_logits: np.ndarray,
    temperature: float = 1.0
) -> float:
    """
    Compute knowledge distillation loss.
    
    L = T^2 * KL(softmax(teacher/T) || softmax(student/T))
    
    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        temperature: Softmax temperature
        
    Returns:
        Distillation loss value
    """
    # Convert to numpy arrays if not already
    student_logits = np.array(student_logits, dtype=np.float64)
    teacher_logits = np.array(teacher_logits, dtype=np.float64)
    
    # Apply temperature scaling
    student_scaled = student_logits / temperature
    teacher_scaled = teacher_logits / temperature
    
    # Compute softmax probabilities
    # For numerical stability, subtract max before exp
    student_max = np.max(student_scaled)
    teacher_max = np.max(teacher_scaled)
    
    student_exp = np.exp(student_scaled - student_max)
    teacher_exp = np.exp(teacher_scaled - teacher_max)
    
    student_probs = student_exp / np.sum(student_exp)
    teacher_probs = teacher_exp / np.sum(teacher_exp)
    
    # KL divergence: KL(P || Q) = sum(P * log(P / Q))
    # where P = teacher_probs, Q = student_probs
    # Avoid log(0) by using a small epsilon
    eps = 1e-10
    student_probs = np.clip(student_probs, eps, 1.0)
    teacher_probs = np.clip(teacher_probs, eps, 1.0)
    
    # KL divergence from teacher to student
    # Using teacher as the reference distribution
    kl_div = np.sum(teacher_probs * np.log(teacher_probs / student_probs))
    
    # Scale by T^2 to keep gradients stable
    loss = (temperature ** 2) * kl_div
    
    return float(loss)