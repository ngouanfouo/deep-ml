def conditional_probability(joint_distribution: dict) -> float:
    """
    Compute conditional probability P(A|B) from a joint probability distribution.

    Args:
        joint_distribution (dict): dictionary with keys ('A','B'), ('A','`B'), ('`A','B'), ('`A','`B')

    Returns:
        float: Conditional probability P(A|B)
    """
    # Extract the relevant probabilities
    p_A_and_B = joint_distribution.get(('A', 'B'), 0.0)
    p_notA_and_B = joint_distribution.get(('`A', 'B'), 0.0)
    
    # Calculate P(B) = P(A∩B) + P(¬A∩B)
    p_B = p_A_and_B + p_notA_and_B
    
    # Calculate conditional probability P(A|B) = P(A∩B) / P(B)
    # Handle edge case where P(B) = 0
    if p_B == 0:
        return 0.0
    
    result = p_A_and_B / p_B
    
    # Round to 4 decimal places
    return round(result, 4)