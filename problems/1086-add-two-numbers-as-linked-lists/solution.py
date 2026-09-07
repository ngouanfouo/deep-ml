def add_two_numbers(l1: list[int], l2: list[int]) -> list[int]:
    """
    Adds two numbers represented as lists of digits in reverse order.
    
    Args:
        l1 (list[int]): First number's digits in reverse order.
        l2 (list[int]): Second number's digits in reverse order.
        
    Returns:
        list[int]: Sum of the numbers as a list of digits in reverse order.
    """
    result = []
    carry = 0
    p1, p2 = 0, 0
    len1, len2 = len(l1), len(l2)
    
    while p1 < len1 or p2 < len2 or carry > 0:
        val1 = l1[p1] if p1 < len1 else 0
        val2 = l2[p2] if p2 < len2 else 0
        
        total = val1 + val2 + carry
        carry = total // 10
        digit = total % 10
        
        result.append(digit)
        
        if p1 < len1:
            p1 += 1
        if p2 < len2:
            p2 += 1
            
    return result