import torch

def verify_code_execution(
    test_cases: list[dict],
    numeric_tolerance: float = 1e-6
) -> dict:
    """
    Verify code execution results for a programming benchmark using PyTorch.
    
    Args:
        test_cases: List of dicts with keys:
            - 'expected': Expected output string
            - 'actual': Actual output string (or None if execution failed)
            - 'status': 'success', 'error', or 'timeout'
        numeric_tolerance: Tolerance for floating-point comparisons
        
    Returns:
        Dict with keys:
            - 'pass_rate': Proportion of passed tests (float, rounded to 4 decimals)
            - 'error_rate': Proportion of execution errors (float, rounded to 4 decimals)
            - 'passed_count': Number of passed tests (int)
            - 'total_count': Total number of tests (int)
            - 'verdicts': List of 'pass', 'fail', or 'error' for each test
    """
    # Handle empty test cases
    if not test_cases:
        return {
            'pass_rate': 0.0,
            'error_rate': 0.0,
            'passed_count': 0,
            'total_count': 0,
            'verdicts': []
        }
    
    verdicts = []
    passed_count = 0
    error_count = 0
    total_count = len(test_cases)
    
    for test in test_cases:
        # Check if execution was successful
        status = test.get('status', 'error')
        
        if status != 'success':
            # Execution error or timeout
            verdicts.append('error')
            error_count += 1
            continue
        
        expected = test.get('expected', '')
        actual = test.get('actual')
        
        # If actual is None (shouldn't happen with status='success', but handle defensively)
        if actual is None:
            verdicts.append('error')
            error_count += 1
            continue
        
        # Strip whitespace for comparison
        expected_str = str(expected).strip()
        actual_str = str(actual).strip()
        
        # Try numeric comparison first
        try:
            expected_num = float(expected_str)
            actual_num = float(actual_str)
            
            # Compare with tolerance
            if abs(expected_num - actual_num) <= numeric_tolerance:
                verdicts.append('pass')
                passed_count += 1
            else:
                verdicts.append('fail')
        except (ValueError, TypeError):
            # Fall back to exact string matching
            if expected_str == actual_str:
                verdicts.append('pass')
                passed_count += 1
            else:
                verdicts.append('fail')
    
    # Compute rates
    pass_rate = round(passed_count / total_count, 4)
    error_rate = round(error_count / total_count, 4)
    
    return {
        'pass_rate': pass_rate,
        'error_rate': error_rate,
        'passed_count': passed_count,
        'total_count': total_count,
        'verdicts': verdicts
    }