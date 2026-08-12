import torch
import re
import math

def verify_math_answer(predicted, ground_truth, tolerance: float = 1e-6) -> torch.Tensor:
    """
    Verify if two mathematical answers are equivalent (supports single or batch inputs).
    
    Args:
        predicted: Single string or list of predicted answer strings
        ground_truth: Single string or list of ground truth answer strings
        tolerance: Numerical tolerance for comparison
    
    Returns:
        torch.BoolTensor of equivalence results
    """
    # Convert single inputs to lists for uniform handling
    if isinstance(predicted, str):
        predicted = [predicted]
        ground_truth = [ground_truth]
    
    if len(predicted) != len(ground_truth):
        raise ValueError("predicted and ground_truth must have same length")
    
    results = []
    
    for pred_str, gt_str in zip(predicted, ground_truth):
        # Clean strings
        pred_str = pred_str.strip()
        gt_str = gt_str.strip()
        
        # Quick exact match (case sensitive)
        if pred_str == gt_str:
            results.append(True)
            continue
        
        # Try to evaluate both strings numerically
        pred_val = evaluate_math_expression(pred_str)
        gt_val = evaluate_math_expression(gt_str)
        
        if pred_val is not None and gt_val is not None:
            # Both are numeric, compare with tolerance
            results.append(abs(pred_val - gt_val) <= tolerance)
        else:
            # If either fails to parse, they're not equivalent (unless exact match already checked)
            results.append(False)
    
    return torch.tensor(results, dtype=torch.bool)


def evaluate_math_expression(expr: str):
    """
    Safely evaluate a mathematical expression string to a float.
    Returns None if expression cannot be parsed.
    """
    expr = expr.strip()
    
    # Handle specific cases
    # Replace 'pi' with math.pi
    expr = expr.replace('pi', str(math.pi))
    expr = expr.replace('π', str(math.pi))
    
    # Handle 'sqrt(...)'
    def replace_sqrt(match):
        inside = match.group(1)
        inner_val = evaluate_math_expression(inside)
        if inner_val is not None and inner_val >= 0:
            return str(math.sqrt(inner_val))
        return match.group(0)
    
    # Handle square roots
    expr = re.sub(r'sqrt\(([^)]+)\)', replace_sqrt, expr)
    
    # Handle fractions like "1/2"
    # We'll evaluate these carefully
    try:
        # Safely evaluate the expression
        # Use eval with restricted globals to prevent arbitrary code execution
        allowed_names = {
            'abs': abs,
            'pow': pow,
            'round': round,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'exp': math.exp,
            'log': math.log,
            'log10': math.log10,
            'pi': math.pi,
            'e': math.e,
            'inf': float('inf'),
            'nan': float('nan'),
        }
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        if isinstance(result, (int, float)):
            return float(result)
        return None
    except:
        return None


def parse_fraction(expr: str):
    """Helper to parse fraction expressions like '1/2'"""
    # Try to parse as a simple fraction
    parts = expr.split('/')
    if len(parts) == 2:
        try:
            num = evaluate_math_expression(parts[0].strip())
            den = evaluate_math_expression(parts[1].strip())
            if num is not None and den is not None and den != 0:
                return num / den
        except:
            pass
    return None