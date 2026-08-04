import torch

def calculate_data_quality_score(data: list, schema: dict) -> dict:
    """
    Calculate data quality metrics for ML pipeline monitoring using PyTorch.
    
    Args:
        data: list of dictionaries representing rows of data
        schema: dictionary defining expected columns and their types
    
    Returns:
        dict with keys: 'completeness', 'type_validity', 'uniqueness_ratio', 'overall_score'
    """
    if not data:
        return {}
    
    columns = list(schema.keys())
    num_rows = len(data)
    num_cols = len(columns)
    total_fields = num_rows * num_cols
    
    # Use tensors to track metrics
    non_null_mask = torch.zeros((num_rows, num_cols), dtype=torch.bool)
    type_valid_mask = torch.zeros((num_rows, num_cols), dtype=torch.bool)
    
    # Track unique records
    unique_records = set()
    
    for i, row in enumerate(data):
        row_tuple = []
        for j, col in enumerate(columns):
            value = row.get(col)
            
            # Check non-null
            if value is not None:
                non_null_mask[i, j] = True
            
            # Check type validity
            type_valid_mask[i, j] = _check_type_validity(value, schema[col])
            
            row_tuple.append(value)
        
        unique_records.add(tuple(row_tuple))
    
    # Calculate metrics
    completeness = (torch.sum(non_null_mask).item() / total_fields) * 100
    type_validity = (torch.sum(type_valid_mask).item() / total_fields) * 100
    uniqueness_ratio = (len(unique_records) / num_rows) * 100
    
    overall_score = 0.4 * completeness + 0.4 * type_validity + 0.2 * uniqueness_ratio
    
    return {
        'completeness': round(completeness, 2),
        'type_validity': round(type_validity, 2),
        'uniqueness_ratio': round(uniqueness_ratio, 2),
        'overall_score': round(overall_score, 2)
    }


def _check_type_validity(value, col_schema):
    """Check if a value matches the expected type."""
    col_type = col_schema['type']
    nullable = col_schema['nullable']
    
    if value is None:
        return nullable
    
    if col_type == 'numeric':
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    elif col_type == 'categorical':
        return isinstance(value, str)
    elif col_type == 'boolean':
        return isinstance(value, bool)
    else:
        return False