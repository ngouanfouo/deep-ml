def partition_parameters(params):
    """
    Partition parameter names into Muon and AdamW groups.
    
    Args:
        params: list of (name: str, shape: tuple[int, ...]) descriptors.
    
    Returns:
        dict with keys 'muon' and 'adamw', each a list of names preserving
        the original order from `params`.
    """
    muon = []
    adamw = []
    
    for name, shape in params:
        # Muon condition: exactly 2 dimensions AND 'embed' not in name
        if len(shape) == 2 and 'embed' not in name:
            muon.append(name)
        else:
            adamw.append(name)
    
    return {'muon': muon, 'adamw': adamw}