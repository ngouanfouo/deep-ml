def analyze_ml_pipeline(tasks: list) -> dict:
    """
    Analyze an ML pipeline DAG for scheduling and critical path.
    
    Args:
        tasks: list of task dicts with:
            - 'id': task identifier (str)
            - 'duration': task duration in minutes (int)
            - 'dependencies': list of task IDs this task depends on
    
    Returns:
        dict with:
            - 'execution_order': topologically sorted list of task IDs
            - 'earliest_start': dict mapping task ID to earliest start time
            - 'earliest_finish': dict mapping task ID to earliest finish time
            - 'latest_start': dict mapping task ID to latest start time
            - 'latest_finish': dict mapping task ID to latest finish time
            - 'slack': dict mapping task ID to slack time
            - 'critical_path': list of task IDs on critical path (in execution order)
            - 'makespan': total time to complete pipeline
    """
    # Handle empty input
    if not tasks:
        return {
            'execution_order': [],
            'earliest_start': {},
            'earliest_finish': {},
            'latest_start': {},
            'latest_finish': {},
            'slack': {},
            'critical_path': [],
            'makespan': 0
        }
    
    # Build task lookup and adjacency
    task_map = {task['id']: task for task in tasks}
    
    # Build dependency graph (children and parent counts)
    children = {task['id']: [] for task in tasks}
    in_degree = {task['id']: 0 for task in tasks}
    
    for task in tasks:
        for dep in task['dependencies']:
            children[dep].append(task['id'])
            in_degree[task['id']] += 1
    
    # Topological sort (execution order)
    execution_order = []
    queue = [task['id'] for task in tasks if in_degree[task['id']] == 0]
    queue.sort()  # alphabetical ordering when multiple tasks available
    
    while queue:
        current = queue.pop(0)
        execution_order.append(current)
        
        for child in children[current]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
                queue.sort()
    
    # Forward pass: earliest start and finish times
    earliest_start = {task['id']: 0 for task in tasks}
    earliest_finish = {task['id']: 0 for task in tasks}
    
    for task_id in execution_order:
        task = task_map[task_id]
        # Earliest start is max of dependencies' finish times
        if task['dependencies']:
            earliest_start[task_id] = max(earliest_finish[dep] for dep in task['dependencies'])
        else:
            earliest_start[task_id] = 0
        earliest_finish[task_id] = earliest_start[task_id] + task['duration']
    
    makespan = max(earliest_finish.values())
    
    # Backward pass: latest start and finish times
    latest_finish = {task['id']: makespan for task in tasks}
    latest_start = {task['id']: makespan for task in tasks}
    
    reversed_order = execution_order[::-1]
    for task_id in reversed_order:
        task = task_map[task_id]
        # If task has children, latest finish is min of children's latest starts
        if children[task_id]:
            latest_finish[task_id] = min(latest_start[child] for child in children[task_id])
        else:
            latest_finish[task_id] = makespan
        latest_start[task_id] = latest_finish[task_id] - task['duration']
    
    # Slack time
    slack = {task['id']: latest_start[task['id']] - earliest_start[task['id']] for task in tasks}
    
    # Critical path (tasks with zero slack)
    critical_path = []
    # Find start of critical path (task with zero slack and no dependencies)
    start_tasks = [task['id'] for task in tasks if not task['dependencies']]
    current = start_tasks[0] if start_tasks else None
    
    while current is not None:
        critical_path.append(current)
        # Find next critical task among children
        critical_children = [child for child in children[current] 
                            if slack[child] == 0]
        current = critical_children[0] if critical_children else None
    
    return {
        'execution_order': execution_order,
        'earliest_start': earliest_start,
        'earliest_finish': earliest_finish,
        'latest_start': latest_start,
        'latest_finish': latest_finish,
        'slack': slack,
        'critical_path': critical_path,
        'makespan': makespan
    }