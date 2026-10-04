def compute_dilation_factor(tasks: list, capacity: int) -> dict:
    n = len(tasks)
    if n == 0:
        return {
            "dilation_factor": 0.0,
            "actual_makespan": 0.0,
            "lower_bound": 0.0,
            "start_times": []
        }

    # ---------- Critical path length ----------
    memo = {}

    def cp(i):
        if i in memo:
            return memo[i]
        deps = tasks[i].get("dependencies", [])
        if not deps:
            val = float(tasks[i]["duration"])
        else:
            val = float(tasks[i]["duration"]) + max(cp(d) for d in deps)
        memo[i] = val
        return val

    critical_path = max(cp(i) for i in range(n))

    # ---------- Total work and lower bound ----------
    total_work = sum(
        float(tasks[i]["duration"]) * float(tasks[i]["resources"]) for i in range(n)
    )
    lower_bound = float(max(critical_path, total_work / capacity))

    # ---------- Greedy list schedule ----------
    completed = [False] * n
    scheduled = [False] * n
    start_times = [-1.0] * n
    finish_times = [-1.0] * n

    running = []          # (finish_time, task_index, resources)
    resources_used = 0.0
    t = 0.0
    completed_count = 0

    while completed_count < n:
        # Process completions at current time
        still_running = []
        for finish, idx, res in running:
            if finish <= t + 1e-12:
                completed[idx] = True
                completed_count += 1
                resources_used -= res
            else:
                still_running.append((finish, idx, res))
        running = still_running

        # Find ready tasks (dependencies completed, not scheduled)
        ready = []
        for i in range(n):
            if not scheduled[i] and not completed[i]:
                deps = tasks[i].get("dependencies", [])
                if all(completed[d] for d in deps):
                    ready.append(i)

        # Schedule ready tasks in index order if resources allow
        for i in ready:
            res = float(tasks[i]["resources"])
            if resources_used + res <= capacity + 1e-12:
                start_times[i] = t
                finish = t + float(tasks[i]["duration"])
                finish_times[i] = finish
                running.append((finish, i, res))
                resources_used += res
                scheduled[i] = True

        if completed_count == n:
            break

        if running:
            t = min(finish for finish, _, _ in running)
        else:
            break

    actual_makespan = float(max(finish_times)) if n > 0 else 0.0

    # ---------- Dilation factor ----------
    if lower_bound > 0:
        dilation = actual_makespan / lower_bound
    else:
        dilation = 0.0

    return {
        "dilation_factor": round(float(dilation), 4),
        "actual_makespan": round(float(actual_makespan), 4),
        "lower_bound": round(float(lower_bound), 4),
        "start_times": [round(float(st), 4) for st in start_times]
    }