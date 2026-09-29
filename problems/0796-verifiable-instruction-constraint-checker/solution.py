import re


def check_constraints(response, constraints):
    """
    response: str — the model's generated text.
    constraints: list[str] — constraint specifications.
    Returns: dict[str, bool] mapping each constraint to whether it is satisfied,
             preserving the input order of constraints.
    """
    result = {}
    words = response.split()
    word_count = len(words)

    for constraint in constraints:
        satisfied = False

        if constraint.startswith('word_count_min:'):
            try:
                n = int(constraint[len('word_count_min:'):])
                satisfied = word_count >= n
            except ValueError:
                satisfied = False

        elif constraint.startswith('word_count_max:'):
            try:
                n = int(constraint[len('word_count_max:'):])
                satisfied = word_count <= n
            except ValueError:
                satisfied = False

        elif constraint.startswith('contains_keyword:'):
            keyword = constraint[len('contains_keyword:'):]
            satisfied = keyword.lower() in response.lower()

        elif constraint.startswith('starts_with:'):
            prefix = constraint[len('starts_with:'):]
            satisfied = response.startswith(prefix)

        elif constraint.startswith('ends_with:'):
            suffix = constraint[len('ends_with:'):]
            satisfied = response.endswith(suffix)

        elif constraint == 'no_bullet_points':
            satisfied = True
            for line in response.split('\n'):
                stripped = line.lstrip()
                # Bullet markers: "- ", "* ", or digits followed by ". "
                if stripped.startswith('- ') or stripped.startswith('* '):
                    satisfied = False
                    break
                if re.match(r'^\d+\.\s', stripped):
                    satisfied = False
                    break

        else:
            satisfied = False

        result[constraint] = satisfied

    return result