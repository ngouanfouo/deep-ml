import numpy as np
from math import factorial, exp

def jacks_car_rental(max_cars, max_move, rental_credit, move_cost,
                     request_lambda, return_lambda, gamma,
                     poisson_upper_bound=11, theta=0.01):
    """
    Solve a simplified car rental MDP using policy iteration.
    """

    def poisson_pmf(lam, n):
        return (lam ** n) * exp(-lam) / factorial(n)

    # Poisson probabilities truncated WITHOUT renormalization
    poisson_probs = []
    for loc in range(2):
        lam_req = request_lambda[loc]
        lam_ret = return_lambda[loc]
        poisson_probs.append({
            'request': [poisson_pmf(lam_req, n) for n in range(poisson_upper_bound)],
            'return': [poisson_pmf(lam_ret, n) for n in range(poisson_upper_bound)],
        })

    policy = [[0 for _ in range(max_cars + 1)] for _ in range(max_cars + 1)]
    value = [[0.0 for _ in range(max_cars + 1)] for _ in range(max_cars + 1)]

    def get_feasible_actions(i, j):
        actions = []
        for a in range(0, min(i, max_move, max_cars - j) + 1):
            actions.append(a)
        for a in range(-1, -min(j, max_move, max_cars - i) - 1, -1):
            actions.append(a)
        actions.sort()  # most negative first for tie-breaking
        return actions

    loc_cache = [{}, {}]

    def compute_location_dynamics(loc, morning_cars):
        if morning_cars in loc_cache[loc]:
            return loc_cache[loc][morning_cars]

        expected_reward = 0.0
        next_dist = {}

        for req in range(poisson_upper_bound):
            req_prob = poisson_probs[loc]['request'][req]
            actual_rentals = min(morning_cars, req)
            reward = actual_rentals * rental_credit
            remaining = morning_cars - actual_rentals

            for ret in range(poisson_upper_bound):
                ret_prob = poisson_probs[loc]['return'][ret]
                prob = req_prob * ret_prob
                end_of_day = min(remaining + ret, max_cars)

                expected_reward += prob * reward
                next_dist[end_of_day] = next_dist.get(end_of_day, 0.0) + prob

        loc_cache[loc][morning_cars] = (expected_reward, next_dist)
        return expected_reward, next_dist

    def expected_action_value(i, j, a):
        morning1 = i - a
        morning2 = j + a
        move_cost_total = abs(a) * move_cost

        reward1, next_dist1 = compute_location_dynamics(0, morning1)
        reward2, next_dist2 = compute_location_dynamics(1, morning2)

        # Probability mass kept after truncation at each location
        mass1 = sum(next_dist1.values())
        mass2 = sum(next_dist2.values())

        # Rental rewards are weighted by the other location's kept mass;
        # the deterministic move cost is subtracted in full
        total_reward = reward1 * mass2 + reward2 * mass1 - move_cost_total

        expected_future = 0.0
        for next1, prob1 in next_dist1.items():
            for next2, prob2 in next_dist2.items():
                expected_future += prob1 * prob2 * value[next1][next2]

        return total_reward + gamma * expected_future

    num_iterations = 0
    policy_stable = False

    while not policy_stable:
        # Policy Evaluation
        while True:
            delta = 0.0
            for i in range(max_cars + 1):
                for j in range(max_cars + 1):
                    v = value[i][j]
                    a = policy[i][j]
                    value[i][j] = expected_action_value(i, j, a)
                    delta = max(delta, abs(v - value[i][j]))
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for i in range(max_cars + 1):
            for j in range(max_cars + 1):
                old_action = policy[i][j]
                best_action = old_action
                best_value = expected_action_value(i, j, old_action)

                for a in get_feasible_actions(i, j):
                    v = expected_action_value(i, j, a)
                    if v > best_value + 1e-9:
                        best_value = v
                        best_action = a
                    elif abs(v - best_value) <= 1e-9 and a < best_action:
                        best_action = a

                if best_action != old_action:
                    policy_stable = False
                    policy[i][j] = best_action

        num_iterations += 1

    value_rounded = [[round(v, 2) for v in row] for row in value]
    return policy, value_rounded, num_iterations