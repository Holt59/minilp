# -*- encoding: utf-8 -*-

from pytest import approx

import minilp


def test_minilp_lp():

    lp = minilp.problem('My first LP problem')
    # Create two continuous variables within [0, 4]:
    x1, x2 = lp.continuous_var_list(2, 0, 4)

    # Add constraints:
    lp.add_constraint(-3 * x1 + 4 * x2 <= 7)
    lp.add_constraint(2 * x2 <= 5)
    lp.add_constraint(6 * x1 + 4 * x2 <= 25)
    lp.add_constraint(2 * x1 - x2 <= 6)

    # Set the objective function:
    lp.set_objective('max', x1 + 2 * x2)

    # Solve the problem:
    res = lp.lp_solve()

    assert res.status == minilp.status.optimal
    assert res.objective == approx(7.5)
    assert res.get_value(x1) == approx(2.5)
    assert res.get_value(x2) == approx(2.5)


def test_minilp_kp():
    N = 5
    p = [1, 4, 5, 3, 5] # profits
    w = [3, 4, 3, 5, 9] # weights
    K = 10 # capacity

    assert N == len(w) and N == len(p)

    # A simple knapsack:
    kp = minilp.problem('Simple knapsack')

    # Create variables, add constraints and set the objective:
    x = kp.binary_var_list(N)
    kp.add_constraint(kp.dot(x, w) <= K)
    kp.maximize(kp.dot(x, p))

    # We can solve the linear relaxation:
    res = kp.lp_solve()

    assert res.status == minilp.status.optimal
    assert res.objective == approx(10.8)

    expected_x = [0, 1, 1, 0.6, 0]
    assert res.get_values(x) == [approx(v) for v in expected_x]
