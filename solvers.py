# -*- encoding: utf-8 -*-

import numpy as np
import operator as oper

from minilp.result import result


class solver:

    def solve(self, problem):
        pass


class scipy:

    status = ['optimal', 'unknown', 'infeasible', 'unbounded']

    def __init__(self):
        from scipy.optimize import linprog
        self.__linprog = linprog
        self.method = 'simplex'

    def solve(self, problem):

        # Number of variables
        nvars = len(problem.variables)

        obj = np.concatenate((
            problem.objective._u[1:],
            np.zeros(max(0, nvars - len(problem.objective._u[1:])))
        ))
        if problem.sense == max:
            obj *= -1
        kargs = {
            'c': obj,
            'bounds': [(v.lb, v.ub) for v in problem.variables]
        }
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for c in problem.constraints:
            if c.oper == oper.le:
                A_ub.append(np.concatenate((
                    c._e._u[1:],
                    np.zeros(max(0, nvars - len(c._e._u[1:]))))))
                b_ub.append(c._r)
            else:
                A_eq.append(np.concatenate((
                    c._e._u[1:],
                    np.zeros(max(0, nvars - len(c._e._u[1:]))))))
                b_eq.append(c._r)

        if A_ub:
            kargs.update({
                'A_ub': A_ub,
                'b_ub': b_ub
            })
        if A_eq:
            kargs.update({
                'A_eq': A_eq,
                'b_eq': b_eq
            })
        kargs['method'] = self.method
        res = self.__linprog(**kargs)
        if res.success and problem.sense == max:
            res.fun *= -1
        return result(res.success, scipy.status[res.status], res.fun, res.x)


class docplex:

    sense = {
        min: 'min',
        max: 'max'
    }

    def __init__(self):
        from docplex.mp. model import Model
        from docloud.status import JobSolveStatus
        self.__model = Model
        self.status = {
            JobSolveStatus.UNKNOWN: 'unknown',
            JobSolveStatus.FEASIBLE_SOLUTION: 'feasible',
            JobSolveStatus.OPTIMAL_SOLUTION: 'optimal',
            JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION: 'unknown',
            JobSolveStatus.INFEASIBLE_SOLUTION: 'infeasible',
            JobSolveStatus.UNBOUNDED_SOLUTION: 'unbounded'
        }

    def solve(self, problem):

        m = self.__model()
        v = m.continuous_var_list(
            len(problem.variables),
            [v.lb if v.lb != np.inf else m.infinity
             for v in problem.variables],
            [v.ub if v.ub != np.inf else m.infinity
             for v in problem.variables])

        # objective
        obj = problem.objective._u[0]
        obj += sum(c * v for c, v in zip(problem.objective._u[1:], v))
        m.set_objective(self.sense[problem.sense], obj)

        # constraints
        for cn in problem.constraints:
            lhs = sum(c * v for c, v in zip(cn.lhs._u[1:], v))
            rhs = cn.rhs
            if cn.oper == oper.eq:
                m.add_constraint(lhs == rhs)
            else:
                m.add_constraint(lhs <= rhs)

        # solve
        m.solve()

        if not m.solution:
            return result(False, self.status[m.get_solve_status()], np.nan,
                          [None] * len(problem.variables))
        return result(True, self.status[m.get_solve_status()],
                      m.solution.objective_value,
                      [m.solution.get_value(x) for x in v])


def get_default_solver():
    solvers = [docplex, scipy]
    for solver in solvers:
        try:
            s = solver()  # try to construct a solver
        except:
            pass
        finally:
            return s
    return None
