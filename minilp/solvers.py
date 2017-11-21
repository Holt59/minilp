# -*- encoding: utf-8 -*-

import numpy as np
import operator as oper

from minilp.result import result


class solver:

    def solve(self, problem):
        pass


class pysimplex:

    eps = 1e-8

    def get_basis(self, A):
        basis = np.zeros(A.shape[1], dtype=int)
        for i in range(1, len(basis)):
            eq1 = A[:, i] == 1
            if eq1.sum() == 1 and (abs(A[~eq1, i]) < self.eps).all():
                basis[i] = eq1.argmax()
        return basis

    def simplex(self, A):
        while (A[0, :-1] < 0).any():
            # Bland's rule
            entx = (A[0, :-1] < 0).argmax()
            if (A[1:, entx] <= self.eps).all():
                # unbounded
                return A, -np.inf, None
            c = A[1:, entx]
            b = A[1:, -1]
            ratios = np.ones(A.shape[0] - 1) * np.inf
            ratios[(c > 0) & (b == 0)] = 0
            m = (c != 0) & (b != 0)
            ratios[m] = b[m] / c[m]
            ratios[ratios < 0] = np.inf
            outy = ratios.argmin() + 1
            A[outy, :] /= A[outy, entx]
            for y in range(len(A)):
                if y != outy:
                    A[y, :] -= A[y, entx] * A[outy, :]
            A[:, entx] = 0
            A[outy, entx] = 1

        basis = self.get_basis(A)
        x = np.zeros(basis.shape)
        for i in range(len(x)):
            if basis[i] != 0:
                x[i] = A[basis[i], -1]
        return A, -A[0, -1], np.array(x[1:-1])

    def solve(self, problem):
        pb = problem

        nrows = 1 + len(pb.constraints) + 2 * len(pb.variables)
        ncols = 1 + len(pb.variables) + len(pb.constraints) \
            + 2 * len(pb.variables) + 1

        # create zeros matrix
        A = np.zeros((nrows, ncols))

        # top-left cell is 1
        A[0, 0] = 1

        # add constraints
        row = 1
        nse = len(pb.variables) + 1
        for cn in pb.constraints:
            A[row, 1:(len(pb.variables) + 1)] = cn.lhs._u[1:]
            if cn.oper == oper.le:
                A[row, nse] = 1
            A[row, -1] = cn.rhs
            row += 1
            nse += 1

        # add variable bounds
        for va in pb.variables:
            if va.ub is not None and va.ub < np.inf:
                A[row, [va._idx, nse, -1]] = 1, 1, va.ub
            if va.lb is not None and va.lb > 0:
                A[row + 1, [va._idx, nse + 1, -1]] = -1, 1, -va.lb
            row += 2
            nse += 2

        # convert max -> min
        if pb.sense == max:
            mul = -1
        else:
            mul = 1

        A[0, 1:len(pb.variables) + 1] = mul * pb.objective._u[1:]

        # drop all zeros rows
        A = A[~np.all(A == 0, axis=1), :]
        # drop all zeros columns
        A = A[:, ~np.all(A == 0, axis=0)]

        nrows, ncols = A.shape

        vrows = []  # number of violated rows
        for row in range(nrows):
            if A[row, -1] < 0:
                A[row, :] *= -1
            if np.sign(A[row, -1]) != np.sign(
                    A[row, len(pb.variables) + 1:-1].sum()):
                vrows.append(row)

        # phase 1
        S = np.zeros((nrows, ncols + len(vrows)))
        S[0, 0] = 1
        S[1:, :ncols - 1] = A[1:, :-1]
        S[1:, -1] = A[1:, -1]
        for i, row in enumerate(vrows):
            S[0, :] -= S[row, :]
            S[row, ncols + i - 1] = 1

        S, z, x = self.simplex(S)

        if z > self.eps:
            return result(False, 'infeasible')

        # phase 2
        basis = self.get_basis(S)
        A = S[:, :ncols]
        A[:, -1] = S[:, -1]
        A[0, 1:len(pb.variables) + 1] = mul * pb.objective._u[1:]
        A[0, len(pb.variables) + 1:] = 0

        for i in range(ncols - 1):
            if basis[i] != 0:
                A[0, :] -= A[0, i] * A[basis[i], :]

        A, z, x = self.simplex(A)

        if x is None:
            return result(False, 'unbounded', mul * (-np.inf))

        return result(True, 'optimal', mul * z, x[:len(pb.variables)])


class scipy:

    status = ['optimal', 'unknown', 'infeasible', 'unbounded']

    def __init__(self):
        from scipy.optimize import linprog
        self.__linprog = linprog
        self.method = 'simplex'

    def solve(self, problem):

        obj = problem.objective._u[1:].copy()
        if problem.sense == max:
            obj *= -1
        kargs = {
            'c': obj,
            'bounds': [(v.lb, v.ub) for v in problem.variables]
        }
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for c in problem.constraints:
            if c.oper == oper.le:
                A_ub.append(c._e._u[1:])
                b_ub.append(c._r)
            else:
                A_eq.append(c._e._u[1:])
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
        from docplex.mp.model import Model
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
            [v.lb if v.lb > -np.inf else -m.infinity
             for v in problem.variables],
            [v.ub if v.ub < np.inf else m.infinity
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


default_solver = None


def set_default_solver(cls):
    global default_solver
    default_solver = cls


def get_default_solver():
    if default_solver is not None:
        return default_solver()
    solvers = [docplex, scipy]
    for solver in solvers:
        try:
            s = solver()  # try to construct a solver
        except:
            pass
        else:
            return s
    return pysimplex()
