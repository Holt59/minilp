from __future__ import annotations

import abc
import typing

import numpy as np

import minilp.exprs
import minilp.problems
import minilp.results
from minilp.modeler import modeler


class solver(abc.ABC):

    """
    Abstract class representing a solver for linear programs (without
    integer variables).
    """

    @abc.abstractmethod
    def solve(self, problem: minilp.problems.problem) -> minilp.results.result:
        """
        Solve the linear relaxation of the given problem.

        Args:
            problem: The problem to solve.

        Returns:
            A solution for the given problem.
        """
        pass


class pysimplex(solver):
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
        # Clean the problem:
        problem._clean()

        pb = problem

        nrows = 1 + len(pb.constraints) + 2 * len(pb.variables)
        ncols = 1 + len(pb.variables) + len(pb.constraints) + 2 * len(pb.variables) + 1

        # create zeros matrix
        A = np.zeros((nrows, ncols))

        # top-left cell is 1
        A[0, 0] = 1

        # add constraints
        row = 1
        nse = len(pb.variables) + 1
        for cn in pb.constraints:
            A[row, 1 : (len(pb.variables) + 1)] = cn.lhs._u[1:]
            if cn.oper == minilp.exprs.comparison_operator.le:
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
        if pb.sense == "max":
            mul = -1
        else:
            mul = 1

        A[0, 1 : len(pb.variables) + 1] = mul * pb.objective._u[1:]

        # drop all zeros rows
        A = A[~np.all(A == 0, axis=1), :]
        # drop all zeros columns
        A = A[:, ~np.all(A == 0, axis=0)]

        nrows, ncols = A.shape

        vrows = []  # number of violated rows
        for row in range(nrows):
            if A[row, -1] < 0:
                A[row, :] *= -1
            if np.sign(A[row, -1]) != np.sign(A[row, len(pb.variables) + 1 : -1].sum()):
                vrows.append(row)

        # phase 1
        S = np.zeros((nrows, ncols + len(vrows)))
        S[0, 0] = 1
        S[1:, : ncols - 1] = A[1:, :-1]
        S[1:, -1] = A[1:, -1]
        for i, row in enumerate(vrows):
            S[0, :] -= S[row, :]
            S[row, ncols + i - 1] = 1

        S, z, x = self.simplex(S)

        if z > self.eps:
            return minilp.results.result(False, minilp.results.status.INFEASIBLE)

        # phase 2
        basis = self.get_basis(S)
        A = S[:, :ncols]
        A[:, -1] = S[:, -1]
        A[0, 1 : len(pb.variables) + 1] = mul * pb.objective._u[1:]
        A[0, len(pb.variables) + 1 :] = 0

        for i in range(ncols - 1):
            if basis[i] != 0:
                A[0, :] -= A[0, i] * A[basis[i], :]

        A, z, x = self.simplex(A)

        if x is None:
            return minilp.results.result(
                False, minilp.results.status.UNBOUNDED, mul * (-np.inf)
            )

        return minilp.results.result(
            True, minilp.results.status.OPTIMAL, mul * z, x[: len(pb.variables)]
        )


class scipy(solver):
    # Mapping between scipy status and minilp status.
    _status = [
        minilp.results.status.OPTIMAL,
        minilp.results.status.UNKNOWN,
        minilp.results.status.INFEASIBLE,
        minilp.results.status.UNBOUNDED,
    ]

    def __init__(self):
        from scipy.optimize import linprog

        self.__linprog = linprog
        self.method = "simplex"

    def solve(self, problem):
        # Clean the problem:
        problem._clean()

        obj = problem.objective._u[1:].copy()
        if problem.sense == "max":
            obj *= -1
        kargs = {"c": obj, "bounds": [(v.lb, v.ub) for v in problem.variables]}
        A_ub, b_ub, A_eq, b_eq = [], [], [], []
        for c in problem.constraints:
            if c.oper == minilp.exprs.comparison_operator.le:
                A_ub.append(c._e._u[1:])
                b_ub.append(c._r)
            else:
                A_eq.append(c._e._u[1:])
                b_eq.append(c._r)

        if A_ub:
            kargs.update({"A_ub": A_ub, "b_ub": b_ub})
        if A_eq:
            kargs.update({"A_eq": A_eq, "b_eq": b_eq})
        kargs["method"] = self.method
        res = self.__linprog(**kargs)
        if res.success and problem.sense == "max":
            res.fun *= -1
        return minilp.results.result(
            res.success, scipy._status[res.status], res.fun, res.x
        )


class docplex(solver):
    def __init__(self):
        from docplex.util.status import JobSolveStatus

        self.status = {
            JobSolveStatus.UNKNOWN: minilp.results.status.UNKNOWN,
            JobSolveStatus.FEASIBLE_SOLUTION: minilp.results.status.FEASIBLE,
            JobSolveStatus.OPTIMAL_SOLUTION: minilp.results.status.OPTIMAL,
            # fmt: off
            JobSolveStatus.INFEASIBLE_OR_UNBOUNDED_SOLUTION:
                minilp.results.status.UNKNOWN,
            # fmt: on
            JobSolveStatus.INFEASIBLE_SOLUTION: minilp.results.status.INFEASIBLE,
            JobSolveStatus.UNBOUNDED_SOLUTION: minilp.results.status.UNBOUNDED,
        }

    def solve(self, problem):
        from docplex.mp.model import Model

        # Clean the problem:
        problem._clean()

        with Model() as m:
            # Create a list of variables:
            v = m.continuous_var_list(
                len(problem.variables),
                [
                    v.lb if v.lb > -modeler.inf else -m.infinity
                    for v in problem.variables
                ],
                [v.ub if v.ub < modeler.inf else m.infinity for v in problem.variables],
            )

            # Set the objective:
            obj = problem.objective._u[0]
            obj += sum(c * v for c, v in zip(problem.objective._u[1:], v))
            m.set_objective(problem.sense, obj)

            # Add the constraints:
            for cn in problem.constraints:
                lhs = sum(c * v for c, v in zip(cn.lhs._u[1:], v))
                rhs = cn.rhs
                if cn.oper == minilp.exprs.comparison_operator.eq:
                    m.add_constraint(lhs == rhs)
                else:
                    m.add_constraint(lhs <= rhs)

            # Disable presolve:
            m.parameters.preprocessing.presolve.set(0)

            # Solve the problem:
            m.solve()

            if not m.solution:
                return minilp.results.result(
                    False,
                    self.status[m.get_solve_status()],
                    np.nan,
                    [None] * len(problem.variables),
                )
            return minilp.results.result(
                True,
                self.status[m.get_solve_status()],
                m.solution.objective_value,
                m.solution.get_values(v),
            )


# The default solver to  use:
default_solver: typing.Optional[typing.Type[solver]] = None


def set_default_solver(solver_class: typing.Type[solver]):
    """Set the type of the default solver to use.

    Args:
        solver_class: Class of the solver to use. Must inherit solver.
    """
    global default_solver
    default_solver = solver_class


def get_default_solver() -> solver:
    """Get a new instance of the default solver.

    If a default solver class has been set using set_default_solver,
    and instance of the class is created and returned. Otherwize,
    we try to create solvers starting with docplex, and returns the
    first one available.

    Returns:
        A new instance of the default solver.
    """
    if default_solver is not None:
        return default_solver()
    solvers = [docplex, scipy]
    for solver in solvers:
        try:
            s = solver()  # try to construct a solver
        except ImportError:
            pass
        else:
            return s
    return pysimplex()
