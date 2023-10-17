# pyright: reportPrivateUsage=false

from __future__ import annotations

import collections.abc
from typing import Iterable, Literal, Sequence, TypeVar

import numpy as np

import minilp.exprs as exprs
import minilp.solvers as solvers
from minilp.modeler import modeler

_T = TypeVar("_T")


class problem(modeler):
    _idx: int = 1
    _sense: Literal["min", "max"]
    _vars: list[exprs.var]
    _cons: list[exprs.cons]
    _obj: exprs.expr

    name: str

    def __init__(self, name: str = ""):
        """
        Create a new problem with the given name and sense for the objective.

        Args:
            name: Name of the problem.
        """
        self._idx = 1
        self._vars = []
        self._cons = []
        self._sense = "min"
        self._obj = exprs.expr(0, self)
        self.name = name

    def _var(
        self,
        lb: float = 0,
        ub: float = modeler.inf,
        cat: type = int,
        name: str | None = None,
    ) -> exprs.var:
        """
        Create a variable of the given category.

        Args:
            lb: Lower bound of the variable (or -inf for unbounded).
            ub: Upper bound of the variable (or inf for unbounded).
            cat: Category of the variable (float or int).
            name: Name of the variable.

        Returns:
            A variable of the given category with the given parameters..
        """
        idx = self._idx
        self._idx += 1
        if name is None:
            name = "_x{}".format(idx)
        self._vars.append(exprs.var(self, idx, lb, ub, cat, name))
        return self._vars[-1]

    def _var_dict(
        self,
        keys: Iterable[_T],
        lb: Iterable[float] | float = 0,
        ub: Iterable[float] | float = modeler.inf,
        cat: type = int,
        prefix: str = "",
    ) -> dict[_T, exprs.var]:
        """
        Create a dictionary of variables of the given category
        with the given parameters.

        The names of the created variables are the concatenation of prefix
        and the key.

        Args:
            keys: Keys of the dictionary to create (one key per variable).
            lb: Lower bounds of the variables (or -inf for unbounded). Either a
                list of values (one value per key), or a single value to use
                for all the keys.
            ub: Upper bounds of the variables (or -inf for unbounded). Either a
                list of values (one value per key), or a single value to use
                for all the keys.
            cat: Category of the variable (float or int).
            prefix: Prefix for the name of the variable.

        Returns:
            A dictionary of variables of the given category with the given parameters..
        """
        keys = tuple(keys)
        if not isinstance(lb, collections.abc.Iterable):
            lb = [lb] * len(keys)
        if not isinstance(ub, collections.abc.Iterable):
            ub = [ub] * len(keys)
        return {
            k: self._var(l, u, cat, "{}{}".format(prefix, k))
            for l, u, k in zip(lb, ub, keys)  # noqa: E741
        }

    def _var_list(
        self,
        n_or_names: int | Iterable[str],
        lb: Iterable[float] | float = 0,
        ub: Iterable[float] | float = modeler.inf,
        cat: type = int,
        prefix: str | None = None,
    ) -> list[exprs.var]:
        """
        Create a list of variables of the given category
        with the given parameters.

        The names of the created variables are the concatenation of prefix
        and the index (if n_or_names is an integer) or the name.

        Args:
            n_or_names: Number of variables or names of the variable.
            lb: Lower bounds of the variables (or -inf for unbounded). Either a
                list of values (one value per key), or a single value to use
                for all the variables.
            ub: Upper bounds of the variables (or -inf for unbounded). Either a
                list of values (one value per key), or a single value to use
                for all the variables.
            cat: Category of the variable (float or int).
            prefix: Prefix for the name of the variable.

        Returns:
            A dictionary of variables of the given category with the given parameters..
        """
        names: Iterable[str] | Iterable[None]
        if not isinstance(n_or_names, collections.abc.Iterable):
            if prefix is None:
                names = [None] * n_or_names
            else:
                names = [str(c) for c in range(n_or_names)]
        else:
            names = list(n_or_names)
        if prefix is not None:
            names = ["{}{}".format(prefix, c) for c in names]
        if not isinstance(lb, collections.abc.Iterable):
            lb = [lb] * len(names)
        if not isinstance(ub, collections.abc.Iterable):
            ub = [ub] * len(names)
        return [self._var(l, u, cat, n) for l, u, n in zip(lb, ub, names)]  # noqa: E741

    def binary_var(self, name: str | None = None) -> exprs.var:
        """
        Create a new binary variable with the given name.

        Args:
            name: Name of the variable. If None, a name will be
                  automatically generated.

        Returns:
            A binary variable with the given parameters.
        """
        return self._var(0, 1, int, name)

    def integer_var(
        self, lb: float = 0, ub: float = modeler.inf, name: str | None = None
    ) -> exprs.var:
        """
        Create a new integer variable with the given bounds and name.

        Args:
            lb: Lower bound of the variable (or -inf for unbounded).
            ub: Upper bound of the variable (or inf for unbounded).
            name: Name of the variable. If None, a name will be
                  automatically generated.

        Returns:
            An integer variable with the given parameters.
        """
        return self._var(lb, ub, int, name)

    def continuous_var(
        self, lb: float = 0, ub: float = modeler.inf, name: str | None = None
    ) -> exprs.var:
        """
        Create a new continuous variable with the given bounds and name.

        Args:
            lb: Lower bound of the variable (or -inf for unbounded).
            ub: Upper bound of the variable (or inf for unbounded).
            name: Name of the variable. If None, a name will be
                  automatically generated.

        Returns:
            A continuous variable with the given parameters.
        """
        return self._var(lb, ub, float, name)

    def binary_var_list(
        self,
        n_or_names: int | Iterable[str],
        prefix: str | None = None,
    ) -> list[exprs.var]:
        """
        Create a list of binary variables.

        The name of the variable is created by concatenating the prefix with
        either the index of the variable (if n_or_names is an int) or the
        name in n_or_names. If prefix is None, names are generated as if multiple
        calls to binary_var had been made.

        Args:
            n_or_names: Number of variables to create or list of variable names.
            prefix: Prefix for the name of the variable.

        Returns:
            A list of binary variables with the given parameters.
        """
        return self._var_list(n_or_names, 0, 1, int, prefix)

    def integer_var_list(
        self,
        n_or_names: int | Iterable[str],
        lb: float = 0,
        ub: float = modeler.inf,
        prefix: str | None = None,
    ) -> list[exprs.var]:
        """
        Create a list of integer variables.

        The name of the variable is created by concatenating the prefix with
        either the index of the variable (if n_or_names is an int) or the
        name in n_or_names. If prefix is None, names are generated as if multiple
        calls to integer_var had been made.

        Args:
            n_or_names: Number of variables to create or list of variable names.
            lb: Lower bound of the variable (or -inf for unbounded), can be a
                single value (same lower bound for all variables) or a list of
                lower bounds.
            ub: Upper bound of the variable (or inf for unbounded), can be a
                single value (same upper bound for all variables) or a list of
                upper bounds.
            prefix: Prefix for the name of the variable.

        Returns:
            A list of integer variables with the given parameters.
        """
        return self._var_list(n_or_names, lb, ub, int, prefix)

    def continuous_var_list(
        self,
        n_or_names: int | Iterable[str],
        lb: float = 0,
        ub: float = modeler.inf,
        prefix: str | None = None,
    ) -> list[exprs.var]:
        """
        Create a list of continuous variables.

        The name of the variable is created by concatenating the prefix with
        either the index of the variable (if n_or_names is an int) or the
        name in n_or_names. If prefix is None, names are generated as if multiple
        calls to continuous_var had been made.

        Args:
            n_or_names: Number of variables to create or list of variable names.
            lb: Lower bound of the variable (or -inf for unbounded), can be a
                single value (same lower bound for all variables) or a list of
                lower bounds.
            ub: Upper bound of the variable (or inf for unbounded), can be a
                single value (same upper bound for all variables) or a list of
                upper bounds.
            prefix: Prefix for the name of the variable.

        Returns:
            A list of continuous variables with the given parameters.
        """
        return self._var_list(n_or_names, lb, ub, float, prefix)

    def binary_var_dict(self, keys: Iterable[_T]) -> dict[_T, exprs.var]:
        """
        Create a dictionary of binary variables.

        Args:
            keys: Keys of the dictionary (used for variable names).

        Returns:
            A dictionary of binary variables.
        """
        return self._var_dict(keys, 0, 1, int)

    def integer_var_dict(
        self, keys: Iterable[_T], lb: float = 0, ub: float = modeler.inf
    ) -> dict[_T, exprs.var]:
        """
        Create a dictionary of integer variables.

        Args:
            keys: Keys for the dictionary (used for variable names).
            lb: Lower bound of the variable (or -inf for unbounded), can be a
                single value (same lower bound for all variables) or a list of
                lower bounds.
            ub: Upper bound of the variable (or inf for unbounded), can be a
                single value (same upper bound for all variables) or a list of
                upper bounds.

        Returns:
            A dictionary of integer variables with the given parameters.
        """
        return self._var_dict(keys, lb, ub, int)

    def continuous_var_dict(
        self, keys: Iterable[_T], lb: float = 0, ub: float = modeler.inf
    ) -> dict[_T, exprs.var]:
        """
        Create a dictionary of continuous variables.

        Args:
            keys: Keys for the dictionary (used for variable names).
            lb: Lower bound of the variable (or -inf for unbounded), can be a
                single value (same lower bound for all variables) or a list of
                lower bounds.
            ub: Upper bound of the variable (or inf for unbounded), can be a
                single value (same upper bound for all variables) or a list of
                upper bounds.

        Returns:
            A dictionary of continuous variables with the given paramaeters.
        """
        return self._var_dict(keys, lb, ub, float)

    def add_constraint(self, constraint: exprs.cons) -> exprs.cons:
        """
        Add the given constraint to the problem.

        Args:
            constraint: Constraint to add to the problem.

        Returns:
            The added constraint.
        """

        if not isinstance(
            constraint, exprs.cons
        ):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(
                "Constraint must be a valid {} instance.".format(
                    ".".join([exprs.cons.__module__, exprs.cons.__name__])
                )
            )

        if constraint._pb != self:
            raise ValueError(
                "Cannot share constraints between different "
                "minilp.problems.problem instances."
            )

        self._cons.append(constraint)
        return self._cons[-1]

    def add_constraints(self, constraints: Iterable[exprs.cons]) -> list[exprs.cons]:
        """
        Add the given constraints to the problem.

        Args:
            constraints: List of constraints to add to the problem.

        Returns:
            The list of constraints added to the problem.
        """
        return [self.add_constraint(c) for c in constraints]

    def del_constraint(self, constraint_or_idx: int | exprs.cons):
        """
        Delete the specified constraint from the problem.

        Args:
            constraint_or_idx: Constraint or index of the constraint
                               to remove.
        """
        idx = constraint_or_idx
        if isinstance(idx, exprs.cons):
            idx = self._cons.index(idx)
        del self._cons[idx]

    def del_constraints(
        self,
        constraints_or_idxs: Iterable[int | exprs.cons],
    ):
        """
        Delete the specified constraints from the problem.

        Args:
            constraints_or_idxs: Constraints or indexes of the constraints
                                 to remove.
        """
        # Create a copy in case the given list is the list of constraints:
        constraints_or_idxs = list(constraints_or_idxs)

        for c in constraints_or_idxs:
            self.del_constraint(c)

    def set_objective(self, sense: Literal["max", "min"], objective: exprs.expr):
        """
        Set the objective of the problem.

        Args:
            sense: Sense of the objective ('min' or 'max').
            objective: Expression of the objective of the problem.
        """
        if sense not in ["min", "max"]:
            raise ValueError("Unrecognized sense for optimization: {}.".format(sense))

        if not isinstance(
            objective, exprs.expr
        ):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError(
                "Objective must be a valid {} instance.".format(
                    ".".join([exprs.expr.__module__, exprs.expr.__name__])
                )
            )

        if objective._pb != self:
            raise ValueError(
                "Cannot share expressions between different "
                "minilp.problems.problem instances."
            )

        self._sense = sense
        self._obj = objective

    def maximize(self, objective: exprs.expr):
        """
        Set the objective value as a maximixation.

        Args:
            objective: Expression of the objective of the problem.
        """
        self.set_objective("max", objective)

    def minimize(self, objective: exprs.expr):
        """
        Set the objective value as a minimization.

        Args:
            objective: Expression of the objective value of the problem.
        """
        self.set_objective("min", objective)

    @property
    def variables(self) -> Sequence[exprs.var]:
        """
        Returns:
            The list of variables of the problem.
        """
        return self._vars

    @property
    def constraints(self) -> Sequence[exprs.cons]:
        """
        Returns:
            The list of constraints of the problem.
        """
        return self._cons

    @property
    def objective(self) -> exprs.expr:
        """
        Returns:
            The objective expression of the problem.
        """
        return self._obj

    @property
    def sense(self) -> Literal["min", "max"]:
        """
        Returns:
            The optimization sense of the problem objective ('min' or 'max').
        """
        return self._sense

    def _clean(self):
        """
        Clean this problem by extending, if necessary, the expression
        with new variables.
        """
        ncols = len(self.variables) + 1
        self._obj._u = np.concatenate(
            (self._obj._u, np.zeros(max(0, ncols - len(self._obj._u))))
        )
        for cn in self._cons:
            cn.lhs._u = np.concatenate(
                (cn.lhs._u, np.zeros(max(0, ncols - len(cn.lhs._u))))
            )
        for vs in self._vars:
            vs._u = np.concatenate((vs._u, np.zeros(max(0, ncols - len(vs._u)))))

    def lp_solve(self, solver: solvers.solver | None = None):
        """
        Solve a relaxation of the problem using the specific solver.

        Args:
            solver: Solver to use to solve the linear relaxation of this problem.

        Returns:
            The solution returned by the given solver for this problem.
        """
        if solver is None:
            solver = solvers.get_default_solver()
        return solver.solve(self)

    def __str__(self):
        s: list[str] = []

        # Type of problem:
        pb_type = "LP"
        if all(x.category == int for x in self.variables):
            pb_type = "ILP"
        elif any(x.category == int for x in self.variables):
            pb_type = "MILP"

        s.append("{} --- {}".format(pb_type, self.name))
        s.append("-" * len(s[0]))
        s.append("{}.   {}".format(self.sense, self.objective))
        if self.constraints:
            s.append("s.t.   {}".format(self.constraints[0]))
            for c in self.constraints[1:]:
                s.append("       {}".format(c))

        return "\n".join(s)

    def __repr__(self):
        return str(self)
