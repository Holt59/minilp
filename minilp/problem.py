# -*- encoding: utf-8 -*-

import collections
import numpy as np

from minilp.expr import var, cons
import minilp.solvers as solvers


class problem:

    sense_repr = {
        min: 'min',
        max: 'max'
    }

    def __init__(self, name='', sense=min):
        """ Create a new problem with the given name and sense for the objective.

        Parameters:
          - name Name of the problem.
          - sense Sense of the objective function.
        """
        self.__idx = 1
        self.__vars = []
        self.__cons = []
        self.__obj = 0
        self.__sense = sense
        self.name = name

    def _var(self, lb=0, ub=np.inf, cat=int, name=None):
        idx = self.__idx
        self.__idx += 1
        if name is None:
            name = '_x{}'.format(idx)
        self.__vars.append(var(self, idx, lb, ub, cat, name))
        return self.__vars[-1]

    def _var_dict(self, keys, lb=0, ub=np.inf, cat=int):
        if not isinstance(lb, collections.Iterable):
            lb = [lb] * len(keys)
        if not isinstance(ub, collections.Iterable):
            ub = [ub] * len(keys)
        return {k: self._var(l, u, cat, k) for l, u, k in zip(lb, ub, keys)}

    def _var_list(self, n, lb=0, ub=np.inf, prefix=None, cat=int):
        if prefix is None:
            prefix = '_x'
        ks = ['{}{}'.format(prefix, i) for i in range(n)]
        vs = self._var_dict(ks, lb, ub, cat)
        return [vs[k] for k in ks]

    def binary_var(self, name=None):
        """ Create a new binary variable with the given name.

        Parameters:
          - name Name of the variable.

        Return: A binary varible. """
        return self._var(0, 1, int, name)

    def integer_var(self, lb=0, ub=np.inf, name=None):
        """ Create a new integer variable with the given bounds and name.

        Parameters:
          - lb Lower bound of the variable (or -np.inf for unbounded).
          - ub Upper bound of the variable (or np.inf for unbounded).
          - name Name of the variable.

        Return: An integer variable. """
        return self._var(lb, ub, int, name)

    def continuous_var(self, lb=-np.inf, ub=np.inf, name=None):
        """ Create a new continuous variable with the given bounds and name.

        Parameters:
          - lb Lower bound of the variable (or -np.inf for unbounded).
          - ub Upper bound of the variable (or np.inf for unbounded).
          - name Name of the variable.

        Return: A continuous variable. """
        return self._var(lb, ub, float, name)

    def binary_var_list(self, n, prefix=None):
        """ Create a list of binary variables with the given prefix.

        Parameters:
          - n Number of binary variables to create.
          - prefix Prefix for the name of the variable.

        Return: A list of binary variables. """
        return self._var_list(n, 0, 1, prefix, int)

    def integer_var_list(self, n, lb=0, ub=np.inf, prefix=None):
        """ Create a list of integer variables with given bounds and prefix.

        Parameters:
          - n Number of integer variables to create.
          - lb Lower bound of the variable (or -np.inf for unbounded), can be a
            single value (same lower bound for all variables) or a list of
            lower bounds.
          - ub Upper bound of the variable (or np.inf for unbounded), can be a
            single value (same upper bound for all variables) or a list of
            upper bounds.
          - prefix Prefix for the name of the variable.

        Return: A list of integer variables. """
        return self._var_list(n, lb, ub, prefix, int)

    def continuous_var_list(self, n, lb=-np.inf, ub=np.inf, prefix=None):
        """ Create a list of continuous variables with given bounds and prefix.

        Parameters:
          - n Number of continuous variables to create.
          - lb Lower bound of the variable (or -np.inf for unbounded), can be a
            single value (same lower bound for all variables) or a list of
            lower bounds.
          - ub Upper bound of the variable (or np.inf for unbounded), can be a
            single value (same upper bound for all variables) or a list of
            upper bounds.
          - prefix Prefix for the name of the variable.

        Return: A list of continuous variables. """
        return self._var_list(n, lb, ub, prefix, float)

    def binary_var_dict(self, keys):
        """ Create a dictionary of binary variables indexed by the given keys.

        Parameters:
          - keys Keys for the dictionary.

        Return: A dictionary of binary variables. """
        return self._var_dict(keys, 0, 1, int)

    def integer_var_dict(self, keys, lb=0, ub=np.inf):
        """ Create a dictionary of integer variables with given bounds indexed
        by the given keys.

        Parameters:
          - keys Keys for the dictionary.
          - lb Lower bound of the variable (or -np.inf for unbounded), can be a
            single value (same lower bound for all variables) or a list of
            lower bounds.
          - ub Upper bound of the variable (or np.inf for unbounded), can be a
            single value (same upper bound for all variables) or a list of
            upper bounds.

        Return: A dictionary of integer variables. """
        return self._var_dict(keys, lb, ub, int)

    def continuous_var_dict(self, keys, lb=-np.inf, ub=np.inf):
        """ Create a dictionary of continuous variables with given bounds indexed
        by the given keys.

        Parameters:
          - keys Keys for the dictionary.
          - lb Lower bound of the variable (or -np.inf for unbounded), can be a
            single value (same lower bound for all variables) or a list of
            lower bounds.
          - ub Upper bound of the variable (or np.inf for unbounded), can be a
            single value (same upper bound for all variables) or a list of
            upper bounds.

        Return: A dictionary of continuous variables. """
        return self._var_dict(keys, lb, ub, float)

    def add_constraint(self, cons):
        """ Add a constraint to the problem.

        Parameter:
          - cons Constraint to add (minilp.expr.cons).

        Return: The constraint. """
        self.__cons.append(cons)
        return self.__cons[-1]

    def add_constraints(self, conss):
        """ Add constraints to the problem.

        Parameter:
          - conss Constraints to add (minilp.expr.cons).

        Return: List containing the constraints. """
        return [self.add_constraint(c) for c in conss]

    def del_constraint(self, cons_or_idx):
        """ Delete the specified constraint from the problem.

        Parameter:
          - cons Constraint (minilp.expr.cons) or index of the constraint
            to remove.
        """
        idx = cons_or_idx
        if isinstance(idx, cons):
            idx = self.__cons.index(idx)
        del self.__cons[idx]

    def del_constraints(self, conss_or_idxs):
        """ Delete the specified constraints from the problem.

        Parameter:
          - cons Constraints (minilp.expr.cons) or indexes of the constraints
            to remove.
        """
        for c in conss_or_idxs:
            self.del_constraint(c)

    def set_objective(self, expr, sense=None):
        """ Set the objective value.

        Parameters:
          - expr Expression of the objective value (minilp.expr.expr).
          - sense Sense of the objective (min or max). If None, the original
            sense of the problem will be used.
        """
        if sense is not None:
            self.__sense = sense
        self.__obj = expr

    @property
    def variables(self):
        """ List of variables of the problem. """
        return self.__vars

    @property
    def constraints(self):
        """ List of constraints of the problem. """
        return self.__cons

    @property
    def objective(self):
        """ Objective of the problem. """
        return self.__obj

    @property
    def sense(self):
        """ Sense of the problem objective (min or max). """
        return self.__sense

    def lp_solve(self, solver=None):
        """ Solve a relaxation of the problem using the specific solver. """
        if solver is None:
            solver = solvers.get_default_solver()
        ncols = len(self.variables) + 1
        self.__obj._u = np.concatenate(
           (self.__obj._u, np.zeros(max(0, ncols - len(self.__obj._u)))))
        for cn in self.__cons:
            cn.lhs._u = np.concatenate(
                (cn.lhs._u, np.zeros(max(0, ncols - len(cn.lhs._u)))))
        for vs in self.__vars:
            vs._u = np.concatenate(
                (vs._u, np.zeros(max(0, ncols - len(vs._u)))))
        return solver.solve(self)

    def __str__(self):
        s = []
        s.append('ILP --- {}'.format(self.name))
        s.append('-' * len(s[0]))
        s.append('')
        s.append('{}.   {}'.format(
            problem.sense_repr[self.sense],
            self.objective))
        if self.constraints:
            s.append('s.t.   {}'.format(self.constraints[0]))
            for c in self.constraints[1:]:
                s.append('       {}'.format(c))

        return '\n'.join(s)
