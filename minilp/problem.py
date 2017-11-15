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
        self.__idx = 1
        self.__vars = []
        self.__cons = []
        self.__obj = 0
        self.sense = sense
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
        return self._var(0, 1, int, name)

    def integer_var(self, lb=0, ub=np.inf, name=None):
        return self._var(lb, ub, int, name)

    def continuous_var(self, lb=-np.inf, ub=np.inf, name=None):
        return self._var(lb, ub, float, name)

    def binary_var_list(self, n, prefix=None):
        return self._var_list(n, 0, 1, prefix, int)

    def integer_var_list(self, n, lb=0, ub=np.inf, prefix=None):
        return self._var_list(n, lb, ub, prefix, int)

    def continuous_var_list(self, n, lb=-np.inf, ub=np.inf, prefix=None):
        return self._var_list(n, lb, ub, prefix, float)

    def binary_var_dict(self, keys):
        return self._var_dict(keys, 0, 1, int)

    def integer_var_dict(self, keys, lb=0, ub=np.inf):
        return self._var_dict(keys, lb, ub, int)

    def continuous_var_dict(self, keys, lb=-np.inf, ub=np.inf):
        return self._var_dict(keys, lb, ub, float)

    def add_constraint(self, cons):
        self.__cons.append(cons)
        return self.__cons[-1]

    def del_constraint(self, cons_or_idx):
        idx = cons_or_idx
        if isinstance(idx, cons):
            idx = self.__cons.index(idx)
        del self.__cons[idx]

    def del_constraints(self, conss_or_idxs):
        for c in conss_or_idxs:
            self.del_constraint(c)

    def add_constraints(self, conss):
        return [self.add_constraint(c) for c in conss]

    def set_objective(self, expr, sense=None):
        if sense is not None:
            self.sense = sense
        self.__obj = expr

    @property
    def variables(self):
        return self.__vars

    @property
    def constraints(self):
        return self.__cons

    @property
    def objective(self):
        return self.__obj

    def lp_solve(self, solver=None):
        """ Solve a relaxation of the problem using the specific solver. """
        if solver is None:
            solver = solvers.get_default_solver()
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
