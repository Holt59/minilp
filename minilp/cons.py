# -*- encoding: utf-8 -*-

import operator as oper

import minilp.expr


class cons:

    """ Class representing a linear constraint. """

    op_repr = {
        oper.le: '<=',
        oper.eq: '=='
    }

    def __init__(self, lhs, cmp, rhs):
        """ Create a new constraints from the given parameters.

        Parameters:
          - lhs minilp.expr representing the left-hand side of the constraint.
          - cmp comparison operator (operator.eq, operator.le, operator.ge).
          - rhs right-hand side (int, float).
        """
        u = lhs._u
        rhs -= u[0]
        u[0] = 0
        if cmp == oper.ge:
            u, cmp, rhs = -u, oper.le, -rhs
        self._e = minilp.expr.expr(u, lhs._pb)
        self._c = cmp
        self._r = rhs

    def __repr__(self):
        return '{} {} {:g}'.format(
            self._e, cons.op_repr[self._c], self._r)

    def __str__(self):
        return repr(self)
