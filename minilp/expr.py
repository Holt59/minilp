# -*- encoding: utf-8 -*-

import numpy as np
import operator as oper


class expr:

    def __init__(self, arr, pb=None):
        # Case 1:
        if isinstance(arr, expr):
            self._u = arr._u
            self._pb = pb
        elif isinstance(arr, np.ndarray) and pb is not None:
            self._u = arr
            self._pb = pb
        else:
            self._u = np.array([arr])
            self._pb = None

    def __pos__(self):
        return expr(self._u, self._pb)

    def __neg__(self):
        return expr(-self._u, self._pb)

    def __add__(self, other):
        other = expr(other)
        ms = max(len(self._u), len(other._u))
        lhs = np.concatenate((self._u, np.zeros(max(0, ms - len(self._u)))))
        rhs = np.concatenate((other._u, np.zeros(max(0, ms - len(other._u)))))
        return expr(lhs + rhs, self._pb)

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = expr(other)
        if (other._u[1:] != 0).any():
            raise ValueError('Cannot multiply expression.')
        return expr(self._u * other._u[0], self._pb)

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return -self + other

    def __rmul__(self, other):
        return self * other

    def __eq__(self, other):
        return cons(self, oper.eq, expr(other), self._pb)

    def __ge__(self, other):
        return cons(self, oper.ge, expr(other), self._pb)

    def __le__(self, other):
        return cons(self, oper.le, expr(other), self._pb)

    def __repr__(self):
        s = ''
        for c, v in zip(self._u[1:], self._pb.variables):
            fmt = None
            if c == 1:
                fmt = ' + {}{}'
                c = ''
            elif c == -1:
                fmt = ' - {}{}'
                c = ''
            elif c < 0:
                c = abs(c)
                fmt = ' - {:g} * {}'
            elif c > 0:
                fmt = ' + {:g} * {}'
            if fmt is not None:
                s += fmt.format(c, v)
        if self._u[0] != 0:
            if self._u[0] > 0:
                s += ' + '
            else:
                s += ' - '
            s += '{:g}'.format(abs(self._u[0]))
        s = s.strip()
        if s:
            if s[0] == '+':
                s = s[2:]
            elif s[0] == '-':
                s = '-' + s[2:]
        return s

    def __str__(self):
        return repr(self)


class var(expr):

    def __init__(self, pb, idx, lb=0, ub=np.inf, cat=int, name=""):
        self._u = np.zeros(idx + 1)
        self._pb = pb
        self._idx = idx
        self._u[idx] = 1
        self.lb = lb
        self.ub = ub
        self.name = name
        self.__cat = cat

    @property
    def category(self):
        """ Category of the variable (either int or float). """
        return self.__cat

    def __repr__(self):
        return self.name

    def __str__(self):
        return repr(self)


class cons:

    """ Class representing a linear constraint. """

    op_repr = {
        oper.le: '<=',
        oper.eq: '=='
    }

    def __init__(self, lhs, cmp, rhs, pb):
        """ Create a new constraints from the given parameters.

        Parameters:
          - lhs minilp.expr representing the left-hand side of the constraint.
          - cmp comparison operator (operator.eq, operator.le, operator.ge).
          - rhs right-hand side (int, float).
        """

        if rhs._pb is None:
            rhs._pb = lhs._pb

        if cmp == oper.ge:
            lhs, cmp, rhs = -lhs, oper.le, -rhs

        ul = lhs._u.copy()
        ur = rhs._u.copy()

        if len(ul) > len(ur):
            ur = np.concatenate((ur, np.zeros(len(ul) - len(ur))))
        if len(ul) < len(ur):
            ul = np.concatenate((ul, np.zeros(len(ur) - len(ul))))

        # compute constant
        b = ur[0] - ul[0]

        # move variables to lhs
        ul[0] = 0  # no constant
        ul[1:] -= ur[1:]
        self._e = expr(ul, pb)
        self._c = cmp
        self._r = b

    @property
    def lhs(self):
        """ Left hand side of the constraint, which is a minilp.expr.expr. """
        return self._e

    @property
    def rhs(self):
        """ Right hand side of the constraint, which is a numeric value. """
        return self._r

    @property
    def oper(self):
        """ Comparison operator of the constraint (either operator.ge or
        operator.eq). """
        return self._c

    def __repr__(self):
        return '{} {} {:g}'.format(
            self._e, cons.op_repr[self._c], self._r)

    def __str__(self):
        return repr(self)
