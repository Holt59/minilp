# -*- encoding: utf-8 -*-

import enum
import numpy as np
import typing

from .modeler import modeler

import minilp.problems


class oper(enum.Enum):

    """ Enumeration class containing the valid comparison operators for
    linear expression (<=, ==, >=). """

    le = '<='
    eq = '=='
    ge = '>='


class expr:

    """ Class representing a linear expression, i.e., a weighted
    sum of variables.

    An expression is associated with a minilp problem and contains
    an array of coefficients for the variable in the problem. """

    def __init__(self,
                 arr: typing.Union['expr', np.ndarray, float],
                 pb: 'minilp.problems.problem'):
        """
        Args:
            arr: Array containing, for each variable in the corresponding
                 problem, the coefficient of the variable in this expression.
            pb: Problem associated with this expression.
        """
        # Case 1:
        if isinstance(arr, expr):
            self._u = arr._u
            self._pb = pb
        elif isinstance(arr, np.ndarray):
            self._u = arr
            self._pb = pb
        else:
            self._u = np.array([arr])
            self._pb = pb

    def __pos__(self) -> 'expr':
        """ Returns: This expression. """
        return expr(self._u, self._pb)

    def __neg__(self) -> 'expr':
        """ Returns: The negation of this expression. """
        return expr(-self._u, self._pb)

    def __add__(self, other: typing.Union['expr', float]) -> 'expr':
        """ Create a new expression by adding the given value or expression to
        this expression.

        Args:
            other: Expression or value to add to this expression.

        Returns:
            A new expression corresponding to the addition of this expresion
            with the given value.
        """
        if not isinstance(other, expr):
            other = expr(other, self._pb)

        if self._pb != other._pb:
            raise ValueError('Cannot add expression from different problems.')
        ms = max(len(self._u), len(other._u))
        lhs = np.concatenate((self._u, np.zeros(max(0, ms - len(self._u)))))
        rhs = np.concatenate((other._u, np.zeros(max(0, ms - len(other._u)))))
        return expr(lhs + rhs, self._pb)

    def __sub__(self, other: typing.Union['expr', float]) -> 'expr':
        """ Create a new expression by substracting the given value or expression
        from this expression.

        Args:
            other: Expression or value to substract from this expression.

        Returns:
            A new expression corresponding to the substraction of this the given
            value from this expression.
        """
        return self + (-other)

    def __mul__(self, other: float) -> 'expr':
        """ Multiply this expression by the given value.

        Args:
            other: Value to multiply this expression.

        Returns:
            A new expression corresponding to the multiplication of this
            expression with the given value.
        """
        if isinstance(other, expr):
            raise ValueError('Cannot multiply expression.')
        other = expr(other, self._pb)
        return expr(self._u * other._u[0], self._pb)

    def __radd__(self, other: typing.Union['expr', float]) -> 'expr':
        """ Create a new expression by adding the given value or expression to
        this expression.

        Args:
            other: Expression or value to add to this expression.

        Returns:
            A new expression corresponding to the addition of this expresion
            with the given value.
        """
        return self + other

    def __rsub__(self, other: typing.Union['expr', float]) -> 'expr':
        """ Create a new expression by substracting this expression to the given value or
        expression.

        Args:
            other: Expression or value from which this expression should be substracted.

        Returns:
            A new expression corresponding to the substraction of this expresion
            from the given value.
        """
        return -self + other

    def __rmul__(self, other) -> 'expr':
        """ Multiply this expression by the given value.

        Args:
            other: Value to multiply this expression.

        Returns:
            A new expression corresponding to the multiplication of this
            expression with the given value.
        """
        return self * other

    def __eq__(self, other: typing.Union['expr', float]) -> 'cons':
        """ Create a new equality constraint between this expression and
        the given value or expression.

        Args:
            other: Value or expression to which this expression should be equal.

        Returns:
            A new equality constraint between this expresion and the given one.
        """
        return cons(self, oper.eq, expr(other, self._pb))

    def __ge__(self, other: typing.Union['expr', float]) -> 'cons':
        """ Create a new greater-or-equal constraint between this expression and
        the given value or expression.

        Args:
            other: Value or expression that should be lower that this expression.

        Returns:
            A new greater-or-equal constraint between this expresion and the given one.
        """
        return cons(self, oper.ge, expr(other, self._pb))

    def __le__(self, other: typing.Union['expr', float]) -> 'cons':
        """ Create a new lower-or-equal constraint between this expression and
        the given value or expression.

        Args:
            other: Value or expression that should be greater that this expression.

        Returns:
            A new lower-or-equal constraint between this expresion and the given one.
        """
        return cons(self, oper.le, expr(other, self._pb))

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

    """ A variable is a simple linear expression with a coefficient of 1. """

    def __init__(self,
                 pb: 'minilp.problems.problem',
                 idx: int,
                 lb: float = 0,
                 ub: float = modeler.inf,
                 cat: type = int,
                 name: str = ''):
        """
        Args:
            pb: Problem containing this variable.
            idx: Index of the variable in the problem.
            lb: Lower bound of the variable (or -inf).
            ub: Upper bound of the variable (or +inf).
            cat: Category of the variable (int or float).
            name: Name of the variable.
        """
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
        """ Category of the variable (int or float). """
        return self.__cat

    def __repr__(self):
        return self.name

    def __str__(self):
        return repr(self)


class cons:

    """ Class representing a linear constraint. """

    # Representation of the operator:
    _op_repr = {
        oper.le: '<=',
        oper.eq: '=='
    }

    def __init__(self,
                 lhs: expr,
                 cmp: oper,
                 rhs: expr):
        """
        Args:
          - lhs: Left-hand side expression of the constraint.
          - cmp: Comparison operator of the constraint.
          - rhs: Right-hand side expression of the constraint.
        """

        if lhs._pb != rhs._pb:
            raise ValueError('Cannot create constraints using expressions '
                             'from different problems.')

        self._pb = lhs._pb

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
        self._e = expr(ul, lhs._pb)
        self._c = cmp
        self._r = b

    @property
    def lhs(self) -> expr:
        """ Left hand side of the constraint. """
        return self._e

    @property
    def rhs(self) -> float:
        """ Right hand side of the constraint. """
        return self._r

    @property
    def oper(self) -> oper:
        """ Comparison operator of the constraint (either oper.ge or oper.eq). """
        return self._c

    def __repr__(self):
        return '{} {} {:g}'.format(
            self._e, cons._op_repr[self._c], self._r)

    def __str__(self):
        return repr(self)
