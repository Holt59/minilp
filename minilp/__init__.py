# -* - encoding: utf-8 -*-

from minilp.expr import cons, expr, var   # noqa
from minilp.problem import problem        # noqa

import minilp.solvers as solvers          # noqa


def dot(lhs, rhs):
    """ Return the dot product of the two given iterables. """
    return sum(l * r for l, r in zip(lhs, rhs))
