from __future__ import annotations

import math
from typing import Iterable, overload

from minilp.exprs import expr


class modeler:
    inf = float("inf")
    nan = float("nan")

    @staticmethod
    def isnan(value: float) -> bool:
        """
        Check if the given value is a NaN (Not a Number).

        Args:
            value: The value to check.

        Returns:
            True if the given value is nan, False otherwize.
        """
        return math.isnan(value)

    @staticmethod
    def sum(iterable: Iterable[expr | float], start: float = 0) -> expr:
        """
        Sum the values in the given iterable.

        Args:
            iterable: Iterable of values to sum.
            start: Starting value.

        Returns:
            The sum of start plus all the values in iterable.
        """
        return sum(iterable, start)  # type: ignore

    @overload
    @staticmethod
    def dot(lhs: Iterable[float], rhs: Iterable[expr]) -> expr:
        ...

    @overload
    @staticmethod
    def dot(lhs: Iterable[expr], rhs: Iterable[float]) -> expr:
        ...

    @staticmethod
    def dot(
        lhs: Iterable[float] | Iterable[expr], rhs: Iterable[float] | Iterable[expr]
    ) -> expr:
        """
        Compute the dot product of two iterables.

        Args:
            lhs: The left iterable for the dot product.
            rhs: The right iterable for the dot product.

        Returns:
            The dot product of the two given iterables.
        """
        return sum(ls * rs for ls, rs in zip(lhs, rhs))  # type: ignore
