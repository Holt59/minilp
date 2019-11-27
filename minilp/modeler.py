# -*- encoding: utf-8 -*-

import numpy as np
import typing


class modeler:

    inf = np.inf
    nan = np.nan

    @staticmethod
    def isnan(value: float) -> bool:
        """ Check if the given value is a NaN (Not a Number).

        Args:
            value: The value to check.

        Returns:
            True if the given value is nan, False otherwize.
        """
        return np.isnan(value)

    @staticmethod
    def sum(iterable: typing.Iterable[typing.SupportsFloat],
            start: float = 0) -> float:
        """ Sum the values in the given iterable.

        Args:
            iterable: Iterable of values to sum.
            start: Starting value.

        Returns:
            The sum of start plus all the values in iterable.
        """
        return sum(iterable, start)

    @staticmethod
    def dot(lhs: typing.Iterable[typing.SupportsFloat],
            rhs: typing.Iterable[typing.SupportsFloat]) -> float:
        """ Compute the dot product of two iterables.

        Args:
            lhs: The left iterable for the dot product.
            rhs: The right iterable for the dot product.

        Returns:
            The dot product of the two given iterables.
        """
        return sum(l * r for l, r in zip(lhs, rhs))
