# pyright: reportPrivateUsage=false

from __future__ import annotations

import enum
from typing import Iterable

from . import exprs
from .modeler import modeler


class solve_status(enum.Enum):

    """
    Enumeration class representing the status of a problem
    solution.
    """

    UNKNOWN = enum.auto()
    FEASIBLE = enum.auto()
    OPTIMAL = enum.auto()
    INFEASIBLE = enum.auto()
    UNBOUNDED = enum.auto()

    def __str__(self):
        return self.name


class result:

    """
    Class representing the solution of a minilp problem.
    """

    _success: bool
    _status: solve_status
    _objective: float
    _values: list[float] | None

    def __init__(
        self,
        success: bool = False,
        status: solve_status = solve_status.UNKNOWN,
        objective: float = modeler.nan,
        variables: Iterable[float] | None = None,
    ):
        """
        Args:
            success: True if the solution is valid, False otherrwize.
            status: Status of the solution.
            objective: Objective value of the solution.
            variables: Values of the variables in this solution.
        """
        self._success = success
        self._status = status
        self._objective = objective
        if variables is None:
            self._values = None
        else:
            self._values = list(variables)

    def get_value(self, variable: exprs.var) -> float:
        """Retrieve the value associated to the given variable.

        Args:
            variable: The variable to retrieve the value for.

        Returns:
            The value associated with the given variable in this solution.
        """
        if self._values is None:
            raise ValueError(
                "No value associated to variable {} in  this solution".format(variable)
            )
        value = self._values[variable._idx - 1]
        return value

    def get_values(self, variables: Iterable[exprs.var]) -> list[float]:
        """Retrieve thes value associated to the given variables.

        Args:
            variables: The variables to retrieve the value for.

        Returns:
            A list containing the value associated with the given variables
            in this solution.
        """
        return [self.get_value(v) for v in variables]

    @property
    def success(self) -> bool:
        """True if this result contains a solution, false otherwize."""
        return self._success

    @property
    def status(self) -> solve_status:
        """Status of this result."""
        return self._status

    @property
    def objective(self) -> float:
        """Objective value of this result or nan."""
        return self._objective

    def __repr__(self):
        return "status = {}, obj. = {}".format(self.status, self.objective)

    def __bool__(self) -> bool:
        """True if this result contains a solution, false otherwize."""
        return self.success
