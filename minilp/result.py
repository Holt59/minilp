# -*- encoding: utf-8 -*-

import numpy as np


class result:

    def __init__(self, success=False, status='unknown',
                 objective=np.nan, variables=None):
        self.__success = success
        self.__status = status
        self.__objective = objective
        self.__vs = variables

    def get_value(self, var):
        """ Retrieve the value associated to the given variable.

        Parameters:
          - var A minilp.expr.var object.

        Return: Value associated with the given variable. """
        value = self.__vs[var._idx - 1]
        return value

    def get_values(self, *args):
        """ Return values associated to the given variables.

        Parameters:
          - vs Iterable of minilp.expr.var.

        Return: List of value associated with the variables. """
        if len(args) == 1:
            args = args[0]
        return [self.get_value(v) for v in args]

    @property
    def success(self):
        """ True if this result contains a solution, false otherwize. """
        return self.__success

    @property
    def status(self):
        """ Status of this result. """
        return self.__status

    @property
    def objective(self):
        """ Objective value of this result or np.nan. """
        return self.__objective

    def __repr__(self):
        return 'status = {}, obj. = {}'.format(self.status, self.objective)

    def __bool__(self):
        return self.success
