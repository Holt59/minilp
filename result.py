# -*- encoding: utf-8 -*-

import numpy as np


class result:

    def __init__(self, success=False, status='unknown',
                 objective=np.nan, variables=None):
        self.success = success
        self.status = status
        self.objective = objective
        self.__vs = variables

    def get_value(self, var):
        """ Retrieve the value associated to the given variable.

        Parameters:
          - var A minilp.var object.

        Return: Value associated with the given variable. """
        value = self.__vs[var._idx - 1]
        return value

    def get_values(self, vs):
        """ Return values associated to the given variables.

        Parameters:
          - vs Iterable of minilp.var.

        Return: List of value associated with the variables. """
        return [self.get_value(v) for v in vs]

    def __repr__(self):
        return 'status = {}, obj. = {}'.format(self.status, self.objective)

    def __bool__(self):
        return self.success
