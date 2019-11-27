# -*- encoding: utf-8 -*-

import numpy as np


class modeler:

    inf = np.inf
    nan = np.nan
    
    @staticmethod
    def isnan(value):
        """ Return True if the given value is nan, False otherwize. """
        return np.isnan(value)
    
    @staticmethod
    def sum(iterable, start=0):
        """ Return the sum of a 'start' value (default: 0) plus an iterable of numbers. """
        return sum(iterable, start)
        
    @staticmethod
    def dot(lhs, rhs):
        """ Return the dot product of the two given iterables. """
        return sum(l * r for l, r in zip(lhs, rhs))