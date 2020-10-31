# -* - encoding: utf-8 -*-

__version__ = "0.0.1"

import minilp.problems
import minilp.results
from minilp.expr import expr, var, cons  # noqa: F401

problem = minilp.problems.problem
result = minilp.results.result
status = minilp.results.solve_status
