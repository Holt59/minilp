# -* - encoding: utf-8 -*-

__version__ = "0.0.1"

import minilp.exprs
import minilp.problems
import minilp.results
import minilp.solvers

cons = minilp.exprs.cons
expr = minilp.exprs.expr
problem = minilp.problems.problem
result = minilp.results.result
solver = minilp.solvers.solver
status = minilp.results.solve_status
var = minilp.exprs.var
