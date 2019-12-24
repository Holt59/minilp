# minilp

`minilp` is a light python package to model
[(Mixed-)Integer Linear Program (MILP)](https://en.wikipedia.org/wiki/Integer_programming)
and solve their linear relaxations.

## Examples

### Linear Programs

You can easily create linear program and solve them using one of the available softwares:

```python
import minilp

lp = minilp.problem("My first LP problem")

# Create two continuous variables within [0, 4]:
x1, x2 = lp.continuous_var_list(2, 0, 4)

# Add constraints:
lp.add_constraint(-3 * x1 + 4 * x2 <= 7)
lp.add_constraint(2 * x2 <= 5)
lp.add_constraint(6 * x1 + 4 * x2 <= 25)
lp.add_constraint(2 * x1 - x2 <= 6)

# Set the objective function:
lp.set_objective("max", x1 + 2 * x2)

# Solve the problem:
res = lp.lp_solve()
print(res)
print('x1 = {:.4f}, x2 = {:.4f}'.format(res.get_value(x1), res.get_value(x2)))
```

Will output:

```
status = optimal, obj. = 7.5
x1 = 2.5000, x2 = 2.5000
```


### Integer Linear Programs

You can model Integer Linear Programs using `minilp`, but the `lp_solve` method will always
solve the linear relaxation.

```python
import minilp

N = 5
p = [1, 4, 5, 3, 5]  # profits
w = [3, 4, 3, 5, 9]  # weights
K = 10  # capacity

assert N == len(w) and N == len(p)

# A simple knapsack:
kp = minilp.problem("Simple knapsack")

# Create N binary variables:
x = kp.binary_var_list(N)

# Add the capacity constraint:
kp.add_constraint(kp.dot(x, w) <= K)

# Set the objective:
kp.maximize(kp.dot(x, p))

# Solve the linear relaxation:
res = kp.lp_solve()
print(res)
print(res.get_values(x))
```

The output is:

```
status = optimal, obj. = 10.8
[0, 1.0, 1.0, 0.6000000000000001, 0]
```

...and, as you can see, the solution contains a non-integer value.
