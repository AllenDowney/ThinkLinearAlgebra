---
jupyter:
  jupytext:
    split_at_heading: true
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

*Think Linear Algebra* is not for sale yet, but if you would like to support this project, you can [contribute through Green Tea Press](https://greenteapress.com/support/?book=tla).

# Null Space

In the previous chapter we saw examples where a system of linear equations has a single solutions.
In this chapter, we'll consider an example where there are many possible solution.
One domain where systems like this are common is conservation of atoms (and sometimes charge) in chemical reactions -- that is, stoichiometry.
If you studied chemistry some time ago, or never, I'll explain what you need to know.

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkLinearAlgebra/blob/main/chapters/nullspace.ipynb).
<!-- #endregion -->

```python tags=["remove-cell"]
# temporary hack to make autoreload work on Colab
import importlib, sys
sys.modules["imp"] = importlib

%load_ext autoreload
%autoreload 2
```

```python tags=["remove-print", "hide-cell"]
from os.path import basename, exists


def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve

        local, _ = urlretrieve(url, filename)
        print("Downloaded " + local)


download("https://github.com/AllenDowney/ThinkLinearAlgebra/raw/main/utils.py")
```

```python tags=["hide-cell", "remove-print"]
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

from utils import decorate
```

```python tags=["remove-cell"]
from utils import set_precision

set_precision(3)
```

## Stoichiometry

In chemistry, a reaction equation specifies a set of chemicals that can react to form another set of chemicals.
In this context, the chemicals are called reagents.

As an example, we'll write the reaction equation for combustion of ethanol -- that is, the reaction of ethanol with molecular oxygen to form carbon dioxide and water.
We'll define SymPy symbols to represent each of these reagents.

```python
import sympy as sp

C2H5OH = sp.Symbol('C_{2} H_5 OH')
O2 = sp.Symbol('O2')
CO2 = sp.Symbol('CO2')
H2O = sp.Symbol('H_{2} O')
```

Now we can define the left hand side and the right hand side of the equation.

```python
lhs = C2H5OH + O2
rhs = CO2 + H2O
```

And we can display it like this.

```python
from IPython.display import display, Math

display(Math(f"{sp.latex(lhs)} \\rightarrow {sp.latex(rhs)}"))
```

In this form the equation is not balanced, because it does not have the same number of atoms, of each kind, on both sides.
For example, there are two carbon atoms on the left and only one on the right.
In chemical reactions (as opposed to nuclear reactions) atoms are conserved.

To balance the equation, we have to find a set of proportions in which these species of chemicals can be combined while conserving atoms.
To do that, we'll introduce unknown coefficients to the equation.

```python
x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
```

We'll rewrite the equations with each species multiplied by a coefficient.

```python
lhs = x1 * C2H5OH + x2 * O2
rhs = x3 * CO2 + x4 * H2O
```

And collect the coefficients and species into lists.

```python
coefficients = [x1, x2, x3, x4]
species = [C2H5OH, O2, CO2, H2O]
```

Now we can use the following function to display the reaction equation.

```python
def display_reaction(coefficients, species, i=2):
    terms = [c * s for c, s in zip(coefficients, species)]
    lhs, rhs = sum(terms[:i]), sum(terms[i:])
    display(Math(f"{sp.latex(lhs)} \\rightarrow {sp.latex(rhs)}"))
```

Here's what the combustion of ethanol looks like with stoichiometric coefficients.

```python
display_reaction(coefficients, species)
```

Now let's solve for the unknown coefficients.
We can do that by writing an equation to represent conservation of each type of atom involved in the reaction.

For example, there are $2 x_1$ carbon atoms on the left and $x3$ on the right, so we can write:

```python
eq_carbon = sp.Eq(2*x1, x3)         
eq_carbon
```

Similarly, there are six hydrogen atoms in each molecule of ethanol and two in each molecule of $H_2$, so we can write:

```python
eq_hydrogen = sp.Eq(6*x1, 2*x4)     
eq_hydrogen
```

Finally, we can write this equation to represent conservation of oxygen atoms.

```python
eq_oxygen = sp.Eq(x1 + 2*x2, 2*x3 + x4)  
eq_oxygen
```

Together, these form a system of linear equations, which we will solve simultaneously.
As we did in the previous chapter, we can write the system as a matrix equation.
We'll use the SymPy function `linear_eq_to_matrix` to assemble the equations into a matrix.

```python
A, b = sp.linear_eq_to_matrix([eq_carbon, eq_hydrogen, eq_oxygen], coefficients)
A
```

The results represent the matrix equation $A x = b$, where $x$ is the vector of coefficients.

```python
x = sp.Matrix(coefficients)
x
```

If we multiply the left hand side, we can confirm that we recover the system of equations.

```python
eqn = sp.Eq(A * x, b)
eqn
```

Because there are no constants in the equations, $b$ is all zeros -- which means that this is a homogeneous equation.
We can use `solve` to solve this system.

```python
solution = sp.solve(eqn, x)
solution
```

The result is a dictionary that represents a system of equations that express $x_1$, $x_2$, and $x_3$ in terms of $x_4$.

```python
equations = [sp.Eq(var, expr) for var, expr in solution.items()]
for eq in equations:
    display(Math(sp.latex(eq)))
```

This is not the most convenient way to represent the solution.
An alternative is to compute the null space of the system, which represents the solution as a list of vectors.

```python
nullspace = A.nullspace()
nullspace
```

This vector represents a solution to the system of equations.

```python
basis_vector = nullspace[0]
basis_vector
```

If we multiply the solution by $A$, we can confirm that the result is $b$.

```python
A * basis_vector
```

And we can use it to write a balanced version of reaction equation.

```python
display_reaction(basis_vector, species)
```

But it is not standard to write reaction equations with fractional coefficients.
So we'll use the following function to find the least common multiplier of the fractional coefficients and multiply through.

```python
def scale_vector(vector):
    denoms = [sp.denom(x) for x in vector]
    lcm = sp.lcm(denoms)
    return vector * lcm
```

Here's the scaled version of the solution.

```python
solution = scale_vector(basis_vector)
solution
```

Any multiple of the solution vector is also a solution.

```python
A * solution
```

Finally here's the balanced reaction equation in standard form.

```python
display_reaction(solution, species)
```

Because every molecule contains an integer number of each atom type, we know that the coefficients can be expressed as integers.
So using SymPy to compute the solution symbolically is a good option.
But we can also find null spaces numerically.

## Finding a Null Space Numerically

To find the null space of $A$ numerically, we'll convert it to a NumPy array.

```python
A_numpy = np.array(A).astype(float)
A_numpy
```

And use the `null_space` function from SciPy.

```python
from scipy.linalg import null_space

ns = null_space(A_numpy)
ns
```

The result is an array where each column is a vector in the null space.
In this example there is only one, so we can select it like this.

```python
vector = ns.T[0]
vector
```

If we multiply by $A$, we can confirm that the result is all zeros, within floating-point error.

```python
A_numpy @ vector
```

We might not recognize this vector as the solution we got from SymPy.
That's because it is normalized so its magnitude is 1, within floating-point error.

```python
np.linalg.norm(vector)
```

If we scale it so the smallest element is 1, its easier to recognize.

```python
denom = np.min(np.abs(vector))
scaled = vector / denom
scaled
```

And again, any multiple of this vector is also a solution.

```python
A_numpy @ scaled
```

## Null Spaces With More Dimensions

Let's consider a more complex example, the oxidation of hydrogen peroxide ($H_2O_2$) by potassium permanganate ($K \mathit{Mn} O_4$).
This reaction is a classic demonstration in chemistry classes because the color of the solution changes dramatically.
It is also used in oxidative treatment of wastewater.

Here are the species we'll need to write the reaction equation.

```python
KMnO4 = sp.Symbol('K \mathit{Mn} O_4')
H2O2 = sp.Symbol('H_{2}O_{2}')
H_plus = sp.Symbol('H^{+}')
Mn2_plus = sp.Symbol('\mathit{Mn}^{2+}')
K_plus = sp.Symbol('K^{+}')

species2 = [KMnO4, H2O2, H_plus, Mn2_plus, O2, H2O, K_plus]
```

Since we have seven species, we'll need seven coefficients.

```python
a, b, c, d, e, f, g = sp.symbols('a b c d e f g')
coefficients2 = [a, b, c, d, e, f, g]
```

Here is the version of the reaction equation with unknown coefficients.

```python
display_reaction(coefficients2, species2, 3)
```

To find the coefficients, we can write four equations, one for each type of atom involved in the reaction.

```python
eq_mn = sp.Eq(a, d)
eq_k = sp.Eq(a, g)
eq_h = sp.Eq(2*b + c, 2*f)
eq_o = sp.Eq(4*a + 2*b, 2*e + f)
```

Notice that three of the reagents are charged ions: $H^{+}$ and $K^{+}$ and $\mathit{Mn}^{2+}$.
So we can write an additional equation that requires charge to be conserved.

```python
eq_charge = sp.Eq(c, 2*d + g)
```

```python
eqns = [eq_mn, eq_k, eq_h, eq_o, eq_charge]
for eqn in eqns:
    display(Math(sp.latex(eqn)))
```

Now we can express this system as a matrix equation.

```python
A, b = sp.linear_eq_to_matrix(eqns, coefficients2)
A
```

Because we have seven coefficients and only five equations, the system is underconstrained.
In fact, it is even less constrained than the previous example: the null space contains two vectors.

```python
nullspace = A.nullspace()
len(nullspace)
```

We'll scale the vectors so the coefficients are integers.

```python
solutions = [scale_vector(basis_vector) for basis_vector in nullspace]
solutions
```

Here is what the first solution looks like as a balanced equation.

```python
display_reaction(solutions[0], species2, 3)
```

This equation represent the decomposition of hydrogen peroxide (H2O2) into water (H2O) and oxygen gas (O2).
This reaction happens slowly at room temperature, and more quickly in the presence of catalase, an enzyme found in living cells, which is the cause of the foaming you might see if you apply hydrogen peroxide to a wound.

Here's the second solution.

```python
display_reaction(solutions[1], species2, 3)
```

Notice that it contains a negative coefficient, which is not meaningful -- there can't be a negative amount of a reagent.
So this solution is mathematically valid in the sense that it solves the equations that conserve atoms and charge, but by itself it does not represent a physically possible reaction.

However, any linear combination of these solutions is also a solution, and some of those combinations are physically possible.
For example, if we combine the solutions in the ratio 4:1, we get the following reaction.

```python
display_reaction(4 * solutions[0] + solutions[1], species2, 3)
```

This equation is balanced in atoms and charge, and it is theoretically possible because all of the coefficients are positive.
In fact this is the reaction that occurs spontaneously when these reagents are combined.

However, if we combine the solutions in a ratio of 2:1, the result is also balanced and theoretically possible.

```python
display_reaction(2 * solutions[0] + solutions[1], species2, 3)
```

But under normal conditions, this reaction does not happen at non-negligible levels, I presume because it is not energetically favored.
And there are many more combinations that are theoretically possible, but also unlikely.

So this is an example where balance of atoms and charge can tell us which reactions are theoretically possible, but stoichiometry alone is not enough to tell us which reactions are thermodynamically likely.

## Discussion

This chapter presents underdetermined systems, where the number of unknowns is greater than the number of equations.
These systems often have many solutions.

The set of solutions forms a **null space**, which is defined by a set of basis vectors.
Every linear combination of these vectors is a solution to the system.

In the ethanol example, the null space had one basis vector, so the solution space was one-dimensional.
In the potassium permanganate example, there were two basis vectors, forming a two-dimensional space of solutions.

In general, there is a relationship between the rank of a matrix and the number of dimensions in its null space, given by the **rank-nullity theorem**:

$$
\text{rank}(A) + \text{nullity}(A) = n
$$

where $n$ is the number of columns in $A$.
The rank tells us how many independent constraints the equations impose.
The nullity tells us how many degrees of freedom remain.

In the next chapter we consider overdetermined systems with no solutions -- so we will try to find the best of the non-solutions.

### Exercise

Balance the equation for the combustion of methanol.

```python
CH3OH = sp.Symbol('C H_{3} OH')
```

```python
lhs = x1 * CH3OH + x2 * O2
rhs = x3 * CO2 + x4 * H2O
```

```python
display_reaction([x1, x2, x3, x4], species)
```

```python
eq_carbon = sp.Eq(x1, x3)         
eq_carbon
```

Hydrogen: 6 from ethanol = 2 per H2O

```python
eq_hydrogen = sp.Eq(4*x1, 2*x4)     
eq_hydrogen
```

Oxygen: 1 in ethanol + 2 per O2 = 2 per CO2 + 1 per H2O

```python
eq_oxygen = sp.Eq(x1 + 2*x2, 2*x3 + x4)  
eq_oxygen
```

```python
A, b = sp.linear_eq_to_matrix([eq_carbon, eq_hydrogen, eq_oxygen], (x1, x2, x3, x4))
A
```

```python
b
```

```python
nullspace = A.nullspace()
len(nullspace)
```

```python
basis_vector = nullspace[0]
basis_vector
```

```python
solution = scale_vector(basis_vector)
solution
```

```python
display_reaction(solution, species)
```

<!-- #region tags=["remove-print"] -->
[Think Linear Algebra](https://allendowney.github.io/ThinkLinearAlgebra/index.html)

Copyright 2025 [Allen B. Downey](https://allendowney.com)

Code license: [MIT License](https://mit-license.org/)

Text license: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
<!-- #endregion -->

```python

```
