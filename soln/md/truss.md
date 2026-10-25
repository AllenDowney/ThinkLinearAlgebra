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

# Truss in the System

In this chapter, we'll use systems of linear equations to analyze trusses.

A **truss** is an assembly of bars connected by joints, usually designed to support a load. Many bridges and roofs are supported by trusses, and if you've ever participated in a model bridge competition, the structure you built was probably a truss.

This picture shows a **Warren truss**, the basis of many railroad bridge designs:

<img width="400" src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/VIEW_OF_BRIDGE_FROM_SOUTH_WITH_ONCOMING_CSX_LOCOMOTIVE_ON_TRACKS_BELOW%2C_LOOKING_NORTHEAST_-_Wills_Creek_Bollman_Bridge%2C_Spanning_CSX_tracks_at_T381%2C_Meyersdale%2C_Somerset_County%2C_HAER_PA%2C56-MEYER.V%2C2-2.tif/lossy-page1-800px-thumbnail.tif.jpg">

This structure is made of three connected triangles with horizontal bars across the top.
It's supported by anchor points at each end, and it carries a deck that supports a roadway.
The load on the truss includes the weight of the roadway, any vehicles on it, and the weight of the truss itself.


To model a structure like this, we'll make some simplifying assumptions:

- We'll treat the structure as two-dimensional, analyzing only one side of the truss.

- We'll assume the joints behave like hinge pins that connect bars at their ends but allow them to rotate freely.

- We'll assume the load is applied only at the joints, not along the length of the bars.

- We'll treat each bar as a spring that pushes or pulls in response to stretching or compression, with force that depends on its size and material.

- We'll assume that the stretching and compression are small enough that the geometry of the truss is unchanged.

These assumptions aren't fully realistic, but they're reasonable approximations.
And they make it easier to analyze the structure -- that is, for a given load, we can compute how much each bar stretches or compresses, and how much force it exerts in response.

The goal is to find an **equilibrium**, where the external forces applied to the truss are balanced by internal spring forces in the bars.

To do this analysis, we'll use **projection matrices**, which we saw in Section XXX, and we'll introduce **block matrices**, which are matrices that contain vectors or submatrices as elements.

<!-- #region tags=["remove-print"] -->
[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkLinearAlgebra/blob/main/chapters/truss.ipynb).
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

```python tags=["remove-print", "hide-cell"]
# Install packages that Colab doesn't have
try:
    import pint
except ImportError:
    %pip install pint
```

```python tags=["hide-cell", "remove-print"]
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sympy as sp

from utils import decorate, gset, gget
```

```python tags=["remove-cell"]
from utils import set_precision

set_precision(3)
```

## The Simplest Truss

We'll start with a truss that contains two bars, connected at a single joint and supported at two points.

To represent the locations of the joint and the support points, we'll use vectors we'll call `A`, `B`, and `C`.
And to represent the vectors, we'll use a SymPy `Matrix`.
The following function takes two coordinates and returns a symbolic vector.

```python
def sp_vector(x, y):
    return sp.Matrix([x, y])
```

Here are the positions of the nodes, using the symbol `L` to represent the length of the bars.

```python
L = sp.symbols('L', positive=True)

pos_A = sp_vector(-L, -L)
pos_B = sp_vector(L, -L)   
pos_C = sp_vector(0, 0)
```

The following diagram shows the configuration of this truss.

```python
from utils import diagram_truss

subs = {L: 0.15}
nodes = pos_A, pos_B, pos_C
diagram_truss(nodes, subs, lim=0.2)
```

One bar connects `A` and `C` and the other connects `B` and `C`.
The triangles represent anchors holding the `A` and `B` in place.
The circle represents a pin joint that can move under load.

To start the analysis of this truss, we'll define symbols to represent the displacement of `C` from its starting position due to force applied to it.
These are the unknowns we will solve for.

```python
u_x, u_y = sp.symbols('u_x u_y')
```

We'll put these symbols in a column vector that represents the displacement of `C`.

```python
u_C = sp_vector(u_x, u_y)
u_C
```

Most materials, when they are compressed or stretched by a small amount, are well modeled as a linear spring, where the reaction force is proportional to the size of the displacement.
So we can write $f = k |u|$, where $f$ is the reaction force, $|u|$ is the size of the displacement, and $k$ is the **spring constant**.

For a bar under compression or tension, we can compute $k$ as a function of the size of the bar and its material properties.
Specifically:

$k = \frac{A E}{L}$ 

where $L$ is the length of the bar, $A$ is the cross-sectional area, and `E` is Young's modulus, a constant that quantifies the stiffness of a material -- that is, its ability to push back when compressed or stretched.

The following symbols represent `A` and `E`.
We'll fill in values for these parameters later.

```python
A, E = sp.symbols('A E', positive=True)
```

Now we can compute the spring constant of the two bars.
To make the analysis more interesting, let's suppose the bar `CB` is twice as thick as `CA`, so it has twice the cross-sectional area.

```python
k_CA = A * E / L
k_CB = 2 * A * E / L
```

Next we'll compute the spring forces in the truss due to a known displacement.

## Internal Force

Soon we will find the displacement of the truss under a given load.
But first let's suppose, temporarily, that we are given the displacement, `u`.

To compute the spring forces in the bars we have to figure out what part of the displacement, `u`, is along the axis of each bar.
First we'll compute vectors that represent the offset of A and B, relative to C.

```python
r_CA = pos_A - pos_C
r_CB = pos_B - pos_C
```

Next we'll compute the vector projection of `u` onto `r_CA` and `r_CB`, as shown in this diagram.

```python
u = sp_vector(2, -8) * 1e-6
scale = 2e4
diagram_truss(nodes, subs, u*scale, lim=0.2, add_pin=False, add_labels=False)
```

$u_{CA}$ is the component of the displacement in the direction of $r_{CA}$, and
$u_{CB}$ is the component of the displacement in the direction of $r_{CB}$.

To compute these vector projections, we'll use this function, which:

1. Computes the unit vector in the direction of `r`, called `r_hat`,

2. Computes `r_hat.T * u`, which is the scalar magnitude of `u` in the direction of `r_hat`, and

3. Multiplies the result by `r_hat`.

The result is a vector that represents the component of `u` in the direction of `r`. 

```python
def vector_projection(u, r):
    r_hat = r.normalized()
    scalar_projection = r_hat.T * u
    return r_hat * scalar_projection
```

Now, instead of a known displacement, `u`, suppose we have the unknown displacement `u_C`.
For any value of `u_C`, we can find the spring forces in the bars by computing the vector projections of the displacement onto the axis of the bars, and multiplying by the spring constants.

```python
f_CA = -k_CA * vector_projection(u_C, r_CA)
f_CB = -k_CB * vector_projection(u_C, r_CB)
```

Here's one of the results, `f_CA`, with the constants factored out.
We'll use `factor_matrix` throughout this chapter to make matrices easier to read by pulling out common factors.

```python
from utils import factor_matrix

factor = -A * E / L / 2
factor_matrix(f_CA, factor)
```

`f_CA` is a column vector that represents the reaction force of bar `CA` if `C` is displaced by `u_C`; similarly `f_CB` represents the reaction force of bar `CB`.

The total force at `C` due to this displacement, which we'll call the **internal force**, is the sum of the reaction forces.

```python
f_int = f_CA + f_CB
```

Now let's find the displacement of the truss under load.

## Equilibrium

Suppose we apply a force at `C` that pulls down in the `y` direction, with no additional force in the `x` direction.
We can represent this **external force** with the following vector, where the `F` represents the magnitude of the force.

```python
F = sp.symbols('F')
f_ext = sp_vector(0, -F)
```

If the truss is not moving, the internal and external forces must add up to zero, so we can write the equation $f_{int} = -f_{ext}$.

```python
eq = sp.Eq(f_int, -f_ext)
```

Which we can display concisely like this.

```python
sp.Eq(factor_matrix(f_int, factor), -f_ext)
```

To analyze the truss, we want to find the displacement `u_C` that solves this equation.
First, let's plug in values for the symbols, assuming that the truss is made of wooden popsicle sticks.

A common popsicle stick is about 15 cm long, 1 cm wide, and 0.2 cm thick.
The Young's modulus of wood is about 3 gigapascal.
Let's suppose the external force is the weight of a one kilogram mass, which is roughly 9.8 N.
We'll use the `pint` library to represent these quantities, including their units.

```python
import pint
ureg = pint.UnitRegistry()

params = {
    L: 15.0 * ureg.cm,
    E: 3.0e9 * ureg.Pa,
    A: 1.0 * ureg.cm * 0.2 * ureg.cm,
    F: 9.8 * ureg.newton,
}
```

To substitute these values into the equation, we'll use this function, which makes a dictionary that maps from symbols to their values, converted to base units.
For example, `L` in centimeters gets converted to meters, and `A` to square meters.

```python
def make_subs(params):
    """Make a dictionary that maps from symbols to quantities."""
    return {symbol: param.to_base_units().magnitude 
            for symbol, param in params.items()}
```

Here are the results in SI units.

```python
subs = make_subs(params)
subs
```

Now we'll use the `subs` method to replace the symbols in the equation with these numbers.

```python
eq_num = eq.subs(subs).simplify()
eq_num
```

And `solve` to find the value of `u_C`.

```python
sol = sp.solve(eq_num, u_C)
sol
```

The result is a dictionary that maps from the unknown components to their values.
To put these values in a vector, we can use `subs` again, and display the result in micrometers.

```python
u_sol = u_C.subs(sol)
sp.Eq(u_C, u_sol * 1e6)
```

So if we hang a 1 kg mass from a truss like this made of popsicle sticks, we expect `C` to be displaced by about 6 micrometers left and 18 micrometers down.

To see whether this result is plausible, let's compare it with the result from a simpler configuration.
If we balance a 1 kg mass on top of a vertical popsicle stick, and hold the bottom fixed, we can compute the displacement like this.

```python
displacement = F * L / A / E
displacement.subs(subs) * 1e6
```

The displacement we calculated for the simple truss is a little smaller than the displacement we expect from a single popsicle stick, so the results of the analysis seem plausible.

The simple truss is made of three popsicle sticks (remember that we doubled the thickness of `CB`), so we might expect it to be three times stiffer.
But the bars are diagonal, so only part of the displacement is along the axis of each bar, and only part of the reaction force is in the vertical direction.

Before we leave this example, let's also remember that our model of the truss is based on the assumption that the displacements are small enough that the bars are not deformed or broken.
With a 1 kilogram mass, the displacements we computed are much smaller than the lengths of the bars, so that assumption seems reasonable.

In contrast, suppose we use the same method to compute the displacement due to a 1000 kg mass.
The result would be about 0.6 cm left and 1.8 cm down.
If you are familiar with popsicle sticks, you probably have a sense that it is not possible to stretch or compress one by more than a centimeter without breaking it.
So the model we used to analyze this truss would not be realistic for such a large external force.

```python tags=["remove-cell"]
sp.Matrix([0.6, 1.8]).norm()
```

We're almost ready to take on a more complex truss, but first we need a new tool, the stiffness matrix.

## The Stiffness Matrix

In the previous section, we wrote this equation with internal force on the left and external force on the right:

```python
sp.Eq(factor_matrix(f_int, factor), -f_ext)
```

That was sufficient to analyze the simple truss, but for more complex trusses, we need a more general method.
Specifically, we'll write a matrix equation $K u = f$, where $u$ is the vector of unknown displacements, $f$ is the vector of external forces, and $K$ is the **stiffness matrix** that represents the relationship between the displacements and the reaction forces.

As an example, we'll rewrite the equation from the previous section in this form.
To start, let's rethink the scalar projection.

$$\mathrm{proj}_{r} u = \hat{r}\, \hat{r}^T u$$

In `vector_projection`, we computed the scalar projection $\hat{r}^T u$ first, to get the magnitude of the projection of $u$ in the direction of $r$, and then multiplied by $\hat{r}$.
In other words, we evaluated the vector projection like this:

$$\mathrm{proj}_{r} u = \hat{r}\, (\hat{r}^T u)$$

As an alternative, we could have multiplied $\hat{r}$ and $\hat{r}^T$ first, like this:

$$\mathrm{proj}_{r} u = (\hat{r}\, \hat{r}^T) u$$

The product $\hat{r}\, \hat{r}^T$ is a **projector matrix**, which we can compute like this.

```python
def projector(r):
    rhat = r.normalized()
    return rhat * rhat.T
```

As an example, here's the projector matrix for the vector `r_CA`.

```python
P_CA = projector(r_CA)
P_CA
```

Now if we multiply `P_CA` and `u_C`, the result is the vector projection of `u_C` onto `r_CA`.

```python
P_CA * u_C
```

If we multiply a projector matrix by the spring constant `k`, the result is a the stiffness matrix.
So `k_CA * P_CA` is the stiffness matrix due to bar `CA`, and `k_CB * P_CB` is the stiffness matrix due to bar `CB`.
Their sum is the stiffness matrix for the whole truss, which we'll call `K`.

```python
P_CB = projector(r_CB)

K = k_CA * P_CA + k_CB * P_CB
factor_matrix(K, A * E / L / 2)
```

If we multiply `K` by a displacement vector, the result is a force vector that represents the internal force due to that displacement.
So we can interpret the stiffness matrix by testing it with small displacements.
First we'll use `subs` to replace the symbols in `K` with numbers.

```python
from utils import sympy_to_numpy

K_np = sympy_to_numpy(K, subs)
K_np
```

Here's an example where we displace node `C` by one micrometer in the `x` direction.

```python
u_example1 = np.array([1, 0]) * 1e-6
f_example1 = -K_np @ u_example1
f_example1
```

The restoring force is 0.6 N in the negative `x` direction and 0.2 N in the positive `y` direction.
And here's an example where we displace `C` by one micrometer in the `y` direction.

```python
u_example2 = np.array([0, 1]) * 1e-6
f_example2 = -K_np @ u_example2
f_example2
```

The restoring force is about 0.2 N in the positive `x` direction and 0.6 N in the negative `y` direction.

More generally, we can use `K` to compute the force that results from any displacement.
To demonstrate, we'll make an array of 9 values from -20 to 20 micrometers.

```python
scale = 1e-6
bound = 20
displacements = np.linspace(-bound, bound, 9) * scale
```

By enumerating all pairs of values, we can make an array of 81 displacement vectors.

```python
pairs = [(x, y) for x in displacements for y in displacements]
U = np.array(pairs).T 
U.shape
```

`U` is matrix-like in the sense that it has two dimensions, but we are using it as a collection of vectors, not a matrix.
If we multiply by the stiffness matrix, the result is a collection of force vectors.

```python
F_field = -K_np @ U
F_field.shape
```

The following figure shows these results.
The location of each arrow shows a hypothetical displacement.
Each arrow shows the direction and relative magnitude of the resulting force due to displacement.
Note that the positions and forces are not in the same units, so the scale of the arrows is arbitrary.

```python
from utils import plot_force_field
    
ticks = np.arange(-20, 21, 5)
lim = scale * (bound + 2)
plot_force_field(nodes, subs, F_field, U, lim, displacements, ticks)
```

Notice that the restoring forces are bigger in the upper left and lower right, which correspond to extension and compression of the thicker bar.

In summary, multiplication by `K` is like a function that takes a displacement vector and returns a corresponding force vector.
A function that maps vectors to other vectors is sometimes called a **vector field**.
And when those vectors represent forces, we call it a **force field**.

Finally, let's confirm that we get the same result using NumPy to solve $K u = f$.
First we'll convert the external force vector to an array.

```python
f_ext_np = sympy_to_numpy(f_ext, subs)
f_ext_np
```

And here's the solution.

```python
import scipy

scipy.linalg.solve(K_np, f_ext_np) * 1e6
```

That's the same as the result we got before we factored out the stiffness matrix.

The key idea of this example is the use of the projector matrix to construct the stiffness matrix.
We'll use this idea again in the next example.

## Warren Trusses

So far we've been working with a simple truss with two bars and only one joint that can move, so we wrote a vector equation with only two components.
Now let's try something a little harder -- we'll analyze a Warren truss.
We'll use NetworkX to represent the joints and bars as nodes and edges; then we'll construct a matrix equation that relates the displacements at the joints to the external forces.

To specify the configuration of the truss, we'll define `N`, which is the number of triangles, `L`, which is the length of the bars, and `H`, which is the height of the triangles.
If the bars are all the same length, the triangles are equilateral, so $H = \frac{\sqrt{3} L}{2}$.

```python
N = 2
L = sp.Symbol('L', positive=True)
H = sp.sqrt(3) * L / 2
```

Now let's make the graph.
To understand the next few blocks of code, you might find it helpful to look ahead to the diagram that shows the results.

```python
G = nx.Graph()
```

First we'll create the nodes that run along the bottom of the truss and the edges that connect them.

```python
origin = sp_vector(0, 0)

for i in range(N + 1):
    pos = origin + sp_vector(i * L, 0)
    G.add_node(f'B{i}', pos=pos)
    if i > 0:
        G.add_edge(f'B{i-1}', f'B{i}')
```

The bottom nodes are assigned positions based on `L`, and labeled with strings like `B0`, `B1`, etc.

Now we'll add the nodes that run along the top, the horizontal edges that connect them, and the diagonal edges that connect the top and bottom nodes.

```python
for i in range(N):
    pos = origin + sp_vector((i + sp.Rational(1, 2)) * L, H)
    G.add_node(f'T{i}', pos=pos)
    G.add_edge(f'B{i}', f'T{i}')
    G.add_edge(f'T{i}', f'B{i+1}')

    if i > 0:
        G.add_edge(f'T{i-1}', f'T{i}')
```

The top nodes are assigned positions based on `L` and `H`, and labeled `T0`, `T1`, etc.
Here's what the graph looks like.

```python
from utils import draw_truss_graph

draw_truss_graph(G, {L: 0.15})
```

For reasons we'll see later, it will be convenient if every node has a unique index starting from `0`.
So we'll loop through the nodes and use `gset` to assign each node an index.

```python
for index, node in enumerate(G.nodes):
    gset(G, node, index=index)
```

Next we'll loop through the nodes and assign a displacement vector to each one.

```python
for node in G.nodes():
    ux, uy = sp.symbols(f'u_{node}_x u_{node}_y')    
    disp = sp_vector(ux, uy)
    gset(G, node, disp=disp)
```

For this example, we'll assume that the roadway is attached to the bottom nodes.
To keep things simple, we'll ignore the weight of the truss itself and assume that the load is attached to a single node, `B1`.
The following loop creates a force vector for each node in the truss and assigns it to an attribute called `f_ext`.

```python
F = sp.Symbol('F')
weighted = ['B1']

for node in G.nodes:
    if node in weighted:
        gset(G, node, f_ext=sp_vector(0, -F))
    else:
        gset(G, node, f_ext=sp_vector(0, 0))
```

Finally, for each bar, we'll compute a projector matrix that represents the projection of a displacement vector along the axis of the bar.

```python
for edge in G.edges:
    node1, node2 = edge
    x1 = G.nodes[node1]['pos']
    x2 = G.nodes[node2]['pos']
    r = x2 - x1
    P = projector(r)
    gset(G, edge, P=P)
```

To keep this example simple, we'll assume that every bar has the same cross-sectional area and Young's modulus -- but it would be easy to generalize by adding these attributes to the edges.

## Block Matrix Equation

At this point we have a NetworkX graph that represents the bars and joints of the truss.
The next step is to write a matrix equation to relate the unknown displacements to the internal and external forces.
To do that, we will construct a **block matrix** which is a way to represent a system of equations where the unknowns are vectors.

To show the structure of the problem, we'll construct with a vector of displacements, where each displacement is itself a vector.

```python
blocks = [sp.MatrixSymbol(f'u_{{{node}}}', 2, 1) for node in G.nodes]
u_block = sp.BlockMatrix([[u] for u in blocks])
```

And another vector that represents the external forces on the nodes.

```python
blocks = [sp.MatrixSymbol(f'f_{{{node}}}', 2, 1) for node in G.nodes]
f_block = sp.BlockMatrix([[f] for f in blocks])
```

Now we'll create a block matrix where each element is a stiffness matrix -- the element $K_{ij}$ represents the stiffness at node $i$ due to its connection to node $j$.

```python
n = len(G.nodes)
blocks = [[sp.MatrixSymbol(f'K_{i}{j}', 2, 2) 
           for j in range(n)] 
           for i in range(n)]
K_block = sp.BlockMatrix(blocks)
```

Here's the equation that relates these matrices.

```python
equation = sp.Eq(K_block * u_block, f_block)
equation
```

Let's think about what each element of the stiffness matrix means.
Each row represents an equation that computes the sum of the internal forces on one node, given the displacements of all of the nodes.
And each column represents the contribution of one displacement to the internal forces on all of the nodes.

Now let's think about how to compute the elements of $K$.
As an example, consider the bar between `B0` and `T0`, which have indices 0 and 3.
The stiffness matrix of that bar contributes to four elements of $K$:

* In row 0, it contributes to $K_{00}$ and $K_{03}$, because the internal force on node 0 depends on the difference of displacements $u_{B0}$ and $u_{T0}$.

* In row 3, it contributes to $K_{30}$ and $K_{33}$, because the internal force on node 3 depends on the same difference of displacements.

In the next section, we'll assemble $K$ by looping through the edges and adding up the contributions from the stiffness matrix of each bar.

## Compute the Block Matrix

To construct the block matrix, we'll loop through the edges and compute a matrix that represents the stiffness of each node due to the other nodes it is connected to.
We'll start by working through a single example, the edge from `B0` to `T0`.
We can use `gget` to get the projector matrix for this edge.

```python
edge = 'B0', 'T0'
P = gget(G, edge, "P")
P
```

So we can compute the stiffness matrix for a single bar like this:

```python
k = A * E / L
kP = -k * P
factor_matrix(kP, k/4)
```

Next we'll get the displacements of the nodes.

```python
node1, node2 = edge
u1 = gget(G, node1, "disp")
u2 = gget(G, node2, "disp")
```

If both ends of the bar are displaced, the total extension or compression is the difference between the displacements, `u1 - u2`.

Now we're ready to assemble the block matrix.
We'll start with a list of lists, where each element is a stiffness matrix full of zeroes.
As we loop through the edges, we'll add to these matrices.

```python
blocks = [[sp.Matrix.zeros(2, 2) for _ in range(n)] for _ in range(n)]
```

Now here's the key to the whole operation.
If `kP` is the stiffness matrix of the bar that connects `u1` and `u2`, the internal force in that bar is `kP * (u1 - u2)`, which we can factor as `kP * u1` and `-kP * u2`.

Now let's figure out how to add `kP` to the block matrix.
Remember that each row of the block matrix represents an equation that computes the total internal force on that node.
And each column represents the contribution of one displacement to each of the nodes.

So each edge contributes to two rows in the matrix, one for each node it connects.
And it contributes to two columns, because it depends on the difference between two displacements.

To break this down, we'll use the following function, which takes two nodes and a stiffness matrix, `kP`.
It uses `gget` to look up the indices of the nodes.
Then it modifies two blocks in the row corresponding to `u1`: it adds `kP` to the column corresponding to `u1` and subtracts `kP` from the column corresponding to `u2`.
These changes represent the stiffness at `node1` because it is connected to `node2`.

```python
def add_block(node1, node2, kP):
    i = gget(G, node1, 'index')
    j = gget(G, node2, 'index')
    
    blocks[i][i] +=  kP
    blocks[i][j] += -kP
```

Now for each edge in the graph, we call `add_block` twice, once to add the stiffness at `node1` because it is connected to `node2`, and once to add the stiffness at `node2` because it is connected with `node1`.

```python
for edge in G.edges:
    node1, node2 = edge
    P = gget(G, edge, "P")

    add_block(node1, node2, k * P)
    add_block(node2, node1, k * P)
```

The result is a list of lists of matrices, each a sum of stiffness matrices.

Finally, we can use the list of lists to make a `BlockMatrix`.
Here's the result, shown with the common factor $k/4$ pulled out.

```python
K_block = sp.BlockMatrix(blocks)
factor_matrix(K_block, factor=k/4)
```

Now let's see if we can solve the matrix equation $K u = f$ (hint: we can't quite yet).

## Shift and Rotate

At this point we have a block matrix with five columns -- one for each displacement -- and five rows -- one for each equation.
So if we are given five external forces, it's tempting to think we can solve for the unknown displacements.
But we can't, because the stiffness matrix we assembled does not have full rank.

Before we can compute its rank, we have to convert `k_block` from a 5x5 block matrix to a 10x10 conventional matrix, using the method `as_explicit`.

```python tags=["hide-output"]
K = K_block.as_explicit()
factor_matrix(K, k/4)
```

Now we can compute the rank, which indicates the number of non-redundant equations it represents.

```python
K.rank()
```

Although the matrix has 10 rows, the rank is only 7, which means that the matrix represents only 7 independent equations.
The rank-nullity theorem implies that it has a three-dimensional nullspace, which we can compute with the `nullspace` method.

```python
nullspace = K.nullspace()
len(nullspace)
```

The result is a list of three vectors.
Each of them is a solution to the homogeneous equation $K u = 0$, which means that there are three ways we can displace the truss that don't depend on the external forces.

If you think about the physical scenario we are modeling, you might be able to figure out what they are.
The equations we have written so far balance the internal and external forces, but they don't constrain the truss from moving in space.
Because the truss is not attached to anything, it is free to shift in the `x` direction, shift in the `y` direction, and rotate.
These unconstrained ways of moving are called **degrees of freedom**.

If we inspect the vectors in the nullspace, we can see how they correspond to these degress of freedom.
We'll use this function to extract the displacements of the nodes, in symbolic form, and put them in a block matrix.

```python
def get_block_vector(G, attr, keep=None):
    blocks = [gget(G, node, attr) for node in G.nodes]
    if keep is not None:
        blocks = [blocks[i] for i in keep]
    return sp.BlockMatrix([[block] for block in blocks])
```

```python
u_sym = get_block_vector(G, 'disp').as_explicit()
```

Now here's the second of the three null vectors, which is the easiest to interpret.

```python
v_null_1 = nullspace[1]
sp.Eq(u_sym, v_null_1)
```

This vector represents a displacement of all nodes by one unit in the `x` direction.

The sum of the other two null vectors is similar -- it displaces all nodes by one unit in the `y` direction.

```python
v_null_2 = (nullspace[0] + nullspace[2])
sp.Eq(u_sym, v_null_2)
```

Finally, here's the difference between the same two null vectors.

```python
v_null_3 = (nullspace[0] - nullspace[2])
sp.Eq(u_sym, v_null_3)
```

Looking at the components of this vector, it's not obvious what it means.
It will be clearer if we plot the effect it has on the truss.
To do that, we'll pack it into a dictionary that maps from the components of the displacement vectors to the elements of the null vector.

```python
rotation = dict(zip(u_sym, v_null_3))
```

We'll use the following function to loop through the nodes in the graph and displace each node using the values in this dictionary.

```python
def displace_truss(G, sol, scale=300):
    G_sol = G.copy()

    for node in G_sol.nodes:
        pos = gget(G_sol, node, 'pos')
        u = gget(G_sol, node, 'disp')
        u_eval = u.subs(sol).evalf() * scale 
        gset(G_sol, node, pos=pos + u_eval)

    return G_sol
```

The `scale` parameter determines the magnitudes of the displacements.

```python
G_rot = displace_truss(G, rotation, scale=0.005)
```

Finally, let's draw the displaced truss on top of an outline of the original.

```python
draw_truss_graph(G, subs, label_nodes=False, alpha=0.3)
draw_truss_graph(G_rot, subs)
```

Now we can see that this null vector represents the third degree of freedom, rotation.

With 10 unknowns and only 7 independent equations, this system is **underdetermined**, which means it has many possible solutions.
But that's because we have not yet taken into account all of the information we have about the physical scenario -- specifically, we know that two of the nodes are fixed in space.

## Reduce the Matrix

Let's assume that the truss is attached firmly at the left and right.
The following loop makes a copy of the graph where the displacement at those nodes is zero.

```python
fixed_nodes = ['B0', f'B{N}']
G_fixed = G.copy()

for node in G.nodes():
    if node in fixed_nodes:
        disp = sp_vector(0, 0)
        gset(G_fixed, node, disp=disp)
```

Since these displacements are no longer unknown, we can remove them from the matrix equation.
First, we'll compute the indices of the remaining nodes that are not fixed.

```python
keep = [gget(G_fixed, node, 'index') 
        for node in G_fixed.nodes 
        if node not in fixed_nodes]
keep    
```

Now we'll select the blocks that correspond to these nodes.

```python
blocks_reduced = [
    [blocks[i][j] 
     for j in keep]
     for i in keep
]
```

And make a new block matrix that has been **reduced** by removing the known displacements.

```python
K_reduced = sp.BlockMatrix(blocks_reduced)
factor_matrix(K_reduced, k/4)
```

Now there are only 6 columns, and the matrix has **full rank** -- that is, the equations it represents are independent. 

```python
K_reduced.as_explicit().rank()
```

And that means the equation $K u = f$ has a unique solution for any vector of forces, $f$. 

To solve the system, we'll use `get_block_vector` to loop through the graph and collect the unknown displacements.

```python
u_reduced = get_block_vector(G_fixed, 'disp', keep)
u_reduced
```

And we can collect the external forces the same way.

```python
f_reduced = get_block_vector(G_fixed, 'f_ext', keep)
f_reduced
```

Now (finally!) we are ready to solve the system.

## Solve the System

Here's the matrix equation, $K u = f$.

```python
eq = sp.Eq(K_reduced * u_reduced, f_reduced)
```

We can use `subs` again to replace the symbols with their values.

```python
eq_num = eq.subs(subs)
```

The result is a numerical matrix equation we can solve for the unknown displacements.

```python
sol = sp.solve(eq_num, u_reduced.as_explicit())
```

Here's what the result looks like.

```python
u_sol = u_reduced.subs(sol)
sp.Eq(u_reduced, factor_matrix(u_sol, 1e-6))
```

And here's what those displacements look like, scaled up by a factor of 300 to make them visible.

```python
G_sol = displace_truss(G_fixed, sol, scale=300)

draw_truss_graph(G, subs, label_nodes=False, alpha=0.3)
draw_truss_graph(G_sol, subs)
```

At `B1`, the displacement in the `y` direction is about 41 micrometers, which is about twice the displacement we computed for the simple truss.
And that makes sense because the span of this truss is about twice the span of the simple truss.

If you go back where we defined `N = 2`, you can set `N` to another value and run the notebook again.
See what happens to the results as you increase `N`.

In some Warren trusses, the roadbed is suspended from the top nodes, rather than attached to the bottom nodes.
Run the analysis with half of the load on each of `T0` and `T1`, rather than all of it on `B0`.
Are the displacements bigger or smaller?


TECHNICAL NOTE: If you've seen truss analysis before, you might have been surprised that I constrained the fixed nodes in both dimensions.
In textbook truss analysis, it is more common to constrain one of the nodes in only one dimension.
In this example, that would reduce the 10x10 underdetermined system to a 7x7 fully determined system.
By fixing two nodes entirely, we reduced it to a 6x6 system, also fully determined.

The conventional approach imposes the minimal number of constraints, but the approach I used here was a little easier to implement, and I think it's a more realistic model of a real bridge, where the two endpoints are generally similar.
But either method is fine.

## Discussion

When I was an undergraduate, I took a class where we learned to analyze trusses.
One of my classmates asked a question that I'm not sure I appreciated at the time, but it stuck with me.
To paraphrase, he asked, "Do we design trusses because they are optimal, or because they are easy to analyze?"
This is a really good question.

As you might guess, the answer is both.
If your goal is to build a structure that supports a given load with a minimum amount of material, a truss may not be optimal -- with the complexity of real-world requirements and constraints, it's hard to say what "optimal" even means.
But for many applications, a truss is a fundamentally good design.

It is also easy to analyze, but that is not as important as it used to be.
Before we had computers, we analyzed structures by hand, so we had to keep them simple.
When I was an undergraduate, we had enough computational power and software capability to analyze more complex structures, but what we could design was still limited by what we could analyze.

Now with modern computational tools, we can analyze almost any structure we can imagine, so analysis is no longer the limiting factor on what we can design and build.
But this is a recent change, and we are only beginning to see its consequences.

<!-- #region tags=["remove-print"] -->
[Think Linear Algebra](https://allendowney.github.io/ThinkLinearAlgebra/index.html)

Copyright 2025 [Allen B. Downey](https://allendowney.com)

Code license: [MIT License](https://mit-license.org/)

Text license: [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by-nc-sa/4.0/)
<!-- #endregion -->

```python

```
