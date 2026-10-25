---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Boids in 3D (prototype)

A prototype for Chapter 3 (`plans/chapter3_boids.md`): Reynolds' Boids, written with the `Vector` class from `utils.py` and animated with Matplotlib in 3D.

Changes from `Boids7.py` in *Think Complexity*:

* Vectors are `utils.Vector` objects, not VPython vectors.
* `align` matches velocity: the neighbors' mean velocity minus the Boid's own velocity.
* All Boids choose their goals before any of them moves (synchronous update).
* The carrot follows a fixed path instead of the mouse.

[Run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkLinearAlgebra/blob/main/plans/prototypes/boids3d.ipynb).

```python
from os.path import basename, exists


def download(url):
    filename = basename(url)
    if not exists(filename):
        from urllib.request import urlretrieve

        local, _ = urlretrieve(url, filename)
        print("Downloaded " + local)


download("https://github.com/AllenDowney/ThinkLinearAlgebra/raw/main/utils.py")
```

```python
import numpy as np
import matplotlib.pyplot as plt

from utils import Vector, decorate
```

## Vectors

A `Vector` supports the operations that define a vector -- addition and multiplication by a scalar -- plus a few that use them.

```python
v = Vector(1, 2, 2)
w = Vector(0, 1, 0)
v + w, 2 * v, v.mag, v.hat()
```

A linear combination is a sum of vectors, each multiplied by a scalar.

```python
3 * v - 2 * w
```

Multiplying two vectors with `*` is not a vector operation.

```python
try:
    v * w
except TypeError as e:
    print(e)
```

## Boids

Each Boid has a position and a velocity, both `Vector` objects.
Its speed is always 1, so its velocity is also its heading.

```python
rng = np.random.default_rng(17)


def random_vector(low, high):
    return Vector(rng.uniform(low, high, size=3))
```

```python
def mean(vectors):
    """The mean of a list of vectors, which is a linear combination."""
    return sum(vectors) / len(vectors)


def toward_mean(vectors, ref):
    """The vector from ref to the mean of vectors, at most length 1."""
    if not vectors:
        return Vector.zero()
    return (mean(vectors) - ref).limit(1)
```

```python
class Boid:
    def __init__(self):
        self.pos = random_vector(0, 1)
        self.vel = random_vector(-1, 1).hat()

    def neighbors(self, objects, radius, angle):
        """Objects within radius whose direction is within angle of our heading."""
        result = []
        for obj in objects:
            if obj is self:
                continue
            offset = obj.pos - self.pos
            if offset.mag < radius and self.vel.angle(offset) < angle:
                result.append(obj)
        return result

    def center(self, boids, radius=1, angle=1):
        """Toward the center of the neighbors we can see."""
        neighbors = self.neighbors(boids, radius, angle)
        return toward_mean([b.pos for b in neighbors], self.pos)

    def avoid(self, objects, radius=0.3, angle=np.pi):
        """Away from the center of anything too close."""
        neighbors = self.neighbors(objects, radius, angle)
        return -toward_mean([b.pos for b in neighbors], self.pos)

    def align(self, boids, radius=0.5, angle=1):
        """Toward the mean velocity of the neighbors we can see."""
        neighbors = self.neighbors(boids, radius, angle)
        return toward_mean([b.vel for b in neighbors], self.vel)

    def love(self, carrot):
        """Toward the carrot."""
        return (carrot.pos - self.pos).limit(1)

    def set_goal(self, boids, carrot, weights):
        """The goal is a linear combination of the four rules."""
        w_center, w_avoid, w_align, w_love = weights
        self.goal = (
            w_center * self.center(boids)
            + w_avoid * self.avoid(boids + [carrot])
            + w_align * self.align(boids)
            + w_love * self.love(carrot)
        ).hat()

    def move(self, mu=0.1, dt=0.1):
        """Turn partway toward the goal, then take a step."""
        self.vel = ((1 - mu) * self.vel + mu * self.goal).hat()
        self.pos = self.pos + dt * self.vel
```

The carrot is not a Boid, but it has a position, so Boids can steer toward it and avoid it.
It follows a loop in 3D.

```python
class Carrot:
    def __init__(self):
        self.pos = Vector(2, 0, 0)

    def move(self, t, period=150):
        theta = 2 * np.pi * t / period
        self.pos = Vector(2 * np.cos(theta), 2 * np.sin(theta), 0.5 * np.sin(2 * theta))
```

```python
class World:
    def __init__(self, n=20, weights=(3, 10, 1, 10)):
        self.boids = [Boid() for _ in range(n)]
        self.carrot = Carrot()
        self.weights = weights
        self.t = 0

    def step(self):
        self.t += 1
        self.carrot.move(self.t)
        for boid in self.boids:
            boid.set_goal(self.boids, self.carrot, self.weights)
        for boid in self.boids:
            boid.move()

    def polarization(self):
        """Length of the mean heading: 1 if all Boids point the same way."""
        return mean([boid.vel for boid in self.boids]).mag
```

## Simulation

We run the simulation and record positions and velocities at each step.
`np.array` converts a list of `Vector` objects to an array with one row per vector, which is the form Matplotlib wants.

```python
world = World(n=20)
num_steps = 150

history = []
polarizations = []
for _ in range(num_steps):
    world.step()
    positions = np.array([boid.pos for boid in world.boids])
    velocities = np.array([boid.vel for boid in world.boids])
    history.append((positions, velocities, world.carrot.pos))
    polarizations.append(world.polarization())
```

Polarization is near 1 when the Boids are flying in the same direction.

```python
plt.plot(polarizations)
decorate(xlabel="Step", ylabel="Polarization", ylim=[0, 1.05])
```

## Animation

Each Boid is drawn as an arrow pointing in the direction it's flying.
The camera circles slowly so it's easier to see the flock in 3D.

```python
from matplotlib import animation
from IPython.display import HTML

fig = plt.figure(figsize=(5, 5), dpi=72)
ax = fig.add_subplot(projection="3d")


def draw_frame(i):
    ax.clear()
    positions, velocities, carrot = history[i]
    ax.quiver(*positions.T, *velocities.T, length=0.3, normalize=True, color="C0")
    ax.scatter(*carrot, color="C3", s=60)
    ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), zlim=(-1.5, 1.5))
    ax.view_init(elev=25, azim=-60 + 0.6 * i)


anim = animation.FuncAnimation(fig, draw_frame, frames=num_steps, interval=67)
plt.close(fig)
```

```python
HTML(anim.to_jshtml())
```
