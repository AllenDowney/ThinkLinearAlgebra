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
v + w, 2 * v, v.mag(), v.hat()
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

The behavior depends on the weights of the four rules, and on how far and how wide each Boid can see.
Here are the parameters from `Boids7.py`, and a version that weakens the carrot, strengthens alignment, and widens the field of view.
With the original parameters, the neighbor rules rarely see anyone, so each Boid orbits the carrot on its own.

```python
original = dict(
    w_center=3, w_avoid=10, w_align=1, w_love=10,
    center_radius=1, align_radius=0.5, avoid_radius=0.3, view_angle=1,
)

flock = dict(
    w_center=3, w_avoid=10, w_align=5, w_love=3,
    center_radius=2, align_radius=1, avoid_radius=0.3, view_angle=2,
)
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
            if offset.mag() < radius and self.vel.angle(offset) < angle:
                result.append(obj)
        return result

    def center(self, boids, radius, angle):
        """Toward the center of the neighbors we can see."""
        neighbors = self.neighbors(boids, radius, angle)
        return toward_mean([b.pos for b in neighbors], self.pos)

    def avoid(self, objects, radius, angle=np.pi):
        """Away from the center of anything too close, in any direction."""
        neighbors = self.neighbors(objects, radius, angle)
        return -toward_mean([b.pos for b in neighbors], self.pos)

    def align(self, boids, radius, angle):
        """Toward the mean velocity of the neighbors we can see."""
        neighbors = self.neighbors(boids, radius, angle)
        return toward_mean([b.vel for b in neighbors], self.vel)

    def love(self, carrot):
        """Toward the carrot."""
        return (carrot.pos - self.pos).limit(1)

    def set_goal(self, boids, carrot, p):
        """The goal is a linear combination of the four rules."""
        self.goal = (
            p["w_center"] * self.center(boids, p["center_radius"], p["view_angle"])
            + p["w_avoid"] * self.avoid(boids + [carrot], p["avoid_radius"])
            + p["w_align"] * self.align(boids, p["align_radius"], p["view_angle"])
            + p["w_love"] * self.love(carrot)
        ).hat()

    def move(self, mu=0.1, dt=0.1):
        """Turn partway toward the goal, then take a step."""
        self.vel = ((1 - mu) * self.vel + mu * self.goal).hat()
        self.pos = self.pos + dt * self.vel
```

The carrot moves like a Boid: at each step it turns a little to the left (toward a goal that is a linear combination of its velocity and a perpendicular vector) and takes a step.
It moves at about a fifth of the Boids' speed, so they can overtake it and circle back.

```python
def perpendicular(v):
    """Rotate 90 degrees counterclockwise in the x-y plane."""
    return Vector(-v.y, v.x, 0)


class Carrot:
    def __init__(self, weight=0.1, mu=0.1, dt=0.021):
        self.pos = Vector(2, 0, 0)
        self.vel = Vector(0, 1, 0)
        self.weight, self.mu, self.dt = weight, mu, dt

    def move(self, t):
        goal = (self.vel + self.weight * perpendicular(self.vel)).hat()
        self.vel = ((1 - self.mu) * self.vel + self.mu * goal).hat()
        self.pos = self.pos + self.dt * self.vel
```

```python
class World:
    def __init__(self, params, n=20):
        self.boids = [Boid() for _ in range(n)]
        self.carrot = Carrot()
        self.params = params
        self.t = 0

    def step(self):
        self.t += 1
        self.carrot.move(self.t)
        for boid in self.boids:
            boid.set_goal(self.boids, self.carrot, self.params)
        for boid in self.boids:
            boid.move()

    def polarization(self):
        """Length of the mean heading: 1 if all Boids point the same way."""
        return mean([boid.vel for boid in self.boids]).mag()
```

## Simulation

The following function runs a simulation and records positions and velocities at each step.
`np.array` converts a list of `Vector` objects to an array with one row per vector, which is the form Matplotlib wants.
Each run starts from the same random state, so runs with different parameters start from the same positions and velocities.

```python
def simulate(params, num_steps=300, n=20, seed=17):
    global rng
    rng = np.random.default_rng(seed)
    world = World(params, n=n)

    history = []
    polarizations = []
    for _ in range(num_steps):
        world.step()
        positions = np.array([boid.pos for boid in world.boids])
        velocities = np.array([boid.vel for boid in world.boids])
        history.append((positions, velocities, world.carrot.pos))
        polarizations.append(world.polarization())
    return history, polarizations
```

```python
num_steps = 300
history_original, polar_original = simulate(original, num_steps)
history_flock, polar_flock = simulate(flock, num_steps)
```

Polarization is near 1 when the Boids are flying in the same direction.

```python
plt.plot(polar_original, label="original")
plt.plot(polar_flock, label="flock")
decorate(xlabel="Step", ylabel="Polarization", ylim=[0, 1.05])
```

Choose which run to animate.

```python
history = history_flock
```

## Animation

Each Boid is drawn as an arrow pointing in the direction it's flying, with a short trail showing where it has been.
The animation shows every other step of the simulation.
The camera sways a little from side to side, which makes it easier to see the flock in 3D.

```python
from matplotlib import animation
from IPython.display import HTML

plt.rcParams["animation.embed_limit"] = 50  # MB

fig = plt.figure(figsize=(7, 7), dpi=100)
ax = fig.add_subplot(projection="3d")

step = 2
trail_length = 8
frames = range(trail_length, num_steps, step)


def draw_frame(i):
    ax.clear()
    positions, velocities, carrot = history[i]

    # trails: each Boid's recent positions
    trail = np.array([p for p, v, c in history[i - trail_length : i + 1]])
    for k in range(trail.shape[1]):
        ax.plot(*trail[:, k].T, color="C0", alpha=0.3, lw=1)

    ax.quiver(*positions.T, *velocities.T, length=0.3, normalize=True, color="C0")
    ax.scatter(*carrot, color="C3", s=60)
    ax.set(xlim=(-3.5, 3.5), ylim=(-3.5, 3.5), zlim=(-2, 2),
           xticks=[], yticks=[], zticks=[])
    ax.view_init(elev=25, azim=-60 + 20 * np.sin(2 * np.pi * i / num_steps))


anim = animation.FuncAnimation(fig, draw_frame, frames=frames, interval=67)
plt.close(fig)
```

```python
HTML(anim.to_jshtml())
```
