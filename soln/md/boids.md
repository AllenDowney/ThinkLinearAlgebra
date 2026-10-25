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

# Boids


[Click here to run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/ThinkLinearAlgebra/blob/main/nb/boids.ipynb).

```python
%load_ext nb_black
```

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
import pandas as pd
import matplotlib.pyplot as plt

from utils import decorate
```

## Boids

In 1987 Craig Reynolds published "Flocks, herds and schools: A distributed behavioral model", which describes an agent-based model of herd behavior. You can download his paper from <https://thinkcomplex.com/boid>.

Agents in this model are called "Boids", which is both a contraction of "bird-oid" and an accented pronunciation of "bird" (although Boids are also used to model fish and herding land animals).

Each agent simulates three behaviors:

Flock centering:

:   Move toward the center of the flock.

Collision avoidance:

:   Avoid obstacles, including other Boids.

Velocity matching:

:   Align velocity (speed and direction) with neighboring Boids.

Boids make decisions based on local information only; each Boid only sees (or pays attention to) other Boids in its field of vision.

In the repository for this book, you will find `Boids7.py`, which contains my implementation of Boids, based in part on the description in Gary William Flake's book, *The Computational Beauty of Nature*.

My implementation uses VPython, which is a library that provides 3-D graphics. VPython provides a `Vector` object, which I use to represent the position and velocity of Boids in three dimensions. You can read about them at <https://thinkcomplex.com/vector>.

## The Boid algorithm

`Boids7.py` defines two classes: `Boid`, which implements the Boid behaviors, and `World`, which contains a list of Boids and a "carrot" the Boids are attracted to.

The `Boid` class defines the following methods:

-   `center`: Finds other Boids within range and computes a vector toward their centroid.

-   `avoid`: Finds objects, including other Boids, within a given range, and computes a vector that points away from their centroid.

-   `align`: Finds other Boids within range and computes the average of their headings.

-   `love`: Computes a vector that points toward the carrot.

Here's the implementation of `center`:

```python
def center(self, boids, radius=1, angle=1):
    neighbors = self.get_neighbors(boids, radius, angle)
    vecs = [boid.pos for boid in neighbors]
    return self.vector_toward_center(vecs)
```

The parameters `radius` and `angle` are the radius and angle of the field of view, which determines which other Boids are taken into consideration. `radius` is in arbitrary units of length; `angle` is in radians.

`center` uses `get_neighbors` to get a list of `Boid` objects that are in the field of view. `vecs` is a list of `Vector` objects that represent their positions.

Finally, `vector_toward_center` computes a `Vector` that points from `self` to the centroid of `neighbors`.

Here's how `get_neighbors` works:

```python
def get_neighbors(self, boids, radius, angle):
    neighbors = []
    for boid in boids:
        if boid is self:
            continue

        # if not in range, skip it
        offset = boid.pos - self.pos
        if offset.mag > radius:
            continue

        # if not within viewing angle, skip it
        if self.vel.diff_angle(offset) > angle:
            continue

        # otherwise add it to the list
        neighbors.append(boid)

    return neighbors
```

For each other Boid, `get_neighbors` uses vector subtraction to compute the vector from `self` to `boid`. The magnitude of this vector is the distance between them; if this magnitude exceeds `radius`, we ignore `boid`.

`diff_angle` computes the angle between the velocity of `self`, which points in the direction the Boid is heading, and the position of `boid`.
If this angle exceeds `angle`, we ignore `boid`.

Otherwise `boid` is in view, so we add it to `neighbors`.

Now here's the implementation of `vector_toward_center`, which computes a vector from `self` to the centroid of its neighbors.

```python
def vector_toward_center(self, vecs):
    if vecs:
        center = np.mean(vecs)
        toward = vector(center - self.pos)
        return limit_vector(toward)
    else:
        return null_vector
```

VPython vectors work with NumPy, so `np.mean` computes the mean of `vecs`, which is a sequence of vectors. `limit_vector` limits the magnitude of the result to 1; `null_vector` has magnitude 0.

We use the same helper methods to implement `avoid`:

```python
def avoid(self, boids, carrot, radius=0.3, angle=np.pi):
    objects = boids + [carrot]
    neighbors = self.get_neighbors(objects, radius, angle)
    vecs = [boid.pos for boid in neighbors]
    return -self.vector_toward_center(vecs)
```

`avoid` is similar to `center`, but it takes into account the carrot as well as the other Boids. Also, the parameters are different: `radius` is smaller, so Boids only avoid objects that are too close, and `angle` is wider, so Boids avoid objects from all directions. Finally, the result from `vector_toward_center` is negated, so it points *away* from the centroid of any objects that are too close.

Here's the implementation of `align`:

```python
def align(self, boids, radius=0.5, angle=1):
    neighbors = self.get_neighbors(boids, radius, angle)
    vecs = [boid.vel for boid in neighbors]
    return self.vector_toward_center(vecs)
```

`align` is also similar to `center`; the big difference is that it computes the average of the neighbors' velocities, rather than their positions. If the neighbors point in a particular direction, the Boid tends to steer toward that direction.

Finally, `love` computes a vector that points in the direction of the carrot.

```python
def love(self, carrot):
    toward = carrot.pos - self.pos
    return limit_vector(toward)
```

The results from `center`, `avoid`, `align`, and `love` are what Reynolds calls "acceleration requests", where each request is intended to achieve a different goal.

## Arbitration

To arbitrate among these possibly conflicting goals, we compute a weighted sum of the four requests:

```python
def set_goal(self, boids, carrot):
    w_avoid = 10
    w_center = 3
    w_align = 1
    w_love = 10

    self.goal = (
        w_center * self.center(boids)
        + w_avoid * self.avoid(boids, carrot)
        + w_align * self.align(boids)
        + w_love * self.love(carrot)
    )
    self.goal.mag = 1
```

`w_center`, `w_avoid`, and the other weights determine the importance of the acceleration requests. Usually `w_avoid` is relatively high and `w_align` is relatively low.

After computing a goal for each Boid, we update their velocity and position:

```python
def move(self, mu=0.1, dt=0.1):
    self.vel = (1 - mu) * self.vel + mu * self.goal
    self.vel.mag = 1
    self.pos += dt * self.vel
    self.axis = self.length * self.vel
```


Read the code to see how the parameters control Boid behaviors.
Experiment with different parameters. What happens if you "turn off" one of the behaviors by setting its weight to 0?

To generate more bird-like behavior, Flake suggests adding a behavior to maintain a clear line of sight; in other words, if there is another bird directly ahead, the Boid should move away laterally. What effect do you expect this rule to have on the behavior of the flock? Implement it and see.

```python
import vpython as vp

vp.sphere()
```

```python
from vpython import vector

null_vector = vector(0, 0, 0)


def random_vector(a, b):
    """Create a vector with each element uniformly distributed in [a, b)."""
    coords = np.random.uniform(a, b, size=3)
    return vector(*coords)


def limit_vector(vect):
    """If the magnitude is greater than 1, set it to 1"""
    if vect.mag > 1:
        vect.mag = 1
    return vect
```

```python
class Boid(vp.cone):
    """A Boid is a VPython cone with a velocity and an axis."""

    def __init__(self, radius=0.03, length=0.1):
        pos = random_vector(0, 1)
        self.vel = random_vector(0, 1).norm()
        super().__init__(pos=pos, radius=radius, length=length)
        self.axis = length * self.vel

    def get_neighbors(self, boids, radius, angle):
        """Return a list of neighbors within a field of view.

        boids: list of boids
        radius: field of view radius
        angle: field of view angle in radians

        returns: list of Boid
        """
        neighbors = []
        for boid in boids:
            if boid is self:
                continue
            offset = boid.pos - self.pos

            # if not in range, skip it
            if offset.mag > radius:
                continue

            # if not within viewing angle, skip it
            diff = self.vel.diff_angle(offset)
            if abs(diff) > angle:
                continue

            # otherwise add it to the list
            neighbors.append(boid)

        return neighbors

    def center(self, boids, radius=1, angle=1):
        """Find the center of mass of other boids in range and
        return a vector pointing toward it."""
        neighbors = self.get_neighbors(boids, radius, angle)
        vecs = [boid.pos for boid in neighbors]
        return self.vector_toward_center(vecs)

    def vector_toward_center(self, vecs):
        """Vector from self to the mean of vecs.

        vecs: sequence of vector

        returns: Vector
        """
        if vecs:
            center = np.mean(vecs)
            toward = vector(center - self.pos)
            return limit_vector(toward)
        else:
            return null_vector

    def avoid(self, boids, carrot, radius=0.3, angle=np.pi):
        """Find the center of mass of all objects in range and
        return a vector in the opposite direction, with magnitude
        proportional to the inverse of the distance (up to a limit)."""
        objects = boids + [carrot]
        neighbors = self.get_neighbors(objects, radius, angle)
        vecs = [boid.pos for boid in neighbors]
        return -self.vector_toward_center(vecs)

    def align(self, boids, radius=0.5, angle=1):
        """Return the average heading of other boids in range.

        boids: list of Boids
        """
        neighbors = self.get_neighbors(boids, radius, angle)
        vecs = [boid.vel for boid in neighbors]
        return self.vector_toward_center(vecs)

    def love(self, carrot):
        """Returns a vector pointing toward the carrot."""
        toward = carrot.pos - self.pos
        return limit_vector(toward)

    def set_goal(self, boids, carrot):
        """Sets the goal to be the weighted sum of the goal vectors."""

        # weights for various rules
        w_avoid = 10
        w_center = 3
        w_align = 1
        w_love = 10

        self.goal = (
            w_center * self.center(boids)
            + w_avoid * self.avoid(boids, carrot)
            + w_align * self.align(boids)
            + w_love * self.love(carrot)
        )
        self.goal.mag = 1

    def move(self, mu=0.1, dt=0.1):
        """Update the velocity, position and axis vectors.

        mu: how fast the boids can turn (maneuverability).
        dt: time step
        """

        self.vel = (1 - mu) * self.vel + mu * self.goal
        self.vel.mag = 1
        self.pos += dt * self.vel
        self.axis = self.length * self.vel
```

```python
class World(object):

    def __init__(self, n=10):
        """Create n Boids and one carrot.

        tracking: indicates whether the carrot follows the mouse
        """
        self.boids = [Boid() for i in range(n)]
        self.carrot = sphere(pos=vector(1, 0, 0), radius=0.1, color=vector(1, 0, 0))
        self.tracking = False

    def step(self):
        """Compute one time step."""
        # move the boids
        for boid in self.boids:
            boid.set_goal(self.boids, self.carrot)
            boid.move()

        # if we're tracking, move the carrot
        if self.tracking:
            self.carrot.pos = scene.mouse.pos
```

```python
n = 20
size = 5

world = World(n)
scene.center = world.carrot.pos
scene.autoscale = False


def toggle_tracking(evt):
    """If we're currently tracking, turn it off, and vice versa."""
    world.tracking = not world.tracking


# when the user clicks, toggle tracking.
scene.bind("click", toggle_tracking)

while 1:
    rate(10)
    world.step()
```

```python

```
