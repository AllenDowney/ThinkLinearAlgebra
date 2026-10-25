# Chapter 3 Plan: Boids

**Status:** exploratory, 2026-10-24. Board: Task 7 (under Task 5). Nothing here is decided yet.

**Question:** could a chapter built around Boids fill slot 3, between Ch 2 (Track: vector addition and subtraction) and Ch 4 (Projection: dot product)? See `inventory.md` §4 and `design.md`.

---

## 1. The existing Boids code

**Sources:**
- `soln/boids.ipynb`: prose adapted from *Think Complexity* plus a copy of the code.
- The original is `~/ThinkComplexity2/code/Boids7.py` (the same file is in `~/ThinkComplexity/soln/`): "Original code by Matt Aasted, modified by Allen Downey", based on Reynolds (1987) and Flake, *The Computational Beauty of Nature*.

**Libraries:** VPython 7 (`from vpython import *`) for 3D rendering, its `vector` class, and mouse interaction; NumPy for random numbers and `np.mean`.

### Structure

- **Helpers**
  - `null_vector`
  - `random_vector(a, b)`: uniform in a cube
  - `limit_vector(v)`: caps the magnitude at 1 by setting `v.mag`
- **`class Boid(vp.cone)`**: a Boid *is* a VPython cone with `pos`, `vel`, and `axis`.
  - `get_neighbors(boids, radius, angle)`: the other Boids within `radius` whose offset is within `angle` of the Boid's velocity (`vel.diff_angle(offset)`), i.e. a field-of-view cone.
  - `vector_toward_center(vecs)`: `mean(vecs) - self.pos`, limited to magnitude 1.
  - Four rules, which Reynolds calls "acceleration requests":
    - `center` (cohesion): toward the centroid of neighbors in view (radius 1, angle 1 rad)
    - `avoid` (separation): away from the centroid of anything too close, Boids or carrot (radius 0.3, angle π)
    - `align`: toward the neighbors' mean velocity (radius 0.5, angle 1)
    - `love`: toward the carrot, limited to magnitude 1
  - `set_goal`: weighted sum `3·center + 10·avoid + 1·align + 10·love`, normalized to magnitude 1.
  - `move(mu=0.1, dt=0.1)`: `vel = (1-mu)·vel + mu·goal`, normalized to speed 1; `pos += dt·vel`; the cone's axis follows `vel`.
- **`class World`**: n Boids plus a red sphere "carrot". `step()` sets the goal and moves each Boid in turn; the carrot can follow the mouse.
- **Main loop:** 20 Boids, `scene.bind('click', toggle_tracking)`, then `while 1: rate(10); world.step()`.

### Issues to fix in a rewrite

1. **`align` mixes units.** It passes velocities to `vector_toward_center`, which subtracts `self.pos`, so the result is (mean velocity − position). **Decision (2026-10-24): use mean neighbor velocity minus own velocity ("steer")**: Reynolds' velocity matching, and the same "neighbors' mean minus mine" pattern as `center`. Implement it by passing the reference to the helper (`toward_mean(vecs, ref)`). Allen compared three versions with `plans/prototypes/boids7_align.py`:
   - `original` and `heading` (mean neighbor velocity) look the same;
   - `steer` looks more aligned;
   - `heading --w-align 5` has a few near collisions, then settles into a neater circle.

   With the default weights (align 1, avoid and love 10), align has little effect, which is probably why the bug went unnoticed.
2. **`avoid`'s docstring is wrong.** It says the magnitude is "proportional to the inverse of the distance", but the code just points away from the centroid with magnitude ≤ 1.
3. **Updates are sequential.** Each Boid moves before the next one computes its goal, so later Boids react to positions that have already changed. A synchronous update (compute every goal, then move everything) is cleaner, and it suits arrays.
4. **The code relies on VPython-specific vector features** (`.mag` setter, `.norm()`, `.diff_angle()`), cone objects, `scene.mouse`, and an infinite `while` loop. None of that carries over to a notebook animation.
5. **One Boid at a time, in Python loops**, which is O(n²) per step. That's fine for 20 Boids, but an array version is both faster and more interesting as linear algebra (see §2).

---

## 2. Topics Boids could demonstrate

Boids is mostly about **scaling and combining vectors**, which is exactly the gap between Ch 2 (add/subtract) and Ch 4 (dot product).

### Core topics (natural fit)

| Topic | Where it shows up in Boids |
|---|---|
| Scalar multiplication | Weights on the rules; `dt · vel` |
| **Linear combination** | `set_goal` is literally a weighted sum of four vectors, and changing the weights changes the behavior |
| Convex combination / interpolation | `(1-mu)·vel + mu·goal`: the new velocity lies on the segment between old velocity and goal; `mu` is maneuverability |
| Repeated interpolation | Applying the velocity update every step is exponential smoothing, which echoes the repeated multiplication of Ch 1 |
| Centroid (mean of vectors) | The `center` and `avoid` rules; the mean as a linear combination with weights 1/n |
| Norm, unit vector, normalize | Speed fixed at 1; goal normalized |
| Clipping a magnitude | `limit_vector`: scale down only if longer than 1 |
| Vector from A to B | `centroid − pos`, `carrot − pos` (review of Ch 2) |
| Arrays of vectors, broadcasting | Positions as an (n, 2) array; all pairwise offsets as `P[:, None] - P[None, :]` with shape (n, n, 2); all distances with `norm(..., axis=2)` |

### Stretch topics (possible, but they push into other chapters)

| Topic | Notes |
|---|---|
| **Neighbor averaging as a matrix product** | Build a 0/1 neighbor matrix `A` (who can see whom), row-normalize it to `W`, and the centroids of every Boid's neighbors are `W @ P`. So `center = W @ P - P` and `align = W @ V`. This connects to Ch 1 (adjacency and transition matrices), and makes "matrix times array of vectors" concrete. |
| **Angle between vectors / field of view** | Needs either the dot product (Ch 4's topic) or a 2D cross product plus dot, via `arctan2(cross, dot)`. The cross version gives a *signed* angle (target on my left or right), and introducing `cross2d` here would fill the hole Ch 5 currently has (`inventory.md` §4). Simplest option: use the dot product operationally, as Ch 1 did, and point forward to Ch 4 for why it works. |
| Local frame (forward, left) | Express a neighbor's offset in the Boid's own coordinates (ahead/behind, left/right) using `v_hat` and a perpendicular. That's a preview of change of basis, and it needs projection, so it's Ch 4 material. |
| Linearity | With the neighbor sets held fixed, `center` and `align` are linear in positions and velocities; `avoid` with distance weighting and the normalization steps are not. This is a nice "where linearity holds and where it breaks" discussion. |

---

## 3. Prep needed before getting into Boids

The chapter would start with these, then build Boids from them:

1. **Scalar multiplication and linear combinations.** Show the geometry (stretching, flipping, combining arrows) using the `plot_vector` helpers from Ch 2. A small motivating example could be mixing two velocities with weights, though Ch 2 already used plane plus wind.
2. **Norm and unit vectors.** Ch 2 already uses `norm` without explaining it, and Ch 4 defines norm and unit vectors. Moving those definitions here means editing Ch 4 to refer back.
3. **Mean of vectors as a linear combination**, and the centroid of a point cloud.
4. **Convex combinations and interpolation:** the segment between two vectors, parameterized by `mu`.
5. **Arrays of vectors and broadcasting.** Ch 5 introduces broadcasting; if Ch 3 does it first, Ch 5 can shrink that section.
6. **Pairwise differences and distance matrices**, if we use the array implementation.
7. **Decide how to handle angles:** dot product as a black box, the 2D cross product, or a field of view without angles (radius only, or a forward-only half-plane test).

### Effects on other chapters

- **Ch 4 (Projection):** it currently defines `norm`, unit vectors, and polar conversion. If Ch 3 takes norm and unit vectors, Ch 4 needs small edits.
- **Ch 5 (Affine):** it could drop or shorten its broadcasting section, and gains an earlier introduction of `cross2d` if Ch 3 covers the cross product.
- **Ch 1:** if Ch 3 uses the neighbor-matrix formulation, it can refer back to adjacency matrices.

---

## 4. Tools for running Boids in Colab

### Can VPython run in Colab now?

**Not yet, but it's getting close.** You remember right: the code uses VPython (`Boids7.py`, `soln/boids.ipynb`). As of 2026-10-24:

- The current release (vpython 7.6.5 on PyPI) does not work in Colab. VPython talks to the browser over a websocket that Colab's sandboxed output frames can't reach. Users have reported failures since 2019, and the only workaround in the vpython-users forum is "use mybinder instead".
- [vpython-jupyter PR #293](https://github.com/vpython/vpython-jupyter/pull/293), opened 2026-08-16 and **still a draft with no reviews**, adds a Colab front end that runs over Jupyter comms. Its author reports it working end to end in Colab, with `rate()` animation supported. Mouse features (`pick`, `waitfor`) don't work, though, so the click-to-track carrot wouldn't either, and the scene doesn't appear in a static Jupyter Book page.

So VPython might work in Colab in a future release, but we can't count on it, and even then it wouldn't render in the published book.

### Alternatives

| Option | Colab | Jupyter Book page | 3D | Interactive | Fit with the book |
|---|---|---|---|---|---|
| **Matplotlib `FuncAnimation` → `to_jshtml`** | Yes | Yes (pre-rendered frames with play controls) | 2D natural; 3D possible (mplot3d, slow) | Play/pause/scrub only | Excellent. Used for pool (Ch 4), *Enterprise* (Ch 5), track (Ch 2). `plot_vectors` can draw each Boid as its velocity arrow. |
| **Plotly animation frames** (`go.Scatter3d` / `go.Cone`) | Yes | Yes (self-contained HTML) | Yes, and the camera can rotate while it plays | Play/scrub, rotate, zoom | Good, but a new library and plotting style; frames are precomputed |
| **K3D** (anywidget) | Yes | Limited (needs a live kernel or a snapshot export) | Yes, WebGL | Live updates from Python | New dependency; less proven in static pages |
| **pythreejs / ipywidgets** | Unreliable | No | Yes | Yes | Not recommended (stale project, widget trouble in Colab) |
| **Custom anywidget / ipycanvas** | Yes | No | 2D (or hand-written three.js) | Real time | Too much JavaScript that isn't linear algebra |
| **Web VPython (glowscript.org)** | — (runs in the browser) | Link out | Yes | Full, including mouse | Good as an optional extra: "run the original 3D version here". Needs a GlowScript port of the code. |
| **Wait for VPython PR #293** | Maybe, later | No | Yes | Partial | Too uncertain to plan around |

### Requirements (Allen, 2026-10-24)

- Keep 3D.
- Vectors are **objects that provide vector behavior** (as VPython's `vector` does), not positions and velocities "compiled down" to NumPy arrays.

### Vector representation

VPython's `vector` class can't be used on its own: importing `vpython` starts the display machinery. The options:

| Option | Pros | Cons |
|---|---|---|
| **Our own `Vector` class** in `utils.py` (or written in the chapter) | Exactly the API Boids needs (`mag`, `hat`, `limit`, `diff_angle`, `dot`, `cross`), close to VPython's, so the `Boids7.py` code barely changes. No install. **Only vector operations are allowed**: `v + w`, `k * v`, `v.dot(w)`, `v.cross(w)`, and `v * w` raises "use dot() or cross()". Defining `__add__` and `__mul__` *is* defining what a vector is, which is a good lesson for a chapter about scaling and combining vectors. | One more home-grown utility; readers learn our API rather than a standard one. |
| **scikit-hep [`vector`](https://pypi.org/project/vector/)** (v1.9) | Mature and pip-installable. Real objects: `vector.obj(x=, y=, z=)`, `.mag`, `.unit()`, `.dot`, `.cross`, `.deltaangle`, `.rotateZ`, … The same API also works on *arrays* of vectors, so a later section could switch representations without changing the code. | Particle-physics vocabulary throughout (`rho`, `eta`, `pt`, momentum and Lorentz vectors), with about 100 methods. The default name `vector.obj(...)` is clunky. |
| **pygame-ce `pygame.math.Vector3`** | A very Boids-friendly API: `lerp` (which is exactly `(1-mu)·v + mu·g`), `clamp_magnitude`, `scale_to_length`, `angle_to`, `dot`, `cross`. | Importing a game library for its vector math is odd; heavy install (SDL); angles in degrees. |
| SymPy `sympy.vector` | Symbolic, like Ch 4 and 5. | Far too slow for a simulation. |

### Rendering 3D with Matplotlib

Yes, Matplotlib can do it. I built a prototype (now `plans/prototypes/boids3d.md`/`.ipynb`, using `utils.Vector`): 20 Boids, the original four rules with the `align` bug fixed, synchronous update, and a carrot on a 3D loop. Each Boid is drawn as a `quiver` arrow, using `FuncAnimation` + `to_jshtml`, and the camera orbits slowly to give a sense of depth.

- Simulation: about 2 s for 150 steps in pure Python with the `Vector` class.
- Rendering: about 12 s; **4.6 MB** of HTML at 72 dpi (7.4 MB at the default resolution). For comparison, `enterprise.html` is 1 MB. Fewer frames, a smaller figure, or MP4/GIF output for the book page would shrink it.
- The flock clearly forms and follows the carrot.
- Limits: no mouse rotation during playback (the orbiting camera stands in for it); mplot3d's depth sorting is approximate; arrows need tuning to read well at small sizes.

### Recommendation

**Our own small `Vector` class plus Matplotlib 3D animation.**

- The `Vector` class keeps the code reading like vector math and close to the original `Boids7.py`. The class itself becomes chapter content: what operations make something a vector, why `v * w` is not one of them, and what a linear combination is. No new dependencies.
- Matplotlib 3D runs in Colab and appears in the published book, and it matches the other animations (design goal G7). Watch the file size.
- Replace the mouse-driven carrot with one on a scripted path; that also makes the animation reproducible.
- Optional extras:
  - an interactive Plotly version (rotate while it plays) as an exercise or a bonus cell;
  - a "see also" link to a Web VPython version of the original;
  - a closing section or exercise on the array formulation (neighbor matrix, `W @ P`), as a contrast with the object version.
- Runner-up: scikit-hep `vector`, if we'd rather use an existing library than write one.
- Revisit VPython if PR #293 is merged and released.

---

## 5. Open questions

1. Is Boids the centerpiece of Ch 3, or one application in a chapter about linear combinations, alongside others (e.g. correlation as a dot product from `penguins`, or color mixing)?
2. Should Ch 3 handle angles with the dot product (a preview of Ch 4) or with the 2D cross product (which fills Ch 5's gap)?
3. Include the neighbor-matrix formulation (`W @ P`)? It's elegant and links to Ch 1, but it may be too much for an early chapter. It could be an exercise or a closing section instead.
4. How much of Ch 4's material (norm, unit vectors) moves to Ch 3?
5. ~~2D only, or 2D in the chapter plus a 3D exercise?~~ Decided: 3D, with vector objects.
6. Does the chapter build the `Vector` class in front of the reader, or import it from `utils.py` and show only the interesting parts?
7. Do later chapters keep using `Vector`, or does the book move to NumPy arrays from Ch 4 on? (The other chapters all use arrays.)

## 6. Next steps (if we go ahead)

- [ ] Allen decides on the questions in §5
- [x] Prototype 3D Boids on `utils.Vector` with a synchronous update and a Matplotlib animation (`plans/prototypes/boids3d.ipynb`, 2026-10-24): runs in about 20 s locally, and polarization climbs to about 1
- [ ] Test it in Colab
- [ ] Draft the chapter outline (sections, examples, exercises)
- [ ] List the edits needed in Ch 4 and Ch 5
