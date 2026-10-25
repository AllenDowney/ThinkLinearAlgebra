# Chapter 6 Plan: Frames of Reference

**Status:** planning started 2026-10-24. Board: Task 8.

**Decided (Allen, 2026-10-24):**
- Frames is Chapter 6, between Ch 5 (Affine) and Ch 7 (Systems).
- The graph of frames uses NetworkX.
- Composed and inverse transforms are cached as edges ("once you've computed it, you might as well save the shortcut").
- A moving linkage is a likely example, and it should be animated.

---

## 1. The existing code

**Source:** `soln/Frame.py`, copied unchanged from `~/ThinkPython2/python_thinking/Frame.py`. © 2012, GPLv3. It runs under Python 3; there's no prose.

### Structure

- **`FrameError(ValueError)`**: raised for frame mismatches.
- **`Vector(array, frame)`**: coordinates plus the frame they're relative to (`None` is the root frame O).
  - `__add__`: requires the same frame.
  - `from_list`.
  - `__str__` prints Craig-style notation, `^{A}[x y z]`.
- **`Rotation(array)`**: a 3×3 matrix.
  - `from_axis(axis, theta)`: axis-angle to matrix (Rodrigues formula, written out by components).
  - `transpose()`, with `inverse = transpose`.
  - `__mul__` applies the rotation to a `Vector` or a `Rotation`.
  - `to_axis` is a stub.
- **`Transform(rot, org, source)`**: maps vectors from `source` to `org.frame`, where `org` is the source origin expressed in the destination frame.
  - `mul_vector`: `R p + org`, with the frame checked.
  - `mul_transform`: composition; requires this transform's source to be the other's destination.
  - `inverse()`: `(Rᵀ, -Rᵀ org)`.
  - The constructor registers the transform with its source frame.
- **`Frame(name)`**: a name plus a dict mapping a destination frame to a `Transform`. A class-level `roster` lists every frame.
- **`Vertex`, `shortest_path`**: a hand-written shortest path over frames. Following a stored transform costs 0, and following one backward (which needs an inverse) costs 1.
- **`main`**: frames O → A → B → C, each rotated 90° about one axis and offset by one unit. It maps a point from C to O step by step, then composes the transforms, inverts the result, and reprints the shortest paths. They change because the composed and inverted transforms were added as edges.

### Issues to fix in the rewrite

1. **Sign bug in `Rotation.from_axis`:** `s = sqrt(1 - c*c)` is never negative, so θ and −θ give the same matrix. (Checked: rotating x̂ by −π/2 about ẑ gives +ŷ.) Use `s = sin(theta)`.
2. **License:** GPLv3, but the book's code is MIT. Allen owns the copyright and can relicense.
3. **Side effects in constructors:** creating a `Transform` changes the frame graph, and `Frame.roster` is global. In a notebook, re-running cells silently adds or overwrites edges.
4. **`Vector` supports only `+`.** It will be replaced by the shared `Vector` class (Ch 3 / Task 7), which has subtraction, scalar multiplication, dot, cross, and so on, plus an optional `frame`.
5. **`Rotation * Rotation` returns a bare array** (which callers wrap); `to_axis` is unimplemented.
6. **The hand-written shortest path** gets replaced by NetworkX with edge weights.
7. **Python 2 leftovers** (`from __future__`, `println`).

---

## 2. Topics the chapter could teach

| Topic | Notes |
|---|---|
| **Coordinates are relative to a frame** | The same point has different coordinates in different frames; adding vectors from different frames is a type error, which `FrameError` makes concrete. |
| **Change of basis** | A frame's axes, expressed in the parent frame, are the columns of the rotation matrix. Ch 5 showed this in 2D (`rotation_matrix` from `v_hat` and `v_perp`). This fills the change-of-basis gap in `inventory.md` §3. |
| **3D rotation** | Axis-angle and the Rodrigues formula; rotation about x, y, z; non-commutativity in 3D (rotate a book 90° about two axes in both orders). |
| **Orthogonal matrices** | Columns are orthonormal, so `R.T @ R = I` and the inverse is the transpose. This connects forward to Q in Ch 10. Possibly mention det = +1 (rotation) vs −1 (reflection). |
| **Rigid transforms** | Rotation plus translation, as a pair `(R, t)` or as a 4×4 homogeneous matrix (the 3D version of Ch 5's 3×3). |
| **Composition** | `T_AO ∘ T_BA ∘ T_CB = T_CO`: associativity in action. |
| **Inverse of a rigid transform** | `(Rᵀ, -Rᵀ t)`. This is the first inverse in the book that we can compute by thinking, a lead-in to Ch 7's general inverse. |
| **Graph of frames** | Frames are nodes and known transforms are edges; finding a transform between two frames is a path search (NetworkX, which ties back to Ch 1). |
| **Caching derived transforms** | Save composed and inverse transforms as edges; invalidate them when a frame moves (§4). |

---

## 3. Applications

### Main example: an animated linkage

**A planar robot arm (open chain), 2–3 links.**
- Each link has its own frame. The transform from link *i* to link *i−1* is a rotation by joint angle θᵢ, then a translation by the link length.
- **Forward kinematics:** where is the gripper, in world coordinates? Compose the transforms along the chain.
- **Animate:** vary the joint angles over time and redraw the arm. Each frame's transform changes on every animation frame, so this example *forces* the cache-invalidation design (§4). That makes the chapter more honest about caching.
- It can start in 2D, using the 3×3 homogeneous transforms from Ch 5, then go to 3D: a base that rotates about the vertical axis plus a planar arm makes a simple 3-DOF arm.
- **Visual:** draw each link's frame axes as small arrows (`plot_vectors` in 2D, `quiver` in 3D) to show the frames moving with the links.

**Stretch: a four-bar linkage (closed chain).**
- A classic mechanism (windshield wipers, bicycle suspensions, crank-rockers). The loop must close, which is a constraint, so the joint angles aren't independent.
- Solving the constraint is nonlinear: `root_scalar` / `fsolve` (as in Ch 4), or linearize it, which could foreshadow Ch 7.
- It gives a great animation of the coupler curve. Maybe an exercise.

### Other candidate examples

- **East-North-Up frame** (from the archived `soln/geography.ipynb`): build a local frame at the GPS track's location from Earth-centered coordinates, using cross products for the axes. Express the run (Ch 2) in local coordinates and compare with Ch 2's flat approximation.
- **Aircraft attitude:** body frame vs. world frame (yaw, pitch, roll). It could use the OpenSky flight data idea, but credentials make that awkward for readers.
- ***Enterprise* in 3D:** the Ch 5 orbit with the ship's frame in 3D. A callback, but probably redundant.
- **Camera frame:** project world points into a camera's view (connects to Ch 5's projective transform). Possibly too much.

---

## 4. Design of the frame graph (NetworkX)

```text
G = nx.DiGraph()
G.add_edge('A', 'O', transform=T_AO, kind='primary')          # given
transform_between(G, 'C', 'O')                                # finds path C→B→A→O,
                                                              # composes, and caches:
G.edges['C', 'O'] == {transform: T_CO, kind: 'derived',
                      via: [('C','B'), ('B','A'), ('A','O')]}
set_transform(G, 'B', 'A', new_T_BA)    # a joint moved:
                                        # replaces the primary edge and deletes every
                                        # derived edge whose `via` includes ('B','A')
```

- **Primary edges** are given; **derived edges** are computed (composed or inverted) and record the path they came from (`via`).
- **`transform_between(G, a, b, cache=True)`** uses `nx.shortest_path` with weights: following an edge forward costs 0, and following one backward (which needs an inverse) costs 1, as in the original. An inverse can also be cached as a derived edge.
- **`set_transform`** updates a primary edge and invalidates the derived edges that depend on it. This is what makes caching safe for the moving linkage.
- **No side effects in constructors:** `Transform(...)` is a value; only these functions change the graph.
- An alternative to discuss: cache only within one time step (clear all derived edges each animation frame). That's simpler, but it shows less.

### Relationship to the shared `Vector` class

- `Vector` (Ch 3) has an optional `frame` attribute that Ch 3 ignores. In Ch 6, `+`, `-`, `dot`, and `cross` raise `FrameError` when frames differ, and `Transform` changes a vector's frame.
- `Rotation` and `Transform` could be classes in the chapter, or in `utils.py` if a later chapter needs them.

---

## 5. Prerequisites and effects on other chapters

- **From Ch 3:** the `Vector` class (operations; the `frame` attribute introduced here).
- **From Ch 4:** the 2D rotation matrix, derived with SymPy. Ch 6 can derive the 3D rotation about each axis the same way.
- **From Ch 5:** homogeneous coordinates and composition of transforms in 2D, and the rotation matrix built from basis vectors. Ch 6 is the 3D version of all of these.
- **3D cross product:** needed for building frames from two directions (ENU, or a "look-at" frame). Not covered yet: Ch 3 might introduce the 2D cross product; Ch 6 could introduce the 3D one.
- **Ch 7:** opens with general systems and inverses. Ch 6 provides the special case where the inverse is easy (orthogonal R), which motivates "what about matrices in general?".
- **Ch 9 (Truss):** rigid-body motion (shift x, shift y, rotate) appears as the null space of the stiffness matrix. Ch 6 makes "rigid motion" a familiar idea before then.
- **Order check:** Ch 5's Discussion section is still notes; it could end with a pointer to Ch 6.

---

## 6. Open questions

1. 2D first, then 3D, or straight to 3D?
2. Represent transforms as `(R, t)` pairs (as in `Frame.py`) or as 4×4 homogeneous matrices (as Ch 5 does in 3×3)? Maybe both, showing they're equivalent.
3. Keep Craig's notation (${}^{A}p$, ${}^{A}_{B}T$)? It's precise but heavy for this audience.
4. Main example: the planar arm, the four-bar linkage, or the arm with the four-bar as an exercise?
5. Is ENU / geography in this chapter, an exercise, or dropped?
6. Inverse kinematics (where should the joints be to put the gripper *here*?) is the natural next question. It's nonlinear in general; mention it, make it an exercise, or skip it?

## 7. Next steps

- [ ] Allen decides on the questions in §6
- [ ] Rewrite `Frame.py` on the shared `Vector` class, with NetworkX and derived-edge caching; fix the sign bug
- [ ] Prototype the animated planar arm (2D, then 3D), with frames drawn on each link
- [ ] Draft the chapter outline
