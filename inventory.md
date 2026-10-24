# Think Linear Algebra: Inventory and Gap Analysis

Written 2026-10-24 from the jupytext versions in `soln/md/` (regenerated the same day). Purpose: see what the eight published chapters cover, what they assume, and what is missing, so we can choose the remaining chapters (Task 2).

Chapter numbers follow the README, which leaves slots 3 and 6 open:

| # | File | Title |
|---|------|-------|
| 1 | eigenvector | The Power of Linear Algebra |
| 2 | track | A Day At the Track |
| 3 | — | *open* |
| 4 | projection | Projection |
| 5 | affine | To Boldly Go |
| 6 | — | *open* |
| 7 | system | Systems |
| 8 | nullspace | Null Space |
| 9 | truss | Truss in the System |
| 10 | regression | Least Squares Regression |

`chapters/centrality.ipynb` is an older copy of Chapter 1, not a separate chapter.

---

## 1. Chapter by chapter

### Ch 1. The Power of Linear Algebra (eigenvector)

- **Application:** random walk in the Gardner Museum (6 rooms); PageRank on Wikispeedia (4,604 pages). Exercise: MFA Boston room graph (Yoshimura et al. 2024).
- **Introduces:** array vs. matrix, adjacency matrix, transition matrix, elementwise product, dot product (as sum of products, operational only), matrix-vector product (row picture), repeated multiplication (power iteration), eigenvector and eigenvalue, linearity, directed vs. undirected graphs, Google matrix.
- **Tools:** NetworkX (`to_numpy_array`, `google_matrix`, `pagerank`), `@`, `np.dot`, `scipy.linalg.eig`, `scipy.sparse.linalg.eigs`, `np.allclose`.
- **Promises:** "we'll come back to these topics in future chapters". No later chapter goes back to eigenvectors in depth (Ch 10 uses eigenvalues once, to get a condition number).

### Ch 2. A Day At the Track (track)

- **Application:** GPS track of a run; distances on a sphere; velocity and acceleration by differencing; dead reckoning from acceleration with noise. Exercises: honeybee map-like navigation, Boston–London chord vs. great circle, plane plus jet stream, spline smoothing.
- **Introduces:** vector addition and subtraction; three pictures of a vector (point, arrow from origin, arrow anywhere); head-to-tail addition; spherical-to-Cartesian conversion; norm as distance; scaling vector (elementwise); free vs. bound vectors; numerical differentiation (`diff`) and integration (`cumsum`); centripetal acceleration.
- **Tools:** `gpxpy`, `haversine`, `np.diff`, `np.cumsum`, `np.vstack`, `splprep`/`splev`, matplotlib animation. Uses `norm`, `sph2cart`, `pol2cart`, `cart2pol` from `utils`.
- **Note:** uses `norm`, `pol2cart` and `cart2pol` before Ch 4 defines them. The text treats `norm` as obvious; the jet-stream exercise relies on the polar functions.

### Ch 4. Projection (projection)

- **Application:** pool-ball collisions (collision prediction, elastic collision, pocketing), with animation. Exercise: unequal masses.
- **Introduces:** norm (2-norm), polar coordinates, scalar and vector projection, unit vectors, dot product derived symbolically, $a \cdot b = \|a\|\|b\|\cos\theta$, orthogonality, vector rejection, rotation matrix derived with SymPy, perpendicular vector (90° rotation), normal and tangent vectors.
- **Tools:** `numpy.linalg.norm`, `arctan2`, `hypot`, SymPy (`symbols`, `Matrix`, `simplify`, `expand_trig`, `subs`), `minimize_scalar`, `root_scalar`.

### Ch 5. To Boldly Go (affine)

- **Application:** contour of the *Enterprise* from a photo; scale, shear, rotate, translate; orbit from an ODE solver; Kepler's second law; animated ship in orbit. Exercises: italics by shear, Asteroids (SVEC shapes with rotation), Star Wars crawl (projective transform).
- **Introduces:** broadcasting, row vs. column vectors, transpose, scale/shear/rotation matrices, rotation matrix built from columns (basis vectors $\hat{i}, \hat{j}$ go to the columns), matrix-matrix product as composition, non-commutativity, homogeneous coordinates, translation matrix, augmented 3×3 matrices, affine transform, 2D cross product as parallelogram area (used, not explained), angular momentum, projective transform (homography, perspective divide).
- **Tools:** scikit-image (`imread`, `rgb2gray`, `find_contours`), shapely (`simplify`), `np.roll`, `np.hstack`, `solve_ivp`, `TextPath`, `cross2d` from `utils`.
- **Note:** `cross2d` appears without an introduction ("the area of that parallelogram is the cross product"), so the cross product needs to come before Ch 5.

### Ch 7. Systems (system)

- **Application:** school-play ticket word problem; circuit analysis built from a NetworkX graph (voltage divider, R2R ladder DAC). Exercises: numerical R2R solution, quarter bridge (strain gauge).
- **Introduces:** system of equations as $Ax = b$, coefficient matrix, augmented matrix, row operations, Gauss-Jordan elimination, identity matrix, matrix inverse (via Gauss-Jordan on $[A \mid I]$), decomposition, LU decomposition (implemented), triangular solves, numerical stability (brief), LAPACK.
- **Tools:** SymPy (`linsolve`, `linear_eq_to_matrix`, `row_join`, `inv`, `LUdecomposition`, `LUsolve`, `solve`), `np.linalg.solve`, `scipy.linalg.lu`, `lu_factor`, `lu_solve`.

### Ch 8. Null Space (nullspace)

- **Application:** balancing chemical reactions: ethanol combustion (1-D null space), permanganate plus peroxide (2-D null space, physically meaningful combinations). Exercise: methanol.
- **Introduces:** homogeneous system, underdetermined system, null space, basis vector, linear combination of solutions, rank, nullity, rank-nullity theorem.
- **Tools:** SymPy `nullspace`, `sp.lcm`/`sp.denom`, `scipy.linalg.null_space`.

### Ch 9. Truss in the System (truss)

- **Application:** popsicle-stick truss with two bars; Warren truss with N triangles; displacements under load; units with `pint`.
- **Introduces:** spring constant ($k = AE/L$), internal vs. external force, equilibrium, projector matrix $\hat{r}\hat{r}^T$, stiffness matrix, force field, block matrices and their assembly, rank deficiency as rigid-body modes (translate x, y, rotate), degrees of freedom, reducing the matrix by fixing nodes, full rank.
- **Tools:** SymPy `BlockMatrix`, `MatrixSymbol`, `as_explicit`, `rank`, `nullspace`; `scipy.linalg.solve`; `pint`; `gset`/`gget`, `factor_matrix`, `sympy_to_numpy` from `utils`.
- **Note:** says projection matrices were "seen in Section XXX", but this chapter is where they are introduced.

### Ch 10. Least Squares Regression (regression)

- **Application:** three-point toy example; GSS political views vs. age, cohort, year; quadratic model; age-period-cohort analysis. Exercise: centering and scaling predictors.
- **Introduces:** design matrix, residuals, RSS as $r \cdot r$, regression by minimization, column space and span (as "the plane"), least squares as projection, QR decomposition, orthonormal matrix, projection matrix $QQ^T$, Gram-Schmidt (implemented, with R), triangular solve, rank of the design matrix, collinearity, condition number, normal equations ($X^TX\beta = X^Ty$), $\kappa(X^TX) = \kappa(X)^2$, condition number from eigenvalues of $X^TX$.
- **Tools:** `scipy.optimize.minimize`, `np.linalg.qr`, `solve_triangular`, `scipy.linalg.lstsq` (gelsy), `np.linalg.cond`, `scipy.linalg.eig`, 3D plotting from `utils`.
- **Promises:** "Another way to find the condition number is to compute the singular values of `X`, but that's a topic for another chapter."

---

## 2. Concept coverage

**I** = introduced, **U** = used, **u** = used briefly or without explanation.

| Concept | 1 | 2 | 4 | 5 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| Vector add / subtract | u | **I** | U | U | | | U | U |
| Scalar multiple, linear combination | u | u | U | U | | **I** (solutions) | | U |
| Norm / length | | u | **I** | U | | u | U | U |
| Unit vector, normalize | | | **I** | U | | | U | U |
| Dot product | **I** (operational) | | **I** (geometric) | | | | U | U |
| Projection, rejection | | | **I** | | | | U (projector) | U |
| Orthogonality | | | **I** | | | | | U |
| Matrix-vector product | **I** | | U | U | U | U | U | U |
| Matrix-matrix product, composition | | | | **I** | u | | | U |
| Transpose | | u | | **I** | | | U | U |
| Rotation matrix | | | **I** | U | | | | |
| Scale / shear / affine / homogeneous | | | | **I** | | | | |
| Basis vectors / change of basis | | | | u | | **I** (null space basis) | | u (span of Q) |
| Cross product / area | | | | u | | | | |
| Determinant | | | | | | | | |
| Systems, $Ax=b$, Gauss-Jordan | | | | | **I** | U | U | U |
| Inverse, identity | | | | | **I** | | | u |
| LU decomposition | | | | | **I** | | | u |
| Rank | | | | | | **I** | U | U |
| Null space, rank-nullity | | | | | | **I** | U | |
| Block matrices | | | | | | | **I** | |
| Least squares, QR, Gram-Schmidt | | | | | | | | **I** |
| Condition number | | | | | u (stability) | | | **I** |
| Eigenvectors / eigenvalues | **I** | | | | | | | u |
| Markov chain / stationary distribution | **I** (informally) | | | | | | | |
| SVD | | | | | | | | mentioned |
| PCA, covariance matrix | | | | | | | | |
| Symbolic vs. numerical (SymPy / NumPy) | u | | **I** | U | U | U | U | U |

---

## 3. Gaps compared with a standard first course

### Covered well

Vectors and their operations, norm, dot product and projection, orthogonality, linear and affine transformations, composition, systems and elimination, inverse, LU, rank, null space, rank-nullity, least squares, QR and Gram-Schmidt, condition number. The applications are varied and physical (graphs, GPS, pool, graphics, circuits, chemistry, structures, survey data).

### Thin

- **Linear combinations, span, basis, linear independence.** These ideas are used throughout but never get a section of their own. "Span" first appears in Ch 10. Linear independence comes up only as rank.
- **Matrix-vector product, column picture.** Ch 1 gives the row picture (dot products). Ch 5 hints at the column picture (rotation matrix columns are where the basis vectors land) but doesn't name it.
- **Change of basis / coordinate systems.** Ch 5's rotation matrix is close. The archived `geography` notebook has an East-North-Up basis built with cross products.
- **Eigenvectors after Ch 1.** Ch 1 introduces them quickly and promises to come back. Nothing returns to them: no eigendecomposition, no diagonalization, no symmetric matrices, no "why power iteration works" beyond the linearity sketch.

### Missing

- **Determinant.** Never appears. It is the natural companion to the 2D cross product, area scaling under a transformation, orientation, and invertibility.
- **Cross product (3D) as a topic.** Ch 5 uses `cross2d` with no introduction.
- **SVD.** Promised in Ch 10.
- **PCA and the covariance / correlation matrix.** These are the most-used linear algebra ideas in data science, and the material exists in draft (`correlation`, `penguins`).
- **Symmetric / positive-definite matrices, Cholesky.** Only in the `penguins` draft.
- **Linear dynamical systems** (repeated application of a matrix, e.g. Leslie population models, Fibonacci, coupled oscillators), beyond the Markov chain in Ch 1.

These are probably out of scope by design: abstract vector spaces, proofs, complex vector spaces, Jordan form.

---

## 4. Dependencies that shape the open slots

**Slot 3 (between Track and Projection).** Ch 2 covers addition and subtraction. Ch 4 assumes scalar multiplication, unit vectors, norm, and polar conversions; it defines norm and polar coordinates itself, though Ch 2 already used them. Ch 5 needs the 2D cross product. A Ch 3 could:

- cover scalar multiplication, linear combinations, span and basis (the missing foundations), and
- introduce the 2D cross product / determinant as signed area, which Ch 5 uses with no introduction.

**Slot 6 (between Affine and Systems).** Ch 5 ends with transformations. Ch 7 opens with a word problem and "solve $Ax = b$". A bridge chapter could ask "how do we undo a transformation?" or "which transformation maps these points to those?". That leads to inverses, the determinant as area scaling and an invertibility test, and setting up a linear system from data. Ch 7 would then pick up the algorithms.

**After Ch 10.** Two forward references (Ch 1 on eigenvectors, Ch 10 on singular values) point to a chapter that doesn't exist yet. An eigen/SVD/PCA chapter fits naturally after regression, because it builds on orthogonality, projection, and the condition number. That would make the book 11 chapters, or Ch 10's SVD sentence would need rewording.

---

## 5. Candidate material in `soln/`

| Notebook | State | Linear algebra content | Fits |
|---|---|---|---|
| **boids** | Prose and code adapted from *Think Complexity*; uses VPython (3D, interactive). Needs a rewrite for NumPy/matplotlib and a linear algebra framing. | Vector add/subtract, scaling, norm, centroid (mean of vectors), normalize, angle between vectors (field of view), **weighted sum of vectors** (`set_goal` is a linear combination of four "acceleration requests"). | Slot 3: linear combinations come naturally. |
| **penguins** | Code only, no prose (2024). | Standardization, correlation as a dot product of standardized vectors, correlation and covariance matrices as $Z^TZ/n$, `eig` of the correlation matrix, power iteration on it, Cholesky to generate correlated data, transforming a circle into an ellipse. | Dot product application (slot 3 or an extension of Ch 4); covariance/Cholesky part goes with PCA. |
| **correlation** ("The Principal Components of Anthropometry") | Code plus some interpretation of the loadings; PCA with sklearn, then repeated with SVD. | PCA, explained variance, loadings, SVD. | After Ch 10 (eigen/SVD/PCA chapter). |
| **ansur_corr** | From *Probably Overthinking It*; correlation exploration, mostly statistics. | Correlation matrix. | Data source for the PCA chapter, not a chapter in itself. |
| **spurious_correlation** | Very short: correlation as the mean product of standardized series. | Dot product. | An example for a dot-product section. |
| **cross** | Placeholder: one 3D cross-product plot and a 2D angle-between using cross plus dot. | Cross product. | Slot 3 (if it covers cross product / determinant). |
| **trilateration** | Code only: linearized least squares, then Gauss-Newton refinement. | Least squares, Jacobian, normal equations. | After Ch 10 (needs least squares), or as a Ch 10 exercise. |
| **geography** (archival) | Leftovers from Ch 2: ENU local basis by cross products, projection onto a basis, OpenSky flights. | Change of basis, 3D cross product, orthonormal basis. | Source material for a change-of-basis section. |
| **pagerank** (archival) | Absorbed into Ch 1. | — | — |

---

## 6. Loose ends noticed in the published chapters

Small fixes to batch into the final polish (Task 2 publish step):

- **Ch 1:** "end up in the library" should be the Lobby; "matrixes"; exercise step 3 says "eigenvector of the adjacency matrix" but means the transition matrix. The museum section normalizes columns, while the PageRank section normalizes rows and uses `GM.T`; a sentence explaining the switch would help.
- **Ch 2:** "In Chapter xxx, we will work with bound vectors".
- **Ch 4:** Colab link points at `nb/projection.ipynb`, not `chapters/`; "18 inches per second" but `r=16`; "after 4 seconds" but `t = 3`; "root_scale".
- **Ch 5:** "as we did in Section xxx"; says `coords` has 67 rows but the text elsewhere says 59; "scaled, shifted, and rotated" should be "scaled, sheared, and rotated"; the Discussion is still notes; the Exercises section starts with three placeholder bullets.
- **Ch 7:** "SciPy provides a function" and "SciPy matrices have a method" should be SymPy (the code uses `A.inv()` and `A.LUdecomposition()`); "use row 3" should be row 2.
- **Ch 8:** the Discussion says the next chapter covers overdetermined systems, but the next chapter is Truss; the methanol exercise still has ethanol comments; "each molecule of $H_2$" should be $H_2O$.
- **Ch 9:** "projection matrices, which we saw in Section XXX" (they are introduced here); "rather than all of it on `B0`" should be `B1`.
- **Ch 10:** "an orthonormal matrix is its own inverse" is wrong. Its *transpose* is its inverse, and here $Q$ isn't square, so only $Q^TQ = I$ holds. Also "Gram-Schmitt", "colinear" vs. "collinear".
