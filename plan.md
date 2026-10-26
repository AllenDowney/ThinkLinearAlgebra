# Think Linear Algebra - Project Status

## Overview

This document tracks the status of the Think Linear Algebra book project.

- **soln/** - Contains the working/development version of all notebooks
- **soln/md/** - Markdown versions of notebooks (converted via jupytext for easier reading)
- **chapters/** - Contains copies of notebooks that are substantially complete
- **jb/** - Contains Jupyter Book configuration and chapters ready for publication

## Completed Chapters (in publication order)

These chapters are substantially complete and included in the Jupyter Book (`jb/_toc.yml`):

1. **eigenvector** (887 lines)
   - Status: ✅ Complete
   - Title: "The Power of Linear Algebra"
   - Topics: vectors, matrices, PageRank algorithm, eigenvectors

2. **projection** (1154 lines)
   - Status: ✅ Complete
   - Title: "Projection"
   - Topics: dot product, projection, Cartesian/polar coordinates, rotation matrices

3. **affine** (1647 lines)
   - Status: ✅ Complete
   - Title: "To Boldly Go"
   - Topics: affine transforms, image processing, homogeneous coordinates, orbits

4. **system** (996 lines)
   - Status: ✅ Complete
   - Title: "Systems"
   - Topics: systems of equations, matrix equations, Gauss-Jordan elimination

5. **nullspace** (534 lines)
   - Status: ✅ Complete
   - Title: "Null Space"
   - Topics: systems with many solutions, stoichiometry, chemical reactions

6. **truss** (1021 lines)
   - Status: ✅ Complete
   - Title: "Truss in the System"
   - Topics: structural analysis, projection matrices, block matrices

7. **regression** (1460 lines)
   - Status: ✅ Complete
   - Title: "Least Squares Regression"
   - Topics: QR decomposition, orthogonality, multiple collinearity, GSS data

8. **track** (~960 lines)
   - Status: ✅ Complete
   - Title: "A Day At the Track"
   - Topics: vector addition/subtraction, GPS data analysis, coordinate transformations, numerical differentiation/integration

## Other Notebooks in Development (soln/)

These notebooks are at various stages of completion:

### Substantial Content

- **boids** (453 lines)
  - Status: 🔄 In progress
  - Title: "Boids"
  - Has implementation, needs more explanatory content

- **pagerank** (405 lines)
  - Status: 📦 Archival
  - Title: "PageRank"
  - Material subsumed into Chapter 1 (eigenvector); will not become a separate chapter

- **penguins** (397 lines)
  - Status: 🔄 In progress
  - Title: "Linear Algebra with Penguins"
  - Has data analysis content

### Early Stage / Experimental

- **correlation** (317 lines)
  - Status: 🔨 Early stage
  - Title: "The Principal Components of Anthropometry"
  - Has some content, needs expansion

- **cross** (256 lines)
  - Status: 🔨 Early stage
  - Title: "Title" (placeholder)
  - Minimal content

- **ansur_corr** (248 lines)
  - Status: 🔄 In progress
  - Title: "Correlated Body Parts"
  - ANSUR dataset analysis

- **spurious_correlation** (172 lines)
  - Status: 🔨 Early stage
  - Title: "Correlation"
  - Minimal content

- **trilateration** (155 lines)
  - Status: 🔨 Early stage
  - Title: "Trilateration"
  - Has "TODO: intro" placeholder

- **geography** (~400 lines)
  - Status: 📦 Archival
  - Title: "Geography"
  - Material spun out from track chapter: 3D coordinate transformations, Earth radius calculations, local ENU coordinate systems, cross products for basis vectors, projection matrices, spherical coordinates diagrams, OpenSky flight data exploration
  - May be revisited for a future advanced chapter on 3D geometry

### Utility Files

- **template** (61 lines)
  - Template notebook for starting new chapters

- **test_notebook** (83 lines)
  - Test file for PyCon 2025 tutorial (not part of book)

## Tools

- **soln/convert_to_md.py** - Script to convert all .ipynb notebooks to markdown format using jupytext
  - Creates markdown versions in `soln/md/` subdirectory
  - Makes notebooks easier to read compared to JSON format
  - Usage: `python convert_to_md.py` (from soln/ directory)

- **soln/clean_gpx_files.py** - Script to prepare GPX files for sharing
  - Processes GPX files with capitalized names (e.g., `Morning_Run.gpx`)
  - Creates cleaned versions with lowercase names (e.g., `morning_run.gpx`)
  - Removes metadata and extensions for privacy, keeping only lat/lon/elevation/time
  - Usage: `python clean_gpx_files.py` (from soln/ directory)

- **soln/download_flight_data.py** - Script to download airplane flight data from OpenSky Network
  - Uses the traffic library to query flight trajectories
  - Downloads flights between specific airports (e.g., Boston to Dublin)
  - Saves trajectory data as CSV for analysis in track.md
  - Requires free OpenSky account (https://opensky-network.org/)
  - Usage: Configure credentials and run `python download_flight_data.py` (from soln/ directory)

## Statistics

- **8 chapters** in publication order (all complete!)
- **10 additional notebooks** at various stages of development
- **2 utility files** (template, test)

Total: 18 notebooks in soln/ (excluding utilities)


## Next Steps

- Review `utils.py` for redundant helper functions that should be combined (PROJECT_BOARD.md, Task 10).

### track: "A Day At the Track" ✅ COMPLETE

Published as Chapter 2. Added to jb/_toc.yml, jb/index.md, and both build scripts.


### Chapter 3: Multiplication

One possible idea for this chapter is to organize around multiplication operations.

#### Current Coverage of Dot Product

**Chapter 1 (eigenvector)**: Brief introduction to dot product (lines 957-960) as "sum of elementwise products", then immediately extends to matrix-vector product (lines 1017-1018). This is a quick operational intro with no geometric motivation.

**Chapter 4 (projection)**: Extensive coverage of dot product:
- Derives scalar projection from polar coordinates using trig
- Uses SymPy to simplify to the dot product formula
- Shows the geometric interpretation: a·b = ‖a‖‖b‖cos(θ)
- Discusses orthogonality (dot product = 0)
- Uses pool ball collision as motivating example

**Chapter 10 (regression)**: Uses dot product for residual sum of squares and checking perpendicularity of residuals to columns of X.

#### Current Coverage of Cross Product

**Chapter 5 (affine)**: Uses 2D cross product (`cross2d`) for:
- Computing area of parallelograms
- Computing angular momentum of orbits

**soln/geography** (archival): Uses 3D cross product to create ENU basis vectors for local coordinate systems.

**soln/cross.md**: Early stage - just a 3D visualization of cross product.

#### Suggested Topics for Chapter 3

**Dot Product Applications:**
- **Pearson correlation**: soln/penguins already has content showing how dot product of standardized vectors computes correlation coefficient (r = dot(z_x, z_y) / n). Could use penguin data as example.
- **Weighted sums and linear combinations** as dot products
- **Similarity measures**: cosine similarity for text/vectors
- **Work**: W = F · d (dot product of force and displacement - contrast with cross product)

**Matrix-Vector Multiplication:**
- **Column picture**: linear combination of matrix columns
- **Row picture**: dot products with each row  
- Build intuition before chapter 4's heavy use of matrix-vector ops

**Cross Product Applications (if included):**
- **Area of parallelogram/triangle**: already in chapter 5, could introduce earlier
- **3D perpendicular vectors**: needed for 3D coordinate systems
- **Angular momentum**: physics application
- **Torque**: τ = r × F (cross product of position and force vectors)
- **Signed area / orientation**: determining left/right turns

#### Open Questions

1. **One chapter or two?** Dot product and matrix-vector mult could be one chapter; cross product could be separate or combined.

2. **Reorganization needed?** Chapter 4 (projection) currently gives the deep treatment of dot product. Options:
   - Move dot product fundamentals to chapter 3, keep projection applications in chapter 4
   - Have chapter 3 focus on different dot product applications (correlation, similarity) while chapter 4 keeps the geometric projection focus
   
3. **Pearson correlation example**: The penguins notebook already computes r = standardized.T @ standardized / n. This nicely shows dot product computing correlation matrix.

4. **Cross product depth**: How much 3D vs 2D? The 2D cross product (determinant) is simpler and more immediately useful for area/orientation.

### Potential future chapters

- **boids** - Flocking simulation
- **penguins** - Linear algebra with penguin data
- **correlation** / **ansur_corr** - PCA and correlation analysis
- **trilateration** - GPS trilateration

