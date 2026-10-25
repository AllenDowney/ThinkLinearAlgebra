# Think Linear Algebra: Design Goals

**Status:** first draft, 2026-10-24. Inferred from the eight published chapters, the README, and `jb/index.md`; Allen to revise. Where a goal is a guess rather than something stated, it says so.

Related documents: `inventory.md` (what the chapters cover), `plan.md` (status), `PROJECT_BOARD.md` (tasks), `.cursor/rules/` (coding and plotting conventions).

---

## 1. Purpose and audience

*Think Linear Algebra* is "a code-first, case-based introduction to the most widely used concepts in linear algebra" (README). It is written for:

- readers "who may have struggled with traditional math instruction, or who want a more intuitive, hands-on way to learn";
- people who want to *use* linear algebra in machine learning, scientific computing, graphics, engineering, and data science.

**Assumed background (inferred):**
- **Python:** comfortable. Readers are expected to follow NumPy, Pandas, and NetworkX code without a tutorial. `chapters/jupyter_intro.ipynb` exists, but it isn't in the table of contents.
- **Math:** high-school algebra and trigonometry. Calculus helps but isn't required ("If you know calculus, you will recognize…", Ch 2).
- **Science:** some comfort with physical reasoning. Physics, chemistry, and circuits are explained from scratch ("If you studied chemistry some time ago, or never, I'll explain what you need to know", Ch 8).

## 2. Core design goals

### G1. Start with a real problem, bring in the theory when it's needed

Each chapter is organized around an application, and the linear algebra shows up as the tool that solves it. The applications are concrete and often physical or personal:

- a museum and Wikipedia (Ch 1)
- the author's own GPS run (Ch 2)
- pool balls (Ch 4)
- the *Enterprise* and *Asteroids* (Ch 5)
- circuits (Ch 7)
- chemical reactions (Ch 8)
- popsicle-stick bridges (Ch 9)
- political views in survey data (Ch 10)

Formalism comes *after* motivation, and only as much as the problem needs.

### G2. Top-down: use the library first, open the box later

"With a computational approach, we have the option to proceed top-down — that is, we can start with libraries that implement the core algorithms of linear algebra, and wait until later to see how they work. With this approach we can get to the good stuff faster." (Ch 1)

So Ch 1 uses `eig` and PageRank before the reader knows what an eigenvector is "deeply". Readers are told explicitly that it's fine not to understand everything yet.

### G3. Then open the box: implement the core algorithms

When a chapter does look inside an algorithm, the reader builds a simple version and checks it against the library:

- power iteration vs. `eig` (Ch 1)
- Gauss-Jordan elimination, inverse, and LU decomposition vs. SymPy, NumPy, and SciPy (Ch 7)
- Gram-Schmidt and QR vs. `np.linalg.qr` (Ch 10)

The goal is understanding, not production code. The text says when the library version is better, and why (efficiency, numerical stability).

### G4. Derive with SymPy instead of proving

The book replaces proofs with symbolic computation. It sets up the problem with symbols and lets SymPy simplify, and the identity falls out:

- the dot product formula from polar coordinates (Ch 4)
- the rotation matrix from angle addition (Ch 4)
- scale, shear, and translation matrices shown acting on a symbolic vector (Ch 5)
- circuit solutions as symbolic formulas, then numbers (Ch 7)
- stiffness matrices with common factors pulled out (Ch 9)

Theorems that are needed (rank-nullity, for example) are stated and interpreted, not proved.

### G5. Compute the same thing several ways and check that they agree

Results are routinely cross-checked: simulation vs. eigenvector, polar vs. Cartesian projection, minimization vs. QR vs. normal equations, hand-built vs. library. `np.allclose` shows up in nearly every chapter. "At this point we have solved the same small matrix equations about ten different ways" (Ch 7). This shows that different methods are the same idea, and it builds trust in the code.

### G6. Small example first, then real scale

Most chapters start with a toy that fits on screen, then scale up:

- a 6-room museum → 4,604 Wikipedia pages (Ch 1)
- a 3-point regression → 70,000 GSS respondents (Ch 10)
- a voltage divider → an N-bit R2R ladder (Ch 7)
- a two-bar truss → a Warren truss with N triangles (Ch 9)
- ethanol → permanganate, with a 2-D null space (Ch 8)

### G7. Make it visible

Vectors are drawn as arrows, transformations as before/after contours, and projections as dropped perpendiculars. 3D figures show column spaces, and force fields show stiffness matrices. Most chapters include an animation: the pool shot, the *Enterprise* in orbit, the GPS run, the rotating view of coplanar bases. The `utils.py` helpers (`plot_vector`, `plot_vectors`, `decorate`, `setup_3D`, …) keep plotting code short so the math stays in front.

### G8. Be honest about models and numbers

The book is open about what its models leave out:

- truss assumptions listed up front
- the spherical Earth vs. an ellipsoid
- the noise that differentiation amplifies and integration accumulates
- "on TV it's better to look cool than be scientifically accurate" (the ship stretch)
- stoichiometry can't tell you which reaction actually happens

Numerical issues are treated as part of the subject: floating-point error, stability, condition number, collinearity.

### G9. Clear vocabulary, connected to notation and code

New terms are **bolded** at first use and defined in plain words. Conventional notation is explained ("$\hat{\beta}$ … the caret is almost universally called a 'hat'"). The book distinguishes ideas readers often confuse: array vs. matrix, free vs. bound vectors, row vs. column vectors, scalar vs. vector projection. Math notation and code sit side by side, so readers learn to translate between them (see "What You'll Learn" in `jb/index.md`).

### G10. A personal, conversational voice

The voice is first person, informal, sometimes funny, and often anecdotal: the 1987 graphics assignment, the undergraduate truss question, "In my defense, the route I ran has sharper turns than a standard track", "Oops." Each chapter's introduction says what we'll do and why it's worth doing.

## 3. Chapter pattern

Inferred from the existing chapters (`.cursor/rules/add-chapter.md` covers the mechanics):

1. **Title and introduction:** the problem, and the concepts it will bring in. Sometimes a reassurance ("That's a lot for just one chapter…").
2. **Colab link and setup cells:** these are hidden or removed in the book (`download`, imports, `set_precision`).
3. **Sections that alternate** between a short explanation and a short code cell. Most code cells are only a few lines, and each is followed by a sentence saying what the result means.
4. **Discussion** (some chapters): recap, connections, and a pointer to the next chapter.
5. **Exercises:** extend the chapter's method to a new case (a bigger museum, a new circuit, a different reaction, italics by shear, the Star Wars crawl). Solutions are cells starting with `# Solution`, which `prep_notebooks.py` removes for the student version.
6. **Footer:** license.

Typical length is 500–1,600 lines of jupytext markdown.

## 4. Scope

**In scope:** the linear algebra people actually use. That means vectors, dot products and projection, transformations, systems and decompositions (LU, QR), rank and null space, least squares, and eigenvectors. SVD and PCA are promised but not yet written (see `inventory.md` §3).

**Out of scope (inferred):** abstract vector spaces, proof technique, complex vector spaces, canonical forms. If the book covers a topic, it's because there's a computation the reader can do with it.

**Tools:** Python with NumPy, SciPy, SymPy, NetworkX, Pandas, and Matplotlib, plus whatever a given application needs (scikit-image, shapely, gpxpy, pint). Everything has to run in Colab.

**Data:** real and downloadable, from the repo's `data/` folder or a stable public source.

## 5. Ordering principles (inferred)

- Concepts are introduced where an application needs them, and a chapter can preview an idea and come back to it later (eigenvectors in Ch 1).
- Broadly, the order goes from vectors (Ch 2–4) to transformations (Ch 5) to solving systems (Ch 7–9) to fitting data (Ch 10).
- Each chapter should stand mostly on its own, but some dependencies are real. For example, Ch 9 relies on rank and null space from Ch 8, and Ch 10 relies on projection from Ch 4. `inventory.md` §4 lists the places where the current order breaks this.

## 6. Open questions for Allen

These are things the existing chapters don't settle:

1. **Prerequisites:** should the book say explicitly what Python and math background it assumes? Is `jupyter_intro` meant to be in the book?
2. **Promises and preview-then-return:** when a chapter promises to come back to a topic, as Ch 1 does with eigenvectors, does the book commit to a later chapter that delivers it?
3. **Symbolic vs. numerical balance:** Ch 7–9 lean heavily on SymPy. Is that the intended balance for the remaining chapters?
4. **Row vs. column vectors:** Ch 5 explains both conventions, but the chapters use a mix. Should there be a house convention for collections of vectors?
5. **Exercises:** how many per chapter, and should they always include a new application or data set?
6. **Animations:** a goal for every chapter, or only where motion is natural?
7. **Length and pacing:** is there a target size for a chapter?
8. **Relation to the other Think books:** are there shared conventions with Think Stats, Think Bayes, and the others worth writing down here?
