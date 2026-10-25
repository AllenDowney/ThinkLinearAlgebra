# Think Linear Algebra Project Board

Numbered tasks for tracking work. Each task has a permanent number; add new tasks at the end. Update status as work progresses.

**Status values:** `Pending`, `In progress`, `DONE`

---

## Task 1: Search engine optimization for the book site

**Status:** Pending

**Plan (Allen, 2026-10-24):** do this together with Task 2, when the work in progress is committed and the book is published, in this repo's own thread. Follow the pattern used for Think Python, Think Stats, Think Bayes, ModSimPy, Think DSP 2 and Think Complexity: `jb/descriptions.yml` written into the copied notebooks by `jb/prep_notebooks.py`, front matter in `index.md`, `only_build_toc_files: true` if `build.sh` copies notebooks that aren't in `_toc.yml`, a test build in a scratch copy, then the `Sitemap:` line in `~/AllenDowney.github.io/robots.txt` and `python3 make_sitemap_index.py` there.

**Context:** Help people searching for the book's topics find `https://allendowney.github.io/ThinkLinearAlgebra/`, using ordinary good practice only. The steps were worked out on *Probably Overthinking It* (2026-10-18), and the shared guide is `~/GreenTeaPress/plans/seo.md` ("Checklist for a Jupyter Book 1 site"). Rollout is tracked in GreenTeaPress Task 22. Audit 2026-10-18: no meta descriptions, no sitemap; the landing-page title repeats.

Do it at the next rebuild:
- [ ] `split_at_heading = true` comes from the user-level `~/.config/jupytext/jupytext.toml` (a repo `jupytext.toml` would override it); check that the first `--update` splits no tagged cells
- [ ] `sphinx-sitemap` + `sphinxext-opengraph`, base URL, social image (`think_linear_algebra_fake_cover.png` until there is a real cover) in `_config.yml`; add the page-view counter `extra_footer` too if it isn't there yet (GreenTeaPress Task 15)
- [ ] Description for each page in the notebook metadata (Allen reviews)
- [ ] Test build, publish; add the `Sitemap:` line to `~/AllenDowney.github.io/robots.txt`; Allen submits it in Search Console
- [ ] GitHub repo website field and README link to the site

---

## Task 2: Fill in the missing chapters and publish

**Status:** Pending

**Context:** Eight chapters are complete and in `jb/_toc.yml` (eigenvector, track, projection, affine, system, nullspace, truss, regression). The README lists seven of them (track is missing), numbered 1, 4, 5, 7–10, so the open slots are **Chapter 3** (between track and projection) and **Chapter 6** (between affine and system). Candidates are in `soln/`: boids, penguins, correlation, cross, trilateration, and others; see `inventory.md` §5. (`chapters/centrality.ipynb` is an older copy of Chapter 1, not a candidate.) Status and details are in `plan.md`.

This is the umbrella task. It is split into Task 3 (baseline commit), Task 4 (inventory), and Task 5 (choose chapters and plan). Tasks for writing each new chapter get added once Task 5 picks them.

- [x] Task 3: baseline commit
- [ ] Task 4: inventory and gap analysis
- [ ] Task 5: choose the chapters and write the plan
- [ ] Write the new chapters (tasks to be added)
- [ ] Fix the loose ends in the published chapters (`inventory.md` §6)
- [ ] Update the README chapter list and numbering to match `_toc.yml`
- [ ] Rebuild and publish (do Task 1 in the same rebuild)
- [ ] Announce (blog, newsletter) and update the greenteapress.com page if there is one

---

## Task 3: Baseline commit of local work

**Status:** DONE (2026-10-24), except changing the OpenSky password, which Allen does

**Context:** As of 2026-10-24 the working copy has about 78 modified or untracked files, built up since the last commit. Commit a clean baseline before starting new chapter work. Do this before Task 5's chapter writing starts.

- [x] Review `git status`; decide what gets committed, ignored, or deleted
- [ ] Keep secrets out (removed from the working tree 2026-10-24; Allen still to change the password): `soln/credentials.json`, and the OpenSky password hard-coded in `soln/geography.ipynb` (and `soln/md/geography.md`). **The password is already pushed to GitHub** (commits 685a3f1 and 19e7cc8 on `origin/main`), so change it on OpenSky, and remove it from the notebook
- [x] Add `.gitignore` entries for build output (`jb/_build/`), editor folders (`.cursor/`, `.vscode/`), large or downloaded data, generated GIF/MP4 files, and stray files (`soln/download?id=...`)
- [x] `soln/md/` is ignored; regenerate with `python soln/convert_to_md.py`
- [x] Commit and push (Allen approved the file list). Left untracked on purpose: `soln/fertility_trends_paper.md`, `soln/scalar_projection_*.md`, `soln/test_notebook.ipynb`, and old scripts (`soln/nullspace.py`, `wiki.py`, `centrality.py`, `scalar_projection_example.py`)

---

## Task 4: Inventory and gap analysis of existing chapters

**Status:** In progress (draft done 2026-10-24; Allen to review)

**Context:** Before choosing the remaining chapters, take stock of what the eight published chapters cover. `soln/convert_to_md.py` now also converts notebooks that exist only in `chapters/`; all 21 markdown files were regenerated on 2026-10-24.

The result is `inventory.md`: per-chapter concepts, applications, and tools (§1); a concept × chapter coverage table (§2); gaps compared with a standard first course (§3); dependencies that shape slots 3 and 6 (§4); candidate notebooks (§5); typos and loose ends in the published chapters (§6).

- [x] Regenerate markdown versions of all notebooks
- [x] Write `inventory.md`
- [ ] Allen reviews; correct anything wrong or missing

---

## Task 5: Choose the remaining chapters and write the plan

**Status:** Pending

**Context:** Use `inventory.md` to decide what fills slots 3 and 6, and whether the book also needs an eigen/SVD/PCA chapter after regression. Chapter 1 and Chapter 10 both point forward to one that doesn't exist yet. Some first ideas from the inventory (not decisions):

- **Slot 3:** linear combinations, span, basis; 2D cross product / determinant as signed area (Chapter 5 uses `cross2d` with no introduction). Boids is a natural application (a Boid's goal is a weighted sum of vectors). Correlation as a dot product (penguins) could also go here, or extend Chapter 4.
- **Slot 6:** undoing and finding transformations: inverse, determinant as area scaling and an invertibility test, solving for the transform that maps one set of points to another. This leads into Chapter 7.
- **After 10:** eigendecomposition, SVD, PCA with ANSUR or penguins data (`correlation`, `penguins`).
- **Decided (Allen, 2026-10-24): Frames is Chapter 6** (`soln/Frame.py`): vectors tagged with a frame of reference, 3D rotations, rigid transforms between frames. Use NetworkX for the graph of frames. Keep the caching of composed and inverse transforms as edges, but mark them as derived edges and invalidate them when a primary transform changes. It shares a `Vector` class with the Boids chapter (Task 7).

- [ ] Explore Boids for slot 3 (Task 7)
- [ ] Decide the number of chapters (10 or 11) and what goes in each open slot
- [ ] Write a planning doc for each chosen chapter: goal, concepts, application and data, outline, exercises, dependencies on neighboring chapters, source notebooks
- [ ] Add a board task for writing each chosen chapter
- [ ] Update `plan.md` to match

---

## Task 6: Design goals document

**Status:** In progress (draft done 2026-10-24; Allen to revise)

**Context:** Record the book's design goals, so that new chapters (Task 5) stay consistent with the existing ones. `design.md` was inferred from the eight published chapters, the README and `jb/index.md`. It ends with open questions for Allen.

- [x] Draft `design.md`
- [ ] Allen revises it and answers the open questions (§6)
- [ ] Use it as the checklist when planning and reviewing new chapters

---

## Task 7: Explore a Chapter 3 built around Boids

**Status:** In progress (planning doc started 2026-10-24)

**Context:** Part of Task 5 (choosing the remaining chapters). Boids (Reynolds 1987; the existing code is from *Think Complexity* and uses VPython) is a candidate centerpiece for slot 3. Its steering rule is a weighted sum of vectors, so it naturally teaches scalar multiplication and linear combinations. The planning doc is `plans/chapter3_boids.md`. It outlines the existing code and its bugs, the topics Boids can demonstrate, the prep the chapter needs, and tools for running in Colab. The recommendation is a small `Vector` class of our own plus Matplotlib 3D animation, because VPython doesn't run in Colab yet.

- [x] Start the planning doc: code outline, topics, prep, tools, recommendation
- [ ] Allen decides on the doc's open questions (§5)
- [x] Prototype 3D Boids with a `Vector` class and a Matplotlib animation (`plans/prototypes/boids3d.md` and `.ipynb`, built on `Vector` in `soln/utils.py`, 2026-10-24); Allen asked for 3D and vector objects
- [x] Write the shared `Vector` class in `soln/utils.py`, with tests in `tests/test_utils.py`
- [ ] Test the prototype in Colab (needs `utils.py` with `Vector` pushed to the root); tune arrows, frame count and file size
- [ ] Draft the chapter outline
- [ ] List the edits needed in Ch 4 and Ch 5

---

## Task 8: Chapter 6, Frames of Reference

**Status:** In progress (planning doc started 2026-10-24)

**Context:** Allen decided on 2026-10-24 that Frames is Chapter 6 (Task 5). It's based on `soln/Frame.py` (from ThinkPython2, 2012): vectors tagged with a frame, 3D rotations, rigid transforms, and a graph of frames, now to use NetworkX. Composed transforms are cached as derived edges and cleared when a primary transform changes. That's needed because the main example is likely an animated linkage whose frames move. The planning doc is `plans/chapter6_frames.md`.

- [x] Start the planning doc
- [ ] Allen decides on the doc's open questions (§6)
- [ ] Rewrite `Frame.py` on the shared `Vector` class (Task 7), with NetworkX and derived-edge caching; fix the sign bug in `Rotation.from_axis`
- [ ] Prototype the animated linkage
- [ ] Draft the chapter outline

---

## Task 9: Test notebooks in a Colab-like environment

**Status:** In progress (CI added 2026-10-24)

**Context:** We can't run notebooks in Colab automatically from CI. Google's new official [Colab CLI](https://github.com/googlecolab/google-colab-cli) has `colab exec` for running a notebook on a Colab runtime, but it isn't yet clear how it authenticates from CI, and it uses Colab compute quota. The partial fix is CI that copies Colab's environment as closely as practical. Google publishes Colab's package versions in [googlecolab/backend-info](https://github.com/googlecolab/backend-info). As of 2026-10-24: Python 3.12.13, numpy 2.1.3, scipy 1.16.3, matplotlib 3.10.0, pandas 2.2.3, sympy 1.14.0, networkx 3.7, scikit-image 0.25.2, shapely 2.2.0, IPython 7.34.0, ipykernel 6.17.1.

Problems found while looking into this (2026-10-24):
- The root `utils.py` (the copy that notebooks download) lacked `label_point`, which the track chapter imports. Fixed: the root `utils.py` is now the only copy, and `soln/` and `chapters/` have symlinks to it.
- **Colab doesn't have `gpxpy` or `haversine` (needed by track) or `pint` (needed by truss), and those chapters don't install them.** They probably fail on Colab.

- [x] Add `%pip install` cells (hidden in the book) for packages Colab lacks: track and truss
- [x] Script that builds a pinned requirements file (`ci/colab_requirements.py` → `ci/requirements-colab.txt`) from Colab's `pip-freeze.txt`, limited to the packages the book uses, so CI installs only what Colab has
- [x] GitHub Actions workflow (`.github/workflows/notebooks.yml`, `ci/run_notebooks.sh`, `make colab-test`): Ubuntu 24.04, Python 3.12, Graphviz, and those pins. It runs each chapter in `jb/_toc.yml` (from `soln/`) and the prototypes, each in a fresh directory with nbmake. Data files are downloaded from GitHub; `utils.py` is copied from the commit being tested. All 9 notebooks pass locally.
- [ ] Optional: a local `make colab-test` using Google's Colab CPU runtime Docker image (the closest match, but very large)
- [ ] Run notebooks on real Colab from this machine with the Colab CLI (`colab exec`). Installed in `~/.local/share/colab-cli-venv` (v0.7.5); Allen has to sign in once
- [ ] Decide on solutions in `chapters/`: `chapters/prep_notebooks.py` only matches `chap*.ipynb`, so the student notebooks still contain all the solutions

More problems found by the first Colab-like run (2026-10-24), now fixed:
- projection: `%load_ext autoreload` fails on Colab (Python 3.12 + IPython 7.34 imports the removed `imp` module). It now has the same workaround cell as the other chapters.
- system: `draw_circuit_graph` needed `pygraphviz`, which Colab lacks. It now uses `nx.nx_pydot.graphviz_layout` (Colab has `pydot` and the Graphviz `dot` program), with identical layouts.
- `requirements.txt` was missing gpxpy, haversine, pint, pydot, scikit-image and shapely.
