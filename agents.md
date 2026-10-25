# Agent Guidelines

Project-specific coding guidelines are in `.cursor/rules/`:

- `project-setup.md` - Environment and file organization
- `dataframes.md` - DataFrame/Series handling
- `plotting.md` - Plotting with decorate() and utils functions
- `add-chapter.md` - How to add new chapters to the book

## Notebooks and markdown

The markdown files in `soln/md/` are canonical. Edit them, then update the notebooks with `python soln/md_to_ipynb.py` (or `make notebooks`), which keeps existing outputs. If a notebook is edited directly, convert it back with `jupytext --to md soln/NAME.ipynb -o soln/md/NAME.md`. Don't hard-wrap lines in markdown.
