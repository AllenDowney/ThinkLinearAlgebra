#!/usr/bin/env python3
"""Update notebooks in soln/ from the markdown in soln/md/, which is canonical.

Uses jupytext --update, which replaces the inputs and keeps the outputs of
cells that are unchanged, so a notebook doesn't have to be rerun after
editing its text.

Usage (from the repo root or soln/):
    python soln/md_to_ipynb.py            # every .md newer than its notebook
    python soln/md_to_ipynb.py boids      # named chapters
    python soln/md_to_ipynb.py --all      # every .md

If you edit a notebook instead, convert it back to markdown:
    jupytext --to md soln/NAME.ipynb -o soln/md/NAME.md
"""

import subprocess
import sys
from pathlib import Path

soln = Path(__file__).resolve().parent
md_dir = soln / "md"

args = sys.argv[1:]
if args == ["--all"]:
    sources = sorted(md_dir.glob("*.md"))
elif args:
    sources = [md_dir / f"{name.removesuffix('.md')}.md" for name in args]
else:
    sources = [md for md in sorted(md_dir.glob("*.md"))
               if not (soln / f"{md.stem}.ipynb").exists()
               or md.stat().st_mtime > (soln / f"{md.stem}.ipynb").stat().st_mtime]

if not sources:
    print("All notebooks are up to date.")

for md in sources:
    notebook = soln / f"{md.stem}.ipynb"
    command = ["jupytext", "--to", "ipynb", str(md), "-o", str(notebook)]
    if notebook.exists():
        command.insert(1, "--update")
    subprocess.run(command, check=True)
    print(f"{md.relative_to(soln)} -> {notebook.name}")
