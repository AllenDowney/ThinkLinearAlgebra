#!/bin/bash
# Run notebooks roughly the way Colab does: each one in a fresh, empty
# directory, so data files are downloaded from GitHub. utils.py is copied
# from this repo, so a commit is tested with its own version of utils.py.
#
# Usage: bash ci/run_notebooks.sh [notebook ...]
# With no arguments, runs the chapters in jb/_toc.yml (from soln/)
# and the prototype notebooks in plans/prototypes/.

set -u

repo=$(cd "$(dirname "$0")/.." && pwd)

if [ $# -gt 0 ]; then
    notebooks=("$@")
else
    notebooks=()
    for name in $(grep -oP 'file:\s*\K\w+' "$repo/jb/_toc.yml"); do
        notebooks+=("$repo/soln/$name.ipynb")
    done
    notebooks+=("$repo"/plans/prototypes/*.ipynb)
fi

work=$(mktemp -d)
failed=()

for notebook in "${notebooks[@]}"; do
    name=$(basename "$notebook" .ipynb)
    dir="$work/$name/nb"
    # affine writes ../jb/enterprise.html, so give it somewhere to write
    mkdir -p "$dir" "$work/$name/jb"
    cp "$notebook" "$dir/"
    cp "$repo/utils.py" "$dir/"

    echo "=== $name"
    if ! (cd "$dir" && python -m pytest -q --nbmake --nbmake-timeout=900 "$name.ipynb"); then
        failed+=("$name")
    fi
done

echo
if [ ${#failed[@]} -gt 0 ]; then
    echo "Failed: ${failed[*]}"
    exit 1
fi
echo "All ${#notebooks[@]} notebooks passed."
