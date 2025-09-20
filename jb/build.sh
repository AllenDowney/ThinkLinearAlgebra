# pip install jupyter-book ghp-import

# Build the Jupyter book version

# copy the notebooks
cp ../soln/eigenvector.ipynb .
cp ../soln/affine.ipynb .
cp ../soln/nullspace.ipynb .
cp ../soln/truss.ipynb .

# add tags to hide the solutions
python prep_notebooks.py

# build the HTML version
jb build .

# copy additional files to the build directory
cp enterprise.html _build/html/

# push it to GitHub
ghp-import -n -p -f _build/html
