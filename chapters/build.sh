# Build the notebook folder and zip file

# copy notebooks from soln

cp ../soln/eigenvector.ipynb .
cp ../soln/track.ipynb .
cp ../soln/projection.ipynb .
cp ../soln/affine.ipynb .
cp ../soln/system.ipynb .
cp ../soln/nullspace.ipynb .
cp ../soln/truss.ipynb .
cp ../soln/regression.ipynb .

# remove the solutions
python prep_notebooks.py

# pip install pytest nbmake

# run nbmake
# TODO: can we get the %%expect magic to work?
# pytest --nbmake chap*.ipynb

# update the utils file
cp ../soln/utils.py .

# commit the changes
git add *.ipynb utils.py
git commit -m "Updating the notebooks"

# build the zip file
cd ../..; zip -r ThinkLinearAlgebraNotebooks.zip \
    ThinkLinearAlgebra/chapters/*.ipynb \
    ThinkLinearAlgebra/chapters/utils.py

# add and commit it
mv ThinkLinearAlgebraNotebooks.zip ThinkLinearAlgebra
cd ThinkLinearAlgebra

git add ThinkLinearAlgebraNotebooks.zip
git commit -m "Updating the zip file"

git push
