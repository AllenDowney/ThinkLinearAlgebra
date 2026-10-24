#!/bin/bash
# Top-level build script for ThinkLinearAlgebra
# This script coordinates building the book and managing data files

set -e

# Add data files to git
echo "Adding data files to git..."
git add data/

# Build the chapters (student notebooks without solutions)
echo "Building chapters..."
cd chapters
bash build.sh
cd ..

# Build the Jupyter Book (HTML website)
echo "Building Jupyter Book..."
cd jb
bash build.sh
cd ..

echo "Build complete!"

