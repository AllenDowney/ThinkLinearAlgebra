#!/usr/bin/env python3
"""
Convert all Jupyter notebooks in soln/ to markdown format using jupytext.
This makes the notebooks easier to read compared to the JSON format.
"""

import os
import subprocess
from pathlib import Path

def convert_notebooks_to_markdown():
    """Convert all .ipynb files in the current directory to markdown."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Create md/ subdirectory if it doesn't exist
    md_dir = script_dir / "md"
    md_dir.mkdir(exist_ok=True)
    
    # Find all .ipynb files in the current directory (not recursive)
    notebooks = list(script_dir.glob("*.ipynb"))

    # Also include notebooks in chapters/ that have no counterpart here
    names = {nb.name for nb in notebooks}
    chapters_dir = script_dir.parent / "chapters"
    notebooks += [nb for nb in chapters_dir.glob("*.ipynb") if nb.name not in names]

    if not notebooks:
        print("No notebooks found in the current directory.")
        return
    
    print(f"Found {len(notebooks)} notebooks to convert")
    print(f"Output directory: {md_dir}")
    print()
    
    # Convert each notebook
    success_count = 0
    for notebook in sorted(notebooks):
        notebook_name = notebook.name
        md_name = notebook.stem + ".md"
        output_path = md_dir / md_name
        
        try:
            # Use jupytext to convert notebook to markdown
            result = subprocess.run(
                ["jupytext", "--to", "markdown", "-o", str(output_path), str(notebook)],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✓ Converted: {notebook_name} -> md/{md_name}")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to convert {notebook_name}: {e.stderr}")
        except FileNotFoundError:
            print("Error: jupytext not found. Please install it with: pip install jupytext")
            return
    
    print()
    print(f"Conversion complete: {success_count}/{len(notebooks)} notebooks converted successfully")

if __name__ == "__main__":
    convert_notebooks_to_markdown()

