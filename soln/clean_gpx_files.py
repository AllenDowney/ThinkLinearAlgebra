#!/usr/bin/env python3
"""
Clean GPX files for sharing.

This script processes GPX files with capitalized names (e.g., Morning_Run.gpx)
and creates cleaned versions with lowercase names (e.g., morning_run.gpx).

The clean_gpx function removes metadata and extensions to create privacy-safe
GPX files containing only latitude, longitude, elevation, and time.

Usage:
    python clean_gpx_files.py
"""

from pathlib import Path
from utils import clean_gpx


def main():
    """Process all capitalized GPX files in the current directory."""
    soln_dir = Path(__file__).parent
    
    # Find all GPX files with capital letters in the name
    gpx_files = sorted(soln_dir.glob("*_*.gpx"))
    
    processed = []
    for input_file in gpx_files:
        # Only process files that have capital letters
        if not input_file.stem[0].isupper():
            continue
            
        # Create output filename with lowercase
        output_name = input_file.name.lower()
        output_file = soln_dir / output_name
        
        print(f"Processing {input_file.name} -> {output_name}")
        clean_gpx(str(input_file), str(output_file))
        processed.append(output_name)
    
    if processed:
        print(f"\nSuccessfully processed {len(processed)} file(s):")
        for name in processed:
            print(f"  - {name}")
    else:
        print("No capitalized GPX files found to process.")


if __name__ == "__main__":
    main()

