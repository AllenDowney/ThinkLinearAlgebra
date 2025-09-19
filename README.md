# Think Linear Algebra, 1st edition

*Think Linear Algebra* is a code-first, case-based introduction to the most widely used concepts in linear algebra, designed for readers who want to understand and apply these ideas — not just learn them in the abstract. Each chapter centers on a real-world problem like modeling traffic in the web, simulating flocking birds, or analyzing electrical circuits. Using Python and powerful libraries like NumPy, SciPy, SymPy, and NetworkX, readers build working solutions that reveal how linear algebra provides elegant, general-purpose tools for thinking and doing.

This book is for readers who may have struggled with traditional math instruction, or who want a more intuitive, hands-on way to learn. By working in Jupyter notebooks, readers get instant feedback as they write code, run simulations, visualize results, and explore what-if scenarios. Rather than beginning with mathematical formalism, *Think Linear Algebra* starts with meaningful applications and builds up the theory when it's needed. The result is a uniquely practical and empowering introduction to linear algebra as a language for solving real problems.

Linear algebra is foundational for machine learning, scientific computing, and computer graphics — fields with enormous demand and growth. From search engines and GPS tracking to signal processing and structural engineering, linear algebra is the language behind many of the technologies that shape our world. This book shows you how to use it effectively in your own work.


## Available Chapters

Here are the chapters that are available now. More coming soon!

**Chapter 1: The Power of Linear Algebra**

* [Click here to run Chapter 1 on Colab](https://colab.research.google.com/github/AllenDowney/ThinkLinearAlgebra/blob/main/chapters/centrality.ipynb)

**Chapter 5: To Boldly Go**

* [Click here to run Chapter 5 on Colab](https://colab.research.google.com/github/AllenDowney/ThinkLinearAlgebra/blob/main/chapters/affine.ipynb)

**Chapter 8: Null Space**

* [Click here to run Chapter 8 on Colab](https://colab.research.google.com/github/AllenDowney/ThinkLinearAlgebra/blob/main/chapters/nullspace.ipynb)

**Chapter 9: Truss In the System**

* [Click here to run Chapter 9 on Colab](https://colab.research.google.com/github/AllenDowney/ThinkLinearAlgebra/blob/main/chapters/truss.ipynb)


## Working with the code

We recommend using **Google Colab** to run the notebooks — it's free, requires no setup, and works in your browser. Simply click the "Click here to run" links above for each chapter.

If you prefer to work locally, follow these steps:

1. **Get the code** (choose one option)
   
   **Option A: Download zip file (simplest)**
   - [Download the zip file from GitHub](https://github.com/AllenDowney/ThinkLinearAlgebra/archive/refs/heads/main.zip)
   - Extract the zip file and `cd ThinkLinearAlgebra`
   
   **Option B: Clone with git**
   ```bash
   git clone --depth 1 https://github.com/AllenDowney/ThinkLinearAlgebra.git
   cd ThinkLinearAlgebra
   ```

2. **Set up your environment** (choose one option)
   
   **Option A: Conda environment (recommended)**
   ```bash
   make create_environment
   conda activate ThinkLinearAlgebra
   make requirements
   ```
   
   **Option B: Install packages individually**
   ```bash
   pip install numpy scipy sympy networkx matplotlib jupyter pandas
   ```

3. **Run the notebooks**

   - **With solutions**: Open notebooks directly from the `soln/` directory
   - **Without solutions (for exercises)**: Use notebooks from the `chapters/` directory, which have blank cells where you can work on exercises
   

## Repository Structure

```
ThinkLinearAlgebra/
├── jb/                     # Jupyter Book source files
│   ├── index.md           # Landing page
│   ├── _config.yml        # Book configuration
│   ├── _toc.yml          # Table of contents
│   └── build.sh          # Build script for HTML version
├── chapters/              # Exercise notebooks (blank cells for student work)
│   ├── *.ipynb           # Chapter notebooks with exercises
│   └── build.sh          # Build script to create zip file
├── soln/                  # Solution notebooks and code
│   ├── *.ipynb           # Individual chapter notebooks with solutions
│   ├── utils.py          # Helper functions and utilities
│   └── *.py              # Additional Python modules
├── data/                  # Datasets used in the book
├── papers/               # Research papers and references
├── tests/                # Unit tests for utility functions
├── requirements.txt      # Python package dependencies
└── README.md            # This file
```

The main content is organized in three ways:
- **`soln/`**: Complete notebooks with solutions for reference
- **`chapters/`**: Exercise notebooks with blank cells for student work
- **`jb/`**: Source files for building the HTML version of the book


## License

This book is available under a Creative Commons license, which means that you are free to copy, distribute, and modify it, as long as you attribute the source and don't use it for commercial purposes.
