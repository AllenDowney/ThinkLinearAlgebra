PROJECT_NAME = ThinkLinearAlgebra
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

create_environment:
	mamba create --name $(PROJECT_NAME) python=$(PYTHON_VERSION) -y
	@echo ">>> mamba env created. Activate with:\nmamba activate $(PROJECT_NAME)"


requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements-dev.txt


## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


tests:
	cd soln; pytest --nbmake *.ipynb


## Run notebooks Colab-style, each in a fresh directory (see ci/)
colab-test:
	bash ci/run_notebooks.sh
