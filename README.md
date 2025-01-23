# Repo to the Thesis „Comparative analysis of deep learning architectures for chain-link fence image segmentation“ 

In this README, I want to explain how I used this repo to come to the results of my thesis.

## Project structure

The purpose of directories is to store:

- artifacts - any kind of ML experiments output
- data - object of experiments, will be provided by an external link
- src - source (all .py files).

## Project tools

In the repository you can find a few python tools suggestions.

### [Black](https://github.com/psf/black)

Python code formatter. You can find predefined configs in `.black.cfg` file.

### [Flake8](https://github.com/PyCQA/flake8)

Linting tool that checks Python codebase for errors, styling issues and complexity. It covers more use cases than `Black` but does not format code. You can find predefined configs in `.flake8` file.

### [Isort](https://pycqa.github.io/isort/)

Python library imports sorter. You can find predefined configs in `.isort.cfg` file.

### [NumPy](https://numpy.org/)

Library for a wide variety of mathematical operations on arrays, essential for machine learning algorithms.

### [Make](https://www.tutorialspoint.com/unix_commands/make.htm)

In context of this repo `make` enables you to define aliases for more complex commands so you don't have to rewrite them every time.

## Usage

### First steps

1. Create a virtual environment, activate it and install all the required packages / libraries.
    ```bash
    python3 -m venv ma_env
    source ma_env/bin/activate
    pip install -r requirements.txt
    ```
2. Extract the data provided via the link in the ./data folder.
3. The repo is ready to use now
If you want to use the HPC of the University Leipzig, it's enough to activate the virtual environment and load a Python Module (3.10.4) & corresponding TensorFlow Module.

### Scripts

In the ./src-Folder there is folder structure, where each subdirectory contains all the files to the corresponding model:
- model_name_model.py: This script contains the model definition
- train_model_name.py: This script is for training a specific model setup, most of the hyperparameters can be set by the VARIABLES in caps.
- optuna_model_name.py: This script performs hyperparemeter tuning of the model.
