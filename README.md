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
4. If you want to use the HPC of the University Leipzig, it's enough to activate the virtual environment and load a Python Module (3.10.4) & corresponding TensorFlow Module.

### Model-Scripts

In the ./src-Folder there is a folder structure, where each subdirectory contains all the files to the corresponding model:
- model_name_model.py: This script contains the model definition
- train_model_name.py: This script is for training a specific model setup, most of the hyperparameters can be set by the VARIABLES in caps.
- optuna_model_name.py: This script performs hyperparemeter tuning of the model.

#### Training-Script
Inputs, that need to be set:
- 4 paths that point to the directories of the training and validation images
- the checkpoint-path where the model should be stored
- the path where the validation predictions should be stored
- the size of the input images when loading + how many image channels should be used when loading the data
- hyperparameters like batch_size, epochs, the patience value, dropout_rate, learning_rate or other model specific hyperparameters

Outputs after the training of the model:
- model_name_checkpoint.keras file in the corresponding subdirectory of the artifacts folder
- wandb log files of the training in "wandb/train_model_name"
  
#### Optuna-Script
Inputs, that need to be set:
- 4 paths that point to the directories of the training and validation images
- the size of the input images when loading + how many image channels should be used when loading the data
- parameter of how many trials should be conducted
- the file path of the db-file where the results of the optuna_scripts are stored
- in the objective function of the script all the hyperparameter and their value range are defined

Outputs after hyperparameter tuning:
DB-file that can be inspected with this command, e.g.:
```bash
optuna-dashboard sqlite:///artifacts/models/deeplab/optuna_deeplab.db
```

### Scripts

    src/convert_binary_to_color_mask.py
  this file was used for the masks provided by [this](https://github.com/chen-du/De-fencing) repo, that were binary in black and white. I needed the masks to be black in the background, but fence structure turquoise.

    src/custom_callbacks.py
this file contains some functions that extend the classic callbacks

    src/data_loader.py
this file contains all functions regarding, loading, preprocessing and building datasets before training or testing

    src/generate_images_from_masks.py
this file generates the real looking images out of background images, greenscreen fence images and the fence masks

    src/image_segmentation_seen_data.py
    src/image_segmentation_unseen_data.py
this two files are responsible for assessing the models' ability for segmenting fences, in both files the following information have to be set:
- path to the model_checkpoint.keras files + the name of the models
- path to the images + masks that need to be tested 
- the results are splitted in
	- prediction masks stored in /data/prediction/model_name
	- wandb log file in /wandb/testing_models


```
  src/loss_functions.py
```
this files contains the functions of how my loss functions are calculated, they are used by every file that need to calculate some kind of error / loss

```
  src/metrics_calculation.py
```
this files contains all the functions for calculating the metrics in all the scripts

```
  src/processing.py
```
this files contains all the functions that didn't fit in other scripts






