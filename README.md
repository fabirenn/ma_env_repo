# Student template

This template was made to suggest basic setup for student python projects.

## Project structure

The purpose of directories is to store:

- artifacts - any kind of ML experiments output (models, plots, segmentation mask, ect.),
- data - object of experiments,
- notebooks - jupyter notebooks (.ipynb files),
- src - source (all .py files).

## Project tools

In the repository you can find a few python tools suggestions.

### [Black](https://github.com/psf/black)

Python code formatter. You can find predefined configs in `.black.cfg` file.

### [Flake8](https://github.com/PyCQA/flake8)

Linting tool that checks Python codebase for errors, styling issues and complexity. It covers more use cases than `Black` but does not format code. You can find predefined configs in `.flake8` file.

### [Isort](https://pycqa.github.io/isort/)

Python library imports sorter. You can find predefined configs in `.isort.cfg` file.

### [PyTest](https://docs.pytest.org/en/)

Python testing framework.

### [MyPy](https://mypy.readthedocs.io/en/stable/getting_started.html)

Optional static type checker for Python that aims to combine the benefits of dynamic (or "duck") typing and static typing.
If you have problem with missing stubs for certain library, add to `mypy.ini` following lines:

```ini
[mypy-LIBRARY_NAME.*]
ignore_missing_imports = True
```

### [NumPy](https://numpy.org/)

Library for a wide variety of mathematical operations on arrays, essential for machine learning algorithms.

### [Make](https://www.tutorialspoint.com/unix_commands/make.htm)

In context of this repo `make` enables you to define aliases for more complex commands so you don't have to rewrite them every time.

## Usage

### First steps

1. Create a virtual environment.
If you never did it before, you may want to read about one of following tools: [python venv](https://docs.python.org/3/tutorial/venv.html#creating-virtual-environments), [conda](https://conda.io/projects/conda/en/latest/index.html), [poetry](https://python-poetry.org/docs/managing-environments/) (more advanced). For switching between python versions read about [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#unixmacos). It's recommended to choose python version available on [scientific cluster](https://www.sc.uni-leipzig.de/02_Resources/Software/Categories/lang_modules/#python) as you will perhaps train models there.
2. Install dependencies.
Development tools like `flake8` are listed in `requirements.txt`. Install them using the tool chosen in the previous step.

### Makefile predefined commands

Run:

- `make flake8` to check your source code quality,
- `make black` to format code,
- `make isort` to order imports,
- `make format` to run `isort` and `black` commands together,
- `make mypy` to check type hints,
- `make test` to run tests,
- `make build` to format run `isort`, `black`, `flake8`, `mypy` and `test` respectively.

Feel free to define your custom commands if you want.

## Protips

1. Write your code in python scripts, not in notebooks, otherwise you'll produce spaghetti code. The only exception can be (but doesn't have to be) Exploratory Data Analysis (EDA) and visualizations.
2. Write [type hints](https://docs.python.org/3/library/typing.html). You can check them with `mypy`.
3. Commit. Probably more often than you usually do. Before you commit, format and check your code (`make build`).
4. Refactor your code every time you finish implementing a new feature.
5. Use code debugger.
6. Create a branch when you bring new feature and keep the main branch away from bugs. Merge the main branch with the feature branch only when you are sure that there are no bugs.
7. Don't be afraid to use typical constructions for python like [comprehension list](https://docs.python.org/3/library/typing.html). In most cases they make your code more readable.
8. Write tests for crucial components, you'll find bugs much quicker. Don't spend too much time on that, you should stay focused on experiments. Keep your tests in `src/tests` directory.
9. Use experiment tracking tools like [Tensor Board](https://www.tensorflow.org/tensorboard), [W&B](https://wandb.ai/site), ect.
10. Check if there is no already existing solution for your problem. A good example is a training / test loop implementation. If you use [tensorflow](https://www.tensorflow.org/tutorials) add [keras](https://www.tensorflow.org/guide/keras) to your code and if you use [pytorch](https://pytorch.org/tutorials/beginner/basics/intro.html), add [fastai](https://docs.fast.ai/examples/migrating_pytorch_verbose.html) or [lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html) to train and evaluate your model. These frameworks provide boilerplate code for deep learning experiments.

## Computing Cluster

Before first use of [HPC](https://www.urz.uni-leipzig.de/unsere-services/servicedetail/service/high-performance-computing-hpc) follow the steps:

1. [Send request](https://service.rz.uni-leipzig.de/scientific-computing/sc-infrastruktur-beantragen-en/) for infrastructure.
2. After you get permission, [log in](https://lab.sc.uni-leipzig.de/jupyter/hub/spawn) and select the machine you want to use. Choose minimal hardware requirements as you won't need computational power to set up your project.
3. [Change access](https://www.sc.uni-leipzig.de/04_File_transfer/#file-sharing-with-access-control-lists-acl) to your workspace to protect your projects from unprivileged users by running:

    ```bash
    chmod -R 700 /work/users/<sc~username>
    chmod -R 700 /home/sc.uni-leipzig.de/<sc~username>
    ```

4. Clone your repository and upload your data.
5. Create a virtual environment.
6. Load libraries and interpreters already available at cluster. This will speed up your program and reduce the amount of used space. Run `module avail` to find preinstalled modules and `module load <tool_name>` to load one ([link](https://www.sc.uni-leipzig.de/02_Resources/Software/#using-software-modules) for more information).
7. Install unavailable libraries (with `pip` or other tool).
8. Go to `File -> Hub Control Panel` and click `Stop My Server`.

Your setup is finished. Now you can log in to the cluster again and select more resources for your experiments. Remember to stop your server every time you finish your work.

For more HPC information follow the [link](https://www.sc.uni-leipzig.de/).
