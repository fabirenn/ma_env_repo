isort:
	python -m isort src

black:
	python -m black --config .black.cfg src

flake8:
	python -m flake8 src

format: isort black

mypy:
	python -m mypy --incremental --install-types --show-error-codes --pretty src

test:
	python -m pytest src

build: isort black flake8 mypy test
