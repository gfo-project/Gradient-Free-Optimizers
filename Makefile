install-test-requirements:
	python -m pip install .[test]

install-build-requirements:
	python -m pip install .[build]

build:
	python -m build

install: build
	pip install dist/*.whl

uninstall:
	pip uninstall -y gradient-free-optimizers
	rm -fr build dist *.egg-info

install-editable:
	pip install -e .

reinstall: uninstall install

reinstall-editable: uninstall install-editable

test-visual:
	cd tests/local; \
		python _visualize_search_paths.py

# API smoke tests (fast, run first)
test-api:
	python -m pytest -x -p no:warnings -rfEX tests/test_api/

# Main test suite
test-main:
	python -m pytest -x -p no:warnings -rfEX tests/test_main/

# Dependency isolation tests (run in isolated environments)
test-no-sklearn:
	python -m pytest -x -p no:warnings -rfEX tests/test_dependencies/test_no_sklearn.py

test-no-scipy:
	python -m pytest -x -p no:warnings -rfEX tests/test_dependencies/test_no_scipy.py

# Legacy target - runs all tests
test-gfo:
	python -m pytest -x -p no:warnings -rfEX tests/ src/gradient_free_optimizers/

test-examples:
	cd tests; \
		python _test_examples.py

test-hyper:
	# test if new version of gfo works with current release of hyperactive
	pip install --upgrade --force-reinstall hyperactive; \
	make install; \
	cd ../Hyperactive; \
		make test

test-timings:
	python -m pytest -x -p no:warnings -rfEX tests/test_memory.py
	python -m pytest -x -p no:warnings -rfEX tests/test_main/test_optimizers/_test_max_time.py
	python -m pytest -x -p no:warnings -rfEX tests/test_main/test_optimizers/_test_memory_warm_start.py

# Full test suite (CI order)
test:
	make test-api
	make test-main
	make test-timings
	make test-examples
	make test-hyper

# Run pre-commit on all files
pre-commit-all:
	pre-commit run --all-files
