.PHONY: test test_noparallel clean build install_venv install docs

test:
	python3 -m pytest tests/ -v -x -nauto

test_noparallel:
	python3 -m pytest tests/ -v -x

clean:
	rm -rf dist/ build/ polygraphy.egg-info/

build: clean
	python3 setup.py bdist_wheel

install_common: build
	-python3 -m pip install colored

install_venv: install_common
	python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl

install: install_common
	python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl --user

docs:
	mkdir -p build/docs
	sphinx-build docs build/docs/
