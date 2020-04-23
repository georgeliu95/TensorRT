test:
	python3 -m pytest tests/ -v -x

clean:
	rm -rf dist/ build/ onnx_graphsurgeon.egg-info/

build: clean
	python3 setup.py bdist_wheel

install_venv: build
	python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl

install: build
	python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl --user
