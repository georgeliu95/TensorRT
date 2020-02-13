test:
	python3 -m pytest tests/ -v

clean:
	rm -rf dist/ build/ onnx_graphsurgeon.egg-info/

build: clean
	python3 setup.py bdist_wheel

install_venv: build
	python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl

install: build
	python3 -m pip install --force-reinstall $(CURDIR)/dist/*.whl --user

upload: build
	twine upload --repository-url https://sc-hw-artf.nvidia.com/api/pypi/sw-dl-tensorrt-pypi/ dist/onnx_graphsurgeon*.whl
