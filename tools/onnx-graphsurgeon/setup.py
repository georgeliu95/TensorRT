import sys
import onnx_graphsurgeon
from setuptools import setup, find_packages

def no_publish():
    blacklist = ['register']
    for cmd in blacklist:
        if cmd in sys.argv:
            raise RuntimeError("Command \"{}\" blacklisted".format(cmd))


REQUIRED_PACKAGES = [
    "numpy",
    "onnx",
]

def main():
    no_publish()
    setup(
        name="onnx_graphsurgeon",
        version=onnx_graphsurgeon.__version__,
        description="ONNX GraphSurgeon",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        url="https://github.com/nvidia/tensorrt/tools/onnx-graphsurgeon",
        author="Pranav Marathe",
        author_email="pranavm@nvidia.com",
        classifiers=[
            'Intended Audience :: Developers',
            'Programming Language :: Python :: 3',
        ],
        install_requires=REQUIRED_PACKAGES,
        packages=find_packages(),
        zip_safe=True,
    )

if __name__ == '__main__':
    main()
