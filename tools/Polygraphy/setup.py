import os
import sys
import polygraphy
from setuptools import setup, find_packages

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
BIN_DIR = os.path.join(ROOT_DIR, "bin")

def no_publish():
    blacklist = ['register']
    for cmd in blacklist:
        if cmd in sys.argv:
            raise RuntimeError("Command \"{}\" blacklisted".format(cmd))


REQUIRED_PACKAGES = [
    "numpy",
]

def main():
    no_publish()
    setup(
        name="polygraphy",
        version=polygraphy.__version__,
        description="Polygraphy: A Deep Learning Inference Prototyping and Debugging Toolkit",
        long_description=open("README.md", "r", encoding="utf-8").read(),
        url="https://github.com/NVIDIA/TensorRT/tree/master/tools/polygraphy",
        author="NVIDIA",
        author_email="svc_tensorrt@nvidia.com",
        classifiers=[
            'Intended Audience :: Developers',
            'Programming Language :: Python :: 3',
        ],
        install_requires=REQUIRED_PACKAGES,
        packages=find_packages(),
        scripts=[os.path.join(BIN_DIR, "polygraphy")],
        zip_safe=True,
    )

if __name__ == '__main__':
    main()
