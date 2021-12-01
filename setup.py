import sys

import setuptools

sys.path.insert(0, "src")
import pytorch_adapt

with open("README.md", "r") as fh:
    long_description = fh.read()


extras_require_ignite = ["pytorch-ignite"]
extras_require_lightning = ["pytorch-lightning"]
extras_require_record_keeper = ["record-keeper >= 0.9.31.dev6"]


setuptools.setup(
    name="pytorch-adapt",
    version=pytorch_adapt.__version__,
    author="Kevin Musgrave",
    description="",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KevinMusgrave/pytorch-adapt",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.0",
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "torchmetrics",
        "pytorch-metric-learning >= 1.0.0.dev5",
    ],
    extras_require={
        "ignite": extras_require_ignite,
        "lightning": extras_require_lightning,
        "record-keeper": extras_require_record_keeper,
    },
)
