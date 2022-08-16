import sys

import setuptools

sys.path.insert(0, "src")
import pytorch_adapt

with open("README.md", "r") as fh:
    long_description = fh.read()


extras_require_detection = ["albumentations >= 1.2.1"]
extras_require_ignite = ["pytorch-ignite == 0.4.9"]
extras_require_lightning = ["pytorch-lightning"]
extras_require_record_keeper = ["record-keeper >= 0.9.32"]
extras_require_timm = ["timm"]
extras_require_docs = [
    "mkdocs-material",
    "mkdocstrings[python]",
    "griffe",
    "mkdocs-gen-files",
    "mkdocs-section-index",
    "mkdocs-literate-nav",
]
extras_require_dev = ["black", "isort", "nbqa", "flake8"]

setuptools.setup(
    name="pytorch-adapt",
    version=pytorch_adapt.__version__,
    author="Kevin Musgrave",
    description="Domain adaptation made easy. Fully featured, modular, and customizable.",
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
        "torchmetrics >= 0.9.3",
        "pytorch-metric-learning >= 1.5.2",
    ],
    extras_require={
        "detection": extras_require_detection,
        "ignite": extras_require_ignite,
        "lightning": extras_require_lightning,
        "record-keeper": extras_require_record_keeper,
        "timm": extras_require_timm,
        "docs": extras_require_docs,
        "dev": extras_require_dev,
    },
)
