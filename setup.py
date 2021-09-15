import sys

import setuptools

sys.path.insert(0, "src")
import pytorch_adapt

with open("README.md", "r") as fh:
    long_description = fh.read()

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
    # install_requires=[
    #     "torch",
    #     "torchvision",
    #     "torchmetrics",
    #     "pytorch-metric-learning",
    #     "pytorch-ignite",
    #     "faiss-gpu",
    #     "record-keeper",
    #     "scikit-learn",
    # ],
    # extras_require={
    #     "dev": [
    #         "black",
    #         "isort",
    #         "nbqa",
    #         "mkdocs-material",
    #         "mkdocstrings",
    #         "mkdocs-section-index",
    #         "mkdocs-literate-nav",
    #     ]
    # },
    install_requires=[],
    extras_require={},
)
