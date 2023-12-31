from setuptools import find_packages
from setuptools import setup


def readfile(filename):
    with open(filename, 'r+') as f:
        return f.read()


setup(
    name="image-eval",
    version="0.1.2",
    description="A library for evaluating image generation models",
    long_description=readfile("README.md"),
    long_description_content_type="text/markdown",
    author="Storia AI",
    author_email="mihail@storia.ai",
    py_modules=["eval"],
    license=readfile("LICENSE"),
    packages=find_packages(include=["image_eval", "image_eval.*"]),
    entry_points={
        "console_scripts": [
            "image_eval= eval:main"
        ]
    },
    install_requires=[
        "image-reward==1.5",
        "lpips==0.1.4",
        "networkx==3.1",
        "piq==0.8.0",
        "pytorch-lightning==2.0.8",
        "streamlit==1.26.0",
        "sympy==1.12",
        "tabulate==0.9.0",
        "torch-fidelity==0.3.0",
    ]
)
