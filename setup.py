from setuptools import find_packages
from setuptools import setup


def readfile(filename):
    with open(filename, 'r+') as f:
        return f.read()


setup(
    name="image-eval",
    version="0.1.8",
    description="A library for evaluating image generation models",
    long_description=readfile("README.md"),
    long_description_content_type="text/markdown",
    author="Storia AI",
    author_email="founders@storia.ai",
    py_modules=["eval"],
    license=readfile("LICENSE"),
    packages=find_packages(include=["image_eval", "image_eval.*"]),
    entry_points={
        "console_scripts": [
            "image_eval= eval:main"
        ]
    },
    install_requires=open("requirements.txt").readlines(),
)
