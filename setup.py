from setuptools import setup


def readfile(filename):
    with open(filename, 'r+') as f:
        return f.read()


setup(
    name="image-eval",
    version="0.1",
    description="",
    long_description=readfile("README.md"),
    author="Storia AI",
    author_email="mihail@storia.ai/julia@storia.ai",
    url="",
    py_modules=["eval"],
    license=readfile("LICENSE"),
    entry_points={
        "console_scripts": [
            "image_eval= eval:main"
        ]
    },
)
