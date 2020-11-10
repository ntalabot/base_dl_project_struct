from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Basic structure for a Deep Learning project repository.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Nicolas Talabot",
    author_email="nicolas.talabot@gmail.com",
    url="https://github.com/ntalabot/base_dl_project_structure",
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
