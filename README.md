# Base DL project structure
Basic structure for a Deep Learning project repository.

# TODO
* src:
  * Add tensorboard
  * Check data parallelism
* scripts: write scripts for training, gridsearch, evaluation, prediction (?)
* docs: write documentation
* Add How to use info in main README

## Project organization
    ├── .gitignore         <- Files/directories to ignore
    ├── LICENSE            <- License (MIT by default)
    ├── README.md          <- The top-level README for developers using this project.
    ├── data               <- Data folder (not tracked by default)
    │
    ├── docs               <- Documentation (not as a Sphinx project)
    │
    ├── models             <- Trained and serialized models (not tracked by default)
    │
    ├── notebooks          <- Jupyter notebooks
    │
    ├── scripts            <- Python scripts
    │
    ├── requirements.txt   <- The requirements file for reproducing the python environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    └── src                <- Source code for use in this project.
        ├── __init__.py    <- Makes src a Python module
        │
        ├── data           <- Scripts to download or generate data
        │   └── data.py
        │
        ├── models         <- Define, save and load models
        │   └── model.py
        │
        ├── train          <- Loss, optimizer, and train functions
        │   └── train.py
        │
        ├── eval           <- Evaluate the model, and make predictions
        │   └── eval.py
        │
        └── utils           <- General util functions
            ├── image.py
            └── visualization.py


## Installation
See installation steps in [Installation](docs/Installation.md).

## Documentation
Documentation available under `docs/`.
* Installation steps: [Installation](docs/Installation.md)
* How to use: [HowToUse](docs/HowToUse.md)
* Description of the package: [Package](docs/Package.md)
* Overview of the different scripts: [Scripts](docs/Scripts.md)
* Overview of the different jupyter notebooks: [Notebooks](docs/Notebooks.md)


## How to use
The scripts can be launched with:
```bash
python scripts/script.py [--option VALUE]
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
