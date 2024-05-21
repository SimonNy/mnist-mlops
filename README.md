# mnist_mlops

Training a basic in a MLOps setup on the mnist dataset.

This repos uses the following stack:

- Python project using pyproject.toml
- Models in pytorch.
- Containerized using docker.
- Documentation in mkdocs.
- Linting and formatting with ruff
- Checking using pre-commit
- CI with Github actions
- Managing experiments with MLflow
- Defining hyperparameters with Hydra

## Get started

```bash
git clone git@github.com:SimonNy/mnist-mlops.git
cd mnist-mlops
make create_environment
conda activate mnist_mlops
make requirements
dvc pull
```

running a pipeline that trains, registers and score a test example with MLflow.

```bash
python run_pipeline.py
```

## Docker

Build training image for windows or linux machine.
```bash
docker build --platform linux/amd64 -f trainer.dockerfile . -t trainer:latest
```

Build training image for wMac with M1/M2
```bash
docker build --platform linux/arm64 -f trainer.dockerfile . -t trainer:latest
```

Mount a shared volume between local environment and dockercontainer
```bash
docker run --name {container_name} -v $(pwd)/models:/models/ trainer:latest
```

## Project structure

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── config               <- Configuration files for paths, hyperparameters, mlflow, etc.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
|   └── test             <- Contains sample data for testing the whole pipeline.
│
├── docker               <- Scripts defining docker images
|
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pipeline.yaml        <- Defines a training pipeline
├── run_pipeline.py      <- Script for running the pipelines defined in pipeline.yaml
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
|   ├── register_model.py   <- script for registering the model with MLflow
|   ├── run_experiment.py   <- script for running the mlflow experiment - utilizes functions in train.py
│   ├── train.py            <- script for training the model
│   └── score.py            <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
