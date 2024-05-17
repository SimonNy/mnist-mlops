# Base Image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mnist_mlops/ mnist_mlops/
COPY data/ data/
RUN mkdir -p models/
RUN mkdir -p reports/figures


WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT [ "python", "-u", "mnist_mlops/train_model.py" ]