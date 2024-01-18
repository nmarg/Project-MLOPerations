# Base image
FROM python:3.11-slim

ENV PYTHONPATH="/:${PYTHONPATH}"

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/testing/ data/testing/
COPY config/ config/

WORKDIR /
RUN mkdir training_outputs
RUN pip install -r requirements.txt
RUN pip install -r requirements_dev.txt
RUN pip install . --no-deps --no-cache-dir

CMD ["python", "-u", "src/train_model.py", "cloud=True"]
