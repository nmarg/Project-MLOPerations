# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/testing data/testing
COPY models/model0 models/model0

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_dev.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

CMD ["uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "80"]