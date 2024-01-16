# Base image
FROM python:3.11-slim

ENV HOST 0.0.0.0
ENV PORT 8080

EXPOSE 8080

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/testing data/testing
COPY data/drifting data/drifting
COPY models/model0 models/model0

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install -r requirements_dev.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT exec uvicorn src.server.main:app --port 8080 --host 0.0.0.0
# CMD exec uvicorn src.server.main:app --port $PORT --host 0.0.0.0
# ["exec", "uvicorn", "src.server.main:app", "--port", $PORT, "--host", "0.0.0.0"]