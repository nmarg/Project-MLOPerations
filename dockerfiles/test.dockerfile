FROM python:3.8-slim

COPY test.py test.py
WORKDIR /

RUN pip install google-cloud-storage

ENTRYPOINT ["python", "test.py"]
