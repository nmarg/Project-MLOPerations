FROM python:3.8-slim

COPY test.py test.py
WORKDIR /

ENTRYPOINT ["python", "test.py"]
