FROM python:3.8-slim

COPY test.py test.py
COPY data/drifting/current_data.csv data/drifting/current_data.csv
WORKDIR /

RUN pip install google-cloud-storage

ENTRYPOINT ["python", "test.py"]
