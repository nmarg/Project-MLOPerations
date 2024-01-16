import numpy as np

from http import HTTPStatus
from io import BytesIO
from csv import writer
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from PIL import Image, ImageStat
from transformers import ViTForImageClassification, ViTImageProcessor
from datetime import datetime
from src.predict_model import predict
from src.data.make_reference_data import calculate_image_params
from google.cloud import storage

app = FastAPI()

model_path = "models/model0"
model = ViTForImageClassification.from_pretrained(model_path)
model.eval()
processor = ViTImageProcessor.from_pretrained(model_path)

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

def save_image_prediction(image, inference):
    image_params = calculate_image_params(image)
    csv_row = list(image_params)

    if inference == "Not attractive":
        inference = 0
    elif inference == "Attractive":
        inference = 1

    csv_row.append(inference)

    with open('data/drifting/current_data.csv', 'a') as f_object:
        now = datetime.now()

        csv_row.append(now.strftime("%m/%d/%Y, %H:%M:%S"))

        writer_object = writer(f_object)

        writer_object.writerow(csv_row)
    
        f_object.close()
    
    upload_to_gcs('project-mloperations-data', 'data/drifting/current_data.csv', 'data/drifting/current_data.csv')


@app.post("/predict/")
async def server_predict(background_tasks: BackgroundTasks, data: UploadFile = File(...)):
    """
    USAGE:
        curl -X 'POST' \
            'http://localhost:8000/predict/' \
            -H 'accept: application/json' \
            -H 'Content-Type: multipart/form-data' \
            -F 'data=@/path/to/your/image.jpg;type=image/jpeg'
    """
    # read the data
    content = await data.read()
    image_stream = BytesIO(content)
    image_stream.seek(0)

    # Open the image using PIL
    image = Image.open(image_stream)
    image_tensor = processor(image, return_tensors="pt")

    # get the predicitons from the model
    inference = predict(model, image_tensor)
    response = {
        "inference": inference,
        "message": HTTPStatus.OK.phrase,
    }

    background_tasks.add_task(save_image_prediction, image, inference)


    return response
