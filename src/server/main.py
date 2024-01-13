from fastapi import FastAPI, UploadFile, File
from src.predict_model import predict, load_attribute_names
from transformers import ViTForImageClassification, ViTImageProcessor
from io import BytesIO
from PIL import Image
from http import HTTPStatus

# TODO: incorporate hydra in this

app = FastAPI()

model_path = "models/model0"
att_names = load_attribute_names()
model = ViTForImageClassification.from_pretrained(model_path)
model.eval()
processor = ViTImageProcessor.from_pretrained(model_path)


@app.post("/predict/")
async def server_predict(data: UploadFile = File(...)):
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
    inference = predict(model, image_tensor, att_names)
    response = {
        "inference": inference,
        "message": HTTPStatus.OK.phrase,
    }
    return response
