import os

import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers.image_processing_utils import BatchFeature

PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "model0")
TEST_DATA_PATH = os.path.join(PROJECT_DIR, "data", "testing", "images", "image_0.jpg")


def transform_image(image_path: str, processor: ViTImageProcessor) -> BatchFeature:
    """
    Loads the image from given path and transforms it so it fits the model input.
    """
    # Load the image
    image = Image.open(image_path)
    image_tensor = processor(image, return_tensors="pt")
    return image_tensor


def predict(model: torch.nn.Module, image: BatchFeature) -> str:
    """
    Predict if the person in the image is attractive
    """
    with torch.no_grad():
        output = model(**image).logits
        output = torch.sigmoid(output)
        attractive = (output > 0.5)[0, 0]
    return "Attractive" if attractive else "Not attractive"


if __name__ == "__main__":
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    model.eval()
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    image = transform_image(TEST_DATA_PATH, processor)
    result = predict(model, image)
    print(f"Model Prediction for {TEST_DATA_PATH}:")
    print(result)
