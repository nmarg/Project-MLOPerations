from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
from typing import List
from transformers.image_processing_utils import BatchFeature


def transform_image(image_path: str, processor: ViTImageProcessor) -> BatchFeature:
    """
    Loads the image from given path and transforms it so it fits the model input.
    """
    # Load the image
    image = Image.open(image_path)
    image_tensor = processor(image, return_tensors="pt")
    return image_tensor


def predict(model: torch.nn.Module, image: BatchFeature) -> List[str]:
    """
    Predict if the person in the image is attractive
    """
    with torch.no_grad():
        output = model(**image).logits
        output = torch.sigmoid(output)
        attractive = (output > 0.5)[0, 0]
    return "Attractive" if attractive else "Not attractive"


if __name__ == "__main__":
    model_path = "models/model0"
    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()
    processor = ViTImageProcessor.from_pretrained(model_path)
    image = transform_image("data/testing/images/image_0.jpg", processor)
    atts = predict(model, image)
    print(atts)
