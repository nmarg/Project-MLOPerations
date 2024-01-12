from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import numpy as np
from typing import List
from transformers.image_processing_utils import BatchFeature


def load_attribute_names() -> List[str]:
    """
    Load the attribute names.
    """
    attributenames = np.loadtxt("data/processed/attributenames.txt", dtype=str, delimiter=",")
    return attributenames


def transform_image(image_path: str, processor: ViTImageProcessor) -> BatchFeature:
    """
    Loads the image from given path and transforms it so it fits the model input.
    """
    # Load the image
    image = Image.open(image_path)
    image_tensor = processor(image, return_tensors="pt")
    return image_tensor


def predict(model: torch.nn.Module, image: BatchFeature, attribute_names: List[str]) -> List[str]:
    """
    Predict the labels in the image
    """
    with torch.no_grad():
        outputs = model(**image).logits
        outputs[outputs > 0] = 1
        outputs[outputs <= 0] = 0
    return [attribute_names[i] for i in range(outputs.size(1)) if outputs[0, i] == 1]


if __name__ == "__main__":
    model_path = "models/model0"
    att_names = load_attribute_names()
    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()
    processor = ViTImageProcessor.from_pretrained(model_path)
    image = transform_image("data/processed/images/image_20.jpg", processor)
    atts = predict(model, image, att_names)
    print(atts)
