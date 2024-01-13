import pytest
from src.predict_model import transform_image, predict
import os
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers.image_processing_utils import BatchFeature
from tests import _PATH_MODEL


@pytest.mark.parametrize("image", ["image_0.jpg", "image_5.jpg", "image_17.jpg"])
@pytest.mark.parametrize("model_path", [os.path.join(_PATH_MODEL, "model0")])
def test_transform_image(image, model_path):
    image_path = os.path.join("data/testing/images", image)
    processor = ViTImageProcessor.from_pretrained(model_path)
    transformed = transform_image(image_path, processor)
    assert type(transformed) == BatchFeature


@pytest.mark.parametrize("image", ["image_1.jpg", "image_3.jpg", "image_9.jpg"])
@pytest.mark.parametrize("model_path", [os.path.join(_PATH_MODEL, "model0")])
def test_predict(image, model_path):
    image_path = os.path.join("data/testing/images", image)
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()

    image = transform_image(image_path, processor)

    atts = predict(model, image)
    assert isinstance(atts, str)
