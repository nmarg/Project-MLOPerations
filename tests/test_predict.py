import pytest
from src.predict_model import load_attribute_names, transform_image, predict
import os
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers.image_processing_utils import BatchFeature
from tests import _PATH_MODEL


def test_load_attribute_names():
    att_names = load_attribute_names()
    assert len(att_names) == 40
    atts_to_check = ["Arched_Eyebrows", "Big_Nose", "High_Cheekbones", "Sideburns"]
    for att in atts_to_check:
        assert att in att_names


@pytest.mark.parametrize("image", ["image_0.jpg", "image_5.jpg", "image_17.jpg"])
@pytest.mark.parametrize("model_path", [os.path.join(_PATH_MODEL, "model0")])
def test_transform_image(image, model_path):
    image_path = os.path.join("data/processed/images", image)
    processor = ViTImageProcessor.from_pretrained(model_path)
    transformed = transform_image(image_path, processor)
    assert type(transformed) == BatchFeature


@pytest.mark.parametrize("image", ["image_1.jpg", "image_3.jpg", "image_9.jpg"])
@pytest.mark.parametrize("model_path", [os.path.join(_PATH_MODEL, "model0")])
def test_predict(image, model_path):
    image_path = os.path.join("data/processed/images", image)
    processor = ViTImageProcessor.from_pretrained(model_path)
    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()

    att_names = load_attribute_names()
    image = transform_image(image_path, processor)

    atts = predict(model, image, att_names)
    assert isinstance(atts, list)
