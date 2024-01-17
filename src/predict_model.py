import os

from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers.image_processing_utils import BatchFeature
import torch
import wandb
import pandas as pd
from sklearn.metrics import precision_score
import yaml


PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "model0")
TEST_DATA_PATH = os.path.join(PROJECT_DIR, "data", "testing", "images")
LABELS_PATH = LABELS_PATH = os.path.join(PROJECT_DIR, "data", "testing", "labels.csv")


# initialize wandb run
wandb.init(
    project="ViT-image-classification",
    entity="mlops_team_77"
)


# getting the light_weight attribute from model_config.yaml
CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "model", "model_config.yaml")
model_config = yaml.safe_load(open(CONFIG_PATH, 'r'))
light_weight = model_config.get('light_weight', None)


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
  

def load_test_data(test_images_directory, labels_path):
    """
    Loads all test images and their labels.
    """
    labels_df = pd.read_csv(labels_path, header=None)
    images, labels = [], []

    if light_weight:
        length= 5000
        labels = labels_df.iloc[:length, 0].values.tolist()
    else:
        length = len(labels_df)
        labels = labels_df.iloc[:, 0].values.tolist()

    for idx in range(length):
        image_name =  f"image_{idx}.jpg"
        image_path = os.path.join(test_images_directory, image_name)

        try:
            images.append(image_path)
        except FileNotFoundError:
            print(f"File not found: {image_path}")  # TODO: also set up as an app log (maybe with wandb, maybe not)
            continue
    
    for i in range(len(labels)):
        labels[i] = "Attractive" if labels[i] == 1 else "Not attractive"

    return images, labels


if __name__ == "__main__":
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    model.eval()
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

    wandb.watch(model, log='all', log_freq=10)

    test_images_paths, true_labels = load_test_data(TEST_DATA_PATH, LABELS_PATH)

    predictions = []

    for image_path in test_images_paths:
        transformed_image = transform_image(image_path, processor)
        result = predict(model, transformed_image)
        predictions.append(result)

        # Log each prediction with its corresponding image to wandb
        wandb.log({
            "Predicted Label": 0 if result == "Attractive" else 1,
            "Image": wandb.Image(image_path, caption=result)
        })


        # print(f"Model Prediction for {image_path}:")
        # print(result)

    precision = precision_score(true_labels, predictions, pos_label = "Attractive")
    wandb.log({'precision': precision})
