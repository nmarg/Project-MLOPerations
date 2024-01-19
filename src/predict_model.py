import logging
import logging.config
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from PIL import Image
from rich.logging import RichHandler
from sklearn.metrics import precision_score
from transformers import ViTForImageClassification, ViTImageProcessor
from transformers.image_processing_utils import BatchFeature

import wandb

PROJECT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "model0")
TEST_DATA_PATH = os.path.join(PROJECT_DIR, "data", "testing", "images")
LABELS_PATH = LABELS_PATH = os.path.join(PROJECT_DIR, "data", "testing", "labels.csv")


# initialize wandb run
wandb.init(project="ViT-image-classification", entity="mlops_team_77")


LOGS_DIR = Path("logs")  # creates a Path object for the directory named `logs`
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # creates this directory

# setting up the app logging configuration
logging_config = {  # 4 parts: version, formatters, handlers, root
    "version": 1,
    "formatters": {  # Formatter -> determines how logs should be formatted -> here: 2 types
        "minimal": {"format": "%(message)s"},
        "detailed": {
            "format": "%(levelname)s %(asctime)s [%(name)s:%(filename)s:%(funcName)s:%(lineno)d]\n%(message)s\n"
        },
    },
    "handlers": {  # Handlers -> in charge of what should happen to different level of logging.
        "console": {
            "class": "logging.StreamHandler",
            "stream": sys.stdout,  # sends logs to the .stdout stream
            "formatter": "minimal",  # uses `minimal` format
            "level": logging.DEBUG,  # for level DEBUG and higher (here, all)
        },
        "info": {
            "class": "logging.handlers.RotatingFileHandler",  # sends it to files - `rotating` = up to some file size
            "filename": Path(LOGS_DIR, "info.log"),  # sends logs to the info.log file in the LOGS_DIR directory
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10,
            "formatter": "detailed",  # uses `detailed` format
            "level": logging.INFO,  # sends messages of level INFO and higher
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": Path(LOGS_DIR, "error.log"),  # sends logs to the error.log file in the LOGS_DIR directory
            "maxBytes": 10485760,  # 10 MB              # maximum bytes size before rotating
            "backupCount": 10,  # number of backup files to keep
            "formatter": "detailed",  # uses `detailed` format, too
            "level": logging.ERROR,  # only the high-level priority messages
        },
    },
    "root": {  # defines the default behaviour of the logging system => applies to all the loggers in the application
        "handlers": ["console", "info", "error"],
        "level": logging.DEBUG,  # minumum log level for the root logger
        "propagate": True,  # should the log be passed to handlers of higher-level priority as well
    },
}

# Apply the logging configuration
logging.config.dictConfig(logging_config)
logger = logging.getLogger(__name__)
logger.root.handlers[0] = RichHandler(markup=True)  # set rich handler




def transform_image(image_path: str, processor: ViTImageProcessor) -> BatchFeature:
    """
    Loads the image from given path and transforms it so it fits the model input.
    """
    try:
        image = Image.open(image_path)
        image_tensor = processor(image, return_tensors="pt")
        return image_tensor
    except Exception as e:
        logger.error(f"An error in transforming the image {image_path}: {e}")
        raise


def predict(model: torch.nn.Module, image: BatchFeature) -> str:
    """
    Predict if the person in the image is attractive
    """
    try:
        with torch.no_grad():
            output = model(**image).logits
            output = torch.sigmoid(output)
            attractive = (output > 0.5)[0, 0]
        return "Attractive" if attractive else "Not attractive"
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise


def load_test_data(test_images_directory, labels_path, light_weight) -> tuple[list, list]:
    """
    Loads all test images and their labels.
    """
    labels_df = pd.read_csv(labels_path, header=None)
    images, labels = [], []

    if light_weight:
        length = 5000
        labels = labels_df.iloc[:length, 0].values.tolist()
    else:
        length = len(labels_df)
        labels = labels_df.iloc[:, 0].values.tolist()

    for idx in range(length):
        image_name = f"image_{idx}.jpg"
        image_path = os.path.join(test_images_directory, image_name)

        try:
            images.append(image_path)
        except FileNotFoundError:
            log_message = f"File not found: {image_path}"
            logger.info(f"Something's wrong with loading images. {log_message}")
            wandb.log({"error": log_message})  # also to wandb
            continue

    for i in range(len(labels)):
        labels[i] = "Attractive" if labels[i] == 1 else "Not attractive"

    return images, labels


if __name__ == "__main__":
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    model.eval()
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)

    # getting the light_weight attribute from model_config.yaml
    CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "model", "model_config.yaml")
    model_config = yaml.safe_load(open(CONFIG_PATH, "r"))
    light_weight = model_config.get("light_weight", None)

    wandb.watch(model, log="all", log_freq=10)

    test_images_paths, true_labels = load_test_data(TEST_DATA_PATH, LABELS_PATH, light_weight)

    predictions = []

    for image_path in test_images_paths:
        transformed_image = transform_image(image_path, processor)
        result = predict(model, transformed_image)
        predictions.append(result)

        # Log each prediction with its corresponding image to wandb
        wandb.log(
            {"Predicted Label": 0 if result == "Attractive" else 1, "Image": wandb.Image(image_path, caption=result)}
        )

    precision = precision_score(true_labels, predictions, pos_label="Attractive")
    wandb.log({"precision": precision})
