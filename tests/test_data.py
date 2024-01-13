import os
from glob import glob

import numpy as np
import pytest
from torch import Tensor
from transformers.image_processing_utils import BatchFeature

from src.data.make_dataset import CelebADataModule, CustomImageDataset
from tests import _DATA_TESTING_ROOT

_TESTING_IMAGE_FOLDER = os.path.join(_DATA_TESTING_ROOT, "images")
datamodule = CelebADataModule(processed_data_dir=_DATA_TESTING_ROOT)
datamodule.setup(light_weight=True)


@pytest.mark.skipif(
    len(sorted(glob(f"{_TESTING_IMAGE_FOLDER}/*.jpg"))) >= 5000,
    reason="Processed images already exist",
)
def test_process_data():
    # Usage: Process Data
    datamodule.process_data(reduced=True)  # process 5k images
    images = sorted(glob(f"{_TESTING_IMAGE_FOLDER}/*.jpg"))
    assert len(images) >= 5000, "Images processed incorrectly. Less than 5000 samples in data/processed folder."


def test_attribute_names():
    attrnames = datamodule.attribute_names()
    assert len(attrnames) == 40, "Attribute names of size less than 40"
    assert attrnames[0] == "5_o_Clock_Shadow", "Attribute names wrong"


def test_dataloaders():
    # Note: testing with light_weight == True
    valloader = datamodule.val_dataloader()
    testloader = datamodule.test_dataloader()
    trainloader = datamodule.train_dataloader()
    assert len(trainloader) == 1
    assert len(valloader) == 1
    assert len(testloader) == 1


def test_example_data():
    example = datamodule.show_examples()
    pixel_vals = example["pixel_values"]
    labels = example["labels"]

    assert isinstance(example, BatchFeature), "Samples are loaded incorrectly. They should be of type BatchFeature."
    assert isinstance(pixel_vals, Tensor), "Samples should contain a tensor in their 'pixel_values' key."
    assert isinstance(labels, Tensor), "Samples should contain a tensor in their 'labels' key."

    size, channels, height, width = pixel_vals.shape
    assert channels == 3, "pixel_values are not storing RGB channels in their axis 1"
    assert height == 224 and width == 224, "pixel_values not stored as 28x28"


def test_custom_image_dataset():
    testimg = sorted(glob(f"{_TESTING_IMAGE_FOLDER}/*.jpg"))[:10]
    testlab = np.genfromtxt(os.path.join(_DATA_TESTING_ROOT, "labels.csv"), delimiter=",")[:10]
    testdataset = CustomImageDataset(testimg, testlab)

    assert testdataset.__len__() == 10, "CustomImageDataset is not initializing with the correct size."

    sample = testdataset.__getitem__(idx=0)
    sample_pixel_vals = sample["pixel_values"]
    sample_labels = sample["labels"]

    assert isinstance(sample, BatchFeature), "Samples are loaded incorrectly. They should be of type BatchFeature."
    assert isinstance(sample_pixel_vals, Tensor), "Samples should contain a tensor in their 'pixel_values' key."
    assert isinstance(sample_labels, Tensor), "Samples should contain a tensor in their 'labels' key."

    size, channels, height, width = sample_pixel_vals.shape
    assert channels == 3, "pixel_values are not storing RGB channels in their axis 1"
    assert height == 224 and width == 224, "pixel_values not stored as 28x28"
