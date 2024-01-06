import csv
import math
import os
from pathlib import Path
from typing import List

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import TensorType, ViTImageProcessor

MAX_DATASET_LENGTH = 202599


class CustomImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        """Custom dataset that loads images and labels, then returns them on demand into a DataLoader.
        Outputs a tuple with the format (label, image) in each output.

        :param image_paths: Paths to the images to be loaded (not directories, specific file paths!)
        :param label_rows: Rows from the label.csv file that correspond to the loaded images
        """
        self.images = images
        self.labels = labels
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        image = self.transform(image)
        label = self.labels[idx]
        return (label, image)


class CelebADataModule:
    def __init__(
        self,
        batch_size: int = 64,
    ):
        """Custom data module class for the CelebA dataset. Used for processing
        & loading of data, train/test splitting and constructing dataloaders.

        :param raw_data_dir: Path object leading to raw data, defaults to "path/to/dir"
        :param processed_data_dir: _description_, defaults to "path/to/dir"
        :param batch_size: _description_, defaults to 64
        """
        super().__init__()
        self.raw_data_dir = Path(__file__).resolve().parent.parent.parent / "data/raw/"
        self.processed_data_dir = (
            Path(__file__).resolve().parent.parent.parent / "data/processed/"
        )
        self.processed_attributes_path = Path.joinpath(
            self.processed_data_dir, "attributenames.txt"
        )
        self.processed_labels_path = Path.joinpath(
            self.processed_data_dir, "labels.csv"
        )
        self.processed_images_path = Path.joinpath(self.processed_data_dir, "images/")

        self.batch_size = batch_size

    def setup(self, use_portion_of_dataset=1.0, train_val_test_split=[0.6, 0.2, 0.2]):
        """Setup the data module, loading .jpg images from data/processed/ and splitting
        training, testing, and validation data.

        Note: if use_percent_of_dataset == 1.0, data will be split
        according to the recommended indices:
            - 1-162770: Training
            - 162771-182637: Validation,
            - 182638-202599 Testing.

        :param use_percent_of_dataset: portion of original dataset to use, defaults to 1.0
        :param train_val_test_split: how to split training, validation
        and test data if use_percent_of_dataset != 1.0, defaults to [0.6, 0.2, 0.2]
        """
        # Load attribute names, labels & image paths
        self.attributenames = np.loadtxt(
            self.processed_attributes_path, dtype=str, delimiter=","
        )
        labels = np.genfromtxt(
            self.processed_labels_path,
            delimiter=",",
        )
        images = sorted((self.processed_images_path).glob("*.jpg"))

        # Calculate portion & splits of dataset
        available_data = math.floor(len(images) * use_portion_of_dataset)
        images = images[:available_data]
        train_idx = [
            162770
            if use_portion_of_dataset == 1.0 and len(images) == MAX_DATASET_LENGTH
            else math.floor(available_data * train_val_test_split[0])
        ]
        val_idx = [
            182637
            if use_portion_of_dataset == 1.0 and len(images) == MAX_DATASET_LENGTH
            else math.floor(
                available_data * (train_val_test_split[0] + train_val_test_split[1])
            )
        ]
        print(
            f"Splitting train/val/test as: [{train_idx[0]}, {val_idx[0]-train_idx[0]}, {len(images)-val_idx[0]}]"
        )

        # Create datasets based on splits
        self.train_dataset = CustomImageDataset(
            images[: train_idx[0]], labels[: train_idx[0]]
        )
        self.val_dataset = CustomImageDataset(
            images[train_idx[0] : val_idx[0]], labels[train_idx[0] : val_idx[0]]
        )
        self.test_dataset = CustomImageDataset(
            images[val_idx[0] :], labels[val_idx[0] :]
        )

    def train_dataloader(self):
        """Return a train dataloader with the requested split specified in self.setup()

        :return: a DataLoader object with the train data as (label, image)
        """
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        """Return the evaluation set dataloader with the requested split specified in self.setup()

        :return: a DataLoader object with the val data as (label, image)
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """Return a test dataloader with the requested split specified in self.setup()

        :return: a DataLoader object with the test data as (label, image)
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def process_data(self, reduced=False):
        """Process images in the raw_data_dir directory and output them into
        the processed_data_dir directory as .jpg files.

        :param reduced: Only process a reduced amount (5k samples), defaults to False
        """
        # read attribute names & save them
        labelfile = Path.joinpath(self.raw_data_dir, "list_attr_celeba.csv")
        with open(labelfile, "r") as file:
            csv_reader = csv.reader(file)
            attributenames = next(csv_reader)
        attributenames = np.asarray(attributenames)[1:]
        np.savetxt(
            self.processed_attributes_path, attributenames, delimiter=",", fmt="%s"
        )
        print(
            f"Successfully saved attribute names under {self.processed_attributes_path}"
        )

        # read labels from raw data & save as labels.csv
        labels = np.genfromtxt(
            Path.joinpath(self.raw_data_dir, "list_attr_celeba.csv"),
            skip_header=1,
            delimiter=",",
        )
        labels = labels[:, 1:]  # drop image_id column, now shape [202599, 40]
        np.savetxt(self.processed_labels_path, labels, delimiter=",")
        print(f"Successfully saved labels under {self.processed_labels_path}")

        # Process images
        raw_images = sorted(
            Path.joinpath(self.raw_data_dir, "images_celeba").glob("*.jpg")
        )
        if len(raw_images) == 0:
            raise Exception(
                f"No images detected in directory {Path.joinpath(self.raw_data_dir, 'images_celeba')}. Make sure the raw input images are set in the right place."
            )
        if reduced:  # for debugging, only process 5000 of the available images
            raw_images = raw_images[:5000]
            all_images = 5000
        else:
            all_images = len(raw_images)

        if not self.processed_images_path.exists():
            self.processed_images_path.mkdir(parents=True)

        processor = ViTImageProcessor().from_pretrained("google/vit-base-patch16-224")
        for raw_image_id, raw_image_path in enumerate(raw_images):
            if raw_image_id % 1000 == 0:
                print(f"Processed {raw_image_id} of {all_images} images")
            image = Image.open(raw_image_path)
            image_tensor = processor.preprocess(
                image, return_tensors=TensorType.PYTORCH
            )
            torchvision.utils.save_image(
                image_tensor["pixel_values"],
                Path.joinpath(self.processed_images_path, f"image_{raw_image_id}.jpg"),
            )

        print("Successfully processed all images.")

    def attribute_names(self) -> List[str]:
        """Return the attribute names of the image labels.

        :return: List[str] of attribute names with dimension [40]
        """
        return self.attributenames

    def show_examples(self):
        print(self.train_dataset[0])


if __name__ == "__main__":
    # Usage: Process Data
    datamodule = CelebADataModule()
    datamodule.process_data(
        reduced=False
    )  # Change reduced=True to process only 5k images

    # Usage: Load Data & Get Dataloaders
    datamodule.setup()
    trainloader = datamodule.train_dataloader()
    valloader = datamodule.val_dataloader()
    testloader = datamodule.test_dataloader()
