import csv
from pathlib import Path
from typing import List

import numpy as np
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from transformers import TensorType, ViTImageProcessor


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
        # Load attribute names
        self.attributenames = np.loadtxt(
            self.processed_attributes_path, dtype=str, delimiter=","
        )

        # Load labels
        labels = np.genfromtxt(
            self.processed_labels_path,
            delimiter=",",
        )

        # Load images

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
            Path.joinpath(self.raw_data_dir, "images_celeba/").glob("*.jpg")
        )

        if reduced:  # for debugging, only process 5000 of the available images
            raw_images = raw_images[:5000]
            all_images = 5000
        else:
            all_images = len(raw_images)

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
                Path.joinpath(
                    self.processed_data_dir, "images/", f"image_{raw_image_id}.jpg"
                ),
            )

        print("Successfully processed all images.")

    # def train_dataloader(self):
    #     return DataLoader(self.mnist_train, batch_size=self.batch_size)

    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)

    # def predict_dataloader(self):
    #     return DataLoader(self.mnist_predict, batch_size=self.batch_size)

    def attribute_names(self) -> List[str]:
        """Return the attribute names of the image labels.

        :return: List[str] of attribute names with dimension [40]
        """
        return self.attributenames


if __name__ == "__main__":
    datamodule = CelebADataModule()
    datamodule.process_data(reduced=True)
    datamodule.setup()
