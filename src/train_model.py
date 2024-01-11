import evaluate
import numpy as np
import torch
from datasets import DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

from data.make_dataset import CelebADataModule


def collate_fn(batch):
    """
    The data collator function.
    Used internally by tranformers.Trainer
    """
    return {
        "pixel_values": torch.cat([x["pixel_values"] for x in batch], dim=0).float(),
        "labels": torch.stack([x["labels"] for x in batch]).float(),
    }


def train():
    """
    Train the model on processed data.
    """

    # initialize the input dataset
    datamodule = CelebADataModule()

    # Usage: Load Data & Get Dataloaders
    datamodule.setup()
    trainloader = datamodule.train_dataloader()
    valloader = datamodule.val_dataloader()
    testloader = datamodule.test_dataloader()

    dataset_dict = DatasetDict(
        {
            "train": trainloader.dataset,
            "validation": valloader.dataset,
            "test": testloader.dataset,
        }
    )

    # metric to compute -> accuracy in this case
    # micro-averaging => the accuracy will be computed globally by counting the total true positives, false negatives, and false positives.
    metric = evaluate.load("accuracy", average="macro")

    def compute_metrics(p):
        """
        Function used internally by tranformers.Trainer
        """
        preds = p.predictions
        preds[preds > 0] = 1
        preds[preds <= 0] = 0
        preds = preds.flatten().astype(np.int32).tolist()
        refs = p.label_ids.flatten().astype(np.int32).tolist()

        acc = metric.compute(predictions=preds, references=refs)
        return acc

    # load the pretrained model
    model_name_or_path = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)

    # create the model for fine-tuning
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path, num_labels=40, ignore_mismatched_sizes=True, problem_type="multi_label_classification"
    )

    # define the training arguments
    training_args = TrainingArguments(
        output_dir="./training_outputs",
        per_device_train_batch_size=6,
        evaluation_strategy="steps",
        num_train_epochs=2,
        fp16=False,
        save_steps=1,
        eval_steps=1,
        logging_steps=1,
        learning_rate=0.01,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )

    # define the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["validation"],
        tokenizer=processor,
    )

    # train and save the model and metrics
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()


if __name__ == "__main__":
    train()
