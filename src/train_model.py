import os

import evaluate
import torch
from datasets import DatasetDict
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

from data.make_dataset import CelebADataModule
import hydra


_SRC_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)


def collate_fn(batch):
    """
    The data collator function.
    Used internally by tranformers.Trainer
    """
    return {
        "pixel_values": torch.cat([x["pixel_values"] for x in batch], dim=0).float(),
        "labels": torch.stack([x["labels"] for x in batch]).float(),
    }


@hydra.main(config_path=os.path.join(_PROJECT_ROOT, "config/model"), config_name="model_config.yaml", version_base=None)
def train(cfg):
    """
    Train the model on processed data.
    """

    # initialize the input dataset
    datamodule = CelebADataModule(cfg.batch_size)

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
    # macro-averaging => the accuracy will be computed globally by counting the total true positives, false negatives, and false positives.
    metric = evaluate.load("accuracy", average=cfg.metric)

    def compute_metrics(p):
        """
        Function used internally by tranformers.Trainer
        """
        preds = p.predictions
        preds[preds > 0] = 1
        preds[preds <= 0] = 0
        preds = preds.flatten()
        refs = p.label_ids.flatten()
        acc = metric.compute(predictions=preds, references=refs)
        return acc

    # load the pretrained model
    model_name_or_path = cfg.pretrained_model_path
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)

    # create the model for fine-tuning
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=40,
        ignore_mismatched_sizes=True,
        problem_type="multi_label_classification",
    )

    # define the training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=6,
        evaluation_strategy="steps",
        num_train_epochs=cfg.epochs,
        save_steps=1,
        eval_steps=1,
        logging_steps=1,
        learning_rate=cfg.lr,
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
    savedir = find_free_directory(cfg.model_output_dir)
    trainer.save_model(savedir)
    print(f"Saved model under {savedir}")
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()


def find_free_directory(savedir):
    # Create folder for save directory, incrementing by 1 until new folder found
    index = 0
    while 1:
        dir = os.path.join(savedir, f"model{index}")
        if os.path.exists(dir):
            index += 1
            continue
        else:
            os.makedirs(dir)
            return dir


if __name__ == "__main__":
    train()
