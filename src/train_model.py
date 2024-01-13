import os

import evaluate
import torch
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments, ViTForImageClassification, ViTImageProcessor, set_seed

from data.make_dataset import CelebADataModule
import hydra
from sklearn.metrics import f1_score


_SRC_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)


def collate_fn(batch):
    """
    The data collator function.
    Used internally by tranformers.Trainer
    """
    data = {
        "pixel_values": torch.cat([x["pixel_values"] for x in batch], dim=0).to(torch.float32),
        "labels": torch.stack([x["labels"] for x in batch]).to(torch.float32),
    }
    return data


@hydra.main(config_path=os.path.join(_PROJECT_ROOT, "config/model"), config_name="model_config.yaml", version_base=None)
def train(cfg):
    """
    Train the model on processed data.
    """

    # set seed
    if cfg.reproducible_experiment:
        set_seed(cfg.seed)

    # initialize the input dataset
    datamodule = CelebADataModule(cfg.batch_size)

    # Usage: Load Data & Get Dataloaders
    datamodule.setup(light_weight=cfg.light_weight)
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
        # """
        # Function used internally by tranformers.Trainer
        # """
        # preds = p.predictions
        # preds[preds > 0] = 1
        # preds[preds <= 0] = 0
        # preds = preds.flatten()
        # refs = p.label_ids.flatten()
        # acc = metric.compute(predictions=preds, references=refs)
        # return acc
        """
        Function used internally by transformers.Trainer
        """
        # Extract the predictions and true labels
        preds = p.predictions
        label_ids = p.label_ids

        # Apply a threshold to turn probabilities into binary predictions
        threshold = 0.5
        preds = (preds > threshold).astype(int)
        print(preds)

        # Compute the F1 score
        f1 = f1_score(label_ids, preds, average="weighted")

        return {"f1": f1}

    # load the pretrained model
    model_name_or_path = cfg.pretrained_model_path
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)

    # create the model for fine-tuning
    model = ViTForImageClassification.from_pretrained(
        model_name_or_path,
        num_labels=cfg.num_labels,
        ignore_mismatched_sizes=True,
        problem_type="multi_label_classification",
    )

    # Manually initialize the classification layer
    # This step is necessary to get reproducible results
    if model.classifier.weight.requires_grad:  # Check if it's a newly added layer
        torch.nn.init.xavier_uniform_(model.classifier.weight)
        torch.nn.init.zeros_(model.classifier.bias)
    model.train()

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
