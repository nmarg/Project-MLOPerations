import os

import torch
from datasets import DatasetDict
from models.model import make_model
from transformers import Trainer, TrainingArguments, ViTImageProcessor, set_seed

from data.make_dataset import CelebADataModule
import hydra
import evaluate
import wandb

_SRC_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_SRC_ROOT)


def collate_fn(batch):
    """
    The data collator function.
    Used internally by tranformers.Trainer
    """
    data = {
        "pixel_values": torch.cat([x["pixel_values"] for x in batch], dim=0).to(torch.float32),
        "labels": torch.stack([x["labels"] for x in batch]).to(torch.float32).unsqueeze(-1),
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

    # Convert the Hydra config to a dictionary to be compatible with wandb
    cfg_dict = cfg_dict = {k: v for k, v in cfg.items()}

    wandb.init(
        project="ViT-image-classification",
        entity="mlops_team_77",
        config=cfg_dict
    )


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

    # load the pretrained model
    model_name_or_path = cfg.pretrained_model_path
    processor = ViTImageProcessor.from_pretrained(model_name_or_path)

    # create the model for fine-tuning
    model = make_model(model_name_or_path, cfg.num_labels)
    model.train()

    # logging gradients with wandb
    wandb.watch(model, log_freq=100)

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
        report_to="wandb",  # reporting to the wandb account
        load_best_model_at_end=True,
    )

    metric = evaluate.load("accuracy")

    def compute_metrics(p):
        """
        Used for calculating the accuracy
        Function used internally by transformers.Trainer
        """
        preds = (p.predictions > 0.5).astype(int)
        return metric.compute(predictions=preds, references=p.label_ids)

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

    # evaluate
    metrics = trainer.evaluate(dataset_dict["test"])
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


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
