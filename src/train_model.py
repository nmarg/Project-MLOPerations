
from data.make_dataset import CelebADataModule
from datasets import DatasetDict
import torch
import numpy as np
from datasets import load_metric
from transformers import TrainingArguments
from transformers import Trainer

datamodule = CelebADataModule()

datamodule.process_data(
      reduced=True
   )  # Change reduced=True to process only 5k images

   # Usage: Load Data & Get Dataloaders
datamodule.setup()
trainloader = datamodule.train_dataloader()
valloader = datamodule.val_dataloader()
testloader = datamodule.test_dataloader()

dataset_dict = DatasetDict({
    'train': trainloader.dataset,
    'validation': valloader.dataset,
    'test': testloader.dataset
})


def collate_fn(batch):
   images = [item[1] for item in batch]  # Extract images from the batch

   # Stack images into a tensor
   stacked_images = torch.stack(images)

   # Convert labels into a single NumPy array and then to a tensor
   labels = [item[0] for item in batch]  # Extract labels from the batch
   stacked_labels = torch.tensor(np.array(labels))

   return {
      'pixel_values': stacked_images,
      'labels': stacked_labels
   }


metric = load_metric("accuracy")
def compute_metrics(p):
   return metric.compute(predictions=np.argmax(p.predictions, axis=1), references=p.label_ids)


from transformers import ViTImageProcessor

model_name_or_path = 'google/vit-base-patch16-224-in21k'
processor = ViTImageProcessor.from_pretrained(model_name_or_path)

from transformers import ViTForImageClassification


model = ViTForImageClassification.from_pretrained(
    model_name_or_path,
    num_labels=40
)


training_args = TrainingArguments(
  output_dir="./vit-base-beans",
  per_device_train_batch_size=16,
  evaluation_strategy="steps",
  num_train_epochs=4,
  fp16=False,
  save_steps=100,
  eval_steps=100,
  logging_steps=10,
  learning_rate=2e-4,
  save_total_limit=2,
  remove_unused_columns=False,
  push_to_hub=False,
  report_to='tensorboard',
  load_best_model_at_end=True,
)

model_name_or_path = 'google/vit-base-patch16-224-in21k'

# print("prepared_ds[train]", type(prepared_ds["train"]))
trainer = Trainer(
   model=model,
   args=training_args,
   data_collator=collate_fn,
   compute_metrics=compute_metrics,
   train_dataset=dataset_dict["train"],
   eval_dataset=dataset_dict["validation"],
   tokenizer=processor,
)

train_results = trainer.train()
trainer.save_model()
trainer.log_metrics("train", train_results.metrics)
trainer.save_metrics("train", train_results.metrics)
trainer.save_state()



