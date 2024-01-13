import torch
from transformers import ViTForImageClassification


def make_model(model_path, num_labels):
    """
    Create the model for fine-tunning from a pretrained one
    """
    model = ViTForImageClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    # this initializes the params in order for the code to be reproducible
    if model.classifier.weight.requires_grad:
        torch.nn.init.xavier_uniform_(model.classifier.weight)
        torch.nn.init.zeros_(model.classifier.bias)
    return model
