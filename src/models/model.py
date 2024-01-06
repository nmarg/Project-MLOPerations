import torch

import hydra
from transformers import ViTForImageClassification

class VitTransformer(torch.nn.Module):
    """ A transformer containing a pretrained ViT model
    
    """
    @hydra.main(config_path="config", config_name="default_config.yaml")
    def __init__(self, config):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(
            config.model.pretrained_model_path,
            num_labels=len(config.model.num_labels)
        )