import transformers
from omegaconf import OmegaConf

from src.models.model import make_model
from src.train_model import train


def mock_cfg():
    """
    Provide configuration for the train function. Usually handled by hydra.
    """
    return OmegaConf.create(
        {
            "pretrained_model_path": "google/vit-base-patch16-224",
            "num_labels": 1,
            "batch_size": 64,
            "metric": "macro",
            "output_dir": "./training_outputs",
            "epochs": 1,
            "lr": 0.01,
            "model_output_dir": "./models",
            "reproducible_experiment": False,
            "seed": 15,
            "light_weight": True,
            "test": True,
            "cloud": False,
        }
    )


def test_train():
    """
    Test the training process. The test will fail only if there is an error thrown in the train method
    """
    train(mock_cfg())


def test_make_model():
    """
    Test the model factory.
    """
    model = make_model("google/vit-base-patch16-224", 1)
    assert type(model) == transformers.models.vit.modeling_vit.ViTForImageClassification
