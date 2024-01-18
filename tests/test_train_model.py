from omegaconf import OmegaConf

from src.train_model import train


def test_model_training():
    # Test the training process with a small dummy dataset
    dummy_config = {
        "pretrained_model_path": "google/vit-base-patch16-224",
        "num_labels": 1,
        "batch_size": 64,
        "metric": "macro",
        "output_dir": "./test_training_outputs",
        "epochs": 1,
        "lr": 0.01,
        "model_output_dir": "./test_models",
        "reproducible_experiment": True,
        "seed": 15,
        "light_weight": True,
        "test": True,
    }
    config = OmegaConf.create(dummy_config)

    train(config)
