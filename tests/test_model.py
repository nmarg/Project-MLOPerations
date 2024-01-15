# Unit tests for model
import torch
from src.models.model import make_model
from transformers import ViTForImageClassification

MODEL_PATH = "google/vit-base-patch16-224"
NUM_LABELS = 1

def test_make_model():
   """
   Positive test for the make_model function.

   Checks if the model is initialized correctly and the output shape of the
   model matches the expected shape given a random input tensor with the specified shape.
   """
   input_shape = (64, 3, 224, 224)
   expected_output_shape = (64, NUM_LABELS)

   # Create the model
   model = make_model(MODEL_PATH, NUM_LABELS)

   # Create a random input tensor with the specified shape
   input_tensor = torch.randn(input_shape)

   # Pass the input through the model
   with torch.no_grad():
      output = model(input_tensor)

   # Check if the model is an instance of ViTForImageClassification
   assert type(model) == ViTForImageClassification
   
   # Check if the number of labels is set correctly
   assert model.config.num_labels == NUM_LABELS
   
   # Check if classifier weights are initialized
   if model.classifier.weight.requires_grad:
      assert torch.all(model.classifier.weight != 0.0)
   
   # Check if classifier biases are initialized
   if model.classifier.bias is not None:
      assert torch.all(model.classifier.bias == 0.0)

   # Check if the output shape matches the expected shape
   assert output.logits.shape == expected_output_shape