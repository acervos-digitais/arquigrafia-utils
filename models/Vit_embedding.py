import torch

from transformers import ViTImageProcessor, ViTModel

from .EmbeddingModel import EmbeddingModel

class Vit(EmbeddingModel):
  MODEL_NAME = "google/vit-large-patch16-224-in21k"
  MODEL_NAME = "google/vit-large-patch32-224-in21k"
  processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
  model = ViTModel.from_pretrained(MODEL_NAME).to(EmbeddingModel.device)

  @classmethod
  def get_embedding(cls, imgs):
    inputs = cls.processor(images=imgs, return_tensors="pt").to(cls.device)

    with torch.no_grad():
      outputs = cls.model(**inputs)
      my_embedding = outputs.last_hidden_state[:, 0, :].detach().squeeze()

    return my_embedding
