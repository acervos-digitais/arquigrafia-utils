import torch

from transformers import CLIPProcessor, CLIPModel

from .EmbeddingModel import EmbeddingModel

class Clip(EmbeddingModel):
  MODEL_NAME = "openai/clip-vit-large-patch14"
  processor = CLIPProcessor.from_pretrained(MODEL_NAME)
  model = CLIPModel.from_pretrained(MODEL_NAME).to(EmbeddingModel.device)

  @classmethod
  def get_embedding(cls, imgs):
    inputs = cls.processor(images=imgs, return_tensors="pt", padding=True).to(cls.device)

    with torch.no_grad():
      my_embedding = cls.model.get_image_features(**inputs).detach().squeeze()

    return my_embedding
