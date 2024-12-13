from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights

from .EmbeddingModel import EmbeddingModel

class EfficientNet(EmbeddingModel):
  MODEL_NAME = "efficientnet_b1"
  model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT).to(EmbeddingModel.device)
  layer = model.features
