from torchvision.models import resnet34, ResNet34_Weights

from .EmbeddingModel import EmbeddingModel

class ResNet(EmbeddingModel):
  MODEL_NAME = "resnet34"
  model = resnet34(weights=ResNet34_Weights.DEFAULT).to(EmbeddingModel.device)
  layer = model._modules.get("avgpool")
