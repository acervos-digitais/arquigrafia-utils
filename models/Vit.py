import torch

from transformers import pipeline
from warnings import simplefilter

from .CaptionModel import CaptionModel

simplefilter(action="ignore")

class Vit(CaptionModel):
  MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
  pipeline = pipeline("image-to-text", model=MODEL_NAME, device="cuda", torch_dtype=torch.float16)
