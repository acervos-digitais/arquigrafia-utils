import torch

from transformers import pipeline
from warnings import simplefilter

from .CaptionModel import CaptionModel

simplefilter(action="ignore")

class Blip(CaptionModel):
  MODEL_NAME = "Salesforce/blip-image-captioning-large"
  pipeline = pipeline("image-to-text", model=MODEL_NAME, device="cuda", torch_dtype=torch.float16)
