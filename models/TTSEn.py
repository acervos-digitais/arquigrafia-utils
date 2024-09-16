from transformers import pipeline
from warnings import simplefilter

from .TTSModel import TTSModel

simplefilter(action="ignore")

class TTSEn(TTSModel):
  MODEL_NAME = "facebook/mms-tts-eng"
  pipeline = pipeline(model=MODEL_NAME, device="cuda")
