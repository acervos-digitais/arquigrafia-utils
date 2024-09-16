from transformers import pipeline
from warnings import simplefilter

from .TTSModel import TTSModel

simplefilter(action="ignore")

class TTSPt(TTSModel):
  MODEL_NAME = "facebook/mms-tts-por"
  pipeline = pipeline(model=MODEL_NAME, device="cuda")
