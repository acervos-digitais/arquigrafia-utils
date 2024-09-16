from transformers import pipeline
from warnings import simplefilter

simplefilter(action="ignore")

class PartOfSpeech:
  MODEL_NAME = "QCRI/bert-base-multilingual-cased-pos-english"
  pipeline = pipeline(model=MODEL_NAME, device="cuda")

  @staticmethod
  def get_nouns(txt):
    pos = PartOfSpeech.pipeline(txt)

    nouns = []
    for o in pos:
      if o["entity"].startswith("NN"):
        if o["word"].startswith("#") and len(nouns) > 1:
          nouns[-1] = nouns[-1] + o["word"].replace("#", "")
        elif not o["word"].startswith("#"):
          nouns.append(o["word"])

    return ", ".join(nouns)


class CaptionModel:
  @classmethod
  def caption(cls, img):
    caption = cls.pipeline(img, max_new_tokens=200)[0]["generated_text"].lower()
    nouns = PartOfSpeech.get_nouns(caption)
    return "Picture of " + nouns
