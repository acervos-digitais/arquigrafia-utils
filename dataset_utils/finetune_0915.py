from .finetune import FTUtils as _FTUtils

class FTUtils(_FTUtils):
  LABELS = [
    "awning",
    "balcony",
    "ramp",
  ]

  SUPERLABELS = [
    "object",
    "object",
    "object",
  ]
  _FTUtils.init(LABELS, SUPERLABELS)
