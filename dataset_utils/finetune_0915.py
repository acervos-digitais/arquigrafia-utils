from .finetune import FTUtils as _FTUtils

class FTUtils(_FTUtils):
  LABELS = [
    "awning",
    "balcony",
    "ramp",
    "sign",
    "notawning",
    "notbalcony",
    "notramp",
    "notsign",
  ]

  SUPERLABELS = [
    "object",
    "object",
    "object",
    "object",
    "notobject",
    "notobject",
    "notobject",
    "notobject",
  ]
  _FTUtils.init(LABELS, SUPERLABELS)
