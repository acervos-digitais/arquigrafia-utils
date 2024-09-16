from warnings import simplefilter
from transformers import AutoModel, AutoTokenizer

from .CPModel import CPModel

simplefilter(action="ignore")

class CPM2_6(CPModel):
  MODEL_NAME = "openbmb/MiniCPM-V-2_6-int4"
  processor = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
  model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
