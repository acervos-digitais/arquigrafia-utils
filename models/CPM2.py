import torch

from warnings import simplefilter
from transformers import AutoModel, AutoTokenizer

from .CPModel import CPModel

simplefilter(action="ignore")

class CPM2(CPModel):
  MODEL_NAME = "openbmb/MiniCPM-V-2"
  processor = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
  model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda", dtype=torch.bfloat16)
