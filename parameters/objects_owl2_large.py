from models.objects_models import Owl2Large as Owl2

OBJECTS_PATH = "./metadata/json/objects_large_owl"
DETR_OBJECTS_PATH = "./metadata/json/objects_large_detr"
CAPTIONS_PATH = "./metadata/json/captions"

DB_FILE_PATH = "./metadata/json/objects_large.json"

COLORS = ["COUNT", "HUE", "PALETTE"]

OBJECTS = [
  {
    "minaret": 0.3,
    "tower": 0.65,
    "railing": 0.4,
    "stair railing": 0.41,
    "guard railing": 0.4,
    "table": 0.375,
    "desk": 0.2,
    "chair": 0.17,
    "sculpture": 0.38,
    "painting": 0.35,
    "vertical pillar": 0.35,
    "stairs": 0.4,
    "stoop steps": 0.35,
    "stoop stairs": 0.35,
  },
  {
    "window": 0.25,
    "room door": 0.26,
    "building door": 0.25,
    "masonry": 0.2,

    "concrete wall": 0.22,
    "exposed concrete": 0.22,
    "concrete structure": 0.22,
    "poured concrete": 0.22,
  
    "glass window": 0.2,
    "glass door": 0.2,
    "mirror": 0.2,
  },
  {
    "wood fence": 0.3,
    "wood railing": 0.35,
    "wood pillar": 0.55,
    "wood door": 0.21,
    "wood board": 0.21,

    "metal fence": 0.4,
    "metal railing": 0.22,
    "wrought": 0.2,
  },
  {
    "tree": 0.2,
    "grass": 0.2,
    "shrub": 0.2,
    "bush": 0.2,
    "flower": 0.2,
    "vegetation": 0.2,
    "greenery": 0.2,
  },
    {
    "person": 0.23,
    "people": 0.23,
    "human": 0.23,
    "animal": 0.8,
    "cat": 0.3,
    "dog": 0.3,
    "pigeon": 0.3,
    "bird": 0.3,
  },
  {
    "water": 0.31,
    "pool": 0.31,
    "reflecting pool": 0.31,
    "pond": 0.31,
    "lake": 0.31,
    "cloud": 0.275,
    "sky": 0.3
  },
  {
    "car": 0.3,
    "truck": 0.4,
    "vehicle": 0.72,
  },
  {
    "street sign": 0.275,
    "placard": 0.275,
    "signboard": 0.275,
    "billboard": 0.275,
    "sign": 0.8
  },
  {
    "awning": 0.3,
    "balcony": 0.33
  }
]

DETR_OBJECTS = {
  "awning": 0.4,
  "balcony": 0.4,
  "ramp": 0.5,
}
