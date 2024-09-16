IMAGES_PATH = "../../imgs/arquigrafia"
OBJECTS_PATH = "./metadata/json/objects"
CAPTIONS_PATH = "./metadata/json/captions"

DB_FILE_PATH = "./metadata/json/objects.json"

OBJECTS = [
  {
    "minaret": 0.25,
    "tower": 0.6,
    "railing": 0.4,
    "stair railing": 0.41,
    "guard railing": 0.4,
    "table": 0.45,
    "desk": 0.25,
    "chair": 0.24,
    "inclined walkway": 0.32,
    "sculpture": 0.4,
    "painting": 0.4,
    "vertical pillar": 0.35,
    "stairs": 0.4,
    "stoop steps": 0.35,
    "stoop stairs": 0.35,
  },
  {
    "window": 0.2,
    "room door": 0.25,
    "building door": 0.22,
    "masonry": 0.2,

    "concrete wall": 0.2,
    "exposed concrete": 0.2,
    "concrete structure": 0.2,
    "poured concrete": 0.2,
  
    "glass window": 0.2,
    "glass door": 0.2,
    "mirror": 0.2,
  },
  {
    "wood fence": 0.3,
    "wood railing": 0.35,
    "wood pilar": 0.3,
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
    "person": 0.35,
    "animal": 0.8,
    "cat": 0.3,
    "dog": 0.3,
    "pigeon": 0.3,
    "bird": 0.3,
  },
  {
    "water": 0.3,
    "pool": 0.3,
    "reflecting pool": 0.3,
    "pond": 0.3,
    "lake": 0.3,
    "cloud": 0.3,
    "sky": 0.3
  },
  {
    "car": 0.32,
    "truck": 0.42,
    "vehicle": 0.75,
  }
]

OBJECT2LABEL = {
  "minaret": "tower",
  "stair railing": "railing",
  "guard railing": "railing",
  "stoop steps": "stairs",
  "stoop stairs": "stairs",
  "desk": "table",
  "room door": "building door",

  "exposed concrete": "concrete wall",
  "concrete structure": "concrete wall",
  "poured concrete": "concrete wall",

  "glass window": "mirror",
  "glass door": "mirror",

  "wood railing": "wood fence",
  "wood pilar":"wood fence",
  "wood door": "wood fence",
  "wood board": "wood fence",

  "metal fence": "wrought",
  "metal railing": "wrought",

  "tree": "greenery",
  "grass": "greenery",
  "shrub": "greenery",
  "bush": "greenery",
  "flower": "greenery",
  "vegetation": "greenery",

  "cat": "animal",
  "dog": "animal",
  "pigeon": "animal",
  "bird": "animal",

  "pool": "water",
  "reflecting pool": "water",
  "pond": "water",
  "lake": "water",

  "car": "vehicle",
  "truck": "vehicle",
}

LABEL2DISPLAY = {
  "tower": "chimney/tower",
  "railing": "railing/banister",
  "inclined walkway": "ramp",
  "stairs": "stairs",
  "table": "table",
  "building door": "door",
  "concrete wall": "concrete",
  "masonry": "masonry",
  "mirror": "glass",
  "wood fence": "wood",
  "wrought": "metal",
  "greenery": "vegetation",
  "chair": "chair",
  "painting": "painting",
  "sculpture": "sculpture",
  "vertical pillar": "pillar",
  "window": "window",

  "animal": "animal",
  "person": "person",
  "water": "water",
  "cloud": "cloud",
  "sky":"sky",

  "vehicle": "vehicle",
}
