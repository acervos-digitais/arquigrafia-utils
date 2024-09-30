IMAGES_PATH = "../../imgs/arquigrafia"
OBJECTS_PATH = "./metadata/json/objects-ft-0915"

DB_FILE_PATH = "./metadata/json/objects-ft-0915.json"

COLORS = None

OBJECTS = [
  {
    "inclined walkway": 0.25,
    "pedestrian ramp": 0.2,
    "ramp": 0.25,
    "door overhang": 0.25,
    "door canopy": 0.25,
    "awning": 0.25,
    "balcony": 0.25,
    "window deck": 0.25,
    "street sign": 0.25,
    "placard": 0.25,
    "signboard": 0.25,
    "billboard": 0.25,
    "sign": 0.8
  },
]

OBJECT2LABEL = {
  "inclined walkway": "ramp",
  "pedestrian ramp": "ramp",
  "door overhang": "awning",
  "window deck": "balcony",
  "street sign": "sign",
  "placard": "sign",
  "signboard": "sign",
  "billboard": "sign",
}

LABEL2DISPLAY = {
  "ramp": "ramp",
  "awning": "awning",
  "balcony": "balcony",
  "sign": "sign",
}
