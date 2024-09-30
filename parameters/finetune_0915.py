IMAGES_PATH = "../../imgs/arquigrafia"
AUDIO_PATH = "../../mp3s/captions"
OBJECTS_PATH = "./metadata/json/objects-ft-0915"
CAPTIONS_PATH = "./metadata/json/captions-ft-0915"

DB_FILE_PATH = "./metadata/json/objects-ft-0915.json"

COLORS = None

OBJECTS = [
  {
    "inclined walkway": 0.32,
    "pedestrian ramp": 0.25,
    "ramp": 0.8,
    "door overhang": 0.25,
    "awning": 0.25,
    "balcony": 0.25,
    "window deck": 0.25,
    "street sign": 0.25,
    "placard": 0.25,
    "signboard": 0.25,
    "billboard": 0.25,
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
