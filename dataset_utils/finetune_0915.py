import datasets
import datetime

class FTUtils():
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

  ID2LABEL = {i:l for i,l in enumerate(LABELS)}
  ID2SUPERLABEL = {i:l for i,l in enumerate(SUPERLABELS)}

  LABEL2ID = {v:int(k) for k,v in ID2LABEL.items()}
  LABEL2SUPERLABEL = {l:sl for l,sl in zip(LABELS, SUPERLABELS)}

  SUPERLABEL2SUPERID = {sl:si for si,sl in enumerate(set([l for l in ID2SUPERLABEL.values()]))}

  DATASET_INFO = {
    "info": {
      "year": 2024,
      "version": "1.0.0",
      "description": "Object Detection dataset to detect architectural objects in images",
      "contributor": "Thiago Hersan",
      "url": "https://huggingface.co/datasets/#",
      "date_created": "%s" % datetime.datetime.now(),
    },
    "categories": [],
    "licenses": [
      { "id": 0, "name": "CC0 1.0", "url": "https://creativecommons.org/publicdomain/zero/1.0/", },
      { "id": 1, "name": "CC BY-NC 2.0", "url": "https://creativecommons.org/licenses/by-nc/2.0/", }
    ],
    "references": [
      { "id": 0, "name": "Training Generative Adversarial Networks with Limited Data", "url": "https://doi.org/10.48550/arXiv.2006.06676" }
    ],
    "images": [],
    "annotations": [],
  }

  for i,l in ID2LABEL.items():
    DATASET_INFO["categories"].append({ "id": i, "name": l, "supercategory": ID2SUPERLABEL[i] })

  FEATURES = datasets.Features({
    "image_id": datasets.Value("int64"),
    "image": datasets.Image(decode=True),
    "image_filename": datasets.Value("string"),
    "width": datasets.Value("int64"),
    "height": datasets.Value("int64"),
    "objects": datasets.Sequence(feature={
      "bbox_id": datasets.Value("int64"),
      "category": datasets.ClassLabel(names=list(LABEL2ID.keys())),
      "bbox": datasets.Sequence(feature=datasets.Value("int64"), length=4),
      "super_category": datasets.ClassLabel(names=list(set(SUPERLABELS))),
      "area": datasets.Value("int64"),
      "is_crowd": datasets.Value("bool")
    })
  })

  @staticmethod
  def get_dataset_info():
    return datasets.DatasetInfo(
      description=FTUtils.DATASET_INFO["info"]["description"],
      homepage=FTUtils.DATASET_INFO["info"]["url"],
      version=FTUtils.DATASET_INFO["info"]["version"],
      license=FTUtils.DATASET_INFO["licenses"][0]["name"],
      features=FTUtils.FEATURES
    )

  @staticmethod
  def xyxy_pct_to_xywh(box, iw, ih):
    return [
      int(box[0] * iw),
      int(box[1] * ih),
      int((box[2] - box[0]) * iw),
      int((box[3] - box[1]) * ih),
    ]

  @staticmethod
  def as_coco(img_id, objects):
    coco_anns = []
    for i in range(len(objects["category"])):
      coco_anns.append({
        "image_id": img_id,
        "category_id": objects["category"][i],
        "iscrowd": 0,
        "area": objects["area"][i],
        "bbox": list(objects["bbox"][i]),
      })

    return {
      "image_id": img_id,
      "annotations": coco_anns,
    }
