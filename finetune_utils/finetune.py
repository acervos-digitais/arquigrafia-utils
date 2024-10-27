import datasets
import datetime

class FTUtils():
  @staticmethod
  def init(labels, superlabels):
    FTUtils.ID2LABEL = {i:l for i,l in enumerate(labels)}
    FTUtils.ID2SUPERLABEL = {i:l for i,l in enumerate(superlabels)}

    FTUtils.LABEL2ID = {v:int(k) for k,v in FTUtils.ID2LABEL.items()}
    FTUtils.LABEL2SUPERLABEL = {l:sl for l,sl in zip(labels, superlabels)}

    FTUtils.SUPERLABEL2SUPERID = {sl:si for si,sl in enumerate(set([l for l in FTUtils.ID2SUPERLABEL.values()]))}

    FTUtils.DATASET_INFO = {
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

    for i,l in FTUtils.ID2LABEL.items():
      FTUtils.DATASET_INFO["categories"].append({ "id": i, "name": l, "supercategory": FTUtils.ID2SUPERLABEL[i] })

    FTUtils.FEATURES = datasets.Features({
      "image_id": datasets.Value("int64"),
      "image": datasets.Image(decode=True),
      "image_filename": datasets.Value("string"),
      "width": datasets.Value("int64"),
      "height": datasets.Value("int64"),
      "objects": datasets.Sequence(feature={
        "bbox_id": datasets.Value("int64"),
        "category": datasets.ClassLabel(names=list(FTUtils.LABEL2ID.keys())),
        "bbox": datasets.Sequence(feature=datasets.Value("int64"), length=4),
        "super_category": datasets.ClassLabel(names=list(set(superlabels))),
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
