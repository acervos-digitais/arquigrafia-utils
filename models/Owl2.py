import torch

from transformers import Owlv2Processor, Owlv2ForObjectDetection
from warnings import simplefilter

simplefilter(action="ignore")

class Owl2:
  OBJ_TARGET_SIZE = torch.Tensor([500, 500])
  MODEL_NAME = "google/owlv2-base-patch16-ensemble"

  obj_model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).to("cuda")
  obj_processor = Owlv2Processor.from_pretrained(MODEL_NAME)

  @classmethod
  def px_to_pct(cls, box, img_w, img_h):
    model_dims = cls.OBJ_TARGET_SIZE
    scale_factor = torch.tensor([max(img_w, img_h) / img_w , max(img_w, img_h) / img_h])
    return [round(x, 4) for x in (box.cpu().reshape(2, -1) / model_dims * scale_factor).reshape(-1).tolist()]

  @classmethod
  # filter if box "too large" or "too small"
  def threshold(cls, score, label, box, tholds, img_w, img_h):
    box_pct = cls.px_to_pct(box, img_w, img_h)
    box_width = box_pct[2] - box_pct[0]
    box_height = box_pct[3] - box_pct[1]
    good_min = box_width > 0.05 and box_height > 0.05
    good_max = box_width < 0.8 or box_height < 0.8
    return good_min and good_max and score > tholds[label.item()]

  @classmethod
  def run_object_detection(cls, img, labels_in, labels_out, tholds):
    input = cls.obj_processor(text=labels_in, images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
      obj_out = cls.obj_model(**input)

    res = cls.obj_processor.post_process_object_detection(outputs=obj_out, target_sizes=[cls.OBJ_TARGET_SIZE])
    slbs = zip(res[0]["scores"], res[0]["labels"], res[0]["boxes"])
    iw, ih = img.size

    detected_objs = [{"score": s, "label": labels_out[l.item()], "box": cls.px_to_pct(b, iw, ih)}
                     for s,l,b in slbs if cls.threshold(s, l, b, tholds, iw, ih)]
    return detected_objs

  @classmethod
  def top_objects(cls, img, labels_in, labels_out, tholds):
    detected_objs = cls.run_object_detection(img, labels_in, labels_out, tholds)

    # only keep the box with highest score per object
    detected_objs_boxes = {}
    high_score = {}

    for o in detected_objs:
      ol = o["label"]
      if (ol not in detected_objs_boxes) or (o["score"] > high_score[ol]):
        detected_objs_boxes[ol] = o["box"]
        high_score[ol] = o["score"]
  
    return detected_objs_boxes

  @classmethod
  def all_objects(cls, img, labels_in, labels_out, tholds):
    detected_objs = cls.run_object_detection(img, labels_in, labels_out, tholds)
    # TODO: reorg to be consistent with top_objects()
    return detected_objs

class Owl2Large(Owl2):
  MODEL_NAME = "google/owlv2-large-patch14-ensemble"
  obj_model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).to("cuda")
  obj_processor = Owlv2Processor.from_pretrained(MODEL_NAME)
