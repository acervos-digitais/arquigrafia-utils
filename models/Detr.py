import torch

from transformers import AutoImageProcessor, AutoModelForObjectDetection

from finetune_utils.finetune_0915 import FTUtils

from warnings import simplefilter

simplefilter(action="ignore")

class Detr():
  MODEL_NAME = "acervos-digitais/conditional-detr-resnet-50-ft-0915-e256-augm3"

  detr_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
  detr_model = AutoModelForObjectDetection.from_pretrained(MODEL_NAME).to("cuda")

  @staticmethod
  def px_to_pct(box, img_w, img_h):
    x0,y0,x1,y1 = box
    return [round(x0 / img_w, 4), round(y0 / img_h, 4), round(x1 / img_w, 4), round(y1 / img_h, 4)]

  @staticmethod
  def run_object_detection(img, tholds=[]):
    if type(tholds) == float:
      tholds = 32*[tholds]
    elif type(tholds) == list and len(tholds) == 1:
      tholds = 32*[tholds[0]]
    min_threshold = min(tholds) - 0.005 if len(tholds) > 0 else 0.25

    input = Detr.detr_processor(images=img, return_tensors="pt")
    pixel_values = input["pixel_values"].to("cuda")
    iw, ih = img.size

    with torch.no_grad():
      outputs = Detr.detr_model(pixel_values=pixel_values, pixel_mask=None)
    
    res = Detr.detr_processor.post_process_object_detection(outputs,
                                                            target_sizes=[(ih, iw)],
                                                            threshold=min_threshold)[0]

    slbs = zip(res["scores"].tolist(), res["labels"].tolist(), res["boxes"].tolist())

    if len(tholds) > 0:
      f_scores = []
      f_labels = []
      f_boxes = []

      for s,l,b in slbs:
        if s > tholds[l]:
          f_scores.append(s)
          f_labels.append(l)
          f_boxes.append(b)

      slbs = zip(f_scores, f_labels, f_boxes)

    detected_objs = [{"score": s, "label": FTUtils.ID2LABEL[l], "box": Detr.px_to_pct(b, iw, ih)} for s,l,b in slbs]
    return detected_objs

  @staticmethod
  def top_objects(img, tholds=[]):
    detected_objs = Detr.run_object_detection(img, tholds)

    # only keep the box with highest score per object
    detected_objs_boxes = {}
    high_score = {}

    for o in detected_objs:
      ol = o["label"]
      if (ol not in detected_objs_boxes) or (o["score"] > high_score[ol]):
        detected_objs_boxes[ol] = o["box"]
        high_score[ol] = o["score"]
  
    return detected_objs_boxes

  @staticmethod
  def all_objects(img, tholds=[]):
    detected_objs = Detr.run_object_detection(img, tholds)
    # TODO: reorg to be consistent with top_objects()
    return detected_objs
