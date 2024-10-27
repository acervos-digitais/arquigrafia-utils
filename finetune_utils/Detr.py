import pytorch_lightning as pl
import torch

from datasets import load_dataset
from functools import partial
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from transformers import AutoImageProcessor, AutoModelForObjectDetection

class DetrDataLoader:
  image_transform = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
    T.RandomEqualize(p=0.5),
    T.RandomPerspective(distortion_scale=0.6, p=0.5),
    T.RandomApply(transforms=[T.RandomRotation(degrees=35)], p=0.5),
    T.RandomApply(transforms=[T.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.25))], p=0.5),
    T.RandomApply(transforms=[T.ColorJitter(brightness=0.5, hue=0.3)], p=0.5)
  ])

  @classmethod
  def as_coco(cls, img_id, objects):
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

  @classmethod
  def transform_batch(cls, batch, transform, image_processor, return_pixel_mask=False):
    images = []
    annotations = []
    for image_id, image, objects in zip(batch["image_id"], batch["image"], batch["objects"]):
      iw, ih = image.size
      objects["bbox"] = tv_tensors.BoundingBoxes(objects["bbox"], format="XYWH", canvas_size=(ih, iw))
      image = tv_tensors.Image(image.convert("RGB"))

      # apply augmentations
      if transform is not None:
        image, bboxes, categories = transform(image, objects["bbox"], objects["category"])
        objects["bbox"] = bboxes
        objects["category"] = categories

      images.append(image)

      # format annotations in COCO format
      formatted_annotations = DetrDataLoader.as_coco(image_id, objects)
      annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")

    if not return_pixel_mask:
      result.pop("pixel_mask", None)

    return result

  @classmethod
  def collate_fn(cls, batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
      data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data


  def __init__(self, dataset_name, processor_name):
    hf_ds = load_dataset(dataset_name)
    self.ds = hf_ds
    self.processor = AutoImageProcessor.from_pretrained(processor_name)
    self.categories = hf_ds["train"].features["objects"].feature["category"].names
    self.id2label = {index: x for index, x in enumerate(self.categories, start=0)}
    self.label2id = {v: k for k, v in self.id2label.items()}

    train_transform = partial(DetrDataLoader.transform_batch, transform=DetrDataLoader.image_transform, image_processor=self.processor, return_pixel_mask=True)
    validation_transform = partial(DetrDataLoader.transform_batch, transform=None, image_processor=self.processor, return_pixel_mask=True)

    train_ds = hf_ds["train"].with_transform(train_transform)
    validation_ds = hf_ds["test"].with_transform(validation_transform)

    self.train = DataLoader(train_ds, collate_fn=DetrDataLoader.collate_fn, batch_size=4, num_workers=8, shuffle=True)
    self.validation = DataLoader(validation_ds, collate_fn=DetrDataLoader.collate_fn, batch_size=4, num_workers=8)

  def getTrain(self):
    return self.train
  def getValidation(self):
    return self.validation


class Detr(pl.LightningModule):
  def __init__(self, model_name, dataloader, lr, lr_backbone, weight_decay):
    super().__init__()
    self.train_dl = dataloader.getTrain()
    self.validation_dl = dataloader.getValidation()
    self.train_ds = dataloader.ds["train"]
    self.validation_ds = dataloader.ds["test"]
    self.processor = dataloader.processor
    self.model = AutoModelForObjectDetection.from_pretrained(
      model_name,
      id2label=dataloader.id2label,
      label2id=dataloader.label2id,
      ignore_mismatched_sizes=True
    ).to("cuda")

    # see https://github.com/PyTorchLightning/pytorch-lightning/pull/1896
    self.lr = lr
    self.lr_backbone = lr_backbone
    self.weight_decay = weight_decay
    self.save_hyperparameters()

  def detr_eval(self, dataset, min_threshold=0.2, thresholds=[]):
    num_correct = 0
    num_preds = 0
    num_labels = 0

    self.model.to("cuda")
    with torch.no_grad():
      for row in dataset:
        img = row["image"]
        iw, ih = img.size

        inputs = self.processor(images=img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to("cuda")

        outputs = self.model(pixel_values=pixel_values, pixel_mask=None)

        ppo = self.processor.post_process_object_detection(outputs,
                                                           target_sizes=[(ih, iw)],
                                                           threshold=min_threshold)[0]

        preds = [l.item() for l in ppo["labels"]]
        scores = [s.item() for s in ppo["scores"]]
        boxes = [b.tolist() for b in ppo["boxes"]]

        if len(thresholds) > 0:
          f_preds = []
          f_scores = []
          f_boxes = []

          for p,s,b in zip(preds, scores, boxes):
            if s > thresholds[p]:
              f_preds.append(p)
              f_scores.append(s)
              f_boxes.append(b)

          preds, scores, boxes = f_preds, f_scores, f_boxes

        labels = row["objects"]["category"]

        cpreds = [1 for p in set(preds) if p in labels]

        num_correct += len(cpreds)
        num_preds += len(preds)
        num_labels += len(labels)
    
    precision = round(num_correct / num_preds, 4) if num_preds != 0 else 0
    recall = round(num_correct / num_labels, 4) if num_labels != 0 else 0
    return precision, recall

  def forward(self, pixel_values, pixel_mask):
    return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)

  def common_step(self, batch, batch_idx):
    pixel_values = batch["pixel_values"]
    pixel_mask = batch["pixel_mask"]
    labels = [{k: v.to(self.device) for k,v in t.items()} for t in batch["labels"]]

    outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

    loss = outputs.loss
    loss_dict = outputs.loss_dict

    return loss, loss_dict

  def training_step(self, batch, batch_idx):
    loss, loss_dict = self.common_step(batch, batch_idx)
    try:
      self.log("training_loss", loss)
      for k,v in loss_dict.items():
        self.log("train_" + k, v.item())

      if batch_idx == 0:
        precision, recall = self.detr_eval(self.train_ds)
        self.log("train_precision", precision)
        self.log("train_recall", recall)
    except Exception as e:
      print("training step exception:", e)

    return loss

  def validation_step(self, batch, batch_idx):
    loss, loss_dict = self.common_step(batch, batch_idx)
    try:
      self.log("validation_loss", loss)
      for k,v in loss_dict.items():
        self.log("validation_" + k, v.item())

      if batch_idx == 0:
        precision, recall = self.detr_eval(self.validation_ds)
        self.log("validation_precision", precision)
        self.log("validation_recall", recall)
    except Exception as e:
      print("validation step exception:", e)

    return loss

  def configure_optimizers(self):
    param_dicts = [
          {"params": [p for n,p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
          {
              "params": [p for n,p in self.named_parameters() if "backbone" in n and p.requires_grad],
              "lr": self.lr_backbone,
          },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=self.lr, weight_decay=self.weight_decay)

    return optimizer

  def train_dataloader(self):
    return self.train_dl

  def val_dataloader(self):
    return self.validation_dl
