import torch

from openai import OpenAI
from transformers import AutoModel, AutoTokenizer, pipeline
from transformers import Owlv2Processor, Owlv2ForObjectDetection


class PartOfSpeech:
  MODEL_NAME = "QCRI/bert-base-multilingual-cased-pos-english"
  pipeline = pipeline(model=MODEL_NAME, device="cuda")

  @staticmethod
  def get_nouns(txt):
    pos = PartOfSpeech.pipeline(txt)

    nouns = []
    for o in pos:
      if o["entity"].startswith("NN"):
        if o["word"].startswith("#") and len(nouns) > 1:
          nouns[-1] = nouns[-1] + o["word"].replace("#", "")
        elif not o["word"].startswith("#"):
          nouns.append(o["word"])

    return ", ".join(nouns)


class CaptionModel:
  @classmethod
  def caption(cls, img):
    caption = cls.pipeline(img, max_new_tokens=200)[0]["generated_text"].lower()
    nouns = PartOfSpeech.get_nouns(caption)
    return "Picture of " + nouns


class CPModel():
  @classmethod
  def caption(cls, image, user_text=None):
    if user_text == None:
      user_text = "Describe the image using only 8 nouns. Focus on architecture and urbanism aspects."

    conversation = [{'role': 'user', 'content': user_text}]

    caption = cls.model.chat(
      image=image,
      msgs=conversation,
      max_length=32,
      context=None,
      tokenizer=cls.processor,
      sampling=True,
      temperature=0.7
    )

    if type(caption) == tuple:
      caption = caption[0]
    return "Picture of " + caption


class Blip(CaptionModel):
  MODEL_NAME = "Salesforce/blip-image-captioning-large"
  pipeline = pipeline("image-to-text", model=MODEL_NAME, device="cuda", torch_dtype=torch.float16)

class Vit(CaptionModel):
  MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
  pipeline = pipeline("image-to-text", model=MODEL_NAME, device="cuda", torch_dtype=torch.float16)


class CPM2(CPModel):
  MODEL_NAME = "openbmb/MiniCPM-V-2"
  processor = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
  model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda", dtype=torch.bfloat16)

class CPM2_6(CPModel):
  MODEL_NAME = "openbmb/MiniCPM-V-2_6-int4"
  processor = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
  model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)


class GPT4:
  URL_TEMPLATE = "https://www.arquigrafia.org.br/arquigrafia-images/IDID_view.jpg"
  client = None

  @staticmethod
  def key(key):
    GPT4.client = OpenAI(api_key=key)

  @staticmethod
  def clean_caption(cap):
    return cap.strip().lower().replace("english: ", "").replace("portuguese: ", "")

  @staticmethod
  def caption(img_url_or_id):
    if img_url_or_id.startswith("http"):
      img_url = img_url_or_id
    else:
      img_url = GPT4.URL_TEMPLATE.replace("IDID", img_url_or_id)

    LSEP = "SEPARATOR"
    CAP_PREFIX = ["Picture of ", "Imagem de "]

    response = GPT4.client.chat.completions.create(
      #model="gpt-4o-mini",
      model="gpt-4o-2024-08-06",
      messages=[{
        "role": "user",
        "content": [
          {"type": "text", "text": "What’s in this image? Answer using only nouns. Answer in english and portuguese."},
          {"type": "text", "text": f"Separate english and portuguese descriptions with the word {LSEP}"},
          {"type": "text", "text": f"Don't include anything else in the reponse other than the list of nouns and the separator {LSEP}"},
          {"type": "image_url", "image_url": {"url": img_url,},
          },
        ],
      }],
      max_tokens=200,
    )

    caps = response.choices[0].message.content.split(LSEP)
    clean_caps = [p + GPT4.clean_caption(c) for p,c in zip(CAP_PREFIX, caps)]
    return tuple(clean_caps)


class EnPt:
  MODEL_NAME = "Helsinki-NLP/opus-mt-tc-big-en-pt"
  pipeline = pipeline(model=MODEL_NAME, device="cuda")

  @staticmethod
  def translate(txt_en):
    to_pt = ">>por<< " + txt_en
    return EnPt.pipeline(to_pt)[0]["translation_text"]


class Owl2:
  OBJ_TARGET_SIZE = torch.Tensor([500, 500])
  MODEL_NAME = "google/owlv2-base-patch16-ensemble"

  obj_model = Owlv2ForObjectDetection.from_pretrained(MODEL_NAME).to("cuda")
  obj_processor = Owlv2Processor.from_pretrained(MODEL_NAME)

  @staticmethod
  def px_to_pct(box, img_w, img_h):
    model_dims = Owl2.OBJ_TARGET_SIZE
    scale_factor = torch.tensor([max(img_w, img_h) / img_w , max(img_w, img_h) / img_h])
    return [round(x, 4) for x in (box.cpu().reshape(2, -1) / model_dims * scale_factor).reshape(-1).tolist()]

  @staticmethod
  # filter if box "too large" or "too small"
  def threshold(score, label, box, tholds, img_w, img_h):
    box_pct = Owl2.px_to_pct(box, img_w, img_h)
    box_width = box_pct[2] - box_pct[0]
    box_height = box_pct[3] - box_pct[1]
    good_min = box_width > 0.05 and box_height > 0.05
    good_max = box_width < 0.8 or box_height < 0.8
    return good_min and good_max and score > tholds[label.item()]

  @staticmethod
  def run_object_detection(img, labels_in, labels_out, tholds):
    input = Owl2.obj_processor(text=labels_in, images=img, return_tensors="pt").to("cuda")
    with torch.no_grad():
      obj_out = Owl2.obj_model(**input)

    res = Owl2.obj_processor.post_process_object_detection(outputs=obj_out, target_sizes=[Owl2.OBJ_TARGET_SIZE])
    slbs = zip(res[0]["scores"], res[0]["labels"], res[0]["boxes"])
    iw, ih = img.size

    detected_objs = [{"score": s, "label": labels_out[l.item()], "box": Owl2.px_to_pct(b, iw, ih)}
                     for s,l,b in slbs if Owl2.threshold(s, l, b, tholds, iw, ih)]
    return detected_objs

  @staticmethod
  def top_objects(img, labels_in, labels_out, tholds):
    detected_objs = Owl2.run_object_detection(img, labels_in, labels_out, tholds)

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
  def all_objects(img, labels_in, labels_out, tholds):
    detected_objs = Owl2.run_object_detection(img, labels_in, labels_out, tholds)
    return detected_objs
