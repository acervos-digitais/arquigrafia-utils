import json

from os import listdir, path

from dominant_colors import hls_order_from_rgb255

def combine_objs_caps(objs_path, caps_path):
  # filename -> image info
  img_data = {}

  # obj name -> image name
  obj_data = {}

  # image name -> color order key
  color_key = {}

  input_objs_files = sorted([f for f in listdir(objs_path) if f.endswith("json")])

  for io_file in input_objs_files:
    input_objs_file_path = path.join(objs_path, io_file)
    input_caps_file_path = path.join(caps_path, io_file)

    with open(input_objs_file_path, "r", encoding="utf8") as objf:
      id = int(io_file.replace(".json", ""))
      img_data[id] = json.load(objf)

      for l in img_data[id]["boxes"].keys():
        obj_data[l] = obj_data.get(l, []) + [id]

      color_key[id] = hls_order_from_rgb255(img_data[id]["dominant_color"]["by_hue"])

      if path.isfile(input_caps_file_path):
        img_data[id].pop("caption", None)
        with open(input_caps_file_path, "r", encoding="utf8") as capf:
          img_data[id]["captions"] = json.load(capf)

  # order each object's file list by color order
  for k in obj_data.keys():
    obj_data[k] = sorted(obj_data[k], key=lambda x: color_key[x])

  return {
    "objects": obj_data,
    "images": img_data,
  }


def export_objs_caps(objs_path, caps_path, out_path):
  objs_caps_data = combine_objs_caps(objs_path, caps_path)

  with open(out_path, "w", encoding="utf8") as outf:
    json.dump(objs_caps_data, outf, separators=(',',':'), sort_keys=True, ensure_ascii=False)


def combine_by_key(in_path, key):
  all_data = {}

  input_files = sorted([f for f in listdir(in_path) if f.endswith("json")])

  for io_file in input_files:
    input_file_path = path.join(in_path, io_file)
    id = int(io_file.replace(".json", ""))

    with open(input_file_path, "r", encoding="utf8") as f:
      img_data = json.load(f)
      all_data[id] = img_data.get(key, img_data)

  return all_data


def export_by_keys(in_path, keys):
  for k in keys:
    key_data = combine_by_key(in_path, k)
    out_file_path = path.join("metadata", "json", f"{k}.json")
    with open(out_file_path, "w", encoding="utf8") as f:
      json.dump(key_data, f, separators=(',',':'), sort_keys=True, ensure_ascii=False)


def export_all_captions(in_path):
  cap_data = combine_by_key(in_path, "captions")
  langs = list(cap_data.values())[0].keys()
  models = list(cap_data.values())[0][list(langs)[0]].keys()

  for l in langs:
    for m in models:
      out_file_path = path.join("metadata", "json", f"captions_{l}_{m}.json")
      cap_data_lm = {id: caps[l][m] for id,caps in cap_data.items()}
      with open(out_file_path, "w", encoding="utf8") as f:
        json.dump(cap_data_lm, f, separators=(',',':'), sort_keys=True, ensure_ascii=False)
