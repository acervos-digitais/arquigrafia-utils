from openai import OpenAI
from .openai_api import OPENAI_API_KEY

class GPT4:
  URL_TEMPLATE = "https://www.arquigrafia.org.br/arquigrafia-images/IDID_view.jpg"
  client = OpenAI(api_key=OPENAI_API_KEY)

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

    cap_content = response.choices[0].message.content
    cap_content_fix = cap_content.replace(LSEP, f" {LSEP} ").replace("separador", f" {LSEP} ")
    caps = cap_content_fix.split(LSEP)

    clean_caps = [p + GPT4.clean_caption(c) for p,c in zip(CAP_PREFIX, caps)]
    return tuple(clean_caps)
