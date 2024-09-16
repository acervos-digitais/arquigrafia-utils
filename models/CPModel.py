from warnings import simplefilter
simplefilter(action="ignore")

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
