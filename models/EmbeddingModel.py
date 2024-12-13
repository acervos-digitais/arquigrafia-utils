import torch 

from torchvision.transforms import v2

class EmbeddingModel:
  device = "cuda" if torch.cuda.is_available() else "cpu"
  img_transform = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ConvertImageDtype(torch.float),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
  ])

  @classmethod
  def copy_data(cls):
    def _copy_data(m, i, o):
      my_embedding = o.detach().cpu()
      if len(my_embedding.shape) > 1:
        my_embedding = torch.mean(my_embedding, (2, 3))
      cls.my_embedding = my_embedding.squeeze()
    return _copy_data

  @classmethod
  def get_embedding(cls, img):
    if type(img) is list:
      img_t = torch.stack(cls.img_transform(img)).to(cls.device)
    else:
      img_t = cls.img_transform(img).to(cls.device)

    if len(img_t.shape) < 4:
      img_t = img_t.unsqueeze(0)

    h = cls.layer.register_forward_hook(cls.copy_data())
    with torch.no_grad():
      h_x = cls.model(img_t)
    h.remove()

    my_embedding = cls.my_embedding.clone()
    del cls.my_embedding

    return my_embedding
