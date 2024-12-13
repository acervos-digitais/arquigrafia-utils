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
  def processor(cls, images):
    if type(images) is list:
      images_t = torch.stack(cls.img_transform(images))
    else:
      images_t = cls.img_transform(images)

    if images_t.dim() < 4:
      images_t = images_t.unsqueeze(0)

    return images_t

  @classmethod
  def get_embedding(cls, imgs):
    imgs_t = cls.processor(imgs).to(cls.device)

    def copy_data(m, i, o):
      my_embedding = o.detach()
      if my_embedding.dim() > 1:
        my_embedding = torch.mean(my_embedding, (2, 3))
      cls.my_embedding = my_embedding.squeeze()

    h = cls.layer.register_forward_hook(copy_data)
    with torch.no_grad():
      h_x = cls.model(imgs_t)
    h.remove()

    my_embedding = cls.my_embedding.clone()
    del cls.my_embedding

    return my_embedding
