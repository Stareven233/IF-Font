import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms.functional as F

from modules.vqgan import VQModel, GumbelVQ


def pil_to_tensor(x: Image.Image) -> torch.Tensor:
  x = F.pil_to_tensor(x).float() 
  x = (x / 255.) * 2 - 1
  return x


def tensor_to_pil(x: torch.Tensor, clamp=True) -> Image.Image:
  x = x.squeeze().detach().cpu()
  if clamp:
    x = torch.clamp(x, -1., 1.)
  x = (x + 1.) / 2.
  if x.ndim == 4:
    x = x.split(1, dim=0)
    x = torch.cat(x, dim=-1).squeeze()

  x = x.permute(1, 2, 0).numpy()
  x = (255 * x).astype(np.uint8)
  x = Image.fromarray(x)
  if not x.mode == 'RGB':
    x = x.convert('RGB')
  return x


class VQAdapter:
  def __init__(self, vqgan_path) -> None:
    super().__init__()
    self.vqgan_path = vqgan_path
    # self.downsample_ratio = int(self.vqgan_path.rsplit('_', 2)[1][1:])
    self.config = OmegaConf.load(f"{self.vqgan_path}/configs/config.yaml").model.params
    self.vqgan = self._load_vqgan()

  def _load_vqgan(self, is_gumbel=False, display=False):
    if display:
      print(yaml.dump(OmegaConf.to_container(self.config)))
  
    if is_gumbel:
      model = GumbelVQ(**self.config)
    else:
      model = VQModel(**self.config)

    sd = torch.load(f"{self.vqgan_path}/checkpoints/model.ckpt", map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.freeze()
    model.train = lambda self: self
    return model
  
  def set_device(self, device):
    self.device = device  # or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.vqgan.to(self.device)

  @property
  def indices_dtype(self) -> np.unsignedinteger:
    t = self.config.n_embed
    if t <= 1<<8:
      dtype = np.uint8
    elif t <= 1<<16:
      dtype = np.uint16
    else:
      raise ValueError(f'vqgan embedding size too big, {t}')
    return dtype
  
  @torch.no_grad()
  def encode_raw(self, x:Image.Image) -> tuple[torch.Tensor]:
    if isinstance(x, Image.Image):
      x = (F.pil_to_tensor(x).float() / 255.) * 2 - 1
    elif isinstance(x, np.ndarray):
      x = torch.from_numpy(x)
    if len(x.shape) == 3:
      x = x[None, ...]
  
    if hasattr(self, 'device'):
      x = x.to(self.device)
    quant, loss, info = self.vqgan.encode(x)
    return quant, loss, info

  @torch.no_grad()
  def encode(self, x:Image.Image) -> torch.Tensor:
    _, _, [_, _, indices] = self.encode_raw(x)
    return indices

  @torch.no_grad()
  def get_codebook(self) -> torch.Tensor:
    return self.vqgan.quantize.embedding.weight.detach()

  @torch.no_grad()
  def lookup_quant(self, indices: torch.Tensor) -> torch.Tensor:
    if isinstance(indices, np.ndarray):
      indices = torch.as_tensor(indices, dtype=torch.long)
    if len(indices.shape) == 1:
      indices.unsqueeze_(0)
    if hasattr(self, 'device'):
      indices = indices.to(self.device)
    indices = indices.type(torch.long)
  
    z_hw = int(indices.shape[1] ** 0.5)
    bhwc = (indices.shape[0], z_hw, z_hw, self.config.embed_dim)
    quant = self.vqgan.quantize.get_codebook_entry(indices.reshape(-1), shape=bhwc)
    return quant

  @torch.no_grad()
  def decode_raw(self, indices: torch.Tensor) -> torch.Tensor:
    x = self.lookup_quant(indices)
    x = self.vqgan.decode(x)
    x = torch.clamp(x, -1., 1.)
    return x

  @torch.no_grad()
  def decode(self, indices: torch.Tensor) -> Image.Image:
    x = self.decode_raw(indices)
    x = tensor_to_pil(x, False)
    return x
