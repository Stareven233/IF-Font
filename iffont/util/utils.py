import random
import time
from pathlib import Path
from collections import abc
import shutil

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw


ROOT_DIR = Path(__file__, '../..').resolve()


def counter(initial_num=0):
  num = initial_num
  def inner():
    nonlocal num
    t = num
    num += 1
    return t
  return inner


def randint(a=None, b=None):
  if a is None:
    b = time.time_ns()
    a = -b
  elif b is None:
    b, a = a, 0
  return random.randint(a, b)


def chain_sequence(*args) -> list:
  ret = []
  for r in args:
    if not isinstance(r, str) and isinstance(r, abc.Iterable):
      ret.extend(r)
      continue
    ret.append(r)
  return ret


class Timer:
  def __init__(self, start=None) -> None:
    self.last = start or time.time()
    self.epoch = 0

  def next_epoch(self, num=1):
    self.epoch += num

  def timeit(self, name='pass'):
    print(f'{self.epoch}-{name}: {time.time() - self.last:.4f}s')
    self.last = time.time()


def draw_single_char(ch, font, size, mode='RGB') -> Image.Image:
  if mode == 'L':
    bg_color, fg_color = (255, 0)
  elif mode == 'RGB':
    bg_color, fg_color = ((255, 255, 255), (0, 0, 0))
  else:
    raise ValueError(f'unknown mode={mode}')

  # text_xy = tuple((x/2 for x in img_size))
  text_xy = [size//2, size//2]
  img = Image.new(mode, (size, size), color=bg_color)
  draw = ImageDraw.Draw(img)

  _, top, _, bottom = draw.textbbox(text_xy, ch, font=font, anchor='mm')
  if top >= bottom:
    return None
  text_xy[1] += (size - bottom - top) // 2

  draw.text(text_xy, ch, fill=fg_color, font=font, anchor='mm')
  return img


def draw_text_img(*text, size=256, canvas_size=256, font=None, mode='RGB') -> np.ndarray:
  font = ImageFont.truetype(font, size=size)
  t = (len(text), canvas_size, canvas_size) + ((3, ) if mode=='RGB' else ())
  imgs = np.zeros(t, dtype=np.uint8)
  for i, t in enumerate(text):
    img = draw_single_char(str(t), font, canvas_size, mode)
    if img is None:
      continue
    imgs[i] = np.asarray(img)
  if mode=='RGB':
    imgs = imgs.transpose(0, 3, 1, 2)
  return imgs


def setup_seed(seed, strict=True):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  if strict:
    torch.backends.cudnn.deterministic = True


def show_cuda_info():
  print(f'{torch.__version__=}')
  if torch.cuda.is_available():
    print(f'{torch.version.cuda=}')
    print(f'{torch.backends.cudnn.version()=}')
    print(f'number of avaliable gpu: {torch.cuda.device_count()}')
    print(f'index of current device: {torch.cuda.current_device()}')
    print(f'device capability: {".".join(map(str, torch.cuda.get_device_capability()))}')
    print(f'device name: {torch.cuda.get_device_name()}')
  else:
    print('No CUDA GPUs are available')


def count_parameters(model: torch.nn.Module):
  # trainable, frozen
  bin1, bin2 = [], []
  for p in model.parameters():
    if p.requires_grad:
      bin1.append(p.numel())
    else:
      bin2.append(p.numel())
  return sum(bin1), sum(bin2)


def show_parameters_num(model: torch.nn.Module):
  b1, b2 = tuple(map(lambda x: x/1e6, count_parameters(model)))
  print(f'[{model.__class__}] number of parameters: {b1+b2:.2f}M = trainable {b1:.2f}M + frozen {b2:.2f}')


def concat_batch_images(*batches, n=-1) -> np.ndarray:
  if isinstance(batches[0], torch.Tensor):
    batches = [b.detach().cpu().numpy() for b in batches]

  shape = batches[0].shape
  assert len(shape) in (3, 4, )
  if len(shape) == 3:
    c = 1
    bs, h, w = shape
  else:
    bs, c, h, w = shape

  n = min(bs, n) if n!=-1 else bs
  img = np.empty((c, len(batches) * h, n * w))
  for i, b in enumerate(batches):
    a = np.split(b, b.shape[0], axis=0)[:n]
    # (h, min(_bs, n)*w)
    a = np.concatenate(a, axis=-1).squeeze()
    img[:, i * h:(i+1) * h, :] = a
  img = img.transpose(1, 2, 0)  # h w c
  return img


def draw_batch_images(*batches, n=4, show=True, save=None, **imshow_kwargs):
  img = concat_batch_images(*batches, n=n)
  h, w, _ = img.shape
  plt.figure(figsize=(w, h), dpi=1)
  plt.xticks([])
  plt.yticks([])
  plt.axis('off')
  plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
  plt.imshow(img, **imshow_kwargs)

  if save is not None:
    plt.savefig(save, bbox_inches='tight')
  if show:
    plt.show()
  plt.clf()
  plt.close()


def backup_codes(dst: str|Path):
  if isinstance(dst, str):
    dst = Path(dst)
  if dst.is_dir() and dst.exists():
    print(f'{dst} already exists, removing')
    shutil.rmtree(dst)
  shutil.copytree(ROOT_DIR, dst, symlinks=True)
