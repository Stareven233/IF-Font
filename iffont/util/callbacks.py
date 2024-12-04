import os
from functools import partial

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch
import torchvision.utils as vutils
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities import rank_zero_only

from util import utils


class SetupCallback(Callback):

  def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
    super().__init__()
    self.resume = resume
    self.now = now
    self.logdir = logdir
    self.ckptdir = ckptdir
    self.cfgdir = cfgdir
    self.config = config
    self.lightning_config = lightning_config

  def on_fit_start(self, trainer, pl_module):
    if trainer.global_rank == 0:
      # Create logdirs and save configs
      os.makedirs(self.logdir, exist_ok=True)
      os.makedirs(self.ckptdir, exist_ok=True)
      os.makedirs(self.cfgdir, exist_ok=True)

      print("Project config")
      print(OmegaConf.to_yaml(self.config))
      OmegaConf.save(self.config, os.path.join(self.cfgdir, f"{self.now}-project.yaml"))

      print("Lightning config")
      print(OmegaConf.to_yaml(self.lightning_config))
      OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}), os.path.join(self.cfgdir, f"{self.now}-lightning.yaml"))

    elif not self.resume and os.path.exists(self.logdir):
      dst, name = os.path.split(self.logdir)
      dst = os.path.join(dst, "child_runs", name)
      os.makedirs(os.path.split(dst)[0], exist_ok=True)
      try:
        os.rename(self.logdir, dst)
      except FileNotFoundError:
        pass


class ImageLogger(Callback):

  def __init__(self, batch_frequency, max_images, clamp=True, increase_log_steps=True, from_zero=True):
    super().__init__()
    self.batch_freq = batch_frequency
    self.max_images = max_images
    self.log_steps = []
    if increase_log_steps:
      self.log_steps = [2**n for n in range(int(np.log2(self.batch_freq)) + 1)]
    self.clamp = clamp
    self.from_zero = from_zero
    self.log_cnt = dict.fromkeys(('train', 'validation', 'test', ), 0)
    self.log_last_batch = (None, 0, )  # batch, id,

    for s in self.log_cnt.keys():
      setattr(self, f'on_{s}_batch_end', partial(self.__log_batch_end, s))
      setattr(self, f'on_{s}_epoch_end', partial(self.__log_epoch_end, s))

  @rank_zero_only
  def _testtube(self, pl_module, images, split):
    for k in images:
      grid = vutils.make_grid(images[k])
      grid = (grid+1.0) / 2.0  # -1,1 -> 0,1; c,h,w

      tag = f"{split}/{k}"
      pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

  @rank_zero_only
  def _tensorboard(self, pl_module, images, split):
    for k in images:
      grid = vutils.make_grid(images[k])
      grid = (grid+1.0) / 2.0  # -1,1 -> 0,1; c,h,w
      tag = f"{split}/{k}"
      pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

  @rank_zero_only
  def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
    root = os.path.join(save_dir, 'images', split)
    name = '-'.join(images.keys())
    images = tuple(images.values())
    bs = images[0].shape[0]
    images = torch.cat(images, dim=0)
    
    grid = vutils.make_grid(images, nrow=bs)
    grid = (grid+1.0) / 2  # -1,1 -> 0,1; c,h,w
    grid = grid.permute(1, 2, 0).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)

    filename = f'e-{current_epoch}_gs-{global_step}_b-{batch_idx}_{name}.png'
    os.makedirs(root, exist_ok=True)
    Image.fromarray(grid).save(os.path.join(root, filename))

  def log_img(self, pl_module, batch, batch_idx, outputs=None, split='train'):
    images: dict[str, torch.Tensor]
    if split == 'test' and outputs is None:
      return
    elif split == 'test':
      images = outputs
    else:
      images = pl_module.log_images(batch, split=split)

    for k, v in images.items():
      if self.max_images >= 0:
        N = min(v.shape[0], self.max_images)
        v = v[:N]

      if isinstance(v, torch.Tensor):
        v = v.detach().cpu().type(torch.float32)
        if self.clamp:
          v = torch.clamp(v, -1., 1.)
      images[k] = v

    self.log_local(pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx)
    self._tensorboard(pl_module, images, split)

  def check_frequency(self, pl_module, batch_idx):
    if not hasattr(pl_module, 'log_images') and callable(pl_module.log_images) and self.max_images > 0:
      return False
    if not self.from_zero and batch_idx == 0:
      return False
    if batch_idx % self.batch_freq == 0 or batch_idx in self.log_steps:
      return True
    return False

  def __log_batch_end(self, split, trainer, pl_module, outputs, batch, batch_idx, *args):
    self.log_last_batch = (batch, batch_idx,)
    if not self.check_frequency(pl_module, batch_idx):
      return
    self.log_cnt[split] += 1
    self.log_img(pl_module, batch, batch_idx, outputs=outputs, split=split)

  def __log_epoch_end(self, split, trainer, pl_module):
    if self.log_cnt[split] == 0:
      self.log_img(pl_module, *self.log_last_batch, split=split)
    self.log_cnt[split] = 0


class TimerCallback(Callback):

  def __init__(self) -> None:
    self.timer = utils.Timer()
    super().__init__()

  def on_fit_start(self, *args, **kwargs) -> None:
    self.timer.timeit('fit_start')
    return super().on_fit_start(*args, **kwargs)

  def on_train_epoch_start(self, *args, **kwargs) -> None:
    self.timer.timeit('train_epoch_start')
    return super().on_train_epoch_start(*args, **kwargs)

  def on_train_batch_start(self, *args, **kwargs) -> None:
    self.timer.timeit('train_batch_start')
    return super().on_train_batch_start(*args, **kwargs)

  def on_before_zero_grad(self, *args, **kwargs) -> None:
    self.timer.timeit('before_zero_grad')
    super().on_before_zero_grad(*args, **kwargs)

  def on_before_backward(self, *args, **kwargs) -> None:
    self.timer.timeit('before_backward')
    return super().on_before_backward(*args, **kwargs)

  def on_after_backward(self, *args, **kwargs) -> None:
    self.timer.timeit('after_backward')
    return super().on_after_backward(*args, **kwargs)

  def on_train_batch_end(self, *args, **kwargs) -> None:
    self.timer.timeit('train_batch_end')
    self.timer.next_epoch()
    return super().on_train_batch_end(*args, **kwargs)

  def on_train_epoch_end(self, *args, **kwargs) -> None:
    self.timer.timeit('train_epoch_end')
    return super().on_train_epoch_end(*args, **kwargs)
