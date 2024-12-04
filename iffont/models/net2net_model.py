import itertools

import torch
import lightning.pytorch as pl
from lightning.pytorch.strategies.ddp import DDPStrategy
from torch.optim import lr_scheduler

from modules import losses
from util.metrics import MetricWrapper
from util import cache


class CacheManagerCL(cache.CacheManager):
  def push(self, *values):
    for k, v in zip(self.cache_map.keys(), values):
      cache = self.cache_map[k]
      cache.push(v)
  
  def pop(self, size=None) -> list:
    ret = []
    for cache in self.cache_map.values():
      if len(cache.data) == 0:
        ret.append(None)
        continue
      r = cache.pop(size)
      r = torch.concat(r, dim=0)
      ret.append(r.detach())
    return ret
  
  def popush(self, new_v, size=None, extra_v=()):
    '''extra_v跟队列里的元素拼接并返回，new_v入队'''
    old_v = self.pop(size)  # [bs*cache_size, 2, n_embd], [bs*cache_size, ]
    self.push(*new_v)
    ret = []
    for t in zip(extra_v, old_v):
      if t[1] is None:
        t = t[:1]
      ret.append(torch.concat(t))
    return ret


class Net2NetModel(pl.LightningModule):
  def __init__(self, gpt, ids_enc, moco_wrapper):
    super().__init__()
    self.netTransformer = gpt
    # self.netTransformer = torch.compile(gpt)
    self.ids_encoder = ids_enc
    # self.quant_encoder = encoder.QuantExtEncoder(self.adapter, gpt.config.n_embd, **quant_enc_config)
    self.moco_wrapper = moco_wrapper
    self.adapter = moco_wrapper.adapter  # util.instantiate_from_config(converter)
    self.metrics_wrapper = MetricWrapper('FID', 'L1', 'LPIPS', 'RMSE', 'SSIM', data_range=(-1, 1))

    self.cache_manager = CacheManagerCL(10, 'cl_s', 'font_id')
    self.cache_feats = dict()

  def _on_start(self):
    self.ids_encoder.set_device(self.device)
    self.moco_wrapper.set_device(self.device)
    self.metrics_wrapper.set_device(self.device)

  def on_fit_start(self) -> None:
    self._on_start()
    return super().on_fit_start()

  def on_predict_start(self) -> None:
    self._on_start()
    return super().on_predict_start()
  
  def on_test_start(self) -> None:
    self._on_start()
    return super().on_test_start()
  
  def init_from_ckpt(self, path, ignore_keys=[]):
    sd = torch.load(path, map_location='cpu')['state_dict']
    for k in sd.keys():
      if any((k.startswith(ik) for ik in ignore_keys)):
        self.print('Deleting key {k} from state_dict.')
        del sd[k]
    self.load_state_dict(sd, strict=False)
    print(f'Restored {self.__class__.__name__} from {path}')

  # def on_save_checkpoint(self, checkpoint) -> None:
  #   for k in tuple(checkpoint["state_dict"].keys()):
  #     if not k.startswith('converter.vqgan.'):
  #       continue
  #     del checkpoint["state_dict"][k]
  #   return super().on_save_checkpoint(checkpoint)

  def forward(self, x_indices, c_indices, x_ids, c_ids):
    ids_embed, ids_embed2 = self.ids_encoder(x_ids)  # [bs, 35, 256]
    self.cache_feats['ids_embed'] = ids_embed

    sim = self.ids_encoder.coverage(x_ids, c_ids)
    x_sss, moco_cl = self.moco_wrapper(c_indices, ids_embed2, sim)
    self.cache_feats['x_sss'] = x_sss
    self.cache_feats['moco_cl'] = moco_cl

    logits = self.netTransformer(x_indices[:, :-1], x_sss, embeddings=ids_embed)
    logits = logits[:, ids_embed.shape[1] - 1:]
    return logits
  
  @torch.no_grad()
  def sample(self, x, c, x_ids, c_ids, steps, temperature=1.0, sample=False, top_k=None, step_range=range):
    assert not self.netTransformer.training
    ids_embed, ids_embed2 = self.ids_encoder(x_ids)
    sim = self.ids_encoder.coverage(x_ids, c_ids)
    x_sss, _ = self.moco_wrapper(c, ids_embed2, sim)

    x = self.netTransformer.generate(
      x, steps, x_sss, ids_embed, 
      temperature=temperature, sample=sample, top_k=top_k, step_range=step_range
    )
    return x

  @torch.no_grad()
  def log_images(self, batch, temperature=1.0, top_k=None, **kwargs):
    self.train(False)
    log = dict()
    N = 4
    
    x, c, x_ids, c_ids, *_ = self.get_data(batch, N)
    ids_embed, ids_embed2 = self.ids_encoder(x_ids)
    sim = self.ids_encoder.coverage(x_ids, c_ids)
    x_sss, _ = self.moco_wrapper(c, ids_embed2, sim)

    log['x'] = self.adapter.decode_raw(x)  # [bs, 3, 128, 128]
    log['c'] = self.adapter.decode_raw(c[:, 0])

    index_sample = self.netTransformer.generate(
      x[:, :x.shape[1] // 2],
      x.shape[1] - (x.shape[1] // 2),
      x_sss,
      ids_embed,
      temperature=temperature,
      sample=True,
      top_k=top_k if top_k is not None else 100,
    )
    log['x_half'] = self.adapter.decode_raw(index_sample)

    index_sample = self.netTransformer.generate(
      x[:, :0],
      x.shape[1],
      x_sss,
      ids_embed,
      temperature=temperature,
      sample=True,
      top_k=top_k if top_k is not None else 100,
    )
    log['x_zero'] = self.adapter.decode_raw(index_sample)

    index_sample = self.netTransformer.generate(x[:, :0], x.shape[1], x_sss, ids_embed, sample=False)
    log['x_zero_det'] = self.adapter.decode_raw(index_sample)
    self.train(True)
    return log

  def get_data(self, batch:dict, N=None):
    x, c = batch['x_idx'], batch['c_idx']
    res = (x, c, batch['x_ch'], batch['c_ch'], batch['font_id'], batch['x_font'], )
    if N is not None:
      res = tuple((_c[:N] for _c in res))
    return res
  
  @torch.no_grad()
  def _metrics_step(self, pred, gt):
    gt = self.adapter.decode_raw(gt)
    pred = self.adapter.decode_raw(pred)
    self.metrics_wrapper.step(pred, gt)
    return dict(gt=gt, pred=pred)

  def _metrics_epoch(self, split, **kwargs):
    keys, values = [], []
    for k, v in self.metrics_wrapper.compute().items():
      keys.append(k)
      values.append(f'{v:.4f}')
      self.log(f'{split}/{k}_epoch', v, **kwargs)
    self.metrics_wrapper.reset()
    res = ';\t'.join((f'{k}:{v}' for k, v in zip(keys, values)))
    keys, values = '\t'.join(keys), '\t'.join(values)
    print('\n', f'metrics in epoch={self.current_epoch}:', res, keys, values, sep='\n')

  def training_step(self, batch, batch_idx):
    x, c, x_ids, c_ids, font_id = self.get_data(batch)[:-1]  # x.shape=torch.Size([bs, 256])
    # x_ids: tuple[tuple(int, str)] - [bs, 2]
    # c_ids: tuple[tuple(tuple(int, str))] - [bs, n_ref, 2]
    logits = self.forward(x, c, x_ids, c_ids)  # logits.shape=torch.Size([bs, 256, 256])
    self.moco_wrapper.momentum_update(self.current_epoch / self.trainer.max_epochs)

    cl_s = self.cache_feats.pop('moco_cl')
    cl_s, font_id = self.cache_manager.popush((cl_s, font_id), extra_v=(cl_s, font_id))
    if isinstance(self.trainer.strategy, DDPStrategy):
      cl_s = self.all_gather(cl_s, sync_grads=True).flatten(0, 1)  # [world_size, bs*cache_size, 2, n_embd] -> [world_size*bs*cache_size, 2, n_embd]
      font_id = self.all_gather(font_id, sync_grads=False).flatten()  # [world_size, bs*cache_size, ] -> [world_size*bs*cache_size, ]

    l_sq = losses.sq(logits, x)
    l_cl = losses.sup_cl(cl_s, labels=font_id) / 2
    self.log('train/l_sq', l_sq, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=logits.shape[0])
    self.log('train/l_cl', l_cl, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=logits.shape[0])
    return l_sq + l_cl

  def validation_step(self, batch, batch_idx):
    x, c, x_ids, c_ids, *_ = self.get_data(batch)
    logits = self.forward(x, c, x_ids, c_ids)
    l_sq = losses.sq(logits, x)
    self.log('validation/l_sq', l_sq, prog_bar=True, logger=True, on_step=True, on_epoch=True, batch_size=logits.shape[0])
    if batch_idx >= 100:
      return
    x_sss = self.cache_feats.pop('x_sss')
    ids_embed = self.cache_feats.pop('ids_embed')
    pred = self.netTransformer.generate(x[:, :0], x.shape[1], x_sss, ids_embed, sample=False)
    self._metrics_step(pred, x)

  def on_validation_epoch_end(self):
    self._metrics_epoch('validation', prog_bar=False, logger=True, on_epoch=True, sync_dist=True)

  def test_step(self, batch, batch_idx):
    gt, c, x_ids, c_ids, *_ = self.get_data(batch)
    pred = self.sample(gt[:, :0], c, x_ids, c_ids, steps=gt.shape[1], sample=False)
    return self._metrics_step(pred, gt)

  def on_test_epoch_end(self):
    self._metrics_epoch('test', prog_bar=True, logger=True)

  def configure_optimizers(self):
    weight_decay = 0.01
    optimizer: torch.optim.Optimizer = self.netTransformer.configure_optimizers(weight_decay, self.learning_rate, (0.9, 0.95), self.device.type)

    other_params = []
    for cn, c in self.named_children():
      if cn == 'netTransformer':
        continue
      other_params.append((p for p in c.parameters() if p.requires_grad))

    optimizer.add_param_group({
      'params': itertools.chain.from_iterable(other_params),
      'betas': (0.9, 0.999),
    })

    scheduler = lr_scheduler.OneCycleLR(
      optimizer,
      max_lr=self.learning_rate,
      total_steps=self.trainer.estimated_stepping_batches,
      pct_start=0.5/self.trainer.max_epochs,
      final_div_factor=10/25,
    )
    
    scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
    return {'optimizer': optimizer, 'lr_scheduler': scheduler}
