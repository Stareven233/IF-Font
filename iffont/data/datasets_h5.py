import os
import random
from pathlib import Path
import sys

import h5py
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl


ROOT_DIR = Path(__file__, '../..').resolve()
sys.path.append(ROOT_DIR.as_posix())
from data import datasets
from data import valid_characters
from data.adapter import VQAdapter


def dataset2h5(h5_file: h5py.File, split: str, dset: datasets.IFFontDataset, adapter: VQAdapter):
  t = dset._get_one(0, 0, 't')['t']
  t = adapter.encode(t)
  font_list = []

  if split not in h5_file:
    h5_dset = h5_file.create_dataset(split, (dset.font_size+1, dset.ch_size, ) + t.shape, dtype=adapter.indices_dtype)
  else:
    h5_dset = h5_file[split]
  
  pbar = tqdm(total=(dset.font_size+1)*dset.ch_size)
  for font_id, font_path in enumerate(dset.fonts_path):
    font_name = os.path.splitext(os.path.split(font_path)[-1])[0]
    font_list.append(font_name)
    pbar.set_description(f'{split}-{font_id}-{font_name}')
    for ch_id in range(dset.ch_size):
      d = dset._get_one(ch_id, font_id, 'image')
      h5_dset[font_id, ch_id] = adapter.encode(d['image']).cpu().numpy()
      pbar.update(1)

  h5_dset.attrs['font_num'] = dset.font_size
  h5_dset.attrs['ch_num'] = dset.ch_size
  h5_dset.attrs['ch_list'] = '|'.join(dset.corpus[:dset.ch_size])
  h5_dset.attrs['font_list'] = '|'.join(font_list)


def make_h5_file(*config_paths:Path):
  config = [OmegaConf.load(c.resolve().as_posix()) for c in config_paths]
  config = OmegaConf.merge(*config)
  data_cfg = config.data
  data_dir = Path(data_cfg.init_args.data_dir)
  h5_file = data_dir / data_cfg.init_args.h5_filename

  h5_file = h5py.File(h5_file, 'w-')
  default_kwargs = {
    'corpus': valid_characters.valid_ch,
    'size': data_cfg.dict_kwargs.img_size,
    'mode': data_cfg.dict_kwargs.img_mode,
  }
  adapter = VQAdapter(config.model.init_args.moco_wrapper.init_args.adapter.init_args.vqgan_path)
  adapter.set_device(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
  print(f'{adapter.device=}')

  split = ('train', 'val', )
  for s in split:
    kwargs = default_kwargs.copy()
    kwargs['ttf_root'] = data_dir / 'fonts' / s
    dataset = datasets.init_dataset(**kwargs)
    print(f'start {s} split...')
    dataset2h5(h5_file, s, dataset, adapter)

  h5_file.attrs['config'] = str(config)
  h5_file.attrs['vqgan_path'] = adapter.vqgan_path
  h5_file.attrs['vqgan_config'] = str(adapter.config)
  h5_file.close()


class IFFontDataset(Dataset):
  def __init__(self, h5_path, font_split, ch_split, num_refs, font_num=None, ch_num=None):
    h5_file = h5py.File(h5_path, 'r')
    h5_dset = h5_file[font_split]
    self.corpus = getattr(valid_characters, f'{ch_split}_ch')
    self.corpus_seen = valid_characters.train_ch

    self.data: torch.Tensor = torch.as_tensor(h5_dset[...], dtype=torch.long)

    self.init_by_h5_attrs(h5_file)  # data_config, vqgan_config, vqgan_path
    self.init_by_h5_attrs(h5_dset)  # font_num, ch_num, ch_list, font_list
    ch_num = len(self.corpus) if ch_num is None else ch_num
    self.ch_num: int = min(ch_num, len(self.corpus), self.ch_num)
    self.content_font_id = self.font_num
    self.font_num: int = min(font_num or self.font_num, self.font_num)
    self.num_refs = num_refs
    h5_file.close()

    self.global_cid = {}
    for i, v in enumerate(self.ch_list):
      self.global_cid[v] = i
    print(f'font: {font_split}-{self.font_num} ch: {ch_split}-{self.ch_num}')

  def init_by_h5_attrs(self, h5_obj):
    new_attrs = []
    for k, v in h5_obj.attrs.items():
      if k.endswith('_config'):
        v = OmegaConf.create(v)
      elif k.endswith('_list'):
        v = v.split('|')
      elif k.endswith('_num'):
        v = v.item()
      setattr(self, k, v)
      new_attrs.append(k)

  def __len__(self):
    return self.font_num * self.ch_num

  def __getitem__(self, index):
    ch: str = self.corpus[index % self.ch_num]
    cid = self.global_cid[ch]
    font_id = index // self.ch_num
    
    c_ch = tuple(random.sample(self.corpus_seen, k=self.num_refs))
    c_cid = tuple(map(lambda i: self.global_cid[i], c_ch))

    # if c_cid[0] == cid:
    #   c_cid[0], c_cid[-1] = c_cid[-1], c_cid[0]
    return {
      'x_idx': self.data[font_id, cid],  # [256]
      'c_idx': self.data[font_id, c_cid],  # [n_ref, 256]
      'content_idx': self.data[self.content_font_id, cid],  # [256]
      'font_id': torch.as_tensor(font_id),
      'x_ch': ch,
      'c_ch': c_ch,
      'x_font': self.font_list[font_id],
    }
  
  @staticmethod
  def collate(data_list):
    res = {}
    for d in data_list:
      for k, v in d.items():
        res.setdefault(k, [])
        res[k].append(v)
    for k, v in res.items():
      if isinstance(v[0], np.ndarray):
        new_v = torch.from_numpy(np.stack(v, axis=0))
      elif isinstance(v[0], torch.Tensor):
        new_v = torch.stack(v, dim=0)
      else:
        new_v = v
      res[k] = new_v
    return res


class IFFontDataModule(pl.LightningDataModule):
  def __init__(self, data_dir, h5_filename, batch_size=1, val_batch_size=2, num_workers=0, **kwargs):
    super().__init__()    
    self.batch_size = batch_size
    self.val_batch_size = val_batch_size
    self._n_ref = kwargs.pop('num_refs', 2)
    self.dset_kwargs = {
      'num_refs': self._n_ref
    }
    self.loader_kwargs = {
      'num_workers': num_workers,
      'collate_fn': IFFontDataset.collate,
      'pin_memory': kwargs.pop('pin_memory', False),
      'prefetch_factor': kwargs.pop('prefetch_factor', 2),
      'persistent_workers': kwargs.pop('persistent_workers', False),
      # 'drop_last': True,
    }
    self.h5_path = os.path.join(data_dir, h5_filename)
    self.is_dev = kwargs.pop('is_dev', False)
    self.only_train_set = kwargs.pop('only_train_set', False)
    self.test_set = kwargs.pop('test_set', 'ufuc')

  def prepare_data(self):
    ...

  def setup(self, stage):
    '''stage: <enum 'TrainerFn'>，但当做str用好像也可'''

    train_kwargs = self.dset_kwargs.copy()
    val_kwargs = self.dset_kwargs.copy()
    train_kwargs['num_refs'] = self._n_ref + (self._n_ref&1)
    if self.is_dev:
      train_kwargs |= {'font_num': 30, 'ch_num': self.batch_size + 1}
      val_kwargs |= {'font_num': 2, 'ch_num': self.val_batch_size + 1}
    if self.only_train_set:
      setattr(self, 'val_dataloader', None)
      setattr(self, 'test_dataloader', None)

    if stage == 'fit':
      self.dset_train = IFFontDataset(self.h5_path, 'train', 'train', **train_kwargs)
      # SFUC
      self.dset_val = IFFontDataset(self.h5_path, 'train', 'val', **val_kwargs)
    elif stage == 'test':
      fc = []
      for t in self.test_set[0], self.test_set[2]:
        if t == 'u':
          fc.append('val')
        elif t == 's':
          fc.append('train')
        else:
          raise ValueError(f'unknown test_set: {self.test_set}')
      self.dset_test = IFFontDataset(self.h5_path, *fc, **val_kwargs)

  def train_dataloader(self):
    d = self.dset_train
    return DataLoader(d, batch_size=self.batch_size, shuffle=True, **self.loader_kwargs)

  def val_dataloader(self):
    d = self.dset_val
    return DataLoader(d, batch_size=self.val_batch_size, shuffle=False, **self.loader_kwargs)

  def test_dataloader(self):
    d = self.dset_test
    return DataLoader(d, batch_size=self.val_batch_size, shuffle=False, **self.loader_kwargs)
  
  def teardown(self, stage: str):
    ...


if __name__ == '__main__':
  make_h5_file(ROOT_DIR / 'config' / 'base.yaml')
