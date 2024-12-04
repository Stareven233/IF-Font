import traceback
import sys
sys.path.append('.')
import os
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
from lightning.pytorch.cli import LightningCLI
from pathlib import Path

from util import utils


def melk(trainer, ckptdir):
  if trainer.global_rank == 0:
    print('Summoning checkpoint.')
    ckpt_path = ckptdir / 'last_melk.ckpt'
    trainer.save_checkpoint(ckpt_path)


class IFFontCLI(LightningCLI):
  def fit(self, model, **kwargs):
    config = self.config.fit
    bs, base_lr = self.datamodule.batch_size, config.base_learning_rate
    accumulate_grad_batches = self.trainer.accumulate_grad_batches
    model.learning_rate = accumulate_grad_batches * bs * base_lr
    print(f'Setting learning rate to {model.learning_rate:.2e} = {accumulate_grad_batches} (accumulate_grad_batches) * {self.trainer.num_devices} (num_devices) * {bs} (batchsize) * {base_lr:.2e} (base_lr)')
    utils.show_parameters_num(model)
    utils.backup_codes(self.ckpt_dir / 'code')

    if config.use_compile:
      model = torch.compile(model)
      
    try:
      print(f'{config.name} start！')
      self.trainer.fit(model, **kwargs)
      print('finish！')
    except Exception:
      if config.save_on_exception:
        melk(self.trainer, self.ckpt_dir / 'ckpt')
      err_log = f'error_{config.name}.log'
      traceback.print_exc(file=open(err_log, 'w', encoding='utf-8'))
      raise

  def test(self, model, **kwargs):
    self.trainer.test(model, **kwargs)

  def add_arguments_to_parser(self, parser):
    parser.add_argument('--name', type=str, help='experiment name, determine the path of logger and ckpt')
    parser.add_argument('--base_learning_rate', type=float, help='base learning rate')
    parser.add_argument('--use_compile', type=bool, help='use torch.compile or not')
    parser.add_argument('--save_on_exception', type=bool, help='whether save checkpoint when Exception is raised')
    parser.link_arguments('model.init_args.gpt.init_args.config.init_args.n_embd', 'model.init_args.moco_wrapper.init_args.c_out')
    parser.link_arguments('model.init_args.gpt.init_args.config.init_args.n_embd', 'model.init_args.ids_enc.init_args.n_embd')

  def parse_arguments(self, parser, args) -> None:
    super().parse_arguments(parser, args)

    config = self.config[self.config.subcommand]
    ckpt_dir = Path(config.trainer.default_root_dir).parent / config.name
    config.trainer.logger.init_args.save_dir = ckpt_dir
    for callback in config.trainer.callbacks:
      if 'ModelCheckpoint' in callback.class_path:
        callback.init_args.dirpath = ckpt_dir / 'ckpt'
    self.ckpt_dir = ckpt_dir

if __name__ == "__main__":
  utils.show_cuda_info()
  torch.set_float32_matmul_precision('high')
  # import torch._dynamo
  # torch._dynamo.config.cache_size_limit = 8
  cli = IFFontCLI(parser_kwargs={'parser_mode': 'omegaconf'}, run=True)
