import importlib
from omegaconf import OmegaConf, DictConfig


def get_obj_from_str(str_, reload=False):
  module, cls = str_.rsplit(".", 1)
  if reload:
    module_imp = importlib.import_module(module)
    importlib.reload(module_imp)
  return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
  if not "target" in config:
    raise KeyError("Expected key `target` to instantiate.")
  return get_obj_from_str(config["target"])(**config.get("params", dict()))


def instantiate_from_config_recursively(config: dict|DictConfig):
  cls = config.get('class_path', None)
  if cls is None:
    raise KeyError('Expected key `class_path` to instantiate.')
  if isinstance(config, DictConfig):
    config = OmegaConf.to_object(config)
  need_init = lambda d: isinstance(d, dict) and 'class_path' in d
  
  params = dict()
  new_v = None
  for k, v in config.get('init_args', dict()).items():
    if need_init(v):
      new_v = instantiate_from_config_recursively(v)
    elif isinstance(v, list):
      new_v = []
      for i in v:
        new_v.append(instantiate_from_config_recursively(i) if need_init(i) else i)
    else:
      new_v = v
    params[k] = new_v

  return get_obj_from_str(cls)(**params)
