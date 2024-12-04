import json
import pickle
from pathlib import Path


def load_json(path, **kwargs) -> dict:
  if not isinstance(path, Path):
    path = Path(path)
  if not path.is_file():
    return None
  with open(path, 'r', encoding='utf-8') as f:
    return json.load(f, **kwargs)


def write_json(path, obj, **kwargs) -> None:
  with open(path, 'w', encoding='utf-8') as f:
    json.dump(obj, f, ensure_ascii=False, **kwargs)


def load_pickle(path, **kwargs):
  if not isinstance(path, Path):
    path = Path(path)
  if not path.is_file():
    return None
  with open(path, 'rb') as f:
    return pickle.load(f, **kwargs)


def write_pickle(path, obj, **kwargs) -> None:
  with open(path, 'wb') as f:
    pickle.dump(obj, f, **kwargs)
