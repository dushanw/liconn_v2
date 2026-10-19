"""Local cache of the authors' public sample crop and FFN checkpoint.

The crop and checkpoint are the ones used by google/ffn
notebooks/jax_ffn_inference_liconn.ipynb (commit 12d680e). They are fetched once
from the public gs://liconn-public bucket and then pinned by sha256, so tests and
examples never touch the network.

Fetch with:  .venv/bin/python -m liconn_v2.data
"""

import hashlib
import os
from pathlib import Path
import urllib.request

import numpy as np

BUCKET = 'liconn-public'
IMAGE_PATH = 'ExPID82_1/image_230130b'
IMAGE_SCALE_INDEX = 1  # 18 x 18 x 24 nm voxels (2x downsampled), as in the notebook.
CROP_XYZ = (1100, 1083, 111)
CROP_SIZE = 500
CHECKPOINT_PREFIX = 'models/ffn/axde_59110972/ckpt-2116'
# Authors' proofread segmentation ("proofread" layer of ng_states/expid82.json),
# same grid as the image at IMAGE_SCALE_INDEX.
PROOFREAD_PATH = 'ExPID82_1/segmentation/231030_agg_240123'

RAW_FILE = 'raw_xyz_uint8.npy'
PROOFREAD_FILE = '231030_agg_240123_crop_xyz_uint64.npy'
CHECKPOINT_FILES = ('ckpt/ckpt-2116.flax', 'ckpt/ckpt-2116.index')

# sha256 of the cached files, recorded on first fetch (2026-10-18).
PINNED_SHA256 = {
    RAW_FILE: 'c9abc91b551da7ad2b5b7ca6ac12ebff3b5c2e9e4a55b05140694e1f50083837',
    PROOFREAD_FILE: 'f5f5b8aae8e052efb99a775319b1e7955ee9a3a6f4682847390662579016c182',
    'ckpt/ckpt-2116.flax': 'dbe2bc6fb4169a33101cb5b688f34dcf7b6b39bdf0a1d9866523e5f95033057e',
    'ckpt/ckpt-2116.index': '92ccbd68e09b55db34cbb36b749926eede07569c656579f9a0beda783582bd98',
}


def cache_dir() -> Path:
  return Path(os.environ.get('LICONN_DATA', Path.home() / 'ResearchFlow' / 'data' / 'liconn-v2'))


def sha256_file(path: Path) -> str:
  h = hashlib.sha256()
  with open(path, 'rb') as f:
    for block in iter(lambda: f.read(1 << 20), b''):
      h.update(block)
  return h.hexdigest()


def verify_cache(root: Path | None = None) -> None:
  """Raises if any cached input is missing or differs from its pinned hash."""
  root = cache_dir() if root is None else root
  for rel, expected in PINNED_SHA256.items():
    path = root / rel
    if not path.is_file():
      raise FileNotFoundError(f'{path} is missing; run: python -m liconn_v2.data')
    actual = sha256_file(path)
    if actual != expected:
      raise ValueError(f'{path} has sha256 {actual}, expected {expected}')


def load_raw_xyz(root: Path | None = None) -> np.ndarray:
  """Returns the uint8 crop exactly as read by the notebook (x, y, z order)."""
  root = cache_dir() if root is None else root
  verify_cache(root)
  return np.load(root / RAW_FILE)


def load_proofread_xyz(root: Path | None = None) -> np.ndarray:
  """Returns the uint64 proofread labels for the same crop (x, y, z order)."""
  root = cache_dir() if root is None else root
  verify_cache(root)
  return np.load(root / PROOFREAD_FILE)


def checkpoint_path(root: Path | None = None) -> str:
  """Path prefix accepted by clu.checkpoint.Checkpoint.load_state."""
  root = cache_dir() if root is None else root
  verify_cache(root)
  return str(root / 'ckpt' / 'ckpt-2116')


def fetch(root: Path | None = None) -> None:
  """Downloads the public crop and checkpoint, then checks the pinned hashes."""
  root = cache_dir() if root is None else root
  (root / 'ckpt').mkdir(parents=True, exist_ok=True)
  for suffix in ('.flax', '.index'):
    target = root / 'ckpt' / ('ckpt-2116' + suffix)
    if not target.is_file():
      url = f'https://storage.googleapis.com/{BUCKET}/{CHECKPOINT_PREFIX}{suffix}'
      urllib.request.urlretrieve(url, target)
  for path, scale_index, name in ((IMAGE_PATH, IMAGE_SCALE_INDEX, RAW_FILE),
                                   (PROOFREAD_PATH, 0, PROOFREAD_FILE)):
    if (root / name).is_file():
      continue
    # Same read as the notebook; anonymous access to the public bucket.
    os.environ.setdefault('GCE_METADATA_ROOT', 'metadata.google.internal.invalid')
    import tensorstore as ts  # pylint: disable=g-import-not-at-top
    context = ts.Context({'cache_pool': {'total_bytes_limit': 1_000_000_000}})
    vol = ts.open({
        'driver': 'neuroglancer_precomputed',
        'kvstore': {'driver': 'gcs', 'bucket': BUCKET},
        'path': path,
        'scale_index': scale_index},
        read=True, context=context).result()[ts.d['channel'][0]]
    x0, y0, z0 = CROP_XYZ
    n = CROP_SIZE
    np.save(root / name, vol[x0:x0 + n, y0:y0 + n, z0:z0 + n].read().result())
  verify_cache(root)


if __name__ == '__main__':
  fetch()
  print('Inputs present and match pinned sha256 in', cache_dir())
