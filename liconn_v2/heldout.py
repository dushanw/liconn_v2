"""Held-out crop B of ExPID82_1 for confirming speed variants (docs/heldout_design.md).

A second 500^3 crop, not overlapping the benchmark crop, with seeds chosen by the
benchmark's rule but excluding every benchmark seed object. Data are the same public
volumes as liconn_v2/data.py, cached separately and pinned by sha256.

Fetch with:  .venv/bin/python -m liconn_v2.heldout
"""

import os
from pathlib import Path

import numpy as np

from liconn_v2 import benchmark
from liconn_v2 import data

CROP_XYZ = (2200, 2600, 111)  # fixed 2026-10-21 before any held-out trace was run
CROP_SIZE = 500
RAW_FILE = 'raw_xyz_uint8.npy'
PROOFREAD_FILE = '231030_agg_240123_crop_xyz_uint64.npy'
PINNED_SHA256 = {
    RAW_FILE: '28431b5f83e7b1f74c2806aa8acafcb4d6264de37d777e10109beaef7e2ed2ce',
    PROOFREAD_FILE: '01a753471f1c5b1831657aaf73175146f15c036962f0d0622b74a4b2a7e11b75',
}
# Objects seeded in the benchmark (docs/benchmark_baseline.md); never held-out seeds.
BENCHMARK_SEED_OBJECTS = frozenset({3574, 13395, 10801, 5506, 218, 57259, 1525, 99141,
                                    56789, 61046, 93375, 97257})


def cache_dir() -> Path:
  return data.cache_dir() / 'heldout-crop-b'


def verify_cache(root: Path | None = None) -> None:
  root = cache_dir() if root is None else root
  for rel, expected in PINNED_SHA256.items():
    path = root / rel
    if not path.is_file():
      raise FileNotFoundError(f'{path} is missing; run: python -m liconn_v2.heldout')
    actual = data.sha256_file(path)
    if actual != expected:
      raise ValueError(f'{path} has sha256 {actual}, expected {expected}')


def load_raw_xyz() -> np.ndarray:
  verify_cache()
  return np.load(cache_dir() / RAW_FILE)


def load_proofread_xyz() -> np.ndarray:
  verify_cache()
  return np.load(cache_dir() / PROOFREAD_FILE)


def select_seeds(gt_zyx: np.ndarray, classes: dict | None = None) -> list:
  """Benchmark rule (rng seed 0, same size limits), minus benchmark seed objects; no notebook seed."""
  classes = benchmark.load_classes() if classes is None else classes
  rng = np.random.default_rng(benchmark.RNG_SEED)
  seeds = []
  for cls in ('axons', 'dendrites'):
    pool = [i for i in benchmark.candidates(gt_zyx, classes[cls]) if i not in BENCHMARK_SEED_OBJECTS]
    chosen = rng.choice(pool, size=min(benchmark.SEEDS_PER_CLASS, len(pool)), replace=False)
    for obj_id in chosen.tolist():
      seeds.append({'name': f'heldout_{cls[:-1]}_{obj_id}', 'class': cls[:-1], 'object_id': int(obj_id),
                    'seed_zyx': benchmark.seed_point(gt_zyx, int(obj_id)), 'candidates': len(pool)})
  return seeds


def fetch() -> None:
  root = cache_dir()
  root.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault('GCE_METADATA_ROOT', 'metadata.google.internal.invalid')
  import tensorstore as ts  # pylint: disable=g-import-not-at-top
  for path, scale_index, name in ((data.IMAGE_PATH, data.IMAGE_SCALE_INDEX, RAW_FILE),
                                   (data.PROOFREAD_PATH, 0, PROOFREAD_FILE)):
    if (root / name).is_file():
      continue
    vol = ts.open({'driver': 'neuroglancer_precomputed', 'kvstore': {'driver': 'gcs', 'bucket': data.BUCKET},
                   'path': path, 'scale_index': scale_index}, read=True).result()[ts.d['channel'][0]]
    x0, y0, z0 = CROP_XYZ
    n = CROP_SIZE
    np.save(root / name, vol[x0:x0 + n, y0:y0 + n, z0:z0 + n].read().result())
  verify_cache(root)


if __name__ == '__main__':
  fetch()
  print('Held-out crop present and matches pinned sha256 in', cache_dir())
