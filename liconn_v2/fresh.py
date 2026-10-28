"""300 fresh seeds for confirming fp16_head32 (docs/fresh_design.md).

Continues the many-seed crop order (liconn_v2/manyseed.py) after the crops already used,
with the same seed rule, excluding every object seeded before. Crops are cached next to
the many-seed crops; their pins and seeds are kept in docs/fresh_crops.json and
docs/fresh_seeds.json.

Fetch with:  .venv/bin/python -m liconn_v2.fresh
"""

import json
from pathlib import Path

import numpy as np

from liconn_v2 import data
from liconn_v2 import manyseed

TARGET_SEEDS = 300
MAX_FRESH_BYTES = 20e9  # owner-approved storage for the fresh crops (12 GB, raised to 16 and 20 GB on 2026-10-23)
DOCS = Path(__file__).resolve().parent.parent / 'docs'
MANIFEST = DOCS / 'fresh_crops.json'
SEEDS_FILE = DOCS / 'fresh_seeds.json'


def first_crop() -> int:
  """The first crop of the pre-registered order not used by the many-seed set."""
  return len(manyseed.load_manifest()['crops'])


def excluded_objects() -> set:
  return manyseed.excluded_objects() | {s['object_id'] for s in manyseed.recorded_seeds()}


def load_manifest() -> dict:
  return json.loads(MANIFEST.read_text())


def _entry(k: int, manifest: dict | None = None) -> dict:
  manifest = load_manifest() if manifest is None else manifest
  for c in manifest['crops']:
    if c['index'] == k:
      return c
  raise KeyError(f'crop {k} is not a fresh crop')


def verify_crop(k: int, manifest: dict | None = None) -> None:
  for rel, expected in _entry(k, manifest)['sha256'].items():
    path = manyseed.crop_dir(k) / rel
    if not path.is_file():
      raise FileNotFoundError(f'{path} is missing; run: python -m liconn_v2.fresh')
    actual = data.sha256_file(path)
    if actual != expected:
      raise ValueError(f'{path} has sha256 {actual}, expected {expected}')


def load_raw_xyz(k: int) -> np.ndarray:
  verify_crop(k)
  return np.load(manyseed.crop_dir(k) / manyseed.RAW_FILE)


def load_proofread_xyz(k: int) -> np.ndarray:
  verify_crop(k)
  return np.load(manyseed.crop_dir(k) / manyseed.PROOFREAD_FILE)


def load_proofread_zyx(k: int) -> np.ndarray:
  return np.ascontiguousarray(np.transpose(load_proofread_xyz(k), [2, 1, 0]))


def recorded_seeds() -> list:
  return json.loads(SEEDS_FILE.read_text())['seeds']


def seeds_for_crop(k: int, gt_zyx: np.ndarray | None = None) -> list:
  """Crop k's recorded seeds, re-derived from the crop to check they follow the rule."""
  recorded = recorded_seeds()
  mine = [s for s in recorded if s['crop'] == k]
  if gt_zyx is not None:
    earlier = {s['object_id'] for s in recorded if s['crop'] < k}
    again = manyseed.select_seeds_for_crop(gt_zyx, k, excluded_objects() | earlier)
    last = k == max(s['crop'] for s in recorded)
    n_earlier = sum(s['crop'] < k for s in recorded)
    if last and len(mine) != min(len(again), TARGET_SEEDS - n_earlier):
      raise ValueError(f'crop {k}: last crop has {len(mine)} seeds, expected {TARGET_SEEDS - n_earlier}')
    if (again[:len(mine)] if last else again) != mine:
      raise ValueError(f'crop {k}: recorded seeds differ from the seed rule')
  return mine


def fetch_and_select() -> list:
  """Fetches crops in the pre-registered order, from the first unused one, until 300
  fresh seeds; records crops and seeds. Existing pins are verified first, never replaced."""
  root = manyseed.cache_dir()
  coarse_path = root / manyseed.COARSE_FILE
  if data.sha256_file(coarse_path) != manyseed.load_manifest()['coarse_sha256']:
    raise ValueError(f'{coarse_path} differs from its pinned sha256')
  pinned = load_manifest() if MANIFEST.is_file() else None
  for c in (pinned or {}).get('crops', []):  # cached crops, before any download
    for name, expected in c['sha256'].items():
      path = manyseed.crop_dir(c['index']) / name
      if path.is_file() and data.sha256_file(path) != expected:
        raise ValueError(f'{path} differs from its pinned sha256 in {MANIFEST}')
  start = first_crop()
  order = [c for c in manyseed.select_crops(np.load(coarse_path), None) if c['index'] >= start]
  image = manyseed._open(data.IMAGE_PATH, data.IMAGE_SCALE_INDEX)  # pylint: disable=protected-access
  seg = manyseed._open(data.PROOFREAD_PATH, 0)  # pylint: disable=protected-access
  crops, seeds, excluded = [], [], excluded_objects()
  stored = 0
  for c in order:
    if len(seeds) >= TARGET_SEEDS:
      break
    per_crop = stored / len(crops) if crops else 0
    if stored + per_crop > MAX_FRESH_BYTES:
      raise RuntimeError(f'{len(seeds)} fresh seeds after {len(crops)} crops ({stored / 1e9:.1f} GB); the next crop would '
                         f'pass the approved {MAX_FRESH_BYTES / 1e9:.0f} GB. Nothing was pinned; ask the owner.')
    manyseed.fetch_crop(c, image, seg)
    stored += sum(f.stat().st_size for f in manyseed.crop_dir(c['index']).iterdir())
    if pinned is not None and len(crops) < len(pinned['crops']) and c != pinned['crops'][len(crops)]:
      raise ValueError(f"crop {c['index']} differs from its pinned entry in {MANIFEST}")
    crops.append(c)
    gt = np.ascontiguousarray(np.transpose(np.load(manyseed.crop_dir(c['index']) / manyseed.PROOFREAD_FILE), [2, 1, 0]))
    s = manyseed.select_seeds_for_crop(gt, c['index'], excluded)[:TARGET_SEEDS - len(seeds)]
    excluded |= {x['object_id'] for x in s}
    seeds += s
    print(f"crop {c['index']} at {c['xyz']} coverage {c['coverage']:.3f}: "
          f"{sum(x['class'] == 'dendrite' for x in s)} dendrites, {sum(x['class'] == 'axon' for x in s)} axons; "
          f"total {len(seeds)}", flush=True)
  if len(seeds) < TARGET_SEEDS:
    raise RuntimeError(f'the pre-registered crop order ran out at {len(seeds)} fresh seeds after {len(crops)} crops. '
                       'Nothing was pinned; ask the owner.')
  manifest = {'design': 'docs/fresh_design.md', 'first_crop': start, 'crops': crops}
  if pinned is not None and manifest != pinned:
    raise ValueError(f'the crops differ from the pinned {MANIFEST}; nothing was written')
  if SEEDS_FILE.is_file() and json.loads(SEEDS_FILE.read_text())['seeds'] != seeds:
    raise ValueError(f'the seeds differ from the recorded {SEEDS_FILE}; nothing was written')
  MANIFEST.write_text(json.dumps(manifest, indent=1) + '\n')
  SEEDS_FILE.write_text(manyseed.seeds_text(seeds).replace('docs/manyseed_design.md', 'docs/fresh_design.md'))
  return seeds


if __name__ == '__main__':
  s = fetch_and_select()
  print(f'{len(s)} fresh seeds recorded in {SEEDS_FILE} (target {TARGET_SEEDS})')
