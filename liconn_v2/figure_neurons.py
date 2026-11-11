"""Whole test-crop neurons as reconstructed by every method (docs/figure_neurons_design.md).

The draw, the verification against the test's records and the part volumes (numpy only). The B200 job
is examples/trace_figure_neurons.py.
"""

import hashlib

import numpy as np

from liconn_v2 import unseen

SEED = 20261111
DRAWN = 3
POPULATION = 4579  # the test objects of crops 41-50 (docs/unseen_results.md)
TRACERS = ('F1', 'F5', 'S10', 'P2', 'Q2')
METHODS = TRACERS + ('segneuron',)
PARTS = {'found': 1, 'missed': 2, 'other': 3, 'unlabelled': 4}


def draw(object_ids: dict, population: int = POPULATION) -> list:
  """object_ids: crop -> the test's object order on that crop. The drawn (crop, object_id), in the
  order drawn. Refuses any other number of objects than the test's."""
  flat = [(k, int(o)) for k in unseen.CROPS for o in object_ids[k]]
  if len(flat) != population or len(set(flat)) != population:
    raise ValueError(f'{len(flat)} objects ({len(set(flat))} distinct), not the test\'s {population}')
  picks = np.random.default_rng(SEED).choice(len(flat), DRAWN, replace=False)
  return [flat[int(i)] for i in picks]


def check_trace(mine: dict, test: dict) -> None:
  """A re-traced object against the test's record of the same object and method."""
  for key in ('object_id', 'steps', 'mask_sha256', 'history_sha256'):
    if mine[key] != test[key]:
      raise ValueError(f"object {test['object_id']}: {key} {mine[key]} differs from the test's {test[key]}")


def check_segneuron(mine: dict, test: dict) -> None:
  """SegNeuron's rebuilt segment for an object against its recorded scores."""
  for key in ('object_id', 'recall', 'precision', 'f1'):
    if mine[key] != test[key]:
      raise ValueError(f"object {test['object_id']}: SegNeuron {key} {mine[key]} differs from the record {test[key]}")


def parts(pred: np.ndarray, gt: np.ndarray, object_id: int) -> tuple:
  """(corner zyx, uint8 volume) in the bounding box of the reconstruction and the labelled object:
  1 found, 2 missed, 3 in another labelled object, 4 unlabelled (PARTS)."""
  obj = gt == object_id
  vol = np.zeros(gt.shape, np.uint8)
  vol[pred & obj] = PARTS['found']
  vol[obj & ~pred] = PARTS['missed']
  vol[pred & ~obj & (gt > 0)] = PARTS['other']
  vol[pred & (gt == 0)] = PARTS['unlabelled']
  return crop_box(vol)


def label_pane(gt: np.ndarray, object_id: int) -> tuple:
  """The labelled object alone, as found (1), in its bounding box."""
  return crop_box((gt == object_id).astype(np.uint8))


def reconstruction(lo: list, vol: np.ndarray, shape: tuple) -> np.ndarray:
  """The full-grid reconstruction a part volume encodes (found, other labelled and unlabelled)."""
  out = np.zeros(shape, bool)
  out[tuple(slice(a, a + s) for a, s in zip(lo, vol.shape))] = np.isin(vol, (PARTS['found'], PARTS['other'],
                                                                           PARTS['unlabelled']))
  return out


def mask_sha256(mask: np.ndarray) -> str:
  """As examples/time_crop11.py:record."""
  return hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()


def check_score(score: dict, test: dict, tol: float = 1e-12) -> None:
  """benchmark.score of SegNeuron's extracted segment against its record: recall and precision within
  tol, and the same merges."""
  for key in ('recall', 'precision'):
    if abs(score[key] - test[key]) > tol:
      raise ValueError(f"object {test['object_id']}: SegNeuron {key} {score[key]} from the segment, {test[key]} recorded")
  if [list(m) for m in score['merges']] != [list(m) for m in test['merges']]:
    raise ValueError(f"object {test['object_id']}: SegNeuron merges differ from the record")


def crop_box(vol: np.ndarray) -> tuple:
  nz = np.nonzero(vol)
  if not nz[0].size:
    raise ValueError('nothing to draw')
  lo = [int(a.min()) for a in nz]
  hi = [int(a.max()) + 1 for a in nz]
  return lo, vol[tuple(slice(a, b) for a, b in zip(lo, hi))].copy()
