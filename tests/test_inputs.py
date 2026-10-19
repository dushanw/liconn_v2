"""Inputs, settings and environment match the authors' example."""

import importlib.metadata
import json
from pathlib import Path
import unittest

import numpy as np

from liconn_v2 import data
from liconn_v2 import reference

FFN_COMMIT = '12d680e21f96e41b6a893859364fa4a3e924e51a'
CONNECTOMICS_COMMIT = '73b76a7f6cd18969a8f6379207a63d121aa36a9c'
ROOT = Path(__file__).resolve().parent.parent


class InputsTest(unittest.TestCase):

  def test_cached_inputs_match_pinned_sha256(self):
    data.verify_cache()

  def test_normalized_crop_matches_notebook_layout(self):
    raw_xyz = data.load_raw_xyz()
    self.assertEqual(raw_xyz.shape, (500, 500, 500))
    self.assertEqual(raw_xyz.dtype, np.uint8)
    raw = reference.normalize(raw_xyz)
    self.assertEqual(raw.dtype, np.float32)
    self.assertTrue(np.isfinite(raw).all())
    x, y, z = reference.SEED_XYZ
    self.assertEqual(raw[z, y, x], (np.float32(raw_xyz[x, y, z]) - 128.0) / 33.)

  def test_settings_match_notebook(self):
    self.assertEqual(reference.DEPTH, 20)
    self.assertEqual(reference.FOV_SIZE, (33, 33, 33))
    self.assertEqual(reference.DELTAS, (8, 8, 8))
    self.assertEqual(reference.VOXEL_SIZE_ZYX, (24, 18, 18))
    self.assertEqual((reference.INIT_ACTIVATION, reference.PAD_VALUE,
                      reference.MOVE_THRESHOLD, reference.SEGMENT_THRESHOLD), (0.95, 0.5, 0.6, 0.6))
    self.assertEqual(reference.SEED_XYZ, (123, 171, 225))
    self.assertEqual(data.CROP_XYZ, (1100, 1083, 111))
    self.assertEqual(data.IMAGE_SCALE_INDEX, 1)

  def test_packages_come_from_pinned_sources(self):
    for name, commit in (('ffn', FFN_COMMIT), ('connectomics', CONNECTOMICS_COMMIT)):
      direct = json.loads(importlib.metadata.distribution(name).read_text('direct_url.json'))
      self.assertEqual(direct['vcs_info']['commit_id'], commit, name)
    import ffn  # pylint: disable=g-import-not-at-top
    import liconn_v2  # pylint: disable=g-import-not-at-top
    self.assertIn('site-packages', ffn.__file__)
    self.assertEqual(Path(liconn_v2.__file__).resolve().parent, ROOT / 'liconn_v2')


if __name__ == '__main__':
  unittest.main()
