"""Re-runnable survey of public LICONN data for training (docs/training_feasibility.md).

  survey_training_data.py [--out DIR]

Reads the gs://liconn-public listing and each volume's info file (metadata), and one
coarse level of the ExPID82_1 proofread segmentation (288 x 288 x 384 nm, 240 x 305 x
50 voxels, about 29 MB) to measure label coverage. No image data. Needs internet access.

Output: DIR/training_data_survey.json
"""

import argparse
import json
import os
from pathlib import Path
import urllib.request

import numpy as np

BUCKET = 'liconn-public'
API = f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o'
VOLUMES = ['ExPID82_1/image_230130b', 'ExPID82_1/segmentation/231030_agg_240123',
           'ExPID124/image', 'ExPID124/seg_230907_agglo231128_combined2', 'ExPID146/image',
           'ExPID09_01/image', 'ExPID150/multiround_fusion_240918']


def _get(url):
  with urllib.request.urlopen(url, timeout=60) as r:
    return json.loads(r.read())


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  top = _get(f'{API}?delimiter=/&fields=prefixes').get('prefixes', [])
  models = [i['name'] for i in _get(f'{API}?prefix=models/ffn/&fields=items(name)').get('items', [])]
  vols = {}
  for v in VOLUMES:
    info = _get(f'https://storage.googleapis.com/{BUCKET}/{v}/info')
    s0 = info['scales'][0]
    vols[v] = {'type': info['type'], 'data_type': info['data_type'],
               'finest_resolution_nm': s0['resolution'], 'finest_size_xyz': s0['size'],
               'scales': len(info['scales'])}
  os.environ.setdefault('GCE_METADATA_ROOT', 'metadata.google.internal.invalid')
  import tensorstore as ts  # pylint: disable=g-import-not-at-top
  seg = ts.open({'driver': 'neuroglancer_precomputed', 'kvstore': {'driver': 'gcs', 'bucket': BUCKET},
                 'path': 'ExPID82_1/segmentation/231030_agg_240123', 'scale_index': 4},
                read=True).result()[ts.d['channel'][0]]
  labels = seg.read().result()
  lab = labels > 0
  zfrac = [float(b.mean()) for b in np.array_split(lab.mean(axis=(0, 1)), 10)]
  result = {
      'bucket_top_level': top,
      'ffn_model_files': models,
      'volumes': vols,
      'expid82_proofread_coverage': {
          'level_nm': [288, 288, 384],
          'shape_xyz': list(lab.shape),
          'labeled_fraction': float(lab.mean()),
          'labeled_fraction_by_z_decile': zfrac,
          'objects': int(len(np.unique(labels[lab]))),
      },
  }
  (out / 'training_data_survey.json').write_text(json.dumps(result, indent=2) + '\n')
  print(json.dumps(result, indent=2))


if __name__ == '__main__':
  main()
