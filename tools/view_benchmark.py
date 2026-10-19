"""Local Neuroglancer view of the 12-seed FFN benchmark against the proofread segmentation.

Layers: raw image crop; FFN object masks (one ID per seed); proofread objects at the
seeds; seed points labelled with seed name and scores. Serves on 127.0.0.1 only.

Reads the cached inputs (python -m liconn_v2.data) and a full benchmark run written to
~/ResearchFlow/data/liconn-v2/runs/baseline-20261018 (examples/run_benchmark.py --out ...).
Usage: .venv-viewer/bin/python tools/view_benchmark.py [port]   (see requirements-viewer.txt)
"""

import json
import sys
import time
from pathlib import Path

import neuroglancer
import numpy as np

DATA = Path.home() / 'ResearchFlow' / 'data' / 'liconn-v2'
RUN = DATA / 'runs' / 'baseline-20261018'


def main():
  port = int(sys.argv[1]) if len(sys.argv) > 1 else 0
  raw = np.load(DATA / 'raw_xyz_uint8.npy')                              # x, y, z
  gt = np.load(DATA / '231030_agg_240123_crop_xyz_uint64.npy')           # x, y, z
  bench = json.loads((RUN / 'benchmark.json').read_text())
  records = {r['name']: r for r in bench['records']}

  ffn = np.zeros(raw.shape, np.uint32)
  legend = []
  with np.load(RUN / 'benchmark_masks.npz') as f:
    for i, s in enumerate(bench['seeds'], start=1):
      name = s['name']
      if 'mask_' + name not in f:
        continue
      crop = np.transpose(f['mask_' + name], [2, 1, 0])                   # zyx -> xyz
      oz, oy, ox = f['offset_zyx_' + name]
      region = ffn[ox:ox + crop.shape[0], oy:oy + crop.shape[1], oz:oz + crop.shape[2]]
      region[crop & (region == 0)] = i   # first seed wins where masks overlap
      legend.append((i, name, s['object_id'], s['seed_zyx']))

  neuroglancer.set_server_bind_address('127.0.0.1', port)
  dims = neuroglancer.CoordinateSpace(names=['x', 'y', 'z'], units='nm', scales=[18, 18, 24])
  viewer = neuroglancer.Viewer()
  with viewer.txn() as st:
    st.dimensions = dims
    st.layers['image'] = neuroglancer.ImageLayer(
        source=neuroglancer.LocalVolume(raw, dims))
    st.layers['proofread (seed objects)'] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(gt, dims),
        segments=sorted({int(o) for _, _, o, _ in legend}))
    st.layers['FFN traces'] = neuroglancer.SegmentationLayer(
        source=neuroglancer.LocalVolume(ffn, dims),
        segments=[i for i, *_ in legend])
    pts = neuroglancer.LocalAnnotationLayer(dimensions=dims)
    for i, name, obj, (z, y, x) in legend:
      r = records[name]
      pts.annotations.append(neuroglancer.PointAnnotation(
          id=str(i), point=[x, y, z],
          description=(f"FFN id {i}: {name} | proofread {obj} | steps {r['steps']} | "
                       f"recall {r['recall']:.2f} | precision {r['precision']:.2f} | "
                       f"merges {len(r['merges'])}")))
    st.layers['seeds'] = pts
    st.layout = '4panel'
    st.position = [legend[-1][3][2], legend[-1][3][1], legend[-1][3][0]]

  print('Neuroglancer URL:', viewer, flush=True)
  print('FFN id legend:', flush=True)
  for i, name, obj, _ in legend:
    print(f'  {i:2d}  {name:16s} proofread {obj}', flush=True)
  while True:
    time.sleep(3600)


if __name__ == '__main__':
  main()
