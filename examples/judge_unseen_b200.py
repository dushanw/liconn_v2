"""Judge the SegNeuron-on-B200 addendum (docs/unseen_b200_segneuron_design.md) from its result files.

  judge_unseen_b200.py --b200 RUN_B200 --v100 RUN/v100 [--out docs/unseen_b200_evidence.json]

RUN_B200 holds crop-K/segneuron.json, design-sha256.txt, gpu-segneuron.txt, source_sha256.txt and
env-infer.txt for K in 41-50 (cluster/mit/unseen_segneuron.sbatch); RUN/v100 is the test's V100 run
directory. The B200 FFN crop times come from docs/unseen_evidence.json. Besides these, nothing is read.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import unseen  # pylint: disable=g-import-not-at-top
from liconn_v2 import unseen_b200  # pylint: disable=g-import-not-at-top

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / 'docs' / 'unseen_evidence.json'


def crop_dir(run: Path, k: int) -> Path:
  return run / f'crop-{k:02d}'


def load(b200: Path, v100: Path, pinned: dict, source: dict) -> tuple:
  """({k: B200 file or None}, {k: V100 file}, {k: device record}); refuses a V100 file that is not the
  test's (by its sha256 in docs/unseen_evidence.json, pinned) and a crop whose job ran under another
  version of the design or not on a B200, and results from different staged sources, from a staged
  source that changed the test's code (source, its sha256 per file), or without the design's inference
  versions."""
  design = unseen_b200.design_sha256()
  files, ref, devices, sources, envs = {}, {}, {}, {}, {}
  for k in unseen.CROPS:
    path = crop_dir(v100, k) / 'segneuron.json'
    if data.sha256_file(path) != pinned[f'v100/crop-{k:02d}/segneuron.json']:
      raise ValueError(f"{path} is not the test's V100 SegNeuron result")
    ref[k] = json.loads(path.read_text())
    d = crop_dir(b200, k)
    if not (d / 'segneuron.json').exists():
      files[k] = None
      continue
    recorded = (d / 'design-sha256.txt').read_text().split()[0]
    if recorded != design:
      raise ValueError(f'crop {k}: job ran under design {recorded}, not {design}')
    files[k] = json.loads((d / 'segneuron.json').read_text())
    gpu = (d / 'gpu-segneuron.txt').read_text().splitlines()
    unseen_b200.check_device(gpu[1], k)
    devices[k] = {'gpu': gpu[1], 'job': gpu[2]}
    sources[k] = (d / 'source_sha256.txt').read_text()
    envs[k] = (d / 'env-infer.txt').read_text()
  unseen_b200.check_staging(sources, envs, source)
  return files, ref, devices


def build(b200: Path, v100: Path) -> dict:
  evidence = json.loads(EVIDENCE.read_text())
  files, ref, devices = load(b200, v100, evidence['result_files_sha256'], evidence['source_sha256'])
  hashed = [crop_dir(b200, k) / f for k in unseen.CROPS for f in ('segneuron.json', 'env-infer.txt')
            if files[k] is not None]
  return {'design': 'docs/unseen_b200_segneuron_design.md', 'design_sha256': unseen_b200.design_sha256(),
          'test_evidence_sha256': data.sha256_file(EVIDENCE),
          'result_files_sha256': {str(p.relative_to(b200)): data.sha256_file(p) for p in hashed},
          'v100_files_sha256': {str(crop_dir(v100, k).relative_to(v100.parent) / 'segneuron.json'):
                                data.sha256_file(crop_dir(v100, k) / 'segneuron.json') for k in unseen.CROPS},
          'devices': devices, 'judged': unseen_b200.judge(files, ref, evidence)}


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--b200', required=True)
  p.add_argument('--v100', required=True)
  p.add_argument('--out', default=str(ROOT / 'docs' / 'unseen_b200_evidence.json'))
  a = p.parse_args()
  r = build(Path(a.b200), Path(a.v100))
  Path(a.out).write_text(json.dumps(r, indent=1, sort_keys=True) + '\n')
  for way, c in r['judged']['comparisons'].items():
    for run, x in c.items():
      print(way, run, x['statement'], x.get('totals', {}).get('break_even_per_crop'))


if __name__ == '__main__':
  main()
