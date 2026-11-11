"""Write docs/unseen_b200_results.md from docs/unseen_b200_evidence.json and the addendum's run records.

  write_unseen_b200_results.py [--b200 RUN_B200] [--v100 RUN/v100] [--out docs/unseen_b200_results.md]

Every number on the page is formatted from the evidence written by examples/judge_unseen_b200.py, or read
from the run records (smoke-run header, environment lists, affinity file sizes, patch size). The run directories
default to the data cache. tests/test_unseen_b200_evidence.py requires the page to equal render().
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import unseen_b200  # pylint: disable=g-import-not-at-top

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / 'docs' / 'unseen_b200_evidence.json'
PAGE = ROOT / 'docs' / 'unseen_b200_results.md'
B200 = data.cache_dir() / 'runs' / 'unseen-b200-segneuron-20261106'
V100 = data.cache_dir() / 'runs' / 'unseen-20261104' / 'v100'
CROPS = [str(k) for k in range(41, 51)]
RUNS = ('F5', 'F1', 'S10', 'P2', 'Q2')
NAMES = {'F1': 'float32 FFN one object at a time (F1)', 'F5': 'batched fast FFN (F5)', 'S10': 'the student (S10)',
         'P2': 'P8 (P2)', 'Q2': 'P4 (Q2)'}


def render(e: dict, b200: Path, v100: Path) -> str:
  """The page's full text from the evidence e and the B200 and V100 run directories."""
  j = e['judged']
  host = {k: e['devices'][k]['job'].split()[1] for k in CROPS}
  jobs = sorted({e['devices'][k]['job'].split()[3].split('_')[0] for k in CROPS})
  job = ', '.join(jobs)
  n_crops = len(CROPS)
  span = f'{CROPS[0]}-{CROPS[-1]}'
  gt = unseen_b200.AGREEMENT_F1
  cpus = sorted({e['devices'][k]['job'].split()[-1] for k in CROPS})
  gpus = sorted({e['devices'][k]['gpu'] for k in CROPS})
  nodes = sorted(set(host.values()))
  sb, sv = j['segneuron_b200'], j['segneuron_v100']
  e2e, ws = j['comparisons']['end_to_end'], j['comparisons']['without_startup']
  a = j['agreement_pooled']
  objects = [j['objects_per_crop'][k] for k in CROPS]
  smoke = json.loads((b200 / 'smoke20' / 'crop-11' / 'segneuron.json').read_text())
  smoke_job = (b200 / 'smoke20' / 'crop-11' / 'gpu-segneuron.txt').read_text().splitlines()[2].split()[3]
  pins = sorted({line for k in CROPS for line in (b200 / f'crop-{k}' / 'env-infer.txt').read_text().split()
                 if line.split('==')[0] in ('torch', 'numpy', 'tifffile')})
  v100_torch = sorted({line for k in CROPS for line in (v100 / f'crop-{k}' / 'env-infer.txt').read_text().split()
                       if line.startswith('torch==')})
  aff = sorted({(b200 / f'crop-{k}' / 'infer' / 'affinities.npy').stat().st_size for k in CROPS})
  patches = {tuple(json.loads((b200 / f'crop-{k}' / 'segneuron.json').read_text())['inference']['crop_size_zyx'])
             for k in CROPS}
  if len(patches) != 1:
    raise ValueError(f'crops ran with different patch sizes: {sorted(patches)}')
  patch = ' x '.join(str(d) for d in patches.pop())

  def tot(d, key):
    return sum(d[k][key] for k in CROPS)

  def crops(ks):
    return ', '.join(str(k) for k in ks) if ks else 'none'

  L = []
  w = L.append
  w(f'# SegNeuron on the B200, crops {span}: results of the timing addendum')
  w('')
  w(f"Design: docs/unseen_b200_segneuron_design.md (sha256 `{e['design_sha256'][:16]}...`). Evidence:")
  w('docs/unseen_b200_evidence.json, written by examples/judge_unseen_b200.py (unchanged since its check) from the')
  w(f'result files of MIT job {job} in the data cache (`runs/unseen-b200-segneuron-20261106`) and the test\'s')
  w('V100 SegNeuron run (`runs/unseen-20261104/v100`, files pinned by docs/unseen_evidence.json). Evidence test:')
  w('tests/test_unseen_b200_evidence.py. This page is written by examples/write_unseen_b200_results.py, and the test')
  w('requires it to be exactly that script\'s output.')
  w('')
  w('This addendum was planned after the unseen-crop test\'s results were known (docs/unseen_results.md);')
  w('its rule and measures were fixed before SegNeuron ran on the B200. It times SegNeuron; it does not')
  w('re-judge accuracy, and the test\'s accuracy claims (from the V100 run) are unchanged.')
  w('')
  w('## Runs')
  w('')
  w(f"- Smoke run on crop {smoke['crop']} (job {smoke_job}, {smoke['smoke_objects']} objects scored): completed. "
    'It checked only that the job runs and is not a result.')
  w(f"- {n_crops} crops (job {job}, one B200 per crop, {', '.join(cpus)} CPUs): every crop finished; failures: "
    f"{len(j['failures'])}; non-finite SegNeuron outputs: none on any crop.")
  w(f"- Every crop's records passed the judge's checks: {'; '.join(gpus)} (GPU and driver), the same staged source for "
    'all crops with every file of the test\'s code unchanged, the inference versions '
    f"{', '.join(pins)}, the design's sha256, the test's checkpoint, threshold, crop files and objects.")
  for n in nodes:
    w(f"- {n}: crops {crops([k for k in CROPS if host[k] == n])}.")
  w('')
  w('## Crop time against each B200 run of the test')
  w('')
  w('SegNeuron\'s crop time end to end (primary) is the wall time of `inference.py` plus post-processing, as on')
  w('the V100; without start-up it is `inference.py`\'s own loop time plus post-processing. The FFN runs\'')
  w('crop times are the test\'s (docs/unseen_evidence.json). Break-even: objects per crop at which the two cost')
  w('the same, per crop (median and range) and from the totals over all crops.')
  for way, title in ((e2e, 'End to end (primary)'), (ws, 'Without start-up')):
    w('')
    w(f'{title}:')
    w('')
    w('| run | statement | crops cheaper | break-even per crop, median (range) | break-even from totals |')
    w('|---|---|---|---|---|')
    for r in RUNS:
      c = way[r]
      b = c['break_even']
      w(f"| {r} | {c['statement']} | {c['crops_segneuron_cheaper']} of {n_crops} | "
        f"{b['median']:.0f} ({b['min']:.0f}-{b['max']:.0f}) | "
        f"{c['totals']['break_even_per_crop']:.0f} |")
  w('')
  w('Crops on which SegNeuron was cheaper, end to end:')
  w('')
  for r in RUNS:
    w(f"- {NAMES[r]}: {crops(e2e[r]['cheaper_on'])}.")
  w('')
  w(f"Totals over all crops, end to end: SegNeuron {e2e['F5']['totals']['segneuron_seconds']:,.0f} s; "
    + '; '.join(f"{r} {e2e[r]['totals']['run_seconds']:,.0f} s" for r in ('F1', 'F5', 'S10', 'P2', 'Q2')) + '.')
  w('')
  w('## Per crop')
  w('')
  w('| crop | objects | node | SegNeuron B200, end to end (s) | inference.py wall | loop | post-processing '
    '| F5 crop time | break-even vs F5 |')
  w('|---|---|---|---|---|---|---|---|---|')
  for k in CROPS:
    x = sb[k]
    f5 = x['end_to_end'] / e2e['F5']['segneuron_over_run'][k]
    w(f"| {k} | {j['objects_per_crop'][k]} | {host[k]} | {x['end_to_end']:.0f} | {x['inference_wall_seconds']:.0f} | "
      f"{x['inference_seconds']:.0f} | {x['post_seconds']:.0f} | {f5:.0f} | "
      f"{e2e['F5']['break_even_per_crop'][k]:.0f} |")
  w('')
  w('## SegNeuron on the B200 against the V100 (totals over all crops)')
  w('')
  w('| | end to end | inference.py wall | loop | post-processing |')
  w('|---|---|---|---|---|')
  for label, d in (('B200', sb), ('V100 (the test\'s run)', sv)):
    w(f"| {label} | {tot(d, 'end_to_end'):,.0f} s | {tot(d, 'inference_wall_seconds'):,.0f} s | "
      f"{tot(d, 'inference_seconds'):,.0f} s | {tot(d, 'post_seconds'):,.0f} s |")
  w('')
  w('Per crop:')
  w('')
  w('| crop | loop, B200 | loop, V100 | post-processing, B200 | post-processing, V100 |')
  w('|---|---|---|---|---|')
  for k in CROPS:
    w(f"| {k} | {sb[k]['inference_seconds']:.0f} s | {sv[k]['inference_seconds']:.0f} s | "
      f"{sb[k]['post_seconds']:.0f} s | "
      f"{sv[k]['post_seconds']:.0f} s |")
  w('')
  w('Post-processing runs on the host CPUs of each cluster, so the post-processing rows compare CPUs, not GPUs.')
  w('')
  w('## Agreement with the V100 run')
  w('')
  w(f"Per-object F1, B200 minus V100, over all {a['objects']:,} objects: mean {a['mean_f1_difference']:+.6f}, "
    f"largest absolute {a['max_abs_f1_difference']:.3f}, {a['objects_abs_difference_gt_0.1']} objects differing by "
    f'more than {gt}. Mean F1 per crop:')
  w('')
  w(f'| crop | B200 | V100 | objects differing by > {gt} |')
  w('|---|---|---|---|')
  for k in CROPS:
    g = j['agreement'][k]
    w(f"| {k} | {g['mean_f1_b200']:.4f} | {g['mean_f1_v100']:.4f} | {g['objects_abs_difference_gt_0.1']} |")
  w('')
  w('The test\'s accuracy claims stand as they were; the agreement only shows that the B200 run is the same method.')
  w('')
  w('## Disclosures')
  w('')
  post = {n: [sb[k]['post_seconds'] for k in CROPS if host[k] == n] for n in nodes}
  loop = [sb[k]['inference_seconds'] for k in CROPS]
  cheap = set(e2e['F5']['cheaper_on'])
  w(f'- **Post-processing time differed between the {len(nodes)} nodes.** ' + '; '.join(
      f"{n}: {min(v):.0f}-{max(v):.0f} s" for n, v in post.items())
    + f" (the GPU loop: {min(loop):.0f}-{max(loop):.0f} s on every crop). Against batched fast FFN, SegNeuron was "
    'cheaper on ' + '; '.join(
      f"{sum(1 for k in CROPS if host[k] == n and int(k) in cheap)} of "
      f"{sum(1 for k in CROPS if host[k] == n)} crops on {n}"
      for n in nodes) + '. Each crop ran once on one node, so node and crop cannot be separated here. On the V100 '
    f"the same crops' post-processing took {min(sv[k]['post_seconds'] for k in CROPS):.0f}-"
    f"{max(sv[k]['post_seconds'] for k in CROPS):.0f} s, which is consistent with a node effect but does not show how "
    'these crops would run on the other B200 node. Whether the count of '
    f"{e2e['F5']['crops_segneuron_cheaper']} depends on node assignment was not measured.")
  w(f'- Crops {span} had been read by every method of the test before this addendum ran.')
  w('- Planned after the test\'s results, including the V100 times and the B200 FFN times, were known; one run')
  w('  per method and crop; FFN\'s B200 times come from the test\'s own jobs, SegNeuron\'s from these later jobs,')
  w('  possibly on other nodes.')
  w(f'- Neither method is optimised for the B200: `inference.py` runs one {patch} patch at a time and')
  w('  post-processing is single-process CPU code. SegNeuron\'s end-to-end time includes process start-up,')
  sizes = ', '.join(f'{s:,}' for s in aff)
  w(f"  model loading and reading and writing its affinities ({sizes} bytes per crop) on MIT")
  w('  pool storage; FFN\'s crop time leaves out loading the network and follows a warm-up.')
  w('- For the batched runs (F5, S10, P2, Q2) crop time divided by objects is the average cost at the crop\'s')
  w(f"  own object count ({min(objects)}-{max(objects)}), so a break-even far from that range is an extrapolation.")
  torch = ', '.join(p for p in pins if p.startswith('torch'))
  w(f"- The B200 run's inference environment has {torch}; the V100 run's has "
    f"{', '.join(v100_torch)} (the same version, another CUDA build).")
  return '\n'.join(L) + '\n'


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--b200', default=str(B200))
  p.add_argument('--v100', default=str(V100))
  p.add_argument('--out', default=str(PAGE))
  a = p.parse_args()
  Path(a.out).write_text(render(json.loads(EVIDENCE.read_text()), Path(a.b200), Path(a.v100)))


if __name__ == '__main__':
  main()
