"""Builds docs/unseen_evidence.json from the copied results of the unseen-crop test (docs/unseen_design.md).

  unseen_evidence.py [--run DIR]

DIR (default: the data cache's runs/unseen-20261104) holds b200/ (MIT, examples/run_unseen.py) and
v100/ (Wahab, examples/run_unseen_wahab.py) as copied from the clusters. The judged numbers come
from examples/judge_unseen.py on those files, rounded to 6 significant digits; every result file is
pinned by its sha256, and the staged source by its list of sha256s.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import unseen  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_unseen  # pylint: disable=g-import-not-at-top,wrong-import-position

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / 'docs' / 'unseen_evidence.json'
RUN_NAME = 'unseen-20261104'
JOBS = {'mit_b200': 27976430, 'wahab_ffn': 7557287, 'wahab_segneuron': 7557288,
        'smoke_mit_b200': 27934261, 'smoke_wahab_ffn': 7556576, 'smoke_wahab_segneuron': 7556577}
B200_TASKS = 50


def run_dir() -> Path:
  return data.cache_dir() / 'runs' / RUN_NAME


def sig(x):
  """Numbers rounded to 6 significant digits, recursively; other values unchanged."""
  if isinstance(x, bool) or x is None or isinstance(x, (int, str)):
    return x
  if isinstance(x, float):
    return float(f'{x:.6g}')
  if isinstance(x, dict):
    return {str(k): sig(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [sig(v) for v in x]
  return x


def result_files(d: Path) -> list:
  """The judged result files, relative to d: crop-K/RUN.json (B200) and crop-K/{ffn,segneuron}.json."""
  out = []
  for k in unseen.CROPS:
    out += [f'b200/crop-{k:02d}/{m}.json' for m in unseen.B200_RUNS]
    out += [f'v100/crop-{k:02d}/{n}.json' for n in ('ffn', 'segneuron')]
  return out


def staged_sources(d: Path) -> dict:
  """The source list of every MIT task and of the Wahab run, which must be one and the same."""
  lists = {p.name: p.read_bytes() for p in sorted((d / 'b200').glob(f"source-{JOBS['mit_b200']}_*.txt"))}
  if set(lists) != {f"source-{JOBS['mit_b200']}_{i}.txt" for i in range(B200_TASKS)}:
    raise ValueError(f'MIT source lists {sorted(lists)} are not those of tasks 0-{B200_TASKS - 1}')
  lists['v100/source_sha256.txt'] = (d / 'v100' / 'source_sha256.txt').read_bytes()
  return lists


def source_map(source: str) -> dict:
  """path -> sha256 from a staged source list ('<sha256>  <path>' per line, as shasum writes it)."""
  return {path: sha for sha, path in (line.split('  ', 1) for line in source.splitlines())}


def source_list(mapping: dict) -> str:
  """The staged source list again, from the mapping (inverse of source_map)."""
  return ''.join(f'{sha}  {path}\n' for path, sha in sorted(mapping.items()))  # the list is sorted by path


def comparison(c: dict) -> dict:
  keep = ('crops', 'per_crop', 'mean', 't_interval', 'bootstrap_interval', 'crops_a_higher', 'crops_b_higher',
          'verdict', 'non_inferior', 'worse_than_ffn', 'better_than_ffn', 'difference_claimed', 'objects')
  return {k: c[k] for k in keep if k in c}


def judged(r: dict) -> dict:
  q1 = r['q1_fast_mode']
  speed_v100 = {k: v for k, v in r['speed_v100'].items() if k != 'per_crop'}
  speed_v100['per_crop_break_even'] = {k: v['break_even_objects'] for k, v in r['speed_v100']['per_crop'].items()}
  return {
      'failures': r['failures'],
      'objects_per_crop': r['objects_per_crop'],
      'object_ids_in_more_than_one_crop': r['object_ids_in_more_than_one_crop'],
      'q1_fast_mode': {k: q1[k] for k in ('limit_objects', 'counts', 'rules', 'passes', 'per_crop', 'by_class',
                                           'reference_already_merged')},
      'q2_ffn_vs_segneuron': {k: comparison(v) if isinstance(v, dict) and 'mean' in v else v
                              for k, v in r['q2_ffn_vs_segneuron'].items()},
      'q3_smaller_ffns': {k: comparison(v) for k, v in r['q3_smaller_ffns'].items()},
      'q3_p8_minus_student': comparison(r['q3_p8_minus_student']),
      'crop_mean_f1': r['crop_mean_f1'],
      'pooled': r['pooled'],
      'counts': r['counts'],
      'runaways': {k: {'runaway_traces': v['runaway_traces'],
                       'reference_stopped_at_first_step': v['reference_stopped_at_first_step'],
                       'of_them_counted_as_runaways': [i for i in v['reference_stopped_at_first_step']
                                                       if i in v['object_ids']]}
                   for k, v in r['runaways'].items()},
      'speed_b200': {m: {k: v[k] for k in ('crop_seconds', 'total_seconds', 'mean_crop_seconds', 'seconds_per_object', 'speedup_vs_F1')}
                     for m, v in r['speed_b200'].items()},
      'speed_v100': speed_v100,
      'ffn_v100_overlap': r['ffn_v100_overlap'],
      'nonfinite_outputs': r['nonfinite_outputs'],
      'devices': r['devices'],
  }


def v100_devices(d: Path) -> dict:
  """GPUs (nvidia-smi, written by cluster/wahab/unseen.sbatch) and nodes (first log line) of the
  20 Wahab jobs."""
  gpus = set()
  for p in (d / 'v100').glob('crop-*/gpu-*.txt'):
    gpus |= {l for l in p.read_text().splitlines() if l and not l.startswith(('name,', 'cpus'))}
  hosts = {}
  for job in (JOBS['wahab_ffn'], JOBS['wahab_segneuron']):
    for p in sorted((d / 'v100').glob(f'{job}_*.out')):
      hosts[p.stem] = p.read_text().splitlines()[0].rsplit(' on ', 1)[1]
  if len(hosts) != 2 * len(unseen.CROPS):
    raise ValueError(f'{len(hosts)} Wahab job logs, expected {2 * len(unseen.CROPS)}')
  return {'gpus': sorted(gpus), 'jobs': hosts}


def judge_output_sha256(r: dict) -> str:
  """sha256 of the judge's output as canonical JSON (sorted keys), as examples/judge_unseen.py
  computes it from the result files."""
  return hashlib.sha256(json.dumps(r, sort_keys=True, default=str).encode()).hexdigest()


def build(d: Path) -> dict:
  r = judge_unseen.judge(d / 'b200', d / 'v100', unseen.load_manifest(), unseen.design_sha256())
  lists = staged_sources(d)
  if len(lists) != B200_TASKS + 1 or len(set(lists.values())) != 1:
    raise ValueError('the MIT tasks and the Wahab run did not all record the same staged source')
  source = next(iter(lists.values())).decode()
  return {
      'what': ('The pre-registered test on the unseen crops 41-50 (docs/unseen_design.md), judged by '
               'examples/judge_unseen.py from the copied result files. Derived numbers are rounded to 6 '
               'significant digits; the result files keep exact values.'),
      'run': RUN_NAME,
      'jobs': JOBS,
      'design_sha256': unseen.design_sha256(),
      'manifest_sha256': data.sha256_file(unseen.MANIFEST),
      'source_list_sha256': hashlib.sha256(source.encode()).hexdigest(),
      'source_files': len(source.splitlines()),
      'source_sha256': source_map(source),
      'all_tasks_same_source': True,
      'source_note': ('Each of the 50 MIT tasks recorded its own source list (cluster/mit/unseen.sbatch checks it '
                      'before running). On Wahab the list was written once when the source was synced into the '
                      'repository clone, before both jobs were submitted, and is one snapshot for all 20 tasks; '
                      'the clone was not changed while they ran.'),
      'result_files_sha256': {f: data.sha256_file(d / f) for f in result_files(d)},
      'judge_output_sha256': judge_output_sha256(r),
      'v100_devices': v100_devices(d),
      'judged': sig(judged(r)),
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run', type=Path, default=None)
  a = ap.parse_args()
  EVIDENCE.write_text(json.dumps(build(a.run or run_dir()), indent=1) + '\n')
  print(EVIDENCE)


if __name__ == '__main__':
  main()
