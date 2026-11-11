"""Report the float32 batching study (docs/fp32_batch_design.md) from its result files.

  judge_fp32_batch.py [--widths WIDTHS_JSON ...] [--first-job WIDTHS_JSON ...] [--other-node WIDTHS_JSON ...]
                      [--test DIR --unseen RUN/b200] --out FILE

WIDTHS_JSON is part 1's widths.json (examples/time_widths.py); with several, in order, a later file
replaces the runs it repeats (settings rerun in their own process; the replaced summaries are kept).
--first-job and --other-node files are reported apart from the main widths report, each run as it is.
DIR holds crop-K/F32x32.json for K in 41-50 (examples/run_unseen_fp32x32.py); RUN/b200 is the test's B200 run directory, whose F1 and F5
files must match their sha256 in docs/unseen_evidence.json. SegNeuron's B200 times come from
docs/unseen_b200_evidence.json. Per-object records stay in the result files; FILE keeps the summaries.
"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import fp32_batch  # pylint: disable=g-import-not-at-top
from liconn_v2 import unseen  # pylint: disable=g-import-not-at-top

DOCS = Path(__file__).resolve().parent.parent / 'docs'


def load_pinned(b200: Path, run: str, pins: dict) -> dict:
  out = {}
  for k in unseen.CROPS:
    path = b200 / f'crop-{k:02d}' / f'{run}.json'
    if data.sha256_file(path) != pins[f'b200/crop-{k:02d}/{run}.json']:
      raise ValueError(f"{path} is not the test's {run} result")
    out[k] = json.loads(path.read_text())
  return out


def labels(paths: list) -> list:
  """Each path relative to the folder holding all of them (with its parent folder for one file)."""
  paths = [p.resolve() for p in paths]
  base = Path(os.path.commonpath([p.parent for p in paths]))
  if len(paths) == 1:
    base = base.parent
  return [str(p.relative_to(base)) for p in paths]


def widths_section(paths: list) -> dict:
  """Merged report of part 1's widths files, in order (a later file replaces the runs it repeats)."""
  ws = [(label, json.loads(p.read_text())) for label, p in zip(labels(paths), paths)]
  merged = fp32_batch.merge_widths(ws)
  return {'files': [{'file': label, 'sha256': data.sha256_file(p), 'device': w['device'],
                     'ffn_image_copy_seconds': w['ffn_image_copy_seconds'], 'runs': list(w['runs'])}
                    for (label, w), p in zip(ws, paths)],
          'run_source': merged['source'], 'cross_job': fp32_batch.cross_job(merged['source']),
          'replaced': merged['replaced'],
          'report': fp32_batch.widths_report(merged['runs'])}


def other_section(paths: list) -> dict:
  """Report of widths files kept apart from the main one (another job or node): each checked, runs
  taken as they are, no replacement."""
  runs, files = {}, []
  for label, p in zip(labels(paths), paths):
    w = json.loads(p.read_text())
    fp32_batch.check_widths_file(w)
    if set(w['runs']) & set(runs):
      raise ValueError(f'{label}: repeats a run')
    runs.update(w['runs'])
    files.append({'file': label, 'sha256': data.sha256_file(p), 'device': w['device'], 'runs': list(w['runs'])})
  return {'files': files, 'report': fp32_batch.widths_report({n: runs[n] for n in fp32_batch.RUNS if n in runs})}


def build(widths=None, test=None, unseen_b200_dir=None, first_job=None, other_node=None) -> dict:
  """The judge's output. widths: part 1's files for the main report; first_job, other_node: files
  reported separately (the first all-settings job; a rerun on another node); test, unseen_b200_dir:
  part 2's folder and the test's B200 run folder."""
  out = {'design': 'docs/fp32_batch_design.md', 'design_sha256': fp32_batch.design_sha256()}
  if widths:
    out['widths'] = widths_section(widths)
  if first_job:
    out['first_job'] = other_section(first_job)
  if other_node:
    out['other_node'] = other_section(other_node)
  if test is not None:
    pins = json.loads((DOCS / 'unseen_evidence.json').read_text())['result_files_sha256']
    f1, f5 = load_pinned(unseen_b200_dir, 'F1', pins), load_pinned(unseen_b200_dir, 'F5', pins)
    files = {k: test / f'crop-{k:02d}' / f'{fp32_batch.TEST_RUN}.json' for k in unseen.CROPS}
    xs = {k: json.loads(p.read_text()) for k, p in files.items() if p.is_file()}
    sn = json.loads((DOCS / 'unseen_b200_evidence.json').read_text())['judged']
    out['test'] = {'files_sha256': {k: data.sha256_file(p) for k, p in files.items() if p.is_file()},
                   'hosts': {k: x['host'] for k, x in xs.items()},
                   'report': fp32_batch.test_report(xs, f1, f5, sn)}
  return json.loads(json.dumps(out))  # as written: str keys


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--widths', type=Path, nargs='+', default=None)
  ap.add_argument('--first-job', type=Path, nargs='+', default=None)
  ap.add_argument('--other-node', type=Path, nargs='+', default=None)
  ap.add_argument('--test', type=Path, default=None)
  ap.add_argument('--unseen', type=Path, default=None)
  ap.add_argument('--out', type=Path, required=True)
  args = ap.parse_args()
  out = build(args.widths, args.test, args.unseen, args.first_job, args.other_node)
  args.out.write_text(json.dumps(out, indent=1) + '\n')
  print(json.dumps(out, indent=1)[:4000])


if __name__ == '__main__':
  main()
