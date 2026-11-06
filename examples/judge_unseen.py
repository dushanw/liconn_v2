"""The pre-stated analysis of the unseen crops 41-50 from the result files (docs/unseen_design.md).

  judge_unseen.py --b200 DIR --v100 DIR --out FILE

--b200 holds crop-K/RUN.json for K in 41-50 and each B200 run (examples/run_unseen.py); --v100
holds crop-K/ffn.json and crop-K/segneuron.json (examples/run_unseen_wahab.py). Besides the result files, the judge reads only
the pinned crop manifest (docs/unseen_crops.json), the design (its sha256) and the pinned proofread
classes (liconn_v2/proofread_classes.json, sha256 checked), for the class split of question 1.

Refused: smoke runs, results recorded under another design, other crop files or checkpoints, runs
recorded with settings other than the design's (network, format, objects at once, box_only, stream
scale, TF32 off), and object lists that differ between files or from a file's own header. A run that is missing (its own
task stopped), marked failed, or has non-finite network outputs is a failure of that method on that
crop, as the design says; it is passed to the analysis as such.
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import headtohead as hh  # pylint: disable=g-import-not-at-top
from liconn_v2 import unseen  # pylint: disable=g-import-not-at-top


def check_header(r: dict, k: int, design_sha256: str, crop_sha256: dict, where: str) -> None:
  if r['crop'] != k or r['smoke_objects'] is not None:
    raise ValueError(f'{where}: not a full run of crop {k}')
  if r['design_sha256'] != design_sha256:
    raise ValueError(f'{where}: recorded under design {r["design_sha256"]}, not {design_sha256}')
  if r['crop_sha256'] != crop_sha256:
    raise ValueError(f'{where}: crop files {r["crop_sha256"]} are not the pinned ones {crop_sha256}')
  if r['object_count'] != len(r['object_ids']):
    raise ValueError(f'{where}: {r["object_count"]} objects but {len(r["object_ids"])} IDs')


def check_records(records: list, ids: list, where: str) -> list:
  if [x['object_id'] for x in records] != ids:
    raise ValueError(f'{where}: the records are not every object of the header, in order')
  return records


def check_settings(summary: dict, m: str, where: str) -> None:
  """The run's recorded network, format, objects at once, box_only and stream scale are the design's."""
  got = (summary['network'], summary['format'], summary['width'], summary['box_only'], summary['stream_scale'])
  if [(type(x), x) for x in got] != [(type(x), x) for x in unseen.RUN_SETTINGS[m]]:  # True is not 1
    raise ValueError(f'{where}: settings {got}, the design fixes {unseen.RUN_SETTINGS[m]}')


def frozen_checkpoints() -> dict:
  return {n: {'path': rel, 'sha256': sha} for n, (rel, sha) in unseen.CHECKPOINTS.items()}


def load(b200: Path, v100: Path, design_sha256: str, manifest: dict):
  """(results: method -> crop -> records, failures: method -> crop -> reason, raw files)."""
  results = {m: {} for m in unseen.METHODS}
  failures = {m: {} for m in unseen.METHODS}
  files = {'b200': {}, 'ffn_v100': {}, 'segneuron': {}}
  for k in unseen.CROPS:
    crop_sha = unseen._entry(k, manifest)['sha256']  # pylint: disable=protected-access
    ids = None
    for m in unseen.B200_RUNS:
      path = b200 / f'crop-{k:02d}' / f'{m}.json'
      if not path.is_file():
        failures[m][k] = 'no B200 result file: the task stopped before the run ended'
        continue
      r = json.loads(path.read_text())
      where = f'B200 crop {k} {m}'
      check_header(r, k, design_sha256, crop_sha, where)
      if r['run_name'] != m or r['tf32'] is not False:
        raise ValueError(f'{where}: run {r["run_name"]} with tf32 {r["tf32"]}')
      if r['teacher_sha256'] != unseen.TEACHER_SHA256:
        raise ValueError(f'{where}: teacher {r["teacher_sha256"]}')
      for n, c in r['checkpoints'].items():
        if c != frozen_checkpoints()[n]:
          raise ValueError(f'{where}: checkpoint {n} {c} is not the design\'s')
      if m in unseen.RUN_CHECKPOINT and unseen.RUN_CHECKPOINT[m] not in r['checkpoints']:
        raise ValueError(f'{where}: no recorded {unseen.RUN_CHECKPOINT[m]} checkpoint')
      if ids is not None and r['object_ids'] != ids:
        raise ValueError(f'{where}: objects differ from the other B200 runs')
      ids = r['object_ids']
      files['b200'].setdefault(k, {})[m] = r
      run = r['run']
      if 'failed' in run:
        failures[m][k] = run['failed']
        continue
      check_settings(run['summary'], m, where)
      if m in unseen.COUNTED_NONFINITE and not isinstance(run['summary'].get('nonfinite_outputs'), int):
        raise ValueError(f'{where}: no non-finite output count')
      if run['summary'].get('nonfinite_outputs'):
        failures[m][k] = f"{run['summary']['nonfinite_outputs']} non-finite network outputs"
      else:
        if run['summary']['objects'] != len(ids):
          raise ValueError(f'{where}: {run["summary"]["objects"]} objects, header {len(ids)}')
        results[m][k] = check_records(run['records'], ids, where)
    for name, method, extra in (('ffn.json', 'V', 'ffn_v100'), ('segneuron.json', 'segneuron', 'segneuron')):
      path = v100 / f'crop-{k:02d}' / name
      if not path.is_file():
        failures[method][k] = f'no V100 result file {name}'
        continue
      x = json.loads(path.read_text())
      check_header(x, k, design_sha256, crop_sha, f'{name} crop {k}')
      if ids is not None and x['object_ids'] != ids:
        raise ValueError(f'{name} crop {k}: objects differ from the B200 run')
      ids = x['object_ids'] if ids is None else ids
      if method == 'V':
        if x['variant'] != unseen.V100_FFN_VARIANT:
          raise ValueError(f'crop {k}: V100 variant {x["variant"]}')
        if x['checkpoint_sha256'] != {f: data.PINNED_SHA256[f] for f in data.CHECKPOINT_FILES}:
          raise ValueError(f'crop {k}: V100 FFN checkpoint {x["checkpoint_sha256"]}')
        results[method][k] = check_records(x['records'], ids, f'{name} crop {k}')
      else:
        if x['threshold'] != unseen.SEGNEURON_THRESHOLD or x['checkpoint_sha256'] != unseen.SEGNEURON_CKPT_SHA256:
          raise ValueError(f'crop {k}: SegNeuron threshold {x["threshold"]} or checkpoint is not the design\'s')
        if x['nonfinite_outputs']:
          failures[method][k] = f"{x['nonfinite_outputs']} non-finite SegNeuron outputs"
          continue
        results[method][k] = check_records(x['objects'], ids, f'{name} crop {k}')
      files[extra][k] = x
  return results, {m: v for m, v in failures.items() if v}, files


def b200_speed(files: dict, failures: dict) -> dict:
  out = {}
  for name in unseen.B200_RUNS:
    if name in failures:
      out[name] = {'not_judged': failures[name]}
      continue
    s = [files['b200'][k][name]['run']['summary'] for k in unseen.CROPS]
    secs = [x['crop_seconds'] for x in s]
    out[name] = {'crop_seconds': dict(zip(unseen.CROPS, secs)), 'total_seconds': float(sum(secs)),
                 'mean_crop_seconds': float(np.mean(secs)),
                 'seconds_per_object': float(sum(secs) / sum(x['objects'] for x in s))}
  if 'total_seconds' in out[unseen.REFERENCE]:
    for name in unseen.B200_RUNS:
      if 'total_seconds' in out[name]:
        out[name]['speedup_vs_F1'] = out[unseen.REFERENCE]['total_seconds'] / out[name]['total_seconds']
  return out


def v100_speed(files: dict, failures: dict) -> dict:
  """Run V against SegNeuron on the V100 (headtohead.speed), per crop and from the ten crops' totals.
  End to end is the primary break-even count, as in the text."""
  if 'V' in failures or 'segneuron' in failures:
    return {'not_judged': {m: failures[m] for m in ('V', 'segneuron') if m in failures}}
  f, sn = files['ffn_v100'], files['segneuron']
  per_crop = {k: hh.speed(f[k]['records'], f[k]['setup_seconds'], sn[k]['inference_wall_seconds'],
                          sn[k]['inference']['inference_seconds'], sn[k]['post_seconds']) for k in unseen.CROPS}
  out = {'per_crop': per_crop}
  for way in ('end_to_end', 'gpu_only'):
    v = [per_crop[k]['break_even_objects'][way] for k in unseen.CROPS]
    out[f'break_even_{way}'] = {'median': float(np.median(v)), 'min': float(min(v)), 'max': float(max(v))}
  n = sum(per_crop[k]['objects'] for k in unseen.CROPS)
  wall = sum(per_crop[k]['ffn']['wall_seconds'] for k in unseen.CROPS)
  trace = sum(per_crop[k]['ffn']['trace_seconds'] for k in unseen.CROPS)
  sn_total = sum(per_crop[k]['segneuron']['total_seconds'] for k in unseen.CROPS)
  sn_gpu = sum(per_crop[k]['segneuron']['inference_seconds'] for k in unseen.CROPS)
  out['totals'] = {'objects': n, 'ffn_wall_seconds': wall, 'ffn_trace_seconds': trace,
                   'segneuron_total_seconds': sn_total, 'segneuron_inference_seconds': sn_gpu,
                   'break_even_end_to_end_per_crop': (sn_total / len(unseen.CROPS)) / (wall / n),
                   'break_even_gpu_only_per_crop': (sn_gpu / len(unseen.CROPS)) / (trace / n)}
  return out


def nonfinite(files: dict) -> dict:
  return {k: {m: r['run']['summary'].get('nonfinite_outputs') for m, r in f.items() if 'summary' in r['run']}
          for k, f in files['b200'].items()}


def classes() -> dict:
  actual = data.sha256_file(benchmark.CLASSES_FILE)
  if actual != unseen.CLASSES_SHA256:
    raise ValueError(f'{benchmark.CLASSES_FILE}: sha256 {actual}, expected {unseen.CLASSES_SHA256}')
  return unseen.class_of(benchmark.load_classes())


def judge(b200: Path, v100: Path, manifest: dict, design_sha256: str) -> dict:
  if manifest['design_sha256'] != design_sha256:
    raise ValueError('the design changed after the crops were fetched')
  results, failures, files = load(b200, v100, design_sha256, manifest)
  return {**unseen.analysis(results, failures, classes()), 'design_sha256': design_sha256,
          'classes_sha256': unseen.CLASSES_SHA256,
          'speed_b200': b200_speed(files, failures), 'speed_v100': v100_speed(files, failures),
          'ffn_v100_overlap': {k: f['overlap'] for k, f in files['ffn_v100'].items()},
          'nonfinite_outputs': nonfinite(files),
          'devices': {'b200': sorted({r['device'] for f in files['b200'].values() for r in f.values()}),
                      'b200_hosts': sorted({r['host'] for f in files['b200'].values() for r in f.values()})}}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--b200', type=Path, required=True)
  ap.add_argument('--v100', type=Path, required=True)
  ap.add_argument('--out', type=Path, required=True)
  a = ap.parse_args()
  result = judge(a.b200, a.v100, unseen.load_manifest(), unseen.design_sha256())
  a.out.write_text(json.dumps(result, indent=1) + '\n')
  print(json.dumps({key: result[key] for key in ('failures', 'q1_fast_mode', 'q3_smaller_ffns')}, indent=1,
                   default=str))


if __name__ == '__main__':
  main()
