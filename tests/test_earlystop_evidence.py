"""The recorded early-stop runs (docs/earlystop_results.md) match their evidence."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from liconn_v2 import data
from liconn_v2 import earlystop

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = json.loads((ROOT / 'docs' / 'earlystop_evidence.json').read_text())
RUNS = data.cache_dir() / 'runs'
FP32 = RUNS / 'earlystop-fp32open-20261022'
FP16 = RUNS / 'earlystop-fp16-20261022'


def _sha(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _key(n) -> str:
  if isinstance(n, ast.Assign):
    return ','.join(t.id for t in n.targets if isinstance(t, ast.Name)) or ast.dump(n)
  if isinstance(n, (ast.Import, ast.ImportFrom)):
    return 'import:' + ','.join(a.asname or a.name for a in n.names)
  if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
    return 'docstring'
  return getattr(n, 'name', None) or ast.dump(n)


def _definitions(path: Path) -> dict:
  return {_key(n): ast.dump(n) for n in ast.parse(path.read_text()).body}


class EarlystopEvidenceTest(unittest.TestCase):

  def test_code_that_ran_is_the_checked_code(self):
    edits = EVIDENCE['post_run_edits']
    for run in (FP32, FP16):
      staged = dict(reversed(line.split()) for line in (run / 'source_sha256.txt').read_text().splitlines())
      for rel, expected in EVIDENCE['run_source_sha256'].items():
        self.assertEqual(staged.get(rel), expected, f'{run.name}: {rel}')
    for rel, expected in EVIDENCE['run_source_sha256'].items():
      if rel not in edits:
        self.assertEqual(_sha(ROOT / rel), expected, rel)

  def test_post_run_edits_change_only_the_declared_definitions(self):
    for rel, edit in EVIDENCE['post_run_edits'].items():
      ran = FP32 / 'source' / rel
      self.assertEqual(_sha(ran), EVIDENCE['run_source_sha256'][rel], rel)
      old, new = _definitions(ran), _definitions(ROOT / rel)
      changed = sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))
      self.assertEqual(changed, sorted(edit['changed_definitions']), rel)

  def test_outputs_are_the_recorded_ones(self):
    for rel, expected in EVIDENCE['outputs_sha256'].items():
      self.assertEqual(_sha(RUNS / rel), expected, rel)

  def test_runs_reproduce_and_record_jobs(self):
    for run, job, variant in ((FP32, 'fp32open', 'fp32'), (FP16, 'fp16', 'fp16')):
      r = json.loads((run / 'earlystop_run.json').read_text())
      self.assertEqual(r['variant'], variant)
      self.assertEqual(r['platform']['slurm_job'], str(EVIDENCE['slurm_jobs'][job]))
      self.assertEqual(r['platform']['xla_flags'], '--xla_gpu_deterministic_ops=true')
      self.assertEqual(len(r['records']), 20)
      self.assertTrue(all(x['reproduces_manyseed'] for x in r['records']))
      self.assertEqual(r['seeds'][:10], list(earlystop.EARLY_STOP_SEEDS))
    r = json.loads((FP32 / 'earlystop_run.json').read_text())
    self.assertEqual(r['open_loop'], sorted(earlystop.CANDIDATES))
    self.assertTrue(all(x['emulation_mismatches'] == 0 for x in r['records']))
    self.assertEqual(r['seeds'], json.loads((FP16 / 'earlystop_run.json').read_text())['seeds'])

  def test_documented_analysis(self):
    doc = EVIDENCE['documented']
    with tempfile.TemporaryDirectory() as d:
      subprocess.run([sys.executable, str(ROOT / 'examples' / 'analyze_earlystop.py'), '--recorded', '--out', d],
                     check=True, capture_output=True)
      v = json.loads((Path(d) / 'earlystop_analysis.json').read_text())
    r4 = lambda x: None if x is None else round(x, 4)
    self.assertEqual(v['problems'], [])
    self.assertEqual(v['A']['frontier_kinds'], doc['frontier_kinds'])
    self.assertEqual(v['A']['first_difference_kinds'], doc['first_difference_kinds'])
    self.assertEqual({n: (f['kind'] if f else None) for n, f in v['A']['frontier'].items()}, doc['frontier_by_seed'])
    self.assertEqual(v['A']['supported'], doc['A_supported'])
    b = v['B']['near_threshold_controls']
    self.assertEqual({'n': b['n'], 'mean_4dp': r4(b['mean']), 'ci95_4dp': [r4(x) for x in b['ci95']]},
                     doc['B_near_controls'])
    self.assertEqual(round(v['B']['all_faces_controls']['mean'], 5), doc['B_all_faces_controls_mean_5dp'])
    self.assertEqual((v['B']['supported'], v['B']['non_finite']), (doc['B_supported'], doc['non_finite']))
    self.assertEqual({k: round(x, 2) for k, x in v['C']['speedup_per_eval'].items()}, doc['C_speedup_2dp'])
    self.assertEqual({k: r4(x['move_decision_flip_rate']) for k, x in v['C']['screens_early_stop'].items()},
                     doc['C_flip_rate_early_stop_4dp'])
    self.assertEqual({k: r4(x['move_decision_flip_rate']) for k, x in v['C']['screens_all'].items()},
                     doc['C_flip_rate_all_4dp'])
    self.assertEqual(v['C']['selected'], doc['C_selected'])
    p = v['exploratory']['plateau']
    self.assertEqual({'moving_faces': p['moving_faces'], 'on_plateau_fraction_3dp': round(p['on_plateau_fraction'], 3),
                      'argmax_shifts': p['argmax_shifts']}, doc['exploratory_plateau'])


if __name__ == '__main__':
  unittest.main()
