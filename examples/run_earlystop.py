"""Early-stop runs (docs/earlystop_design.md): traces with a move-decision recorder.

  run_earlystop.py --variant fp32|fp16 --manyseed-run DIR --reference-run DIR
                   [--open-loop] [--shift C] [--verdict FILE] [--seed NAME ...] [--out DIR]

Traces the 10 early-stop seeds and their 10 controls (or the named seeds) with the
authors' network in the given format and records, at every step, the six face maxima
the movement policy sees. Each trace must reproduce the many-seed run in DIR (one
crop-XX/benchmark.json per crop) bitwise. Controls are picked from the float32 many-seed run (--reference-run),
whatever the variant. With --open-loop (float32 only), the network
input of every step is also evaluated in fp16, fp16_head32 and fp16_res32, and the time
per evaluation of each is measured; --shift C instead evaluates fp16_shift (fp16 plus C).

Output: DIR/earlystop_run.json and DIR/steps/<seed>.json
"""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top

from liconn_v2 import data  # pylint: disable=g-import-not-at-top
from liconn_v2 import earlystop  # pylint: disable=g-import-not-at-top
from liconn_v2 import manyseed  # pylint: disable=g-import-not-at-top
from liconn_v2 import precision  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_policy_benchmark import provenance  # pylint: disable=g-import-not-at-top,wrong-import-position


def manyseed_records(run_dir: Path) -> dict:
  out = {}
  for f in sorted(run_dir.glob('crop-*/benchmark.json')):
    for r in json.loads(f.read_text())['records']:
      out[r['name']] = r
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--variant', required=True, choices=('fp32', 'fp16'))
  p.add_argument('--manyseed-run', required=True, help='many-seed run of this variant (for the bitwise check)')
  p.add_argument('--reference-run', required=True, help='float32 many-seed run (for picking controls)')
  p.add_argument('--open-loop', action='store_true')
  p.add_argument('--shift', type=float)
  p.add_argument('--verdict', default=None, help='many-seed verdict json (to pick controls)')
  p.add_argument('--seed', action='append', default=[])
  p.add_argument('--out', default='outputs')
  a = p.parse_args()
  if (a.open_loop or a.shift is not None) and a.variant != 'fp32':
    raise SystemExit('open-loop evaluation runs along float32 traces only')
  out = Path(a.out)
  (out / 'steps').mkdir(parents=True, exist_ok=True)

  seeds = {s['name']: s for s in manyseed.recorded_seeds()}
  recorded = manyseed_records(Path(a.manyseed_run))
  if a.seed:
    names = a.seed
  else:
    verdict = json.loads(Path(a.verdict).read_text())
    ref = {n: {**r, 'crop': seeds[n]['crop'], 'class': seeds[n]['class']}
           for n, r in manyseed_records(Path(a.reference_run)).items()}
    names = list(earlystop.EARLY_STOP_SEEDS) + earlystop.select_controls(verdict, ref)

  started = time.time()
  records, timing = [], {}
  with reference.Tracer(data.checkpoint_path()) as tracer:
    precision.install(tracer, a.variant)
    open_loop = {}
    if a.open_loop:
      open_loop = {v: earlystop.apply_fn(tracer.params, v) for v in earlystop.CANDIDATES}
    if a.shift is not None:
      open_loop = {'fp16_shift': earlystop.shifted(earlystop.apply_fn(tracer.params, 'fp16'), a.shift)}
    crop_cache = {}
    last_input = None
    for name in names:
      s = seeds[name]
      if s['crop'] not in crop_cache:
        crop_cache = {s['crop']: reference.normalize(manyseed.load_raw_xyz(s['crop']))}
      raw = crop_cache[s['crop']]
      rec = earlystop.Recorder(tracer, open_loop)
      try:
        trace = tracer.trace(raw, tuple(s['seed_zyx']), instrument=rec)
      finally:
        rec.close()
      last_input = rec._last_input  # pylint: disable=protected-access
      if len(rec.steps) != trace.iterations:
        raise RuntimeError(f'{name}: {len(rec.steps)} recorded steps for {trace.iterations} iterations')
      want = recorded[name]
      r = {'name': name, 'class': s['class'], 'crop': s['crop'], 'variant': a.variant,
           'steps': len(rec.steps), 'iterations': trace.iterations,
           'canvas_sha256': reference.canvas_sha256(trace.seed),
           'history_sha256': reference.history_sha256(trace.history)}
      r['reproduces_manyseed'] = (r['canvas_sha256'] == want['canvas_sha256']
                                  and r['history_sha256'] == want['history_sha256']
                                  and r['iterations'] == want['steps'])
      if open_loop:
        r['emulation_mismatches'] = sum(1 for st in rec.steps if not st.get('emulation_matches', False))
      earlystop.save_steps(out / 'steps' / f'{name}.json', rec.steps)
      records.append(r)
      print(json.dumps(r), flush=True)
    if open_loop and last_input is not None:
      timing['fp32'] = precision.time_per_eval(tracer._exc._apply_fn, np.asarray(last_input))  # pylint: disable=protected-access
      for v, f in open_loop.items():
        timing[v] = precision.time_per_eval(f, np.asarray(last_input))
  result = {'design': 'docs/earlystop_design.md', 'variant': a.variant, 'open_loop': sorted(open_loop),
            'shift': a.shift, 'seeds': names, 'records': records,
            'seconds_per_eval': timing, 'wall_seconds_total': time.time() - started, 'platform': provenance()}
  (out / 'earlystop_run.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
  main()
