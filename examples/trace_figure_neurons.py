"""Re-run the drawn test-crop objects of one crop with every method, verified (docs/figure_neurons_design.md).

  trace_figure_neurons.py --run-dir DIR --test-dir DIR --sn-dir DIR --crop K --out DIR

One B200. DIR (--run-dir) is the MIT run folder distil-20261031; --test-dir the test's B200 result folder
(unseen-20261104 on MIT, b200/ in the data cache; crop-K/NAME.json); --sn-dir SegNeuron's B200 run (unseen-b200-segneuron-20261106,
crop-K/infer/ and crop-K/segneuron.json). The drawn objects come from liconn_v2.figure_neurons.draw over
the test's F1 object lists. Each tracer is built and run as in examples/run_unseen.py (F1 on the drawn
objects one at a time; F5, S10, P2 and Q2 on the whole crop, 32 at once); SegNeuron's segmentation is
rebuilt as in examples/run_unseen_wahab.py. Each drawn object is checked against the test's record before
its part volume is written to OUT/crop-K/OBJECT_METHOD.npz; OUT/crop-K/figure.json lists what was written
and every verification failure.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # pylint: disable=g-import-not-at-top
import torch  # pylint: disable=g-import-not-at-top

from liconn_v2 import benchmark  # pylint: disable=g-import-not-at-top
from liconn_v2 import distil  # pylint: disable=g-import-not-at-top
from liconn_v2 import factorised as fz  # pylint: disable=g-import-not-at-top
from liconn_v2 import figure_neurons as fn  # pylint: disable=g-import-not-at-top
from liconn_v2 import reference  # pylint: disable=g-import-not-at-top
from liconn_v2 import segneuron  # pylint: disable=g-import-not-at-top
from liconn_v2 import unseen  # pylint: disable=g-import-not-at-top
from liconn_v2 import vitseg  # pylint: disable=g-import-not-at-top

sys.path.insert(0, str(Path(__file__).resolve().parent))
import distil_teacher  # pylint: disable=g-import-not-at-top,wrong-import-position
import run_segneuron  # pylint: disable=g-import-not-at-top,wrong-import-position
import run_unseen as ru  # pylint: disable=g-import-not-at-top,wrong-import-position
import time_crop11 as tc  # pylint: disable=g-import-not-at-top,wrong-import-position
import train_distil  # pylint: disable=g-import-not-at-top,wrong-import-position

WIDTH = 32


def network(name, teacher, image, run_dir, dev):
  """The network of run `name`, built as in examples/run_unseen.py."""
  ru.check_run(name)
  record = {}
  teacher = copy.deepcopy(teacher)  # each run its own copy, as each test run was its own process
  if name == 'S10':
    ckpt = torch.load(ru.checked(run_dir, 'student', record), map_location='cpu', weights_only=True)
    student = fz.student_from_teacher(teacher, dilations=tuple(ckpt['dilations']))
    student.decoder.load_state_dict(ckpt['decoder'])
    encoder = student.encoder.to(dev).eval()
    decoder = student.decoder.to(dev).eval()
    with torch.no_grad():
      encoder(torch.zeros((1, 1, 64, 64, 64), device=dev))
      features = fz.encode_volume(encoder, image, tile=train_distil.TILE, dtype=torch.float16, out_device=dev)
    if sum(int((~torch.isfinite(c)).sum()) for c in features):
      raise ValueError('non-finite encoder features')
    return distil.StudentNet(decoder, features, dtype=torch.float16, fmt='fp16_head32', scale=tc.SCALE[name])
  kind = tc.RUNS[name][0]
  if kind in tc.PRUNED:
    c = torch.load(ru.checked(run_dir, kind, record), map_location='cpu', weights_only=True)
    module = distil.load_checkpoint(teacher, c).to(dev)
  else:
    module = teacher.to(dev).eval()
  return distil.TeacherNet(module, image, dev, fmt=tc.RUNS[name][1], scale=tc.SCALE.get(name, 1.0))


def traces(name, net, shape, seeds, drawn, dev) -> dict:
  """object_id -> trace for the drawn objects, traced as the test traced them."""
  _, _, width, box_only = tc.RUNS[name]
  if width == 1:
    tracer = distil.TorchTracer(net, shape, dev)
    tracer.trace(seeds[0]['seed_zyx'], box_only=box_only)  # warm-up, as in the test
    return {s['object_id']: tracer.trace(s['seed_zyx'], box_only=box_only) for s in seeds if s['object_id'] in drawn}
  if width != WIDTH:
    raise ValueError(f'{name}: width {width}')
  tracer = distil.BatchTracer(net, shape, dev, width)
  tracer.trace_all([s['seed_zyx'] for s in seeds[:width]], box_only=box_only)  # warm-up, as in the test
  out = tracer.trace_all([s['seed_zyx'] for s in seeds], box_only=box_only)
  return {s['object_id']: t for s, t in zip(seeds, out) if s['object_id'] in drawn}


def save(folder: Path, object_id: int, method: str, pane: tuple, stats: dict) -> str:
  lo, vol = pane
  name = f'{object_id}_{method}.npz'
  np.savez_compressed(folder / name, lo=np.array(lo), parts=vol)
  stats['file'] = name
  return name


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--run-dir', type=Path, required=True)
  ap.add_argument('--test-dir', type=Path, required=True)
  ap.add_argument('--sn-dir', type=Path, required=True)
  ap.add_argument('--crop', type=int, required=True)
  ap.add_argument('--out', type=Path, required=True)
  ap.add_argument('--methods', nargs='+', choices=fn.METHODS, default=list(fn.METHODS),
                  help='a subset, each in its own process as in the test; merged into figure.json')
  args = ap.parse_args()
  k = args.crop
  ids = {c: json.loads((args.test_dir / f'crop-{c:02d}' / 'F1.json').read_text())['object_ids'] for c in unseen.CROPS}
  drawn = [o for c, o in fn.draw(ids) if c == k]
  if not drawn:
    raise ValueError(f'no drawn object on crop {k}')
  raw, gt = ru.load_crop(k, None)  # crops 41-50 only, sha256 checked
  seeds = vitseg.tuning_objects(gt)
  if [s['object_id'] for s in seeds] != ids[k]:
    raise ValueError(f"crop {k}: objects differ from the test's")
  folder = args.out / f'crop-{k:02d}'
  folder.mkdir(parents=True, exist_ok=True)
  report = start_report(folder, k, drawn, args.methods)
  for o in drawn:
    report['panes'][str(o)]['label'] = {}
    save(folder, o, 'label', fn.label_pane(gt, o), report['panes'][str(o)]['label'])

  distil.device_settings(tf32=False)
  dev = torch.device('cuda')
  teacher, manifest = distil_teacher.load_export(args.run_dir / 'export')
  if manifest['teacher.pt'] != unseen.TEACHER_SHA256:
    raise ValueError(f"teacher export {manifest['teacher.pt']} is not the test's")
  image = reference.normalize(raw)
  by_id = {s['object_id']: s for s in seeds}
  for name in [m for m in fn.TRACERS if m in args.methods]:
    test = {r['object_id']: r for r in json.loads((args.test_dir / f'crop-{k:02d}' / f'{name}.json').read_text())['run']['records']}
    net = network(name, teacher, image, args.run_dir, dev)
    for o, t in traces(name, net, image.shape, seeds, set(drawn), dev).items():
      mine = tc.record(t, by_id[o], gt, None)
      try:
        fn.check_trace(mine, test[o])
      except ValueError as e:
        report['failures'].append({'method': name, 'object_id': o, 'error': str(e)})
        continue
      pane = fn.parts(tc.mask_of(t, gt.shape), gt, o)
      if fn.mask_sha256(fn.reconstruction(*pane, gt.shape)) != test[o]['mask_sha256']:  # the saved volume itself
        report['failures'].append({'method': name, 'object_id': o, 'error': 'part volume does not encode the mask'})
        continue
      stats = {key: test[o][key] for key in ('f1', 'recall', 'precision', 'merges', 'steps')}
      save(folder, o, name, pane, stats)
      report['panes'][str(o)][name] = stats
    write(folder, report)  # each finished method is kept if a later one fails
    del net
    torch.cuda.empty_cache()
    print(name, 'done', flush=True)

  if 'segneuron' not in args.methods:
    finish(folder, report)
    return
  aff, fg = run_segneuron.load_outputs(args.sn_dir / f'crop-{k:02d}' / 'infer')
  a = segneuron.vitseg_layout(aff, fg)
  frag = vitseg.fragments(a)
  seg = vitseg.segmentation_at(frag, vitseg.agglomerate(*vitseg.fragment_edges(frag, a)), unseen.SEGNEURON_THRESHOLD)
  recorded = json.loads((args.sn_dir / f'crop-{k:02d}' / 'segneuron.json').read_text())
  if int(frag.max()) != recorded['fragments'] or recorded['threshold'] != unseen.SEGNEURON_THRESHOLD:
    raise ValueError(f"crop {k}: {int(frag.max())} fragments, {recorded['fragments']} recorded")
  test = {r['object_id']: r for r in recorded['objects']}
  rows = segneuron.object_accounting(seg, gt, [by_id[o] for o in drawn])['objects']
  for row in rows:
    o = row['object_id']
    try:
      fn.check_segneuron(row, test[o])
    except ValueError as e:
      report['failures'].append({'method': 'segneuron', 'object_id': o, 'error': str(e)})
      continue
    z = tuple(int(v) for v in by_id[o]['seed_zyx'])
    pred = seg == seg[z] if seg[z] else np.zeros(gt.shape, bool)  # a seed in background: an empty segment
    pane = fn.parts(pred, gt, o)
    try:
      fn.check_score(benchmark.score(fn.reconstruction(*pane, gt.shape), gt, by_id[o]['seed_zyx']), test[o])
    except ValueError as e:
      report['failures'].append({'method': 'segneuron', 'object_id': o, 'error': str(e)})
      continue
    stats = {'f1': test[o]['f1'], 'recall': test[o]['recall'], 'precision': test[o]['precision'],
             'merges': len(test[o]['merges'])}
    save(folder, o, 'segneuron', pane, stats)
    report['panes'][str(o)]['segneuron'] = stats
  finish(folder, report)


def start_report(folder: Path, k: int, drawn: list, methods) -> dict:
  """The crop's report before `methods` run: an earlier process's figure.json of the same draw, with the
  panes and failures of `methods` removed (they are re-run), or a new one."""
  report = {'design': 'docs/figure_neurons_design.md', 'crop': k, 'drawn': drawn, 'panes': {}, 'failures': []}
  if (folder / 'figure.json').exists():  # an earlier process of this crop: keep its other methods
    report = json.loads((folder / 'figure.json').read_text())
    if report['crop'] != k or report['drawn'] != drawn:
      raise ValueError(f"crop {k}: figure.json is crop {report['crop']} drawing {report['drawn']}, not {drawn}")
    report['failures'] = [f for f in report['failures'] if f['method'] not in methods]
  for o in drawn:
    panes = report['panes'].setdefault(str(o), {})
    for m in methods:
      panes.pop(m, None)
  return report


def write(folder: Path, report: dict) -> None:
  tmp = folder / 'figure.json.tmp'
  tmp.write_text(json.dumps(report, indent=1) + '\n')
  tmp.replace(folder / 'figure.json')


def finish(folder: Path, report: dict) -> None:
  write(folder, report)
  print(json.dumps({'crop': report['crop'], 'drawn': report['drawn'], 'panes': {o: sorted(p) for o, p in report['panes'].items()},
                    'failures': report['failures']}), flush=True)


if __name__ == '__main__':
  main()
