# SegNeuron fine-tuned on LICONN crops 0-11: design

Owner decisions 2026-10-28, after the zero-shot result (`docs/segneuron_results.md`: crop-12 F1
0.50 ours / 0.53 FRMC masked at native resolution; neuron-background AUC 0.92, same-vs-different
neuron AUC 0.67-0.69): fine-tune SegNeuron on LICONN; about one V100-day; report crop 12 first
(the test crops 13-40 are not read in this milestone); training packages in a new environment
on Wahab, leaving the zero-shot inference environment unchanged.

The question: does supervised fine-tuning on crops 0-11 teach the pretrained network the
boundaries between touching neurons, which it lacked zero-shot? Answered on crop 12 by the same
outputs-against-labels diagnostics and segmentation scores as the zero-shot run, so the two are
directly comparable.

## What is reused unchanged

From the pinned SegNeuron checkout (commit `ea6f0c96`, `Train_and_Inference/`), imported, not
copied or edited:

- the network `MNet(1, kn=(32,64,96,128,256), FMU='sub')`;
- the targets: `seg_to_affgraph(label, mknhood3d(1), pad='replicate')` for the 3 affinities and
  `label != 0` for the foreground (as in `supervised_provider.py`);
- the loss: `WeightedBCE` on the affinities plus `WeightedBCE` on the foreground, summed, mask
  all ones (`loss1 + loss2` in `supervised_train.py`);
- the optimiser: Adam, betas (0.9, 0.999), eps 0.01, weight decay 1e-6, amsgrad
  (`supervised_train.py`);
- the learning-rate schedule: their polynomial decay (`calculate_lr`, power 1.5) from a base to an
  end rate;
- the augmentations of `supervised_provider.py`: `SimpleAugment` flips/transposes; then each with
  probability 0.25: intensity, Gaussian noise (std 0.01-0.2), Gaussian blur (kernel 3-9, sigma
  0-2), cutout, elastic (control points 4x40x40, jitter 0x2x2);
- the training patch 20 x 128 x 128 (z, y, x), cut at that size and divided by 255 before the
  augmentations, as in their provider (the elastic step pads by reflection internally and crops
  back);
- inference and FRMC for evaluation (`inference.py`, `FRMC_post.py`), as in the zero-shot run.

## What differs from the authors' supervised training (disclosed)

- **Data:** crops 0-11 of the LICONN volume (raw uint8 and proofread labels, 500^3 at
  24 x 18 x 18 nm), the crops the ViT trained on. The authors' sampler is hard-wired to their 10
  EM datasets, so our script samples a crop uniformly, then a random patch in it. Their
  dataset-mixing augmentations (Fourier style mixing and spatial mixing with patches from other EM
  datasets, each with probability 0.25 in their config) need other datasets and are left out;
  their multi-z-ratio subsampling is left out because LICONN has one z spacing. Their loss mask
  `bmask` is all ones except for ignore labels on some of their datasets, so all ones here is
  their rule for data without an ignore label; the config field `if_ignore_bg` is not read by the
  code.
- **Labels renumbered per patch:** each patch's labels are renumbered 0, 1, ..., K before
  augmentation, because the authors' elastic step stores labels as uint16 and LICONN ids would
  wrap. The renumbering keeps the partition of the voxels, so the targets are unchanged.
- **Initialisation:** all weights from `SegNeuronModel.ckpt` (their `continue_train` path). Their
  `pre_train` path drops the decoder and output layers because it starts from the
  self-supervised encoder; here the whole supervised model is the starting point.
- **Learning rate:** base 1e-3 decaying to 1e-5 (power 1.5) over the whole run, no warm-up, with
  the step counted from 1 as in their loop. The authors' supervised config
  (`Train_and_Inference/config/SegNeuron.yaml`) trains from the pretrained encoder at base 0.01
  to 1e-4, decaying over the first 200,000 of 400,000 iterations and flat after. Fine-tuning a
  fully trained network uses a lower rate to keep what it has learned. This is a choice, not a
  tuned value.
- **Batch size:** 8 if it fits a 16 GB V100, else 4 (decided by the timing probe and recorded).
- **Length:** iterations chosen from a timing probe so that training takes about 20 h of the
  24 h job, leaving time for validation and saving.
- **Validation:** at step 0 (the released model), every 1000 iterations and at the end, the same
  loss on 32 fixed 20 x 128 x 128 patches of crop 12 without augmentation (positions from a fixed
  seed). The checkpoint with the lowest validation loss is the model evaluated. The authors' loop
  has no held-out validation (their README says the validation fields in the config are not
  implemented). Because the loss is dominated by voxels away from neuron-neuron contacts, each
  validation also records, as a diagnostic that does not choose anything, the fraction of
  different-neuron pairs predicted at or above 0.5 and of same-neuron pairs below 0.5.
- **Crop 12 is apart from the training crops:** at least 500 voxels (9 um) from each of crops
  0-11, so no voxel and no contact is shared (checked by `prep` and a unit test, from
  `docs/manyseed_crops.json`). A neuron can still pass through both a training crop and crop 12.
  All 13 crops are 54-60% labelled (`coverage` in that file).
- **Labels:** label 0 of the proofread segmentation is treated as background, as in the zero-shot
  scoring and the ViT. Unproofread neurites inside label 0 therefore count as background in
  training, so fine-tuning may shrink the predicted foreground; recall lost to background and
  seeds in predicted background are reported to show it. The authors' EM labels are denser.

## Evaluation (crop 12 only)

The chosen checkpoint goes through the zero-shot pipeline at native resolution, unchanged:
`inference.py`; `FRMC_post.py` at the same 7 betas; our pipeline with the threshold chosen on crop
12; the same per-object accounting and threshold-free output scores (`segneuron.output_scores`,
`segneuron.ours`, `segneuron.object_accounting`). Crop 12 now chooses the checkpoint, the
threshold and beta, so the numbers are optimistic, as before, and more so than zero-shot, where
crop 12 chose only the threshold and beta: here it also chooses the checkpoint. The step-0
validation loss and the whole validation curve are reported. Reported next to zero-shot native
and the ViT. No pass/fail verdict; the test crops are a later owner decision.

Fine-tuning against `label != 0` also moves the foreground output towards the labelled fraction,
which alone changes F1 at the fixed foreground cut 0.5. The answer to the question above is
therefore read first from the threshold-free same-vs-different-neuron AUCs and the share of recall
lost to other segments versus background, and only then from F1.

## Code

New files only, so the recorded zero-shot run stays checkable (its evidence test hashes
`segneuron.py`, `run_segneuron.py` and `segneuron.sbatch`):

- `liconn_v2/segneuron_ft.py`: numpy-only parts, testable without torch: crop loading for
  training, patch sampling, fixed validation positions, learning-rate schedule, run summary.
- `examples/train_segneuron.py`: the torch training loop; imports the authors' modules from the
  pinned checkout.
- `examples/run_segneuron_ft.py`: prep of the training crops, and scoring of the fine-tuned
  outputs with the zero-shot functions.
- `cluster/wahab/segneuron_ft.sbatch`: probe, train, infer and post/score modes.
- `tests/test_segneuron_ft.py`; afterwards results, evidence and an evidence test.

## Decision after this milestone (owner)

Whether to spend the test crops on the fine-tuned model, train longer, or stop.
