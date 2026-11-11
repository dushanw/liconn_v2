# SegNeuron on the B200, crops 41-50: timing addendum (owner-approved 2026-11-06)

Nothing in this plan has been run. It is an addendum to the unseen-crop test (docs/unseen_design.md,
docs/unseen_results.md), written after that test's results were known.

## Why

In the test, SegNeuron ran only on the V100 (its environment existed only on Wahab), so its cost was
compared only with FFN traced one object at a time on the same V100 (break-even 181 objects per crop
end to end). Batched fast FFN (run F5) and the smaller FFNs ran on the B200, where SegNeuron was not
timed, so the cost of the dense method against the fastest FFN setting is unknown. On 2026-11-06 the
owner asked for both methods to be timed on the same GPU. This addendum times SegNeuron on the B200
and compares it with the B200 runs of the test.

## Question

On the B200, on crops 41-50, is the fine-tuned SegNeuron cheaper per crop than batched fast FFN (F5),
than float32 FFN one object at a time (F1), and than the three smaller FFNs (S10, P2, Q2)?

## What runs

- Fine-tuned SegNeuron, exactly as run V100 SegNeuron of the test: the authors' `inference.py`
  (github.com/yanchaoz/SegNeuron, commit `ea6f0c963a2f60f0f130244da27e2b2d753f8ee9`, unmodified, clean
  tree checked by every job) with the fine-tuned `best.ckpt` (sha256 `e726bb43...f228`, checked by
  every job), then our post-processing at the fixed threshold 0.6, through the test's own script
  (`examples/run_unseen_wahab.py`, steps `prep` and `segneuron`, unchanged), so the inputs, outputs,
  scoring and recorded times are those of the V100 run.
- One B200 per crop (MIT `mit_normal_gpu`, `--gres=gpu:b200:1`, 8 CPUs, 160 GB, 2 h), as the test's
  B200 runs; one job per crop (an array of ten), each writing crop-K/segneuron.json.
- Inference environment: a separate virtual environment with the V100 run's versions (numpy 2.5.4,
  tifffile 2026.10.15, torch 2.7.1), except that torch is the CUDA 12.8 build (`+cu128`, which
  supports the B200) instead of `+cu126`. Its package list is recorded by every job.
- Post-processing (watershed, agglomeration, segmentation, scoring) runs on the node's CPUs, as it
  did on Wahab, with the test's MIT environment for the FFN runs.
- No FFN run is repeated: F1, F5, S10, P2 and Q2 times are those of the test (docs/unseen_evidence.json).

A smoke run on crop 11 (validation; 20 objects scored, the whole crop segmented) comes first and
checks only that the job completes. It is not reported as a result.

## Measures (fixed now)

Per crop k, SegNeuron's crop time T_SN(k) is defined as on the V100 (`headtohead.speed`, end to
end): the wall time of the `inference.py` process plus the CPU post-processing time. For each
B200 run R of the test, T_R(k) is its crop time (`crop_seconds`: traces plus image copy, or plus
the encoder for the student), as in the test's speed table.

- **Primary**: for R = F5, the number of crops on which SegNeuron was cheaper (T_SN(k) < T_R(k));
  the break-even object count per crop, T_SN(k) / (T_R(k) / n_k), with n_k the crop's objects, as
  median and range over crops; and from the ten crops' totals, (sum T_SN / 10) / (sum T_R / N).
  The statement follows the counts: "SegNeuron was cheaper than batched fast FFN on every test crop"
  (cheaper on 10 of 10), "batched fast FFN was cheaper than SegNeuron on every test crop" (dearer on
  10 of 10), or "SegNeuron was cheaper on c of 10 crops" otherwise, with ", equal on e" when e crops
  tie.
- The same for R = F1, S10, P2 and Q2, reported beside it with the same wording.
- Also reported: SegNeuron's `inference.py` loop time alone (`inference_seconds`, the GPU part), its
  post-processing time, and both against the V100 run's; and the primary measures with T_SN taken
  as inference loop time plus post-processing (leaving out process start-up and model loading,
  which FFN's crop time also leaves out).
- Agreement with the V100 run (a check that it is the same method, not a re-judgment): per-object F1
  differences between the B200 and V100 SegNeuron runs (mean, largest absolute, and the number of
  objects differing by more than 0.1), and each crop's mean F1 on both. The test's accuracy claims
  (question 2, from the V100 SegNeuron run) are not changed by this addendum, whatever it shows.

Every number above is reported whatever it shows. A job that fails for an infrastructure reason may be
rerun unchanged, and the failure is recorded: the rerun first renames the earlier attempt's folder
(crop-K.before-JOB-TIME), so only the rerun's own records and result are judged. Non-finite SegNeuron outputs or a crash caused by the
method on the B200 are a result, reported as such; the crop is then not timed, and the primary
statement is "not judged" for that crop. A recorded time that is not a finite, non-negative number
is treated the same way. The judge refuses a result whose job did not report an NVIDIA B200.

## Code (written and checked before anything runs)

- `cluster/mit/unseen_segneuron.sbatch`: the job; checks this plan's sha256 against the staged copy,
  the SegNeuron commit and clean tree, the checkpoint's sha256 and the staged source; records the GPU,
  the node and both environments; refuses crops other than 41-50 (11 with SMOKE).
- `cluster/mit/setup_segneuron.sh`: the one-time inference environment and SegNeuron checkout.
- `liconn_v2/unseen_b200.py`: the measures above, from result files alone (the judge also refuses
  crops staged from different sources, from a staged source that changes any file of the test's
  code pinned in docs/unseen_evidence.json, or without the inference versions above); `examples/judge_unseen_b200.py`
  writes docs/unseen_b200_evidence.json; tests in `tests/test_unseen_b200.py`.

## Disclosed in advance

- This is not part of the pre-registered test. It was planned after the test's results, including the
  V100 times and the B200 FFN times, were known. Its rule and measures are fixed before SegNeuron runs
  on the B200, but its question was chosen knowing the earlier numbers.
- Crops 41-50 have been read by every method of the test.
- FFN's B200 times come from jobs of 5-6 November that may have run on other B200 nodes; SegNeuron's
  come from later jobs. One run per method and crop; run-to-run spread is not measured.
- Both methods run code as it was in the test: neither is optimised for the B200 here. SegNeuron's
  `inference.py` runs one 20 x 128 x 128 patch at a time, and our post-processing is CPU code
  written for the earlier comparisons; a faster implementation of either would change the comparison.
- `inference.py` reads its input from and writes its outputs (about 1.5 GB of affinities per crop) to the
  cluster's shared storage, MIT pool storage here and Wahab scratch on the V100; that time is in the
  end-to-end measure on both, and the loop-only time leaves it out.
- SegNeuron runs cold in a fresh process per crop, as on the V100: its first patches carry CUDA
  initialisation, which is in both of its times, while FFN's runs warm up before timing. FFN's crop
  time also leaves out loading the network; SegNeuron's end-to-end time includes it.
- For the batched runs (F5, S10, P2, Q2, 32 objects at once) and the student's once-per-crop encoder,
  crop time divided by objects is the average cost at the crop's own object count, not a fixed cost
  per object, so a break-even far from that count is an extrapolation.
- Post-processing runs on the host CPUs of each cluster, so B200 and V100 post-processing times
  compare those CPUs, not the GPUs.
- The B200 SegNeuron run uses a different CUDA build of the same torch version, so its outputs may
  differ slightly from the V100 run's; the agreement measures show by how much.

## Compute

Eleven jobs of about 10-20 min each on one B200 (the smoke run and ten crops), about 3 GPU-hours.
