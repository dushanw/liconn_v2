# Method source for the FFN inference speed work

This file collects the original sources that the speed work must stay faithful to.
Direct quotes are marked as such; everything else is a paraphrase by the implementer
(Claude) and should be checked against the cited source.

## 1. Paper

Tavakoli M., Lyudchik J., Januszewski M., et al. "Light-microscopy-based connectomic
reconstruction of mammalian brain tissue." Nature 642, 398-410 (2025).
doi:10.1038/s41586-025-08985-1. Open access (CC BY 4.0): https://pmc.ncbi.nlm.nih.gov/articles/PMC12158774/

Methods, "Neuronal instance segmentation" (quote): "We used the original convolutional
stack architecture with a 33 × 33 × 33-voxel FOV, a step size of 8 voxels" (the paper
also states the network depth was extended to 20 residual modules).

Paraphrased from the same Methods section:
- FFN code: https://github.com/google/ffn, git hash 12d680e. Inference notebook:
  https://github.com/google/ffn/blob/master/notebooks/jax_ffn_inference_liconn.ipynb
- Input data for the FFN were CLAHE-processed and downsampled 2x laterally (and 2x axially
  for 200-nm-step datasets).
- Training: batch size 128, learning rate 1e-4, AdamW, 32 NVIDIA V100 GPUs for up to 30
  days (~8.8 million steps). Training is out of scope for this project.
- The full pipeline after FFN inference (oversegmentation consensus, end-point
  agglomeration with U-net FFNs, class-consistency filtering, manual proofreading) is out
  of scope for the current milestone.
- The paper reports no runtime for FFN inference.

Data availability (paraphrased): image data are public in browsable form via Neuroglancer
from the `gs://liconn-public` bucket; further data at https://doi.org/10.15479/AT:ISTA:18697.

## 2. Authors' runnable example (google/ffn notebook at commit 12d680e)

Source: `notebooks/jax_ffn_inference_liconn.ipynb`, google/ffn, Apache License 2.0,
commit 12d680e21f96e41b6a893859364fa4a3e924e51a ("Add a demo notebook showing JAX FFN
inference on LICONN data", 2025-01-07). Code cells, verbatim (installs, plotting and
Neuroglancer viewer cells omitted):

```python
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# Ensure tensorstore does not attempt to use GCE credentials
os.environ['GCE_METADATA_ROOT'] = 'metadata.google.internal.invalid'

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from clu import checkpoint
from connectomics.jax.models import convstack
import jax
import matplotlib.pyplot as plt
import numpy as np
import tensorstore as ts

from ffn.inference import inference
from ffn.inference import inference_utils
from ffn.inference import inference_pb2
from ffn.inference import executor
from ffn.training import model as ffn_model

# Check for GPU presence. If this fails, use "Runtime > Change runtime type".
assert jax.devices()[0].platform in ('gpu', 'tpu')

# Use sample LICONN data (2x downsampled).
context = ts.Context({'cache_pool': {'total_bytes_limit': 1_000_000_000}})
img = ts.open({
    'driver': 'neuroglancer_precomputed',
    'kvstore': {'driver': 'gcs', 'bucket': 'liconn-public'},
    'path': 'ExPID82_1/image_230130b',
    'scale_index': 1},
    read=True, context=context).result()[ts.d['channel'][0]]

# Load a 500^3 subvolume for local processing.
x0, y0, z0 = 1100, 1083, 111
raw = img[x0:x0+500, y0:y0+500, z0:z0+500].read().result()
raw = np.transpose(raw, [2, 1, 0])
raw = (raw.astype(np.float32) - 128.0) / 33.  # normalize data for inference

# Load sample model checkpoint.
!gcloud storage cp gs://liconn-public/models/ffn/axde_59110972/ckpt-2116* .

ckpt = checkpoint.Checkpoint('')
state = ckpt.load_state(state=None, checkpoint='ckpt-2116')

# Instantiate model for inference.
model = convstack.ResConvStack(convstack.ConvstackConfig(depth=20, padding='same', use_layernorm=True))
fov_size = 33, 33, 33
model_info = ffn_model.ModelInfo(deltas=(8, 8, 8), pred_mask_size=fov_size, input_seed_size=fov_size, input_image_size=fov_size)

@jax.jit
def _apply_fn(data):
  return model.apply({'params': state['params']}, data)

iface = executor.ExecutorInterface()
counters = inference_utils.Counters()
exc = executor.JAXExecutor(iface, model_info, _apply_fn, counters, 1)
exc.start_server()

options = inference_pb2.InferenceOptions(init_activation=0.95, pad_value=0.5, move_threshold=0.6, segment_threshold=0.6)
cv = inference.Canvas(model_info, exc.get_client(counters), raw, options, voxel_size_zyx=(24, 18, 18))

# Trace a single neurite.
pos_xyz = (123, 171, 225)
cv.segment_at(pos_xyz[::-1], dynamic_image=inference.DynamicImage(), vis_update_every=10)

# Visualize results in neuroglancer.
from scipy.ndimage import label
seg = (label(cv.seed > 0)[0] == 1).astype(np.uint64)
```

## 3. Deliberate, disclosed deviations for a local CPU reference run

These are setup changes. None of them changes the model, weights, data, thresholds or
FFN movement logic.
1. CPU only: the `assert ... in ('gpu', 'tpu')` check is dropped (owner decision:
   local CPU for now). CPU timings are not GPU timings.
2. No `DynamicImage` visualization (`dynamic_image=None`). In FFN at 12d680e,
   `visualize_state` only reads the canvas, so seed updates are unaffected.
3. Inputs come from a local cache (`~/ResearchFlow/data/liconn-v2`, override with
   `LICONN_DATA`). The cache holds the same crop read once from `gs://liconn-public`
   with tensorstore (stored before transpose and normalization), plus the checkpoint
   files fetched over public HTTPS. Their sha256 values are pinned and checked:
   - `raw_xyz_uint8.npy` (NumPy .npy, uint8, shape (500, 500, 500), x/y/z order as read):
     c9abc91b551da7ad2b5b7ca6ac12ebff3b5c2e9e4a55b05140694e1f50083837
   - `ckpt/ckpt-2116.flax`: dbe2bc6fb4169a33101cb5b688f34dcf7b6b39bdf0a1d9866523e5f95033057e
   - `ckpt/ckpt-2116.index`: 92ccbd68e09b55db34cbb36b749926eede07569c656579f9a0beda783582bd98
4. Package versions are pinned in requirements.txt (FFN at the paper's hash; other
   versions contemporary with it). The notebook installs the latest FFN instead.
   The connectomics pin (73b76a7f, the last main commit before FFN 12d680e) and the other
   version pins are the implementer's inference, not stated by the paper or notebook.
   Different jax/jaxlib/flax/connectomics versions may change XLA CPU arithmetic and
   therefore the bitwise reference.
5. JAX is forced onto the CPU (`JAX_PLATFORMS=cpu`, verified with `jax.default_backend()`).
6. One upfront call of the compiled model on zeros of shape (1, 33, 33, 33, 2) before the
   trace, so compile time is reported separately. The model is a pure function.
7. `Canvas(..., counters=counters, keep_history=True)`: the counters record timings and
   history records visited centers. Neither is read by the inference or movement logic.
   The notebook-faithful probe without deviations 6-7 gave the same canvas sha256.

## 4. Speed-work contract (owner decisions, 2026-10-18)

- Target: FFN inference (not training, not agglomeration).
- Equivalence: exact match. A faster implementation must reproduce the reference
  float32 FFN canvas (`cv.seed`) bitwise, and therefore the same mask, the same
  iteration count and the same visit order.
- Hardware: local CPU (Apple M5 Max) for now.

Initial CPU profile of the reference trace (one probe run, before any changes):
359 FFN steps, about 134 s in `executor-inference` out of about 142 s wall time. Network
evaluation dominates; image fetch, movement policy and canvas updates are minor.

Reference fingerprint (liconn_v2/expected_reference.json). A verbatim-notebook probe
script, with no upfront compile and no history recording, gave the same canvas sha256
(cfd7f4d7...) in two separate processes. The milestone module reproduced it a third time.
The notebook's `label(cv.seed > 0)[0] == 1` mask contains the seed voxel in this trace.
That definition is not a general seed-component extractor. Label 1 is the first component
in (z, y, x) scan order, and the threshold is logit 0 (p = 0.5), not segment_threshold 0.6.
The reference therefore also records the seed's component at logit(0.6) (94,534 voxels,
compared with 94,905 voxels for the notebook mask).

Known consequence of the exact-match choice (raised in the Cursor method check): most
speedups of the network itself (batching, layout or fusion changes, other XLA flags, GPU)
change floating-point summation order and are expected to fail a bitwise canvas match,
even when the mask is unchanged. Whether to allow a tolerance is an owner decision for a
later milestone.
