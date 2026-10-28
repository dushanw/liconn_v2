# SegNeuron: source excerpts

Short excerpts used by `docs/segneuron_design.md`, quoted verbatim with their sources.

## Paper

Y. Zhang, J. Guo, H. Zhai, J. Liu, H. Han. "SegNeuron: 3D Neuron Instance Segmentation in Any EM
Volume with a Generalist Model." MICCAI 2024. https://papers.miccai.org/miccai-2024/paper/0518_paper.pdf

- Outputs (Sec. 2): "we aim to train a generalist model for neuron segmentation, capable of
  predicting voxel affinities along the x-, y-, and z-axes for any EM volume." "the foreground mask
  is denoted by F ∈ {0, 1}^(D×H×W), where 1 represents the neuronal region and 0 represents the
  plasma membrane and myelin in G."
- Training data (Sec. 2.1): "EMNeuron integrates 16 volume EM datasets (4 in-house, 12 publicly
  available), covering diverse species, sample preparation protocols, imaging techniques, and
  voxel resolutions." "...adjusting tangential resolutions (x, y) to 6∼9 nm, unifying membrane
  thickness, and masking out unlabeled myelin and glial regions."
- Foreground restriction (Sec. 2.2): "we filter background voxels in each channel of the predicted
  affinity map a using the predicted foreground mask f, as a^r_{c,d,h,w} ≡ min(a_{c,d,h,w},
  f_{d,h,w}). Such foreground-restricted affinity map a^r then serves as input for graph-based
  segmentation."
- Segmentation (Sec. 3.2): "Instance results are obtained by solving the multicut problem with the
  Kernighan-Lin solver [4]."
- Architecture choice (Sec. 3.3): "Transformer-based UNETR struggles to model diverse voxel
  resolutions and thus exhibits poor performance in affinities segmentation. SwinUNETR alleviates
  this problem through a heuristic structure, however it still lags behind CNN-based models by a
  wide margin in performance." "we adopt MNet as the architecture for SegNeuron".
- Fine-tuning (Sec. 3.3): "we achieve further performance improvements and obtain the best results
  on out-of-distribution datasets by finetuning SegNeruon under the same settings."
- Metrics (Sec. 3.1): "the variation of information (VI) [24] and adapted Rand error (ARE) [3] ...
  with the background being ignored during evaluation."

## Repository README

github.com/yanchaoz/SegNeuron, commit ea6f0c963a2f60f0f130244da27e2b2d753f8ee9.

- Input: "Input is a nonempty **3D `uint8` TIFF or NPY volume in `(z, y, x)` order** ... Choose the
  imaging scale before running inference; the script does not resize, resample or reinterpret
  axes. The paper targets approximately 5–10 nm x/y sampling."
- Tiling: "Inference retains MNet's architecture, divides intensities by 255, and blends
  overlapping `20 × 128 × 128` tiles with Gaussian weights and stride `10 × 64 × 64`."
- Outputs: "`affinities.npy` | `float32` probabilities, shape `(3, z, y, x)`; channels connect to
  offsets `(-1,0,0)`, `(0,-1,0)`, `(0,0,-1)`". "The auxiliary head is trained against
  `label != 0`; despite its historical filename, its output is foreground/interior confidence,
  not a membrane probability to invert. FRMC combines it with each affinity channel using the
  original elementwise minimum."
- FRMC: "The environment pins `python-elf=0.8.1`". "The default `0.25` reproduces the original
  parameter, not an accuracy guarantee on a new dataset."

## Code excerpts (verbatim, MIT licence, commit ea6f0c9)

`Train_and_Inference/utils/aff_util.py`, affinity targets (lines 81-112):

```python
def seg_to_affgraph(seg,nhood,pad=''):
    # constructs an affinity graph from a segmentation
    # assume affinity graph is represented as:
    # shape = (e, z, y, x)
    # nhood.shape = (edges, 3)
    shape = seg.shape
    nEdge = nhood.shape[0]
    aff = np.zeros((nEdge,)+shape,dtype=np.int32)

    for e in range(nEdge):
        aff[e, \
            max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] = \
                        (seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] == \
                         seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] ) \
                        * ( seg[max(0,-nhood[e,0]):min(shape[0],shape[0]-nhood[e,0]), \
                            max(0,-nhood[e,1]):min(shape[1],shape[1]-nhood[e,1]), \
                            max(0,-nhood[e,2]):min(shape[2],shape[2]-nhood[e,2])] > 0 ) \
                        * ( seg[max(0,nhood[e,0]):min(shape[0],shape[0]+nhood[e,0]), \
                            max(0,nhood[e,1]):min(shape[1],shape[1]+nhood[e,1]), \
                            max(0,nhood[e,2]):min(shape[2],shape[2]+nhood[e,2])] > 0 )
    if nEdge==3 and pad == 'replicate': # pad the boundary affinity
        aff[0,0] = (seg[0]>0).astype(aff.dtype)                                                                  
        aff[1,:,0] = (seg[:,0]>0).astype(aff.dtype)                                                              
        aff[2,:,:,0] = (seg[:,:,0]>0).astype(aff.dtype)   

    return aff
```

`Train_and_Inference/utils/seg_util.py`, neighbourhood (lines 46-64; `mknhood3d(1)` evaluates to `[[-1,0,0],[0,-1,0],[0,0,-1]]`):

```python
def mknhood3d(radius=1):
    # Makes nhood structures for some most used dense graphs.
    # The neighborhood reference for the dense graph representation we use
    # nhood(1,:) is a 3 vector that describe the node that conn(:,:,:,1) connects to
    # so to use it: conn(23,12,42,3) is the edge between node [23 12 42] and [23 12 42]+nhood(3,:)
    # See? It's simple! nhood is just the offset vector that the edge corresponds to.

    ceilrad = np.ceil(radius)
    x = np.arange(-ceilrad,ceilrad+1,1)
    y = np.arange(-ceilrad,ceilrad+1,1)
    z = np.arange(-ceilrad,ceilrad+1,1)
    [i,j,k] = np.meshgrid(z,y,x)

    idxkeep = (i**2+j**2+k**2)<=radius**2
    i=i[idxkeep].ravel(); j=j[idxkeep].ravel(); k=k[idxkeep].ravel();
    zeroIdx = np.array(len(i) // 2).astype(np.int32);

    nhood = np.vstack((k[:zeroIdx],i[:zeroIdx],j[:zeroIdx])).T.astype(np.int32)
    return np.ascontiguousarray(np.flipud(nhood))
```

`Train_and_Inference/supervised_provider.py`, targets (lines 159-160):

```python
        lb_affs1 = seg_to_affgraph(lb1, mknhood3d(1), pad='replicate').astype(np.float32)
        bounds1 = np.float32(lb1 != 0)
```

`Train_and_Inference/inference.py`, tiling and blending (lines 45-107):

```python
def tile_layout(shape):
    """Use the original overlapping grid, including safe padding for tiny/odd inputs."""
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError("Expected three positive Z,Y,X dimensions")
    counts = tuple(max(1, (n - c) // s + 2) for n, c, s in zip(shape, CROP_SIZE, STRIDE))
    padded_shape = tuple(c + (count - 1) * s for c, count, s in zip(CROP_SIZE, counts, STRIDE))
    padding = tuple(((p - n) // 2, (p - n + 1) // 2) for n, p in zip(shape, padded_shape))
    starts = tuple(tuple(i * step for i in range(count)) for count, step in zip(counts, STRIDE))
    return padding, starts


def gaussian_weight():
    """Original Gaussian blending weights (sigma=0.2, nonzero floor=1e-6)."""
    import numpy as np

    zz, yy, xx = np.meshgrid(
        *(np.linspace(-1, 1, n, dtype=np.float32) for n in CROP_SIZE), indexing="ij"
    )
    distance = np.sqrt(zz * zz + yy * yy + xx * xx)
    return 1e-6 + np.exp(-(distance ** 2 / (2.0 * 0.2 ** 2)))


def infer_volume(model, volume, device="cpu", progress=None):
    """Return float32 affinities (3,Z,Y,X) and boundaries (Z,Y,X), without GT."""
    import numpy as np
    import torch

    validate_volume(volume)
    padding, starts = tile_layout(volume.shape)
    padded = np.pad(volume, padding, mode="reflect")
    sums = np.zeros((4,) + padded.shape, dtype=np.float32)
    weights = np.zeros(padded.shape, dtype=np.float32)
    patch_weight = gaussian_weight()
    total = len(starts[0]) * len(starts[1]) * len(starts[2])
    model = model.to(device).eval()

    with torch.inference_mode():
        for index, position in enumerate(product(*starts), 1):
            region = tuple(slice(start, start + size) for start, size in zip(position, CROP_SIZE))
            patch = np.ascontiguousarray(padded[region], dtype=np.float32) / 255.0
            tensor = torch.from_numpy(patch[None, None]).to(device)
            affinities, boundaries = model(tensor)
            expected = (1, 3) + CROP_SIZE, (1, 1) + CROP_SIZE
            if tuple(affinities.shape) != expected[0] or tuple(boundaries.shape) != expected[1]:
                raise ValueError("Model output shapes must be (1,3,20,128,128) and (1,1,20,128,128)")
            prediction = torch.cat((affinities, boundaries), dim=1)[0].float().cpu().numpy()
            if not np.isfinite(prediction).all() or prediction.min() < 0 or prediction.max() > 1:
                raise ValueError("Model output must contain finite probabilities in [0,1]")
            sums[(slice(None),) + region] += prediction * patch_weight
            weights[region] += patch_weight
            if progress is not None:
                progress(index, total)

    if not np.all(weights > 0):
        raise RuntimeError("Inference grid did not cover the volume")
    sums /= weights[None]
    # A convex blend is a probability; remove only float32 accumulation roundoff.
    np.clip(sums, 0.0, 1.0, out=sums)
    # An explicit end preserves singleton axes and avoids the empty [0:-0] case.
    original = tuple(slice(pad[0], pad[0] + size) for pad, size in zip(padding, volume.shape))
    affinities = sums[(slice(0, 3),) + original].copy()
    boundaries = sums[(3,) + original].copy()
    return affinities, boundaries
```

`Postprocess/FRMC_post.py`, FRMC (lines 36-88) and the foreground fusion in `main`:

```python
def post_mc(affs, beta=0.25):
    """Segment merge affinities of shape (3, z, y, x), returning uint32 IDs.

    Channels correspond to offsets (-1,0,0), (0,-1,0), (0,0,-1).
    The original watershed settings and ELF Kernighan-Lin multicut are kept.
    Output IDs start at 1; cluster 0 from the solver is a neuron, not background.
    """
    affs = _probabilities(affs, "affinities", 4)
    if affs.shape[0] != 3:
        raise ValueError("affinities must have shape (3, z, y, x)")
    if isinstance(beta, (bool, np.bool_)) or not np.isscalar(beta):
        raise ValueError("beta must be a finite number strictly between 0 and 1")
    try:
        beta = float(beta)
    except (TypeError, ValueError) as exc:
        raise ValueError("beta must be a finite number strictly between 0 and 1") from exc
    if not np.isfinite(beta) or not 0 < beta < 1:
        raise ValueError("beta must be a finite number strictly between 0 and 1")

    feats, mc, ws = _load_elf()
    split_affs = 1.0 - affs
    boundary_input = np.maximum(split_affs[1], split_affs[2])
    watershed = np.empty(boundary_input.shape, dtype=np.uint64)
    offset = 0
    for z in range(watershed.shape[0]):
        wsz, _ = ws.distance_transform_watershed(
            boundary_input[z], threshold=0.25, sigma_seeds=2.0
        )
        wsz = np.asarray(wsz)
        if wsz.shape != watershed.shape[1:] or wsz.dtype.kind not in "ui" or np.any(wsz <= 0):
            raise RuntimeError(f"ELF watershed returned invalid or unassigned fragments at z={z}")
        # Dense zero-based RAG nodes, disjoint across slices even for sparse IDs.
        ids, inverse = np.unique(wsz, return_inverse=True)
        watershed[z] = inverse.reshape(wsz.shape).astype(np.uint64) + offset
        offset += len(ids)

    rag = feats.compute_rag(watershed)
    if rag.numberOfEdges == 0:
        # Exact edgeless-graph solution; ELF's cost scaling needs nonempty edges.
        node_labels = np.arange(rag.numberOfNodes, dtype=np.uint64)
    else:
        offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1]]
        costs = feats.compute_affinity_features(rag, split_affs, offsets)[:, 0]
        edge_sizes = feats.compute_boundary_mean_and_length(rag, boundary_input)[:, 1]
        if not np.isfinite(costs).all() or not np.isfinite(edge_sizes).all() or np.any(edge_sizes <= 0):
            raise RuntimeError("ELF returned invalid edge probabilities or sizes")
        costs = mc.transform_probabilities_to_costs(costs, edge_sizes=edge_sizes, beta=beta)
        node_labels = mc.multicut_kernighan_lin(rag, costs)
    segmentation = feats.project_node_labels_to_pixels(rag, node_labels)
    labels, inverse = np.unique(segmentation, return_inverse=True)
    if len(labels) > np.iinfo(np.uint32).max:
        raise OverflowError("Too many instances for uint32 output")
    return (inverse.reshape(watershed.shape) + 1).astype(np.uint32)

        # Keep the original foreground-head fusion and probability direction.
        segmentation = post_mc(np.minimum(affinities, boundaries[None]), args.beta)
```

`Train_and_Inference/inference.py`, output files (from `main`):

```python
    np.save(args.output_dir / "affinities.npy", affinities, allow_pickle=False)
    tifffile.imwrite(args.output_dir / "boundaries.tif", boundaries, photometric="minisblack", metadata={"axes": "ZYX"})
        "boundaries": {"file": "boundaries.tif", "axes": "ZYX", "dtype": "float32"},
```
