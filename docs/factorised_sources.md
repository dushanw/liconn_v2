# Factorised FFN: original sources

Collected for independent review of `liconn_v2/factorised.py` (docs/factorised_design.md).

## 1. The authors' network definition (verbatim)

`connectomics/jax/models/convstack.py`, google-research/connectomics at commit 73b76a7f (the pin
in requirements.txt), Copyright 2024 The Google Research Authors, Apache License 2.0. The
LICONN notebook builds it as
`convstack.ResConvStack(convstack.ConvstackConfig(depth=20, padding='same', use_layernorm=True))`
(docs/method_source.md; liconn_v2/reference.py), so features = 32, num_convs = 2,
kernel 3 x 3 x 3, out_features = 1.

```python
@struct.dataclass
class ConvstackConfig:
  """Config settings for residual convstacks.

  Attributes:
    features: number of feature maps
    depth: number of residual modules
    padding: padding mode to use for convolutions ('same', 'valid')
    dim: number of spatial dimensions
    num_convs: number of convolutions in the residual module
    use_layernorm: whether to use layer normalization; this has been observed to
      stabilize the training of FFNs, particularly in the case of deeper models.
    out_features: number of output feature maps
    enumerate_layers: If true, layer names will be prefixed with their number
      within the model. This parameter affects only the way how model params are
      names, not the behavior.
    kernel_shape: The 3d shape of the convolution kernel
    native_input_size: The native spatial size of the model input. The model may
      be able to process input of different size, but the configured input is
      usually expected to work the best. Changing this parameter does not affect
      the inference.
  """

  features: int | Iterable[int] = 32
  depth: int = 12  # number of residual modules
  padding: str = 'same'
  dim: int = 3
  num_convs: int = 2
  use_layernorm: bool = True
  out_features: int = 1
  enumerate_layers: bool = False
  kernel_shape: tuple[int, int, int] = (3, 3, 3)
  native_input_size: tuple[int, int, int] | None = None


class ResConvStack(nn.Module):
  """Residual convstack."""

  config: ConvstackConfig

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies the convstack to the input.

    Args:
      x: [batch, z, y, x, channels]-shaped input.

    Returns:
      convstack output
    """
    cfg = self.config

    layer_naming = param_util.LayerNaming(self.config.enumerate_layers)

    if isinstance(cfg.features, int):
      features = itertools.repeat(cfg.features)
    else:
      features = iter(cfg.features)

    labels = 'abcdefghijklmnopqrstuvwxyz'

    x = nn.Conv(
        next(features),
        self.config.kernel_shape[: cfg.dim],
        padding=cfg.padding,
        name=layer_naming.get_name('pre_a'),
    )(x)
    if cfg.use_layernorm:
      x = nn.LayerNorm()(x)
    for i in range(1, cfg.num_convs):
      x = nn.relu(x)
      x = nn.Conv(
          next(features),
          self.config.kernel_shape[: cfg.dim],
          padding=cfg.padding,
          name=layer_naming.get_name(f'pre_{labels[i]}'),
      )(x)

    for i in range(cfg.depth):
      mod_input = x
      if cfg.use_layernorm:
        x = nn.LayerNorm()(x)
      for j in range(0, cfg.num_convs):
        x = nn.relu(x)
        x = nn.Conv(
            next(features),
            self.config.kernel_shape[: cfg.dim],
            padding=cfg.padding,
            name=layer_naming.get_name(f'res{i}{labels[j]}'),
        )(x)

      if x.shape != mod_input.shape:
        crop_shape_zyx = x.shape[1 : 1 + cfg.dim]
        x += util.center_crop(mod_input, crop_shape_zyx)
      else:
        x += mod_input

    if cfg.use_layernorm:
      x = nn.LayerNorm()(x)
    x = nn.relu(x)
    return nn.Conv(
        cfg.out_features,
        (1, 1, 1)[: cfg.dim],
        name=layer_naming.get_name('output'),
    )(x)

```

## 2. flax.linen.LayerNorm (flax 0.10.2, Apache License 2.0), paraphrased

Defaults: epsilon = 1e-6, reduction over the last (channel) axis, learned scale and bias,
`use_fast_variance=True`: mean mu = E[x], variance = max(0, E[x^2] - mu^2), computed in at
least float32; y = (x - mu) / sqrt(var + epsilon) * scale + bias.

## 3. How the FFN input is formed (paraphrase; verified bitwise in earlier milestones)

The executor feeds the network [image box, seed box] along the channel axis
(liconn_v2/pruned.InputRecorder; liconn_v2/fasttrace.py). The image is normalised as
(raw - 128) / 33 (notebook). The seed box holds the canvas logits: never-predicted voxels are
the pad value 0.5, a logit of 0; the seed starts at activation 0.95.

## 4. Design precedents (from the 2026-10-31 literature search; not re-checked here)

- Kirillov et al., Segment Anything, ICCV 2023: heavy image encoder once per image, light
  prompt-conditioned mask decoder.
- Cheng & Schwing, XMem, ECCV 2022; Cheng et al., Cutie, CVPR 2024; Ravi et al., SAM 2,
  arXiv:2408.00714: video object segmentation with per-frame features shared across objects.
- Januszewski et al., Nat Methods 15:605 (2018), doi:10.1038/s41592-018-0049-4: FFN.
