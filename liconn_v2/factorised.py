"""Factorised FFN: a PyTorch port of the authors' network and an encoder/decoder student.

The authors' network (connectomics.jax.models.convstack.ResConvStack, depth 20, 32 features)
joins the image and the mask logit at its first convolution. The student splits it:
an image encoder that runs once over the volume, and a mask decoder that runs at every
tracing step on the encoder features of the box and the box's mask logits.
See docs/factorised_design.md.

Tensors are (batch, channels, z, y, x). Flax tensors are (batch, z, y, x, channels).
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

FEATURES = 32
DEPTH = 20  # residual modules in the authors' network
LN_EPSILON = 1e-6  # flax.linen.LayerNorm default
ENCODER_DEPTH = 16
DECODER_DILATIONS = (1, 2, 4, 1)
CONV_FLOP = 2 * FEATURES * FEATURES * 27  # one 3^3 32 -> 32 convolution, per output voxel


class ChannelLayerNorm(nn.Module):
  """flax.linen.LayerNorm over the channel axis, with its fast variance (E[x^2] - E[x]^2)."""

  def __init__(self, channels=FEATURES):
    super().__init__()
    self.scale = nn.Parameter(torch.ones(channels))
    self.bias = nn.Parameter(torch.zeros(channels))

  def forward(self, x):
    mu = x.mean(1, keepdim=True)
    var = torch.clamp((x * x).mean(1, keepdim=True) - mu * mu, min=0.0)
    y = (x - mu) * torch.rsqrt(var + LN_EPSILON)
    return y * self.scale.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)


def conv3(cin=FEATURES, cout=FEATURES, dilation=1):
  return nn.Conv3d(cin, cout, 3, padding=dilation, dilation=dilation)


class Residual(nn.Module):
  """x + conv(ReLU(conv(ReLU(LayerNorm(x)))))."""

  def __init__(self, dilation=1):
    super().__init__()
    self.norm = ChannelLayerNorm()
    self.conv_a = conv3(dilation=dilation)
    self.conv_b = conv3(dilation=dilation)

  def forward(self, x):
    return x + self.conv_b(F.relu(self.conv_a(F.relu(self.norm(x)))))


class Head(nn.Module):
  """LayerNorm, ReLU, 1x1x1 convolution to one logit."""

  def __init__(self):
    super().__init__()
    self.norm = ChannelLayerNorm()
    self.output = nn.Conv3d(FEATURES, 1, 1)

  def forward(self, x):
    return self.output(F.relu(self.norm(x)))


class ConvStack(nn.Module):
  """The authors' network: [image, mask logit] -> logit. `dilations` (one per block; default all 1)
  is for pruned copies (pruned_from_teacher); it sets the depth when given."""

  def __init__(self, depth=DEPTH, dilations=None):
    super().__init__()
    dilations = (1,) * depth if dilations is None else tuple(dilations)
    self.pre_a = conv3(2, FEATURES)
    self.pre_norm = ChannelLayerNorm()
    self.pre_b = conv3()
    self.blocks = nn.ModuleList(Residual(d) for d in dilations)
    self.head = Head()

  def forward(self, x):
    x = self.pre_b(F.relu(self.pre_norm(self.pre_a(x))))
    for b in self.blocks:
      x = b(x)
    return self.head(x)


class Encoder(nn.Module):
  """Image -> features: the authors' stem (image channel only) and their first K modules."""

  def __init__(self, depth=ENCODER_DEPTH):
    super().__init__()
    self.pre_a = conv3(1, FEATURES)
    self.pre_norm = ChannelLayerNorm()
    self.pre_b = conv3()
    self.blocks = nn.ModuleList(Residual() for _ in range(depth))

  @property
  def radius(self) -> int:
    """Receptive radius in voxels: two stem convolutions and two per module."""
    return 2 + 2 * len(self.blocks)

  def forward(self, image):
    x = self.pre_b(F.relu(self.pre_norm(self.pre_a(image))))
    for b in self.blocks:
      x = b(x)
    return x


class Decoder(nn.Module):
  """(features, mask logit) -> logit: features + MaskStem(mask), then the remaining modules."""

  def __init__(self, dilations=DECODER_DILATIONS):
    super().__init__()
    self.mask_a = conv3(1, FEATURES)
    self.mask_b = conv3()
    nn.init.zeros_(self.mask_b.weight)  # at initialisation the decoder ignores the mask
    nn.init.zeros_(self.mask_b.bias)
    self.blocks = nn.ModuleList(Residual(d) for d in dilations)
    self.head = Head()

  def forward(self, features, mask):
    x = features + self.mask_b(F.relu(self.mask_a(mask)))
    for b in self.blocks:
      x = b(x)
    return self.head(x)


class FactorisedFFN(nn.Module):

  def __init__(self, encoder_depth=ENCODER_DEPTH, dilations=DECODER_DILATIONS):
    super().__init__()
    self.encoder = Encoder(encoder_depth)
    self.decoder = Decoder(dilations)

  def forward(self, x):
    """Same input as the authors' network: [image, mask logit] on one box."""
    return self.decoder(self.encoder(x[:, :1]), x[:, 1:])


# ------------------------------------------------------------------ weights from the checkpoint

def _conv_weight(kernel) -> torch.Tensor:
  """Flax (z, y, x, in, out) -> torch (out, in, z, y, x). Both are cross-correlations."""
  return torch.from_numpy(np.array(np.transpose(np.asarray(kernel, np.float32), (4, 3, 0, 1, 2)), order='C'))


def _set_conv(conv, p, in_channels=None):
  w = _conv_weight(p['kernel'])
  if in_channels is not None:
    w = w[:, in_channels]
  with torch.no_grad():
    conv.weight.copy_(w)
    conv.bias.copy_(torch.from_numpy(np.array(p['bias'], np.float32)))


def _set_norm(norm, p):
  with torch.no_grad():
    norm.scale.copy_(torch.from_numpy(np.array(p['scale'], np.float32)))
    norm.bias.copy_(torch.from_numpy(np.array(p['bias'], np.float32)))


def _depth(params) -> int:
  return sum(1 for k in params if k.startswith('res') and k.endswith('a'))


def teacher_from_params(params) -> ConvStack:
  """The authors' network with their weights (flax names as in liconn_v2.earlystop._module)."""
  depth = _depth(params)
  m = ConvStack(depth)
  _set_conv(m.pre_a, params['pre_a'])
  _set_norm(m.pre_norm, params['LayerNorm_0'])
  _set_conv(m.pre_b, params['pre_b'])
  for i, b in enumerate(m.blocks):
    _set_norm(b.norm, params[f'LayerNorm_{i + 1}'])
    _set_conv(b.conv_a, params[f'res{i}a'])
    _set_conv(b.conv_b, params[f'res{i}b'])
  _set_norm(m.head.norm, params[f'LayerNorm_{depth + 1}'])
  _set_conv(m.head.output, params['output'])
  return m.eval()


def student_from_teacher(teacher: ConvStack, encoder_depth=ENCODER_DEPTH,
                         dilations=DECODER_DILATIONS, seed=0) -> FactorisedFFN:
  """Student initialised from the teacher: encoder = stem (image channel) + modules 0..K-1,
  decoder = modules K..depth-1 + head; the mask stem is new (its last convolution is zero)."""
  depth = len(teacher.blocks)
  if encoder_depth + len(dilations) != depth:
    raise ValueError(f'encoder depth {encoder_depth} + {len(dilations)} decoder modules != {depth}')
  torch.manual_seed(seed)
  s = FactorisedFFN(encoder_depth, dilations)
  with torch.no_grad():
    s.encoder.pre_a.weight.copy_(teacher.pre_a.weight[:, :1])
    s.encoder.pre_a.bias.copy_(teacher.pre_a.bias)
    s.encoder.pre_norm.load_state_dict(teacher.pre_norm.state_dict())
    s.encoder.pre_b.load_state_dict(teacher.pre_b.state_dict())
    for i, b in enumerate(s.encoder.blocks):
      b.load_state_dict(teacher.blocks[i].state_dict())
    for i, b in enumerate(s.decoder.blocks):
      b.load_state_dict(teacher.blocks[encoder_depth + i].state_dict())
    s.decoder.head.load_state_dict(teacher.head.state_dict())
  return s


def pruned_from_teacher(teacher: ConvStack, keep, dilations) -> ConvStack:
  """A pruned FFN (docs/prune_design.md): the teacher's stem, its residual blocks `keep` (0-based,
  in this order) with per-block `dilations`, and its head, all copied. The teacher is unchanged."""
  keep, dilations = [int(i) for i in keep], tuple(int(d) for d in dilations)
  if len(keep) != len(dilations) or not keep:
    raise ValueError(f'{len(keep)} blocks with {len(dilations)} dilations')
  if len(set(keep)) != len(keep) or min(keep) < 0 or max(keep) >= len(teacher.blocks):
    raise ValueError(f'blocks {keep} are not distinct blocks of a {len(teacher.blocks)}-block network')
  m = ConvStack(dilations=dilations)
  with torch.no_grad():
    for name in ('pre_a', 'pre_norm', 'pre_b', 'head'):
      getattr(m, name).load_state_dict(getattr(teacher, name).state_dict())
    for b, i in zip(m.blocks, keep):
      b.load_state_dict(teacher.blocks[i].state_dict())
  return m.eval()


# ------------------------------------------------------------------ dense encoding

def tile_starts(n: int, tile: int) -> list:
  """Starts of the output tiles that cover [0, n)."""
  return list(range(0, n, tile))


@torch.no_grad()
def encode_volume(encoder: Encoder, image_zyx, tile=128, dtype=torch.float32, out_device='cpu') -> torch.Tensor:
  """Encoder features (C, Z, Y, X) of a whole volume.

  Definition: the image is zero-padded by the receptive radius r, the encoder runs on it, and the
  padding is cropped off. Each feature then depends only on the image within r voxels (zeros
  outside the volume), so tiling with a halo of r gives the same features as one pass. The
  output is written tile by tile on `out_device`."""
  image = torch.as_tensor(np.asarray(image_zyx, np.float32))
  r = encoder.radius
  shape = image.shape
  out = torch.empty((FEATURES, *shape), dtype=dtype, device=out_device)
  padded = F.pad(image[None, None], (r, r, r, r, r, r))
  dev = next(encoder.parameters()).device
  for z in tile_starts(shape[0], tile):
    for y in tile_starts(shape[1], tile):
      for x in tile_starts(shape[2], tile):
        ez, ey, ex = min(z + tile, shape[0]), min(y + tile, shape[1]), min(x + tile, shape[2])
        # The tile with halo, cut from the zero-padded volume; the encoder's own 'same' padding
        # then only touches halo voxels, which are cropped away.
        inp = padded[..., z:ez + 2 * r, y:ey + 2 * r, x:ex + 2 * r].to(dev)
        f = encoder(inp)[0, :, r:r + ez - z, r:r + ey - y, r:r + ex - x]
        out[:, z:ez, y:ey, x:ex] = f.to(out.device, dtype)
  return out


def box_features(features: torch.Tensor, center_zyx, size=(33, 33, 33)) -> torch.Tensor:
  """Features of the box centred at center_zyx, (1, C, *size)."""
  sl = tuple(slice(c - s // 2, c - s // 2 + s) for c, s in zip(center_zyx, size))
  return features[(slice(None), *sl)][None]


# ------------------------------------------------------------------ check inputs

OBJECT_LOGIT = float(np.log(0.95 / 0.05))  # the notebook's initial activation 0.95, as a logit


def realistic_box(image_zyx, labels_zyx, center_zyx, known_radius=8, size=33) -> np.ndarray:
  """A network input [image, mask logit] (2, size, size, size) shaped like one mid-trace:
  voxels within known_radius of the centre are known (object +logit(0.95) if they carry the
  centre's label, otherwise -logit(0.95)); the rest are unknown (logit 0, FFN's pad value 0.5)."""
  h = size // 2
  sl = tuple(slice(c - h, c + h + 1) for c in center_zyx)
  img = np.asarray(image_zyx[sl], np.float32)
  lab = np.asarray(labels_zyx[sl])
  g = np.mgrid[-h:h + 1, -h:h + 1, -h:h + 1]
  known = (g ** 2).sum(0) <= known_radius ** 2
  obj = lab == lab[h, h, h]
  mask = np.where(known, np.where(obj, OBJECT_LOGIT, -OBJECT_LOGIT), 0.0).astype(np.float32)
  return np.stack([img, mask])


def check_centers(shape, n, size=33, seed=0) -> list:
  """n box centres drawn uniformly (fixed seed) with the whole box inside the volume."""
  rng = np.random.default_rng(seed)
  h = size // 2
  return [tuple(int(rng.integers(h, s - h)) for s in shape) for _ in range(n)]


# ------------------------------------------------------------------ cost

def flop_per_voxel() -> dict:
  """Multiply-adds x 2 per output voxel (LayerNorm and ReLU ignored)."""
  def encoder(k):
    return 2 * 1 * FEATURES * 27 + CONV_FLOP + 2 * k * CONV_FLOP
  def decoder(n):
    return 2 * 1 * FEATURES * 27 + CONV_FLOP + 2 * n * CONV_FLOP + 2 * FEATURES
  teacher = 2 * 2 * FEATURES * 27 + CONV_FLOP + 2 * DEPTH * CONV_FLOP + 2 * FEATURES
  return {'teacher': teacher, 'encoder': encoder(ENCODER_DEPTH),
          'decoder': decoder(DEPTH - ENCODER_DEPTH)}


def box_gflop(blocks: int, box=(33, 33, 33)) -> float:
  """GFLOP of one evaluation of the authors' network with `blocks` residual blocks on a box (the
  teacher with 20; pruned copies with fewer). Dilation does not change the count."""
  per_voxel = 2 * 2 * FEATURES * 27 + CONV_FLOP + 2 * blocks * CONV_FLOP + 2 * FEATURES
  return per_voxel * int(np.prod(box)) / 1e9


def encoder_input_voxels(shape_zyx, tile=None, radius=2 + 2 * ENCODER_DEPTH) -> int:
  """Voxels the encoder actually runs on to encode a volume: each tile (or, with tile=None, the
  whole volume) plus its halo of `radius` on every side (encode_volume)."""
  if tile is None:
    return int(np.prod([s + 2 * radius for s in shape_zyx]))
  per_axis = [[min(t + tile, s) - t + 2 * radius for t in tile_starts(s, tile)] for s in shape_zyx]
  return int(np.prod([sum(a) for a in per_axis]))


def crop_flop(steps: int, shape_zyx, tile=None, box=(33, 33, 33)) -> dict:
  """FLOP to trace `steps` steps in a volume: teacher vs student (dense encoder including its
  halo overhead for the given tiling, plus decoder steps)."""
  v = flop_per_voxel()
  n = int(np.prod(box))
  teacher = steps * n * v['teacher']
  encoder = encoder_input_voxels(shape_zyx, tile) * v['encoder']
  decoder = steps * n * v['decoder']
  return {'teacher': teacher, 'encoder': encoder, 'decoder': decoder, 'student': encoder + decoder,
          'ratio': teacher / (encoder + decoder)}
