# SegNeuron training code: verbatim excerpts

From github.com/yanchaoz/SegNeuron at commit `ea6f0c963a2f60f0f130244da27e2b2d753f8ee9`, MIT licence,
folder `Train_and_Inference/` unless stated. Used by `docs/segneuron_finetune_design.md`. The
checkout's README (Training): "Supervised training expects raw/instance pairs ... in the 10
directories listed in supervised_provider.py. Its dataset-balanced sampling retains the original
76-volume order ...; arbitrary custom folder collections require adapting that sampler." and "the
historical validation-related YAML fields do not implement a held-out validation loop."

## supervised_train.py: build_model (pre_train and continue_train)

```python
def build_model(cfg, writer):
    print('Building model on ', end='', flush=True)
    t1 = time.time()
    device = torch.device('cuda:0')

    print('load mnet!')

    model = MNet(1, kn=(32, 64, 96, 128, 256), FMU='sub').cuda()

    if cfg.MODEL.pre_train:
        ckpt_path = cfg.MODEL.pretrain_path
        print('Load pre-trained model from' + ckpt_path)
        checkpoint = torch.load(ckpt_path)
        pretrained_dict = OrderedDict()
        state_dict = checkpoint['model_weights']
        for k, v in state_dict.items():
            name = k.replace('module.', '') if 'module' in k else k
            pretrained_dict[name] = v

        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           (k in model_dict) and ('up' not in k) and (
                                   'outputs' not in k)}  # 1. filter out unnecessary keys
        print(pretrained_dict)
        model_dict.update(pretrained_dict)  # 2. overwrite entries in the existing state dict
        model.load_state_dict(model_dict)

    if cfg.MODEL.continue_train:
        ckpt_path = cfg.MODEL.continue_path
        print('Load pre-trained model from' + ckpt_path)
        checkpoint = torch.load(ckpt_path)
        new_state_dict = OrderedDict()
        state_dict = checkpoint['model_weights']
        for k, v in state_dict.items():
            name = k.replace('module.', '') if 'module' in k else k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
```

## supervised_train.py: calculate_lr

```python
def calculate_lr(iters):
    if iters < cfg.TRAIN.warmup_iters:
        current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(float(iters) / cfg.TRAIN.warmup_iters,
                                                                  cfg.TRAIN.power) + cfg.TRAIN.end_lr
    else:
        if iters < cfg.TRAIN.decay_iters:
            current_lr = (cfg.TRAIN.base_lr - cfg.TRAIN.end_lr) * pow(
                1 - float(iters - cfg.TRAIN.warmup_iters) / cfg.TRAIN.decay_iters, cfg.TRAIN.power) + cfg.TRAIN.end_lr
        else:
            current_lr = cfg.TRAIN.end_lr
    return current_lr
```

## supervised_train.py: loss and step

```python
    if cfg.TRAIN.loss_func == 'MSELoss':
        criterion = WeightedMSE()
    elif cfg.TRAIN.loss_func == 'BCELoss':
        criterion = WeightedBCE()
    else:
        raise AttributeError("NO this criterion")

    while iters < cfg.TRAIN.total_iters:
        # train
        model.train()
        iters += 1
        t1 = time.time()
        inputs, target, bound, bmask = train_provider.next()

        # decay learning rate
        if cfg.TRAIN.end_lr == cfg.TRAIN.base_lr:
            current_lr = cfg.TRAIN.base_lr
        else:
            current_lr = calculate_lr(iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

        optimizer.zero_grad()
        pred, pred_b = model(inputs)

        ##############################
        loss1 = criterion(pred, target, weight=bmask)
        loss2 = criterion(pred_b, bound, weight=bmask)
        loss = loss1 + loss2
        loss.backward()
        ##############################

        if cfg.TRAIN.weight_decay is not None:
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.data = param.data.add(-cfg.TRAIN.weight_decay * group['lr'], param.data)
        optimizer.step()
```

## supervised_train.py: optimiser

```python
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.base_lr, betas=(0.9, 0.999),
                                     eps=0.01, weight_decay=1e-6, amsgrad=True)
```

## supervised_provider.py: patch, augmentations and targets

```python
        # normal augumentation
        [imgs1, lb1] = self.simple_aug([imgs1, lb1])
        imgs1, lb1 = self.apply_perturbations(imgs1, lb1)

        lb_affs1 = seg_to_affgraph(lb1, mknhood3d(1), pad='replicate').astype(np.float32)
        bounds1 = np.float32(lb1 != 0)
```

## supervised_provider.py: perturbations

```python
    def perturbations_init(self):
        self.per_intensity = Intensity()
        self.per_gaussnoise = GaussNoise(min_std=self.min_noise_std, max_std=self.max_noise_std, norm_mode='trunc')
        self.per_gaussblur = GaussBlur(min_kernel=self.min_kernel_size, max_kernel=self.max_kernel_size,
                                       min_sigma=self.min_sigma, max_sigma=self.max_sigma)
        self.per_cutout = Cutout(model_type=self.model_type)
        self.per_misalign = Elastic(control_point_spacing=[4, 40, 40], jitter_sigma=[0, 0, 0], prob_slip=0.2,
                                    prob_shift=0.2, max_misalign=17, padding=20)
        self.per_elastic = Elastic(control_point_spacing=[4, 40, 40], jitter_sigma=[0, 2, 2], padding=20)

    def apply_perturbations(self, data, mask):
        if random.random() < 0.25:
            data = self.per_intensity(data)
        if random.random() < 0.25:
            data = self.per_gaussnoise(data)
        if random.random() < 0.25:
            data = self.per_gaussblur(data)
        if random.random() < 0.25:
            data = self.per_cutout(data)
        if random.random() < 0.25:
            data, mask = self.per_elastic(data, mask)

        return data, mask
```

## loss/loss.py: WeightedBCE

```python
class WeightedBCE(nn.Module):
    """Weighted binary cross-entropy.
    """
    def __init__(self, size_average=True, reduce=True):
        super().__init__()
        self.size_average = size_average
        self.reduce = reduce

    def forward(self, pred, target, weight=None):
        #_assert_no_grad(target)
        if pred.min()<0:
            # pred = (pred-pred.min())/(pred-pred.min()).max()
            pred = F.relu(pred)

        return F.binary_cross_entropy(pred, target, weight)
```

## supervised_provider.py: patch cut and scaling

```python
        random_z = random.randint(0, raw_data_shape[0] - self.crop_from_origin[0])
        random_y = random.randint(0, raw_data_shape[1] - self.crop_from_origin[1])
        random_x = random.randint(0, raw_data_shape[2] - self.crop_from_origin[2])

        imgs1 = used_data[random_z:random_z + self.crop_from_origin[0], \
                random_y:random_y + self.crop_from_origin[1], \
                random_x:random_x + self.crop_from_origin[2]].copy()
        lb1 = used_label[random_z:random_z + self.crop_from_origin[0], \
              random_y:random_y + self.crop_from_origin[1], \
              random_x:random_x + self.crop_from_origin[2]].copy()

        imgs1 = imgs1.astype(np.float32) / 255.0
```

## supervised_provider.py: loss mask (bmask)

```python
        # area mask
        if True:
            if type_dataset == 10 or type_dataset == 0 or type_dataset == 8 or type_dataset_r2 == 0 or type_dataset_r2 == 8:
                bmask = 1 - np.uint(lb == 10000)
            elif (type_dataset == 1 and k == 33) or (type_dataset_r2 == 1 and r2 == 33):
                bmask = 1 - (np.uint(lb == 121) + np.uint(lb == 122))
            else:
                bmask = np.ones_like(lb)
```

## supervised_provider.py: output shapes

```python
        # extend dimension
        imgs = imgs[np.newaxis, ...]
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)

        lb_affs = np.ascontiguousarray(lb_affs, dtype=np.float32)

        bounds = bounds[np.newaxis, ...]
        bounds = np.ascontiguousarray(bounds, dtype=np.float32)

        bmask = bmask[np.newaxis, ...]
        bmask = np.ascontiguousarray(bmask, dtype=np.float32)

        return imgs, lb_affs, bounds, bmask
```

## utils/augmentation.py: ElasticAugment.forward

```python
    def forward(self, imgs, mask):
        '''Args:
            imgs: numpy array, [Z, Y, Z], it always is float and 0~1
            mask: numpy array, [Z, Y, Z], it always is uint16
        '''
        if self.padding is not None:
            imgs = np.pad(imgs, ((0,0), \
                                (self.padding,self.padding), \
                                (self.padding,self.padding)), mode='reflect')
            mask = np.pad(mask, ((0,0), \
                                (self.padding,self.padding), \
                                (self.padding,self.padding)), mode='reflect')
        transform = self.create_transformation(imgs.shape)
        img_transform = apply_transformation(imgs,
                                         transform,
                                         interpolate=False,
                                         outside_value=0,  # imgs.dtype.type(-1)
                                         output=np.zeros(imgs.shape, dtype=np.float32))
        seg_transform = apply_transformation(mask,
                                         transform,
                                         interpolate=False,
                                         outside_value=0,  # mask.dtype.type(-1)
                                         output=np.zeros(mask.shape, dtype=np.uint16))  # dtype=np.float32
        # seg_transform[seg_transform < 0] = 0
        # seg_transform[seg_transform > 60000] = 0
        if self.padding is not None and self.padding != 0:
            img_transform = img_transform[:, self.padding:-self.padding, self.padding:-self.padding]
            seg_transform = seg_transform[:, self.padding:-self.padding, self.padding:-self.padding]
        return img_transform, seg_transform
```

## inference.py: checkpoint loading

```python
    """Strictly load official model_weights, state_dict, or a bare tensor state dict."""
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a state dictionary or contain model_weights/state_dict")
    state = checkpoint.get("model_weights", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint has no non-empty model state dictionary")
    cleaned = {}
    for key, value in state.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError("Model state must map parameter names to tensors")
        name = key[7:] if key.startswith("module.") else key
        if name in cleaned:
            raise ValueError("Checkpoint has duplicate parameter names after removing module. prefix")
        cleaned[name] = value
    model.load_state_dict(cleaned, strict=True)
```

## Train_and_Inference/config/SegNeuron.yaml (the supervised training config)

```yaml
NAME: 'SegNeuron'

MODEL:
  model_type: 'superhuman'  # Same-size prediction; disable the legacy MALA crop.
  pre_train: True
  pretrain_path: '/***/***'
  continue_train: False
  continue_path: '/***/***'

TRAIN:
  resume: False
  if_valid: True
  cache_path: './caches/'
  save_path: './models/'
  pad: 0
  loss_func: 'BCELoss'
  opt_type: 'adam'
  display_freq: 100
  total_iters: 400000
  warmup_iters: 0
  base_lr: 0.01
  end_lr: 0.0001
  save_freq: 2000
  valid_freq: 2000
  decay_iters: 200000
  weight_decay: ~
  power: 1.5
  batch_size: 8
  num_workers: 32
  if_cuda: True
  random_seed: 666
  min_valid_iter: 10000
  freq_mix_prob: 0.25
  spa_mix_prob: 0.25

DATA:
  min_noise_std: 0.01
  max_noise_std: 0.2
  min_kernel_size: 3
  max_kernel_size: 9
  min_sigma: 0
  max_sigma: 2
  data_folder: '/***/***'
  data_folder_val: '/***/***'
  start_slice: 0
  end_slice: 100
  val_start: 0
  val_end: 100
  predict_split: False
  if_ignore_bg: True
```
