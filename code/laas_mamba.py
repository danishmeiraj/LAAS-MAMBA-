#!/usr/bin/env python
"""
LAAS-Mamba: Lesion-Aware Adaptive Scan Mamba with Background-Invariant Training
for lab-to-field plant disease recognition (train on PlantVillage, test zero-shot on PlantDoc).

Designed for Kaggle (2 x T4). Pure PyTorch, no mamba_ssm compilation needed.
The SSM core is a Mamba-2 / SSD-style selective state space layer (input-dependent dt, B, C,
scalar decay per head) written in its quadratic "dual" form, which is cheap for ~200 tokens.

Usage (Kaggle, 2 GPUs):
  python laas_mamba.py --selftest
  torchrun --standalone --nproc_per_node=2 laas_mamba.py \
      --pv_root /kaggle/input/plantvillage-dataset --plantdoc_root /kaggle/input/<plantdoc> \
      --scan ours --bit --name ours

Contributions implemented
  1. LAAS  : each block predicts a token-level lesion-saliency map, sorts tokens
             (background -> healthy leaf -> lesion) and scans them in that order (asc + desc),
             next to ordinary spatial scans.
  2. Pseudo-lesion supervision from colour statistics (no manual masks).
  3. BIT   : background-invariant training = leaf cut-out + random affine + synthetic/real
             backgrounds, with prediction/feature consistency between clean and swapped views,
             plus saliency-weighted pooling.
  4. Rigorous protocol: leaf/near-duplicate grouped splits, PlantDoc zero-shot, background-swap
             test, corruption robustness, params/throughput.
"""
import os, re, sys, math, json, time, random, argparse
from datetime import timedelta
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from PIL import Image
import torchvision.transforms as T

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize(x):
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


def norm_key(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


# =====================================================================================
# 1. Data: discovery, leaf-grouped splits, PlantDoc mapping
# =====================================================================================
def find_class_root(root):
    """Locate the PlantVillage *colour* class folders (names like 'Apple___Apple_scab') anywhere under root.
    Works with Kaggle's /kaggle/input/datasets/<owner>/<slug>/... layout and with the older flat layout."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"'{root}' does not exist. Contents of /kaggle/input: "
                                f"{os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'n/a'}")
    cands, color = [], []
    for dp, dn, _ in os.walk(root):
        if sum('___' in d for d in dn) >= 10:
            cands.append(dp)
        elif os.path.basename(dp).lower() == 'color' and len(dn) >= 10:
            color.append(dp)
    cands = cands or color
    if not cands:
        raise FileNotFoundError(f"No PlantVillage class folders found under {root}. "
                                f"Did you add the PlantVillage dataset to the notebook (Add Input)?")
    pref = ([c for c in cands if os.path.basename(c).lower() == 'color']
            or [c for c in cands if not any(k in c.lower() for k in ('segmented', 'grayscale'))] or cands)
    print(f'[data] PlantVillage class root: {pref[0]}', flush=True)
    return pref[0]


def find_plantdoc_root(base='/kaggle/input'):
    """Return the folder holding PlantDoc class folders (train/test), or '' if not attached."""
    parents = []
    for dp, dn, _ in os.walk(base):
        if sum(norm_key(d) in PLANTDOC_MAP for d in dn) >= 5:
            parents.append(dp)
    if not parents:
        return ''
    return parents[0] if len(parents) == 1 else os.path.commonpath(parents)


def list_images(class_root):
    classes = sorted(d for d in os.listdir(class_root) if os.path.isdir(os.path.join(class_root, d)))
    items = []
    for ci, c in enumerate(classes):
        for f in sorted(os.listdir(os.path.join(class_root, c))):
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                items.append((os.path.join(class_root, c, f), ci))
    return classes, items


_AUG = re.compile(r'(_?(flip(lr|tb)|\d+deg|newgrsa|newpixel|final_masked))+$', re.I)


def base_name(path):
    return _AUG.sub('', os.path.splitext(os.path.basename(path))[0]).lower()


def _small_gray(path):
    try:
        bx = getattr(Image, 'Resampling', Image).BOX
        return np.asarray(Image.open(path).convert('L').resize((32, 32), bx), dtype=np.float32)
    except Exception:
        return np.zeros((32, 32), np.float32)


@torch.no_grad()
def near_dup_pairs(S, labels, dev, thr):
    """Pairs (i,j) inside the same class whose 32x32 grey thumbnails correlate > thr under any of the 8
    flips/rotations -> catches re-saved / flipped / rotated copies of the same leaf photo."""
    prep = lambda t: F.normalize(t.reshape(len(t), -1) - t.reshape(len(t), -1).mean(1, keepdim=True), dim=1)
    X, out = torch.from_numpy(S), []
    for c in np.unique(labels):
        ids = np.where(labels == c)[0]
        A = X[ids].to(dev)
        base = prep(A)
        for k in range(4):
            for fl in (False, True):
                v = torch.rot90(A, k, (1, 2))
                v = v.flip(2) if fl else v
                ii, jj = torch.nonzero((base @ prep(v).T) > thr, as_tuple=True)
                keep = ii != jj
                if keep.any():
                    out.append(np.stack([ids[ii[keep].cpu().numpy()], ids[jj[keep].cpu().numpy()]], 1))
    return np.concatenate(out) if out else np.zeros((0, 2), int)


def build_splits(args, out_dir):
    cache = os.path.join(out_dir, 'splits.json')
    if not os.path.exists(cache) and args.prev_dir and os.path.exists(os.path.join(args.prev_dir, 'splits.json')):
        import shutil
        shutil.copy(os.path.join(args.prev_dir, 'splits.json'), cache)
    if os.path.exists(cache):
        return json.load(open(cache))
    from sklearn.model_selection import StratifiedGroupKFold
    root = find_class_root(args.pv_root)
    classes, items = list_images(root)
    paths = [p for p, _ in items]
    labels = np.array([y for _, y in items])
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    first = {}
    for i, p in enumerate(paths):
        k = base_name(p)
        union(i, first[k]) if k in first else first.__setitem__(k, i)
    if args.hash_groups:
        print('[data] loading thumbnails to find near-duplicates ...', flush=True)
        with Pool(min(8, os.cpu_count() or 1)) as pool:
            S = np.stack(pool.map(_small_gray, paths, chunksize=256))
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pairs = near_dup_pairs(S, labels, dev, args.dup_thr)
        for a_, b_ in pairs:
            union(int(a_), int(b_))
        print(f'[data] near-duplicate pairs (corr>{args.dup_thr}): {len(pairs)}', flush=True)
    groups = np.array([find(i) for i in range(n)])
    _, counts = np.unique(groups, return_counts=True)
    print(f'[data] {n} images, {len(counts)} groups, largest group = {counts.max()}', flush=True)
    if counts.max() > 300:
        print('[data] WARNING: very large group -> raise --dup_thr (e.g. 0.998) to avoid over-merging', flush=True)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    folds = [te for _, te in sgkf.split(paths, labels, groups)]
    sp = {'classes': classes,
          'test': [items[i] for i in folds[0]],
          'val': [items[i] for i in folds[1]],
          'train': [items[i] for i in np.concatenate(folds[2:])]}
    json.dump(sp, open(cache, 'w'))
    return sp


# PlantDoc folder (normalised) -> tokens identifying the unique PlantVillage class
PLANTDOC_MAP = {
    'applescableaf': ('apple', 'scab'), 'appleleaf': ('apple', 'healthy'),
    'applerustleaf': ('apple', 'cedar'), 'bellpepperleaf': ('pepper', 'healthy'),
    'bellpepperleafspot': ('pepper', 'bacterial'), 'blueberryleaf': ('blueberry',),
    'cherryleaf': ('cherry', 'healthy'), 'corngrayleafspot': ('corn', 'gray'),
    'cornleafblight': ('corn', 'northern'), 'cornrustleaf': ('corn', 'rust'),
    'peachleaf': ('peach', 'healthy'), 'potatoleafearlyblight': ('potato', 'early'),
    'potatoleaflateblight': ('potato', 'late'), 'raspberryleaf': ('raspberry',),
    'soyabeanleaf': ('soybean',), 'soybeanleaf': ('soybean',),
    'squashpowderymildewleaf': ('squash',), 'strawberryleaf': ('strawberry', 'healthy'),
    'tomatoearlyblightleaf': ('tomato', 'early'), 'tomatoseptorialeafspot': ('tomato', 'septoria'),
    'tomatoleaf': ('tomato', 'healthy'), 'tomatoleafbacterialspot': ('tomato', 'bacterial'),
    'tomatoleaflateblight': ('tomato', 'late'), 'tomatoleafmosaicvirus': ('tomato', 'mosaic'),
    'tomatoleafyellowvirus': ('tomato', 'yellow'), 'tomatomoldleaf': ('tomato', 'mold'),
    'tomatotwospottedspidermitesleaf': ('tomato', 'spider'),
    'grapeleaf': ('grape', 'healthy'), 'grapeleafblackrot': ('grape', 'blackrot'),
}


def build_plantdoc(root, pv_classes):
    if not root or not os.path.isdir(root):
        root = find_plantdoc_root('/kaggle/input')
    if not root:
        print('[plantdoc] NOT FOUND -> zero-shot evaluation skipped (attach a PlantDoc dataset to enable it)')
        return []
    pvn = [norm_key(c) for c in pv_classes]
    items, found, skipped = [], set(), set()
    for dp, _, fn in os.walk(root):
        k = norm_key(os.path.basename(dp))
        imgs = [f for f in fn if os.path.splitext(f)[1].lower() in IMG_EXT]
        if not imgs:
            continue
        if k not in PLANTDOC_MAP:
            skipped.add(os.path.basename(dp))
            continue
        hits = [i for i, nm in enumerate(pvn) if all(t in nm for t in PLANTDOC_MAP[k])]
        if len(hits) != 1:
            print(f'[plantdoc] WARNING ambiguous mapping for {dp}: {hits}')
            continue
        found.add(hits[0])
        items += [(os.path.join(dp, f), hits[0]) for f in imgs]
    print(f'[plantdoc] {len(items)} images, {len(found)} mapped classes; unmatched folders: {sorted(skipped)[:8]}')
    return items


class ImgDS(Dataset):
    def __init__(self, items, size, train):
        self.items, self.size = items, size
        self.tf = (T.Compose([T.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.85, 1.18)),
                              T.RandomHorizontalFlip(), T.RandomVerticalFlip()])
                   if train else T.Resize((size, size)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        im = Image.open(p)
        im.draft('RGB', (self.size * 2, self.size * 2))
        im = self.tf(im.convert('RGB'))
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1), y


def load_bg_bank(folder, size, dev, n=1500):
    fs = [os.path.join(dp, f) for dp, _, fn in os.walk(folder) for f in fn
          if os.path.splitext(f)[1].lower() in IMG_EXT]
    random.shuffle(fs)
    arr = []
    for f in fs[:n]:
        im = Image.open(f)
        im.draft('RGB', (size * 2, size * 2))
        arr.append(torch.from_numpy(np.asarray(im.convert('RGB').resize((size, size))).copy()).permute(2, 0, 1))
    print(f'[bg] loaded {len(arr)} background images')
    return torch.stack(arr).to(dev)


# =====================================================================================
# 2. GPU-side pseudo-lesion masks, background swap, photometric aug
# =====================================================================================
@torch.no_grad()
def pseudo_masks(x, border=8, tau_f=0.12, tau_g=0.05, dark=0.18):
    """x (B,3,H,W) in [0,1]. Returns leaf mask fg and lesion-candidate mask les, (B,1,H,W) in {0,1}.
    fg  : colour distance to the border-median background colour.
    les : leaf pixels that are not 'green' (g'-r' small) or very dark.  (heuristic, tune tau_*)"""
    B = x.shape[0]
    edge = torch.cat([x[:, :, :border].flatten(2), x[:, :, -border:].flatten(2),
                      x[:, :, :, :border].flatten(2), x[:, :, :, -border:].flatten(2)], 2)
    bg = edge.median(2).values.view(B, 3, 1, 1)
    fg = ((x - bg).pow(2).sum(1, keepdim=True).sqrt() > tau_f).float()
    fg = (F.avg_pool2d(fg, 7, 1, 3) > 0.5).float()
    gr = (x[:, 1:2] - x[:, 0:1]) / (x.sum(1, keepdim=True) + 1e-6)
    les = (((gr < tau_g) | (x.mean(1, keepdim=True) < dark)).float()) * fg
    les = (F.avg_pool2d(les, 5, 1, 2) > 0.5).float() * fg
    return fg, les


def tok_target(fg, les, patch):
    """token relevance target: background 0, healthy leaf 0.5, lesion 1.0  -> (B,L)"""
    return F.avg_pool2d(0.5 * fg + 0.5 * les, patch).flatten(1)


_PAL = torch.tensor([[.36, .26, .16], [.22, .40, .16], [.50, .50, .50],
                     [.62, .52, .32], [.15, .15, .15], [.55, .45, .35]])


def synth_bg(B, H, W, dev, bank=None):
    pal = _PAL.to(dev)
    base = pal[torch.randint(0, len(pal), (B,), device=dev)]
    rnd = torch.rand(B, 3, device=dev)
    m = (torch.rand(B, 1, device=dev) < 0.3).float()
    base = (1 - m) * base + m * rnd
    up = lambda t: F.interpolate(t, (H, W), mode='bilinear', align_corners=False)
    bg = (base.view(B, 3, 1, 1) + up(torch.randn(B, 3, 4, 4, device=dev) * 0.12)
          + up(torch.randn(B, 3, 24, 24, device=dev) * 0.06) + torch.randn(B, 3, H, W, device=dev) * 0.03)
    if bank is not None:
        r = bank[torch.randint(0, len(bank), (B,), device=dev)].float() / 255
        flip = torch.rand(B, device=dev) < 0.5
        r = torch.where(flip.view(B, 1, 1, 1), r.flip(3), r)
        use = (torch.rand(B, 1, 1, 1, device=dev) < 0.6).float()
        bg = use * r + (1 - use) * bg
    return bg.clamp(0, 1)


@torch.no_grad()
def bg_swap(x, fg, les, bank=None, scale=(0.55, 1.05), rot=30., trans=0.15):
    """Cut the leaf out, random-affine it (scale<1 mimics distant field shots), paste on new background.
    Returns swapped image, warped fg, warped les."""
    B, _, H, W = x.shape
    dev = x.device
    ang = (torch.rand(B, device=dev) * 2 - 1) * rot * math.pi / 180
    s = torch.empty(B, device=dev).uniform_(*scale)
    th = torch.zeros(B, 2, 3, device=dev)
    th[:, 0, 0], th[:, 0, 1] = torch.cos(ang) / s, -torch.sin(ang) / s
    th[:, 1, 0], th[:, 1, 1] = torch.sin(ang) / s, torch.cos(ang) / s
    th[:, 0, 2] = (torch.rand(B, device=dev) * 2 - 1) * trans
    th[:, 1, 2] = (torch.rand(B, device=dev) * 2 - 1) * trans
    grid = F.affine_grid(th, x.shape, align_corners=False)
    w = F.grid_sample(torch.cat([x * fg, fg, les], 1), grid, mode='bilinear',
                      padding_mode='zeros', align_corners=False)
    leaf, a, l2 = w[:, :3], w[:, 3:4], w[:, 4:5]
    out = leaf + (1 - a) * synth_bg(B, H, W, dev, bank)
    ok = (fg.mean((1, 2, 3)) > 0.03).view(B, 1, 1, 1)           # mask failed -> keep original
    return (torch.where(ok, out, x).clamp(0, 1), torch.where(ok, a, fg), torch.where(ok, l2, les))


@torch.no_grad()
def photometric(x):
    B, dev = x.shape[0], x.device
    r = lambda a: 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * a
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * r(.25) + m) * r(.25)
    g = x.mean(1, keepdim=True)
    x = (x - g) * r(.25) + g
    x = x + torch.randn_like(x) * 0.02 * (torch.rand(B, 1, 1, 1, device=dev) < 0.3).float()
    bl = (torch.rand(B, 1, 1, 1, device=dev) < 0.2)
    x = torch.where(bl, F.avg_pool2d(x, 5, 1, 2, count_include_pad=False), x)
    return x.clamp(0, 1)


# =====================================================================================
# 3. Model
# =====================================================================================
def ssd_scan(x, dt, A, Bm, Cm, D):
    """Selective SSM (Mamba-2 / SSD quadratic form).
    x (B,K,L,H,P)  dt (B,K,L,H)  A (K,H)<0  Bm,Cm (B,K,L,N)  D (K,H)  ->  (B,K,L,H,P)
    h_t = exp(dt_t A) h_{t-1} + dt_t B_t x_t ;  y_t = C_t h_t + D x_t"""
    cs = torch.cumsum(dt * A[None, :, None, :], 2).permute(0, 1, 3, 2)      # (B,K,H,L)
    seg = cs.unsqueeze(-1) - cs.unsqueeze(-2)                                # [t,s] = cs_t - cs_s
    L = seg.shape[-1]
    mask = torch.tril(torch.ones(L, L, dtype=torch.bool, device=seg.device))
    decay = torch.exp(seg.masked_fill(~mask, float('-inf')))                 # (B,K,H,L,L)  <= 1
    G = torch.einsum('bkln,bksn->bkls', Cm, Bm)                              # C_t . B_s
    y = torch.einsum('bkhls,bkshp->bklhp', decay * G.unsqueeze(2), x * dt.unsqueeze(-1))
    return y + x * D[None, :, None, :, None]


def ssd_scan_ref(x, dt, A, Bm, Cm, D):
    """naive sequential reference (for the self-test)"""
    B, K, L, H, P = x.shape
    h = x.new_zeros(B, K, H, P, Bm.shape[-1])
    ys = []
    for t in range(L):
        a = torch.exp(dt[:, :, t] * A[None])
        h = a[..., None, None] * h + (dt[:, :, t, :, None, None] * x[:, :, t, :, :, None]) * Bm[:, :, t, None, None, :]
        ys.append((h * Cm[:, :, t, None, None, :]).sum(-1) + D[None, :, :, None] * x[:, :, t])
    return torch.stack(ys, 2)


SCANS = {'cross4': ['row', 'rowr', 'col', 'colr'],      # VMamba-style spatial baseline
         'bi2': ['row', 'rowr'],
         'laas2': ['les', 'lesr'],                      # adaptive only
         'ours': ['row', 'rowr', 'les', 'lesr'],        # same compute as cross4
         'ours6': ['row', 'rowr', 'col', 'colr', 'les', 'lesr']}


class DropPath(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = (torch.rand(x.shape[0], *([1] * (x.ndim - 1)), device=x.device) > self.p).to(x.dtype)
        return x * keep / (1 - self.p)


class LAASBlock(nn.Module):
    def __init__(self, dim, grid, scans, d_state=16, expand=2, headdim=32, use_sal=True, levels=4):
        super().__init__()
        self.d_in, self.N, self.P = expand * dim, d_state, headdim
        self.H = self.d_in // headdim
        self.grid, self.scans, self.levels, K = grid, scans, levels, len(scans)
        self.norm = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, 2 * self.d_in, bias=False)
        self.conv = nn.Conv2d(self.d_in, self.d_in, 3, padding=1, groups=self.d_in)
        self.x_proj = nn.Parameter(torch.randn(K, self.d_in, self.H + 2 * d_state) * self.d_in ** -0.5)
        dt = torch.exp(torch.rand(K, self.H) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))        # inverse softplus
        self.A_log = nn.Parameter(torch.log(torch.empty(K, self.H).uniform_(1, 16)))
        self.D = nn.Parameter(torch.ones(K, self.H))
        self.out_norm = nn.LayerNorm(self.d_in)
        self.out_proj = nn.Linear(self.d_in, dim, bias=False)
        self.sal = (nn.Sequential(nn.Conv2d(dim, max(dim // 4, 8), 3, padding=1), nn.GELU(),
                                  nn.Conv2d(max(dim // 4, 8), 1, 1)) if use_sal else None)
        L = grid * grid
        ar = torch.arange(L)
        col = ar.view(grid, grid).t().reshape(-1)
        st = {'row': ar, 'rowr': ar.flip(0), 'col': col, 'colr': col.flip(0)}
        for k in scans:
            if k in st:
                self.register_buffer(f'idx_{k}', st[k].clone(), persistent=False)
        self.register_buffer('pos', ar.float(), persistent=False)

    def build_idx(self, sal, B):
        L = self.grid ** 2
        out = []
        for k in self.scans:
            if k in ('les', 'lesr'):
                lvl = torch.floor(sal.detach().clamp(0, 0.999) * self.levels)   # tier: bg < leaf < lesion
                order = (lvl * L + self.pos[None]).argsort(1)                     # raster order inside a tier
                out.append(order if k == 'les' else order.flip(1))
            else:
                out.append(getattr(self, f'idx_{k}')[None].expand(B, -1))
        return torch.stack(out, 1)                                               # (B,K,L)

    def forward(self, x):
        B, L, C = x.shape
        g = self.grid
        h = self.norm(x)
        sal = None
        if self.sal is not None:
            sal = torch.sigmoid(self.sal(h.transpose(1, 2).reshape(B, C, g, g))).flatten(1)   # (B,L)
        u, z = self.in_proj(h).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2).reshape(B, -1, g, g)).flatten(2).transpose(1, 2))
        if sal is not None:
            u = u * (1 + sal.unsqueeze(-1))        # gives task loss a gradient path into the saliency head
        idx = self.build_idx(sal, B)
        K, D = idx.shape[1], u.shape[-1]
        ix = idx.unsqueeze(-1).expand(B, K, L, D)
        uk = torch.gather(u.unsqueeze(1).expand(B, K, L, D), 2, ix)
        with torch.autocast(device_type=x.device.type, enabled=False):
            uk = uk.float()
            pr = torch.einsum('bkld,kdm->bklm', uk, self.x_proj.float())
            dtr, Bm, Cm = pr.split([self.H, self.N, self.N], -1)
            dt = F.softplus(dtr + self.dt_bias[None, :, None, :].float())
            y = ssd_scan(uk.view(B, K, L, self.H, self.P), dt, -torch.exp(self.A_log.float()),
                         Bm, Cm, self.D.float()).reshape(B, K, L, D)
        y = torch.zeros_like(y).scatter(2, ix, y).sum(1)                         # back to spatial order
        y = self.out_proj(self.out_norm(y) * F.silu(z))
        return y, (sal if sal is not None else x.new_zeros(0))


class LAASMamba(nn.Module):
    def __init__(self, num_classes, img=224, patch=16, dim=128, depth=8, scan='ours', d_state=16,
                 expand=2, headdim=32, drop_path=0.1, wpool=True, use_sal=None, use_ckpt=True):
        super().__init__()
        self.patch, self.grid, self.use_ckpt = patch, img // patch, use_ckpt
        scans = SCANS[scan]
        self.has_sal = any(k in ('les', 'lesr') for k in scans) if use_sal is None else use_sal
        self.wpool = wpool and self.has_sal
        self.embed = nn.Conv2d(3, dim, patch, patch)
        self.pos = nn.Parameter(torch.zeros(1, self.grid ** 2, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([LAASBlock(dim, self.grid, scans, d_state, expand, headdim, self.has_sal)
                                     for _ in range(depth)])
        self.dps = nn.ModuleList([DropPath(drop_path * i / max(depth - 1, 1)) for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        z = self.embed(x).flatten(2).transpose(1, 2) + self.pos
        sals = []
        for blk, dp in zip(self.blocks, self.dps):
            if self.use_ckpt and self.training:
                dz, s = checkpoint(blk, z, use_reentrant=False)
            else:
                dz, s = blk(z)
            z = z + dp(dz)
            sals.append(s)
        z = self.norm(z)
        if self.wpool:
            w = sals[-1]
            feat = (z * w.unsqueeze(-1)).sum(1) / (w.sum(1, keepdim=True) + 1e-4)
        else:
            feat = z.mean(1)
        return self.head(feat), feat, (torch.stack(sals, 1) if self.has_sal else None)


class TimmWrap(nn.Module):
    """baselines (resnet50, convnext_tiny, vit_small_patch16_224, ...) trained with the same pipeline"""
    def __init__(self, name, num_classes, pretrained=False, img=224):
        super().__init__()
        import timm
        for kw in ({'drop_path_rate': 0.1, 'img_size': img}, {'drop_path_rate': 0.1}, {}):
            try:
                self.m = timm.create_model(name, pretrained=pretrained, num_classes=num_classes, **kw)
                break
            except TypeError:
                continue

    def forward(self, x):
        f = self.m.forward_head(self.m.forward_features(x), pre_logits=True)
        return self.m.get_classifier()(f), f, None


def build_model(args, nc):
    if args.arch.startswith('timm:'):
        return TimmWrap(args.arch[5:], nc, args.pretrained, args.img)
    use_sal = {'auto': None, '0': False, '1': True}[args.sal]
    return LAASMamba(nc, args.img, args.patch, args.dim, args.depth, args.scan, args.dstate,
                     2, 32, args.drop_path, args.wpool, use_sal, args.ckpt)


# =====================================================================================
# 4. Loss, evaluation, train
# =====================================================================================
def compute_loss(logits, feat, sal, y, tgt, args):
    logits = logits.float()
    parts = {}
    yy = torch.cat([y, y]) if args.bit else y
    loss = parts['ce'] = F.cross_entropy(logits, yy, label_smoothing=0.1)
    if args.bit:
        l1, l2 = logits.chunk(2)
        p1, p2 = F.log_softmax(l1, -1), F.log_softmax(l2, -1)
        kl = 0.5 * (F.kl_div(p1, p2, log_target=True, reduction='batchmean') +
                    F.kl_div(p2, p1, log_target=True, reduction='batchmean'))
        f1, f2 = feat.float().chunk(2)
        fc = (1 - F.cosine_similarity(f1, f2, dim=-1)).mean()
        parts['kl'], parts['feat'] = kl, fc
        loss = loss + args.w_cons * kl + args.w_feat * fc
    if sal is not None and args.w_sal > 0:
        s = sal.float()
        sl = F.mse_loss(s, tgt[:, None, :].expand_as(s))
        parts['sal'] = sl
        loss = loss + args.w_sal * sl
    return loss, parts


def corrupt(x, kind, sev):
    if kind == 'noise':
        return (x + torch.randn_like(x) * [0.05, 0.10][sev]).clamp(0, 1)
    if kind == 'blur':
        return F.avg_pool2d(x, [5, 9][sev], 1, [2, 4][sev], count_include_pad=False)
    if kind == 'dark':
        return (x * [0.6, 0.35][sev]).clamp(0, 1)
    if kind == 'bright':
        return (x * [1.5, 1.9][sev]).clamp(0, 1)
    return x


@torch.no_grad()
def evaluate(core, loader, dev, mode='clean', subset=None, bank=None, kind=None, sev=0):
    from sklearn.metrics import f1_score, accuracy_score
    core.eval()
    if mode == 'bg':
        torch.manual_seed(1234)
    P, G = [], []
    amp = dev.type == 'cuda'
    for xb, yb in loader:
        x = xb.to(dev, non_blocking=True).float() / 255
        if mode == 'bg':
            fg, les = pseudo_masks(x)
            x, _, _ = bg_swap(x, fg, les, bank)
        if mode == 'corrupt':
            x = corrupt(x, kind, sev)
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            lg = core(normalize(x))[0].float()
        if subset is not None:
            m = torch.full_like(lg, float('-inf'))
            m[:, subset] = 0
            lg = lg + m
        P.append(lg.argmax(-1).cpu())
        G.append(yb)
    P, G = torch.cat(P).numpy(), torch.cat(G).numpy()
    return {'acc': float(accuracy_score(G, P)), 'f1': float(f1_score(G, P, average='macro'))}


@torch.no_grad()
def throughput(core, args, dev, bs=64, iters=15):
    core.eval()
    x = torch.randn(bs, 3, args.img, args.img, device=dev)
    amp = dev.type == 'cuda'
    for i in range(iters + 3):
        if i == 3:
            if amp:
                torch.cuda.synchronize()
            t0 = time.time()
        with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
            core(x)
    if amp:
        torch.cuda.synchronize()
    return bs * iters / (time.time() - t0)


@torch.no_grad()
def visualize(core, items, dev, args, path, bank=None, n=6):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not getattr(core, 'has_sal', False) or not items:
        return
    core.eval()
    ds = ImgDS(random.sample(items, min(n, len(items))), args.img, False)
    x = torch.stack([ds[i][0] for i in range(len(ds))]).to(dev).float() / 255
    fg, les = pseudo_masks(x)
    x2, fg2, les2 = bg_swap(x, fg, les, bank)
    g = core.grid
    up = lambda s: F.interpolate(s.view(-1, 1, g, g), size=args.img, mode='nearest')[:, 0].cpu()
    s1, s2 = core(normalize(x))[2][:, -1], core(normalize(x2))[2][:, -1]
    t1 = F.interpolate((0.5 * fg + 0.5 * les), size=args.img)[:, 0].cpu()
    cols = [('input', x), ('pseudo target', t1), ('pred. saliency', up(s1)), ('bg-swapped', x2), ('pred. saliency', up(s2))]
    fig, ax = plt.subplots(len(x), 5, figsize=(12, 2.4 * len(x)))
    ax = np.atleast_2d(ax)
    for r in range(len(x)):
        for c, (nm, im) in enumerate(cols):
            a = ax[r, c]
            a.axis('off')
            if im.ndim == 4:
                a.imshow(im[r].permute(1, 2, 0).cpu().numpy())
            else:
                a.imshow(im[r].numpy(), cmap='jet', vmin=0, vmax=1)
            if r == 0:
                a.set_title(nm, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def full_eval(core, args, dev, sp, pd_items, bank, out_dir):
    kw = dict(batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_dl = DataLoader(ImgDS(sp['test'], args.img, False), **kw)
    R = {'arch': args.arch, 'scan': args.scan, 'bit': args.bit, 'name': args.name,
         'params_M': sum(p.numel() for p in core.parameters()) / 1e6}
    R['pv_test'] = evaluate(core, test_dl, dev)
    R['pv_test_bgswap'] = evaluate(core, test_dl, dev, mode='bg', bank=bank)
    R['bg_reliance_drop'] = R['pv_test']['acc'] - R['pv_test_bgswap']['acc']
    if pd_items:
        sub = sorted({y for _, y in pd_items})
        R['plantdoc_zeroshot'] = evaluate(core, DataLoader(ImgDS(pd_items, args.img, False), **kw), dev, subset=sub)
    if args.robust:
        R['robust'] = {f'{k}{s}': evaluate(core, test_dl, dev, mode='corrupt', kind=k, sev=s)['acc']
                       for k in ('noise', 'blur', 'dark', 'bright') for s in (0, 1)}
    R['throughput_img_s'] = throughput(core, args, dev)
    json.dump(R, open(os.path.join(out_dir, 'results.json'), 'w'), indent=2)
    print(json.dumps(R, indent=2))
    if args.viz:
        visualize(core, sp['test'], dev, args, os.path.join(out_dir, 'viz_pv.png'), bank)
        visualize(core, pd_items, dev, args, os.path.join(out_dir, 'viz_plantdoc.png'), bank)


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def setup_dist():
    if int(os.environ.get('WORLD_SIZE', 1)) > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=60))
        lr = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(lr)
        return True, dist.get_rank(), dist.get_world_size(), torch.device('cuda', lr)
    return False, 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(args):
    ddp, rank, world, dev = setup_dist()
    random.seed(args.seed + rank); np.random.seed(args.seed + rank); torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True
    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)
    if rank == 0:
        build_splits(args, args.out)
    if ddp:
        barrier()
    sp = json.load(open(os.path.join(args.out, 'splits.json')))
    classes = sp['classes']
    if args.max_train:
        sp['train'] = sp['train'][:args.max_train]
    pd_items = build_plantdoc(args.plantdoc_root, classes) if (args.plantdoc_root and rank == 0) else []
    bank = load_bg_bank(args.bg_dir, args.img, dev) if args.bg_dir else None

    model = build_model(args, len(classes)).to(dev)
    core = model
    if rank == 0:
        print(f'[model] {args.arch} scan={args.scan} params={sum(p.numel() for p in model.parameters())/1e6:.2f}M')
    if ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dev.index], broadcast_buffers=False)

    tr_ds = ImgDS(sp['train'], args.img, True)
    sampler = torch.utils.data.distributed.DistributedSampler(tr_ds, world, rank, shuffle=True, drop_last=True) if ddp else None
    tr_dl = DataLoader(tr_ds, batch_size=args.bs, sampler=sampler, shuffle=sampler is None, drop_last=True,
                       num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    val_dl = DataLoader(ImgDS(sp['val'], args.img, False), batch_size=64, num_workers=args.workers, pin_memory=True)

    decay, no_decay = [], []
    for n, p in core.named_parameters():
        (no_decay if p.ndim < 2 or n.endswith(('pos', 'A_log', '.D', 'dt_bias')) else decay).append(p)
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.wd}, {'params': no_decay, 'weight_decay': 0}], lr=args.lr)
    amp = dev.type == 'cuda'
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    steps_ep = len(tr_dl)
    total, warm = args.epochs * steps_ep, args.warmup * steps_ep
    lr_at = lambda s: args.lr * (s / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))

    start, best, stopped, t_start = 0, 0.0, False, time.time()
    last = os.path.join(out, 'last.pt')
    if args.resume and not os.path.exists(last) and args.prev_dir and rank == 0:
        import shutil
        for fn in ('last.pt', 'best.pt', 'log.jsonl'):
            src = os.path.join(args.prev_dir, args.name, fn)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(out, fn))
                print(f'[resume] copied {src}', flush=True)
    barrier()
    if args.resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev)
        core.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        start, best = ck['epoch'] + 1, ck['best']
        if rank == 0:
            print(f'[resume] from epoch {start}')

    if not args.eval_only:
        for ep in range(start, args.epochs):
            model.train()
            if sampler is not None:
                sampler.set_epoch(ep)
            t0, agg = time.time(), {}
            for it, (xb, yb) in enumerate(tr_dl):
                step = ep * steps_ep + it
                for gp in opt.param_groups:
                    gp['lr'] = lr_at(step)
                x = xb.to(dev, non_blocking=True).float() / 255
                y = yb.to(dev, non_blocking=True)
                with torch.no_grad():
                    fg, les = pseudo_masks(x)
                    t1 = tok_target(fg, les, args.patch)
                    v1 = normalize(photometric(x))
                    if args.bit:
                        x2, fg2, les2 = bg_swap(x, fg, les, bank)
                        inp = torch.cat([v1, normalize(photometric(x2))])
                        tgt = torch.cat([t1, tok_target(fg2, les2, args.patch)])
                    else:
                        inp, tgt = v1, t1
                with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=amp):
                    logits, feat, sal = model(inp)
                loss, parts = compute_loss(logits, feat, sal, y, tgt, args)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                for k, v in parts.items():
                    agg[k] = agg.get(k, 0) + v.detach()
                if rank == 0 and it % 100 == 0:
                    print(f'ep{ep} it{it}/{steps_ep} loss={float(loss):.3f} ' +
                          ' '.join(f'{k}={float(v):.3f}' for k, v in parts.items()) +
                          f' {(it+1)*args.bs*world/(time.time()-t0):.0f} img/s', flush=True)
            if rank == 0:
                va = evaluate(core, val_dl, dev)
                rec = {'epoch': ep, 'val_acc': va['acc'], 'val_f1': va['f1'], 'time': time.time() - t0,
                       **{k: float(v) / steps_ep for k, v in agg.items()}}
                print('[epoch]', rec, flush=True)
                with open(os.path.join(out, 'log.jsonl'), 'a') as f:
                    f.write(json.dumps(rec) + '\n')
                if va['acc'] >= best:        # model selection on PlantVillage-val ONLY (never on PlantDoc)
                    best = va['acc']
                    torch.save({'model': core.state_dict(), 'args': vars(args)}, os.path.join(out, 'best.pt'))
                torch.save({'model': core.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                            'epoch': ep, 'best': best}, last)
            barrier()
            if args.time_budget_h > 0:
                flag = torch.tensor([float(time.time() - t_start > args.time_budget_h * 3600)], device=dev)
                if ddp:
                    dist.broadcast(flag, 0)
                if flag.item() > 0:
                    stopped = True
                    if rank == 0:
                        print(f'[budget] time budget reached after epoch {ep}; checkpoint saved. '
                              f'Re-run the same command with --resume to continue.', flush=True)
                    break

    if stopped:
        if ddp:
            dist.destroy_process_group()
        return
    if rank == 0:
        bp = args.ckpt_path or os.path.join(out, 'best.pt')
        core.load_state_dict(torch.load(bp, map_location=dev)['model'])
        full_eval(core, args, dev, sp, pd_items, bank, out)
    if ddp:
        barrier()
        dist.destroy_process_group()


# =====================================================================================
# 5. Self-test (run this first on Kaggle; needs no dataset)
# =====================================================================================
def selftest():
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    # (a) SSD quadratic form == sequential recurrence
    B, K, L, H, P, N = 2, 3, 37, 4, 8, 6
    x = torch.randn(B, K, L, H, P)
    dt = F.softplus(torch.randn(B, K, L, H) - 2)
    A = -torch.rand(K, H) * 8 - 0.5
    Bm, Cm, D = torch.randn(B, K, L, N), torch.randn(B, K, L, N), torch.randn(K, H)
    err = (ssd_scan(x, dt, A, Bm, Cm, D) - ssd_scan_ref(x, dt, A, Bm, Cm, D)).abs().max().item()
    print(f'[1] scan vs recurrence max err = {err:.2e}'); assert err < 1e-3
    # (b) causality: changing the last scanned token must not change earlier outputs
    x2 = x.clone(); x2[:, :, -1] += 1
    d = (ssd_scan(x, dt, A, Bm, Cm, D) - ssd_scan(x2, dt, A, Bm, Cm, D)).abs()[:, :, :-1].max().item()
    print(f'[2] causality leak = {d:.2e}'); assert d < 1e-6
    # (c) pseudo masks / bg swap on a synthetic leaf
    img = torch.full((2, 3, 224, 224), 0.5)
    img[:, :, 60:170, 50:180] = torch.tensor([.2, .5, .15]).view(1, 3, 1, 1)
    img[:, :, 100:120, 100:130] = torch.tensor([.45, .3, .15]).view(1, 3, 1, 1)
    fg, les = pseudo_masks(img)
    print(f'[3] fg area={fg.mean():.3f} (exp ~0.28)  lesion area={les.mean():.3f} (exp ~0.012)')
    s, a, l2 = bg_swap(img, fg, les)
    assert s.shape == img.shape and a.shape == fg.shape
    # (d) every scan preset: forward/backward, all params receive gradients, adaptive idx are permutations
    for sc in SCANS:
        m = LAASMamba(5, 64, 16, 64, 2, sc, 8, drop_path=0.1, use_ckpt=True).to(dev).train()
        xin = torch.randn(4, 3, 64, 64, device=dev)
        lg, ft, sl = m(xin)
        loss = lg.logsumexp(-1).mean() + (sl.mean() if sl is not None else 0)
        loss.backward()
        miss = [n for n, p in m.named_parameters() if p.grad is None]
        blk = m.blocks[0]
        sal = torch.rand(4, 16, device=dev)
        idx = blk.build_idx(sal, 4) if sl is not None or blk.sal is None else None
        if blk.sal is not None:
            ok = all((i.sort().values == torch.arange(16, device=dev)).all() for r in idx for i in r)
            assert ok, 'scan index is not a permutation'
        print(f'[4] {sc:7s} out={tuple(lg.shape)} params={sum(p.numel() for p in m.parameters())/1e6:.2f}M no-grad-params={miss}')
        assert not miss
    # (e) near-duplicate detector finds flipped / rotated copies, ignores unrelated images
    S = np.random.rand(6, 32, 32).astype(np.float32)
    S[4] = S[1][:, ::-1]
    S[5] = np.rot90(S[2])
    found = {tuple(sorted(p)) for p in near_dup_pairs(S, np.zeros(6, int), dev, 0.995).tolist()}
    print(f'[5] near-dup pairs found = {sorted(found)}'); assert found == {(1, 4), (2, 5)}
    # (f) dataset discovery on a fake Kaggle tree
    import tempfile
    tmp = tempfile.mkdtemp()
    for nm in ('color', 'segmented'):
        for i in range(12):
            os.makedirs(os.path.join(tmp, 'datasets', 'u', 'plantvillage', nm, f'Crop{i}___healthy'))
    os.makedirs(os.path.join(tmp, 'datasets', 'u', 'plantdoc', 'train', 'Tomato leaf'))
    for nm in list(PLANTDOC_MAP)[:6]:
        os.makedirs(os.path.join(tmp, 'datasets', 'u', 'plantdoc', 'train', nm))
    r = find_class_root(tmp)
    assert r.endswith(os.path.join('plantvillage', 'color')), r
    assert find_plantdoc_root(tmp).endswith('train')
    print('[6] dataset discovery OK')
    print('SELFTEST PASSED')


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--selftest', action='store_true')
    p.add_argument('--pv_root', default='/kaggle/input', help='any folder above the PlantVillage colour images')
    p.add_argument('--plantdoc_root', default='')
    p.add_argument('--bg_dir', default='', help='optional folder of NON-target background photos')
    p.add_argument('--out', default='/kaggle/working/runs')
    p.add_argument('--name', default='ours')
    p.add_argument('--arch', default='mamba', help="'mamba' or 'timm:<name>' e.g. timm:resnet50")
    p.add_argument('--pretrained', action='store_true')
    p.add_argument('--scan', default='ours', choices=list(SCANS))
    p.add_argument('--sal', default='auto', choices=['auto', '0', '1'])
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--dstate', type=int, default=16)
    p.add_argument('--img', type=int, default=224)
    p.add_argument('--patch', type=int, default=16)
    p.add_argument('--drop_path', type=float, default=0.1)
    p.add_argument('--wpool', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--ckpt', action=argparse.BooleanOptionalAction, default=True, help='gradient checkpointing')
    p.add_argument('--bit', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--w_cons', type=float, default=1.0)
    p.add_argument('--w_feat', type=float, default=0.5)
    p.add_argument('--w_sal', type=float, default=2.0)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--bs', type=int, default=32, help='per GPU (x2 views when --bit)')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=0.05)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--hash_groups', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--dup_thr', type=float, default=0.995)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--prev_dir', default='', help="'runs' folder of a previous session/commit to resume from")
    p.add_argument('--time_budget_h', type=float, default=0,
                   help='stop cleanly after the epoch in which this many hours have elapsed (0 = off)')
    p.add_argument('--eval_only', action='store_true')
    p.add_argument('--ckpt_path', default='')
    p.add_argument('--robust', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--viz', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--max_train', type=int, default=0, help='debug: limit train images')
    return p.parse_args()


if __name__ == '__main__':
    a = get_args()
    selftest() if a.selftest else train(a)
