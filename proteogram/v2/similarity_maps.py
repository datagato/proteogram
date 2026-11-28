"""Exact decomposition of proteogram similarity into image regions and residues.

Search embeds a proteogram as the global average of ResNet18's last feature map
(layer4: C=512 channels on an h x w grid, 16 x 16 for a 500 x 500 input) and
compares two proteograms by cosine similarity. Because the embedding is an
average, the cosine splits exactly over pairs of grid cells:

    q = (1/Nq) sum_a F_q(a),   t = (1/Nt) sum_b F_t(b)
    cos(q, t) = sum_a sum_b  F_q(a) . F_t(b) / (Nq * Nt * |q| * |t|)

so the Nq x Nt matrix ``pair`` (query cell a x target cell b) sums to the
cosine. Its row sums give a query map, its column sums a target map, and its
largest entries say which query region matched which target region. The layer4
features follow a ReLU, so every entry is >= 0.

The maps are pair-specific but share a lot between pairs, since a cell that is
strong in general scores high against most partners. ``center`` (the corpus
mean embedding mu) gives a centered map, F_q(a) . (t - mu), which keeps what
this pair shares beyond what every protein shares. It sums to
q . (t - mu) / (|q| |t|), not to the cosine.

Mapping back to the protein
---------------------------
A proteogram is an L x L image of one domain (L = residues). The upper triangle
holds VdW attractive (R), VdW repulsive (G) and C-alpha distance (B) for
residue pair (i, j). The lower triangle holds electrostatic attractive (R),
electrostatic repulsive (G) and hydrophobicity (B), rotated 180 degrees, so
lower pixel (p, q) is residue pair (L-1-p, L-1-q) (see proteogram.py). Cell
maps are upsampled to the native L x L grid (undoing the search-time resize or
pad) with total mass preserved, then split into residue-pair matrices for the
two halves and summed into per-residue scores that add up to the map total.
A last-layer cell is centred on its block of pixels but sees well beyond it,
so block-level placement is an approximation; ``deletion_insertion`` tests it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Model and preprocessing (must match measure_similarity_v2.py)
# ---------------------------------------------------------------------------

def load_resnet18_checkpoint(model_path: str, device: str | torch.device = 'cpu'):
    """Load a ResNet18 classifier checkpoint as search does.

    Accepts the self-describing {'state_dict', 'meta'} format written by
    train_multiple_models_randomized_eval.py (multi-head models save the
    backbone + main head) or a bare state dict. Embedding checkpoints
    (--loss triplet_hierarchy) put a biased Linear after the pooling, so the
    exact split does not hold for them and they are rejected.

    Returns (model in eval mode, meta dict).
    """
    loaded = torch.load(model_path, map_location=device, weights_only=False)
    meta = {}
    if isinstance(loaded, nn.Module):
        raise ValueError('Pickled nn.Module checkpoints are not supported; save a state_dict.')
    if isinstance(loaded, dict) and 'state_dict' in loaded:
        state_dict, meta = loaded['state_dict'], dict(loaded.get('meta') or {})
    else:
        state_dict = loaded
    if meta.get('kind') == 'embedding':
        raise ValueError('Embedding checkpoints (triplet loss) are not supported: the exact '
                         'similarity split needs the pooled feature itself as the embedding.')
    if 'conv_proj.weight' in state_dict or 'fc2.weight' in state_dict:
        raise ValueError('Only ResNet18 checkpoints are supported.')
    model = tv_models.resnet18(weights=None)
    if 'fc.1.weight' in state_dict:
        model.fc = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(model.fc.in_features, state_dict['fc.1.weight'].shape[0]))
    elif 'fc.weight' in state_dict:
        model.fc = nn.Linear(model.fc.in_features, state_dict['fc.weight'].shape[0])
    else:
        raise ValueError('Cannot find a ResNet18 fc head in the checkpoint.')
    model.load_state_dict(state_dict)
    return model.to(device).eval(), meta


def search_grid(meta: dict, default_grid: int = 200, default_resize: bool = False):
    """(grid size, resize?) the checkpoint was trained with, as search resolves it."""
    grid = meta.get('input_size') or meta.get('max_image_size') or default_grid
    resize = meta.get('resize')
    return int(grid), bool(default_resize if resize is None else resize)


def pad_to_size(img: Image.Image, target: int = 200, fill: int = 128) -> Image.Image:
    """Centre-pad with gray to target x target, crop top-left if larger.
    Copy of measure_similarity_v2.pad_to_size so attribution sees what search sees."""
    arr = np.array(img.convert('RGB'))
    H, W = arr.shape[0], arr.shape[1]

    def get_pad(curr, tgt):
        d = tgt - curr
        if d <= 0:
            return (0, 0)
        p1 = d // 2
        return (p1, d - p1)

    arr = np.pad(arr, (get_pad(H, target), get_pad(W, target), (0, 0)), constant_values=fill)
    arr = arr[:target, :target, :]
    return Image.fromarray(arr.astype(np.uint8))


@dataclass
class Geometry:
    """How a native L x L proteogram sits in the model's input grid."""
    L: int                # native size (residues)
    grid: int             # model input size
    resize: bool          # True: L x L scaled to grid; False: centre-padded / cropped
    offset: int = 0       # pad mode: first native row/col in the grid

    def native_rows(self, g0: float, g1: float):
        """Native rows covered by input-grid rows [g0, g1)."""
        if self.resize:
            a, b = g0 * self.L / self.grid, g1 * self.L / self.grid
        else:
            a, b = g0 - self.offset, g1 - self.offset
        return max(0.0, a), min(float(self.L), b)


_NORMALIZE = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])


def preprocess_array(native: np.ndarray, grid: int, resize: bool) -> torch.Tensor:
    """Normalized 1x3xGxG tensor from a native LxLx3 uint8 array, as search does it."""
    img = Image.fromarray(native.astype(np.uint8), 'RGB')
    prepped = img.resize((grid, grid)) if resize else pad_to_size(img, grid)
    return _NORMALIZE(prepped).unsqueeze(0)


def load_proteogram(path: str, grid: int, resize: bool, drop_channels=None):
    """Return (normalized tensor 1x3xGxG, native RGB array LxLx3, Geometry).

    drop_channels: channels the model was trained without (checkpoint meta
    'drop_channels'); they are blanked as in training, and the returned native
    array is the masked image the model sees.
    """
    native = np.array(Image.open(path).convert('RGB'))
    if drop_channels:
        native = mask_channels(native, channels_present(drop_channels))
    L = native.shape[0]
    if native.shape[0] != native.shape[1]:
        raise ValueError(f'{path}: proteogram is not square ({native.shape[:2]}).')
    geom = Geometry(L, grid, resize, offset=0 if resize else max(0, (grid - L) // 2))
    return preprocess_array(native, grid, resize), native, geom


def layer4_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """ResNet18 last-block feature map (B, 512, h, w); its mean is the search embedding."""
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))
    return model.layer4(model.layer3(model.layer2(model.layer1(x))))


@torch.no_grad()
def embed(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Search embedding (B, 512): avgpool of layer4, as Img2Vec computes it."""
    return layer4_features(model, x).mean(dim=(2, 3))


# ---------------------------------------------------------------------------
# Exact decomposition
# ---------------------------------------------------------------------------

@dataclass
class Decomposition:
    cos: float                      # cosine similarity of the two embeddings
    pair: np.ndarray                # (Nq, Nt) cell-pair contributions, sums to cos
    query_cells: np.ndarray         # (hq, wq) row sums of pair
    target_cells: np.ndarray        # (ht, wt) column sums of pair
    query_centered: Optional[np.ndarray] = None   # (hq, wq), uses t - mu
    target_centered: Optional[np.ndarray] = None  # (ht, wt), uses q - mu
    centered_total: Optional[float] = None        # q.(t - mu)/(|q||t|)

    @property
    def lift(self) -> np.ndarray:
        """pair / (row x column marginals / cos): > 1 where a query cell and a
        target cell match more than their overall strength predicts."""
        expected = np.outer(self.query_cells.ravel(), self.target_cells.ravel()) / self.cos
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(expected > 0, self.pair / expected, 0.0)


@torch.no_grad()
def decompose(fq: torch.Tensor, ft: torch.Tensor, center: Optional[torch.Tensor] = None) -> Decomposition:
    """Split cos(mean fq, mean ft) over cell pairs. fq, ft: (C, h, w) layer4 maps."""
    fq, ft = fq.double(), ft.double()
    C, hq, wq = fq.shape
    _, ht, wt = ft.shape
    Aq, At = fq.reshape(C, -1), ft.reshape(C, -1)
    Nq, Nt = Aq.shape[1], At.shape[1]
    q, t = Aq.mean(1), At.mean(1)
    nq, nt = q.norm(), t.norm()
    cos = float(q @ t / (nq * nt))
    pair = (Aq.T @ At) / (Nq * Nt * nq * nt)
    d = Decomposition(cos=cos, pair=pair.cpu().numpy(),
                      query_cells=pair.sum(1).reshape(hq, wq).cpu().numpy(),
                      target_cells=pair.sum(0).reshape(ht, wt).cpu().numpy())
    if center is not None:
        mu = center.double().to(Aq.device).reshape(-1)
        d.query_centered = ((Aq.T @ (t - mu)) / (Nq * nq * nt)).reshape(hq, wq).cpu().numpy()
        d.target_centered = ((At.T @ (q - mu)) / (Nt * nq * nt)).reshape(ht, wt).cpu().numpy()
        d.centered_total = float(q @ (t - mu) / (nq * nt))
    return d


# ---------------------------------------------------------------------------
# Back to the native proteogram and to residues
# ---------------------------------------------------------------------------

def cells_to_native(cells: np.ndarray, geom: Geometry) -> np.ndarray:
    """Upsample an (h, w) cell map to the native L x L proteogram, preserving mass.

    Each cell's value is spread over its block of the input grid (bilinear, so
    block edges are smooth), then the search-time resize or pad is undone. In
    pad mode the mass that falls on padding is dropped, since no residues are
    there; the returned map then sums to less than the cell map.
    """
    h, w = cells.shape
    G = geom.grid
    t = torch.from_numpy(np.ascontiguousarray(cells, dtype=np.float64))[None, None]

    def upsample(n):
        """Bilinear upsample to n x n as a density (sums to cells.sum()). Bilinear
        edges shift a little mass, so non-negative maps are rescaled exactly."""
        out = F.interpolate(t, size=(n, n), mode='bilinear', align_corners=False)[0, 0].numpy()
        out *= (h * w) / float(n * n)
        if (cells >= 0).all() and out.sum() > 0:
            out *= cells.sum() / out.sum()
        return out

    if geom.resize:
        # The whole L x L image was scaled onto the grid, so the grid maps
        # straight back onto it.
        return upsample(geom.L)
    grid_map = upsample(G)
    L, o = geom.L, geom.offset
    out = np.zeros((L, L))
    n = min(L, G - o)
    out[:n, :n] = grid_map[o:o + n, o:o + n]
    return out


@dataclass
class ResidueMaps:
    upper: np.ndarray   # (L, L) residue-pair scores, i < j only: VdW att / VdW rep / distance
    lower: np.ndarray   # (L, L) residue-pair scores, i < j only: elec att / elec rep / hydrophobicity
    diag: np.ndarray    # (L,) native diagonal pixels (mix both halves)
    score_upper: np.ndarray = field(default=None)   # (L,) per residue
    score_lower: np.ndarray = field(default=None)

    @property
    def score(self) -> np.ndarray:
        return self.score_upper + self.score_lower

    @property
    def pairs(self) -> np.ndarray:
        return self.upper + self.lower


def native_to_residues(native: np.ndarray) -> ResidueMaps:
    """Split a native L x L map into residue-pair and per-residue scores.

    Upper pixel (i, j), i < j, is residue pair (i, j). Lower pixel (p, q), p > q,
    is residue pair (L-1-p, L-1-q). A diagonal pixel (i, i) holds residue i's
    upper value and residue L-1-i's lower value, so half goes to each. Each pair
    gives half its score to each residue, so per-residue scores sum to the map.
    """
    L = native.shape[0]
    iu = np.triu_indices(L, 1)
    upper = np.zeros((L, L))
    upper[iu] = native[iu]
    flipped = native[::-1, ::-1]         # flipped[i, j] = native[L-1-i, L-1-j]
    lower = np.zeros((L, L))
    lower[iu] = flipped[iu]
    diag = np.diag(native).copy()
    s_up = 0.5 * (upper.sum(1) + upper.sum(0)) + 0.5 * diag
    s_lo = 0.5 * (lower.sum(1) + lower.sum(0)) + 0.5 * diag[::-1]
    return ResidueMaps(upper, lower, diag, s_up, s_lo)


@dataclass
class CellRegion:
    """Residues a last-layer cell covers in one proteogram."""
    cell: int
    row: int
    col: int
    half: str                 # 'upper', 'lower' or 'diagonal'
    rows: tuple               # native rows [r0, r1)
    cols: tuple               # native cols [c0, c1)
    residues: np.ndarray      # bool (L,), residues the cell's pixels involve

    def label(self) -> str:
        """Residue segments, 1-based in proteogram order."""
        L = len(self.residues)
        r0, r1 = self.rows
        c0, c1 = self.cols
        up = f'{r0 + 1}-{r1}x{c0 + 1}-{c1}'
        lo = f'{L - r1 + 1}-{L - r0}x{L - c1 + 1}-{L - c0}'
        return {'upper': f'upper {up}', 'lower': f'lower {lo}',
                'diagonal': f'diagonal {up} / {lo}'}[self.half]


def cell_regions(shape: tuple, geom: Geometry) -> list[CellRegion]:
    """Map every cell of an (h, w) grid to native rows/cols and residues."""
    h, w = shape
    L = geom.L
    out = []
    for r in range(h):
        for c in range(w):
            ra, rb = geom.native_rows(r * geom.grid / h, (r + 1) * geom.grid / h)
            ca, cb = geom.native_rows(c * geom.grid / w, (c + 1) * geom.grid / w)
            r0, r1 = int(np.floor(ra)), int(np.ceil(rb))
            c0, c1 = int(np.floor(ca)), int(np.ceil(cb))
            res = np.zeros(L, dtype=bool)
            if r1 <= r0 or c1 <= c0:            # cell lies on padding
                half = 'upper'
            elif r1 <= c0 + 1:
                half = 'upper'
            elif r0 + 1 >= c1:
                half = 'lower'
            else:
                half = 'diagonal'
            if r1 > r0 and c1 > c0:
                if half in ('upper', 'diagonal'):
                    res[r0:r1] = True
                    res[c0:c1] = True
                if half in ('lower', 'diagonal'):
                    res[L - r1:L - r0] = True
                    res[L - c1:L - c0] = True
            out.append(CellRegion(r * w + c, r, c, half, (r0, r1), (c0, c1), res))
    return out


def top_links(score: np.ndarray, k: int, distinct: bool = True) -> list[tuple[int, int, float]]:
    """Top-k (query cell, target cell, value) entries of a cell-pair matrix.
    distinct=True skips entries whose query or target cell is already used,
    so the links point at k different regions on each side."""
    order = np.argsort(score, axis=None)[::-1]
    out, used_q, used_t = [], set(), set()
    for idx in order:
        a, b = np.unravel_index(idx, score.shape)
        if distinct and (a in used_q or b in used_t):
            continue
        out.append((int(a), int(b), float(score[a, b])))
        used_q.add(a)
        used_t.add(b)
        if len(out) == k:
            break
    return out


# ---------------------------------------------------------------------------
# Channels: Shapley split of the cell-pair matrix over the six channels
# ---------------------------------------------------------------------------

#: The six proteogram channels as (name, half, RGB index); see proteogram.py.
#: Upper = row < col (and the diagonal), lower = row > col.
CHANNELS = (('vdw_attractive', 'upper', 0), ('vdw_repulsive', 'upper', 1),
            ('distance', 'upper', 2), ('elec_attractive', 'lower', 0),
            ('elec_repulsive', 'lower', 1), ('hydrophobicity', 'lower', 2))


CHANNEL_NAMES = tuple(c[0] for c in CHANNELS)


def channels_present(drop) -> tuple:
    """Six booleans (CHANNELS order) with the channels named in `drop` set False.
    `drop` is a list of names or a comma-separated string; empty keeps all."""
    if isinstance(drop, str):
        drop = [d for d in drop.split(',') if d.strip()]
    drop = [d.strip() for d in (drop or [])]
    unknown = sorted(set(drop) - set(CHANNEL_NAMES))
    if unknown:
        raise ValueError(f'Unknown channel(s) {unknown}; choose from {", ".join(CHANNEL_NAMES)}.')
    return tuple(name not in drop for name in CHANNEL_NAMES)


def mask_channels(native: np.ndarray, present, fill: str = 'black') -> np.ndarray:
    """Copy of a native LxLx3 proteogram with the absent channels blanked.

    present: six booleans in CHANNELS order. A blanked channel is set to 0
    ('black', the proteogram's "no interaction" value) or 128 ('gray') in its
    half only. The diagonal counts as upper.
    """
    out = native.copy()
    L = native.shape[0]
    upper = np.triu(np.ones((L, L), dtype=bool))
    val = 0 if fill == 'black' else 128
    for keep, (_, half, ch) in zip(present, CHANNELS):
        if not keep:
            region = upper if half == 'upper' else ~upper
            out[..., ch][region] = val
    return out


@dataclass
class ChannelSplit:
    """Exact Shapley split of one side's cell-pair matrix over the six channels.

    Channels are removed from one protein (the `side`) while the other stays
    intact, so v(S) = cos(side with channels S, partner). phi[k] is channel k's
    Shapley value per cell pair (Nq, Nt); sum_k phi[k] = full - empty, where
    full is the usual pair matrix and empty is the split of cos(blank, partner).
    """
    side: str
    phi: np.ndarray          # (6, Nq, Nt)
    full: np.ndarray         # (Nq, Nt), equals Decomposition.pair
    empty: np.ndarray        # (Nq, Nt), side fully blanked
    values: np.ndarray       # (64,) cosine for every channel subset (bit k = channel k present)

    @property
    def totals(self) -> np.ndarray:
        """(6,) each channel's Shapley value for the whole cosine."""
        return self.phi.sum(axis=(1, 2))

    def cells(self, k: int) -> np.ndarray:
        """Channel k's map on this side's cells (row sums for the query side,
        column sums for the target side), flattened."""
        return self.phi[k].sum(1) if self.side == 'query' else self.phi[k].sum(0)


@torch.no_grad()
def channel_split(model: nn.Module, native_side: np.ndarray, partner_features: torch.Tensor,
                  side: str, grid: int, resize: bool, fill: str = 'black',
                  batch_size: int = 16) -> ChannelSplit:
    """Shapley values of the six channels for every cell pair.

    Evaluates all 2^6 channel subsets of `native_side` (64 forward passes)
    against the partner's unchanged layer4 features, computes the exact cell
    split for each, and combines them with Shapley weights. Since every subset's
    split is exact and Shapley values are linear, the channel values add up per
    cell pair as well as in total.
    """
    from math import factorial
    n = len(CHANNELS)
    device = partner_features.device
    subsets = [tuple(bool(s >> k & 1) for k in range(n)) for s in range(2 ** n)]
    feats = []
    for i in range(0, len(subsets), batch_size):
        xs = torch.cat([preprocess_array(mask_channels(native_side, p, fill), grid, resize)
                        for p in subsets[i:i + batch_size]]).to(device)
        feats.append(layer4_features(model, xs))
    feats = torch.cat(feats)
    pairs, values = [], []
    for f in feats:
        d = decompose(f, partner_features) if side == 'query' else decompose(partner_features, f)
        pairs.append(d.pair)
        values.append(d.cos)
    pairs = np.stack(pairs)
    w = [factorial(s) * factorial(n - s - 1) / factorial(n) for s in range(n)]
    phi = np.zeros((n,) + pairs.shape[1:])
    for s in range(2 ** n):
        size = bin(s).count('1')
        for k in range(n):
            if not s >> k & 1:
                phi[k] += w[size] * (pairs[s | 1 << k] - pairs[s])
    return ChannelSplit(side, phi, pairs[-1], pairs[0], np.array(values))


# ---------------------------------------------------------------------------
# Faithfulness: does removing the top cells lower the similarity?
# ---------------------------------------------------------------------------

@torch.no_grad()
def deletion_insertion(model: nn.Module, xq: torch.Tensor, t_emb: torch.Tensor,
                       order: np.ndarray, shape: tuple, fractions: np.ndarray,
                       fill: str = 'black') -> tuple[np.ndarray, np.ndarray]:
    """Cosine to the target as query cells are removed (deletion) or restored
    onto a blank image (insertion) in the given order.

    A cell is removed by setting its block of the input grid to `fill`
    ('black' = pixel 0, the proteogram's "no interaction" value, or 'gray' =
    128, the pad value). Returns (deletion curve, insertion curve), one cosine
    per fraction. A faithful order makes deletion fall fast (low area) and
    insertion rise fast (high area) compared with a random order.
    """
    device = xq.device
    h, w = shape
    G = xq.shape[-1]
    px = 0.0 if fill == 'black' else 128 / 255.0
    blank_val = torch.tensor([(px - m) / s for m, s in zip(IMAGENET_MEAN, IMAGENET_STD)],
                             dtype=xq.dtype, device=device).view(1, 3, 1, 1)
    blank = blank_val.expand_as(xq)
    rows = [(int(round(r * G / h)), int(round((r + 1) * G / h))) for r in range(h)]
    cols = [(int(round(c * G / w)), int(round((c + 1) * G / w))) for c in range(w)]
    n = h * w

    def mask_for(cells):
        m = torch.zeros((1, 1, G, G), dtype=xq.dtype, device=device)
        for cell in cells:
            (r0, r1), (c0, c1) = rows[cell // w], cols[cell % w]
            m[..., r0:r1, c0:c1] = 1
        return m

    dels, inss = [], []
    for f in fractions:
        m = mask_for(order[:int(round(f * n))])
        dels.append(xq * (1 - m) + blank * m)
        inss.append(blank * (1 - m) + xq * m)
    batch = torch.cat(dels + inss)
    embs = layer4_features(model, batch).mean(dim=(2, 3))
    cos = F.cosine_similarity(embs, t_emb.expand_as(embs), dim=1).cpu().numpy()
    k = len(fractions)
    return cos[:k], cos[k:]
