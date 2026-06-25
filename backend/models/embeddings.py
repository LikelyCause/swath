"""Geospatial foundation-model embeddings -> unsupervised segmentation.

Runs a *pretrained* GeoFM encoder (Prithvi-EO-2.0 or Clay) over the AOI, extracts
per-patch embeddings, and k-means clusters them into self-similar regions. No
fine-tuning and no task head: this shows what a foundation model considers
similar, zero-shot. Output is a colorized cluster overlay + legend, exactly like
the land-cover task, so the frontend renders it with no new plumbing.

Both backbones consume the same 6-band Sentinel-2 stack the land-cover ingest
already produces (B02,B03,B04,B8A,B11,B12). Geometry probed from terratorch:
  - Prithvi-EO-2.0-300M: 224x224 input -> 14x14 patch grid, 1024-dim, cls-first
  - Clay v1 base:        256x256 input -> 32x32 patch grid,  768-dim, cls-first
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from backend.geo.chips import load_chip
from backend.models.device import get_device
from backend.models.prithvi_model import BAND_MEANS, BAND_STDS
from backend.progress import set_stage

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Sentinel-2 band centers (micrometres) for B02,B03,B04,B8A,B11,B12 — Clay's
# channel embedding is wavelength-driven, so it needs these.
S2_WAVES = [0.49, 0.56, 0.665, 0.865, 1.61, 2.19]

# Backbone configs: terratorch registry name + probed ViT geometry. `means/stds`
# default to the Prithvi HLS stats (a reasonable reflectance normalization for
# both; clustering structure is robust to the exact values).
BACKBONES = {
    "prithvi-eo-2-300": {"tt": "terratorch_prithvi_eo_v2_300", "px": 224, "grid": 14, "clay": False},
    "clay-v1-base": {"tt": "timm_clay_v1_base", "px": 256, "grid": 32, "clay": True},
}

# Qualitative palette for unsupervised clusters (labels carry no fixed meaning).
CLUSTER_COLORS = [
    (228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
    (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191),
    (153, 153, 153), (102, 194, 165), (252, 141, 98), (141, 160, 203),
]

_MODELS: dict[str, torch.nn.Module] = {}


def _get_backbone(model_id: str) -> torch.nn.Module:
    if model_id not in _MODELS:
        cfg = BACKBONES[model_id]
        set_stage("model", f"Loading {model_id} foundation encoder…")
        from terratorch.registry import BACKBONE_REGISTRY  # noqa: PLC0415 - heavy import

        m = BACKBONE_REGISTRY.build(cfg["tt"], pretrained=True).to(get_device()).eval()
        _MODELS[model_id] = m
    return _MODELS[model_id]


def _kmeans(x: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> np.ndarray:
    """Plain Lloyd's k-means with k-means++ init (no sklearn dependency)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = [int(rng.integers(n))]
    for _ in range(k - 1):
        d = np.min(np.stack([((x - x[c]) ** 2).sum(1) for c in idx]), axis=0)
        s = d.sum()
        idx.append(int(rng.choice(n, p=d / s)) if s > 0 else int(rng.integers(n)))
    centers = x[idx].copy()
    labels = np.full(n, -1, dtype=np.int64)
    for _ in range(iters):
        new = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(2).argmin(1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                centers[j] = x[labels == j].mean(0)
    return labels


def embed_and_cluster(chip_id: str, model_id: str, k: int = 6) -> dict:
    meta = load_chip(chip_id)
    cfg = BACKBONES[model_id]
    npy = DATA_DIR / f"{chip_id}_lc.npy"
    if not npy.exists():
        raise FileNotFoundError("Embeddings input missing; re-fetch Sentinel-2 imagery.")

    arr = np.load(npy).astype(np.float32)  # (6, 3, 224, 224)
    x = arr[:, -1]  # latest timestep -> (6, 224, 224)
    means = np.array(BAND_MEANS, np.float32)[:, None, None]
    stds = np.array(BAND_STDS, np.float32)[:, None, None]
    t = torch.from_numpy((x - means) / stds).unsqueeze(0)  # (1, 6, 224, 224)
    if cfg["px"] != t.shape[-1]:
        t = F.interpolate(t, size=(cfg["px"], cfg["px"]), mode="bilinear", align_corners=False)

    dev = get_device()
    model = _get_backbone(model_id)
    set_stage("infer", f"Extracting {model_id} patch embeddings…")
    kwargs = {"waves": torch.tensor(S2_WAVES, device=dev)} if cfg["clay"] else {}
    with torch.inference_mode():
        out = model(t.to(dev), **kwargs)
    tokens = (out[-1] if isinstance(out, (list, tuple)) else out)[0]  # (N+cls, D)
    g = cfg["grid"]
    patches = tokens[-(g * g):].float().cpu().numpy()  # drop leading cls/register token(s)
    patches /= np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8  # cosine k-means

    set_stage("infer", f"Clustering embeddings into {k} regions…")
    labels = _kmeans(patches, k).reshape(g, g)

    rgba = np.zeros((g, g, 4), np.uint8)
    counts = np.bincount(labels.ravel(), minlength=k)
    for c in range(k):
        m = labels == c
        if m.any():
            rgba[m, 0], rgba[m, 1], rgba[m, 2] = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
            rgba[m, 3] = 175
    disp_w, disp_h = meta.get("size_px", [g, g])
    overlay = Image.fromarray(rgba, "RGBA").resize((disp_w, disp_h), Image.NEAREST)
    out_png = DATA_DIR / f"{chip_id}_emb_{model_id}.png"
    overlay.save(out_png)

    total = int(counts.sum()) or 1
    legend = [
        {"class": f"Cluster {c + 1}", "color": "#%02x%02x%02x" % CLUSTER_COLORS[c % len(CLUSTER_COLORS)],
         "pct": round(100.0 * counts[c] / total, 1)}
        for c in np.argsort(counts)[::-1] if counts[c] > 0
    ]
    return {
        "task": "embeddings",
        "overlay_url": f"/data/{out_png.name}",
        "bounds": meta["bounds"],
        "legend": legend,
    }
