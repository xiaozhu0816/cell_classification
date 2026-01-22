"""Generate Grad-CAM overlays for the exact same images used in the ShowMe figure.

Goal
----
For the paper we want explainability visualizations (CAM) on *the same* example
frames readers see in the montage.

Inputs
------
- A trained checkpoint (multitask or classification) storing "model_state".
- A ShowMe manifest JSON produced by `generate_showme_figure.py`.

Outputs
-------
- A 2xN montage of CAM overlays aligned with the ShowMe timepoints.
- Individual per-panel overlay PNGs (optional, but handy for inspection).

Notes
-----
- For multitask checkpoints we backprop through the classification head.
- We intentionally operate directly on TIFF frames + the same crop/contrast
  pipeline as `generate_showme_figure.py` (center 512x512 etc.), so the CAM
  aligns visually with the published figure.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
from PIL import Image

from models import build_model, build_multitask_model
from utils import load_config


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks not initialized")
        grads = self.gradients
        activations = self.activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )
        cam_min, cam_max = cam.min(), cam.max()
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        return cam.squeeze(0).squeeze(0).cpu()


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate CAM overlays for ShowMe montage frames")
    p.add_argument("--config", required=True, help="Config used to build the model")
    p.add_argument("--checkpoint", required=True, help="Path to checkpoint (best.pt)")
    p.add_argument("--manifest", required=True, help="ShowMe manifest JSON from generate_showme_figure.py")
    p.add_argument("--out", required=True, help="Output PNG path for the CAM montage")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cmap", default="jet")
    p.add_argument("--alpha", type=float, default=0.5)
    return p.parse_args()


def _read_stack(path: Path) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        arr = tif.asarray()
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim == 3:
        return arr
    if arr.ndim == 4:
        return arr[:, 0, ...]
    raise ValueError(f"Unsupported TIFF array ndim={arr.ndim} shape={arr.shape} for {path}")


def _center_crop_border(frame: np.ndarray, crop_border_fraction: Optional[float]) -> np.ndarray:
    if not crop_border_fraction or crop_border_fraction <= 0:
        return frame
    h, w = frame.shape[-2], frame.shape[-1]
    dy = int(round(h * crop_border_fraction))
    dx = int(round(w * crop_border_fraction))
    y0, y1 = dy, max(dy + 1, h - dy)
    x0, x1 = dx, max(dx + 1, w - dx)
    return frame[..., y0:y1, x0:x1]


def _center_crop_pixels(frame: np.ndarray, size: Optional[int]) -> np.ndarray:
    if size is None:
        return frame
    size = int(size)
    if size <= 0:
        return frame
    h, w = frame.shape[-2], frame.shape[-1]
    if h < size or w < size:
        size = min(h, w)
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return frame[..., y0 : y0 + size, x0 : x0 + size]


def _prepare_display(frame: np.ndarray, lo_pct: float, hi_pct: float) -> np.ndarray:
    frame = frame.astype(np.float32)
    lo = np.percentile(frame, lo_pct)
    hi = np.percentile(frame, hi_pct)
    if hi <= lo:
        hi = lo + 1.0
    frame = (frame - lo) / (hi - lo)
    return np.clip(frame, 0, 1)


def _to_3ch(frame01: np.ndarray) -> np.ndarray:
    if frame01.ndim == 2:
        return np.stack([frame01] * 3, axis=-1)
    if frame01.ndim == 3 and frame01.shape[-1] == 3:
        return frame01
    raise ValueError(f"Unexpected display frame shape {frame01.shape}")


def _overlay_cam(image01_3ch: np.ndarray, cam01: np.ndarray, cmap_name: str, alpha: float) -> np.ndarray:
    cmap = cm.get_cmap(cmap_name)
    heat = cmap(cam01)[:, :, :3]
    overlay = alpha * heat + (1 - alpha) * image01_3ch
    return np.clip(overlay, 0, 1)


def _build_model_from_checkpoint(cfg: Dict, checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, str]:
    """Return (model, kind) where kind is 'multitask' or 'single'."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_cfg = cfg.get("model", {})

    # Detect multitask vs single-task.
    # Our multitask configs don't set `model.type`, so we also look at the
    # checkpoint contents (regressor head params, etc.).
    model_type = str(model_cfg.get("type", ""))
    state = ckpt.get("model_state", {})
    looks_multitask = (
        "multitask" in model_type.lower()
        or any(k.startswith("regressor.") for k in state.keys())
        or any(k.startswith("classifier.") for k in state.keys())
    )

    if looks_multitask:
        model = build_multitask_model(model_cfg).to(device)
        kind = "multitask"
    else:
        model = build_model(model_cfg).to(device)
        kind = "single"
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, kind


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    hours: List[float] = list(manifest["hours"])
    frames_per_hour = float(manifest.get("frames_per_hour", 2.0))
    crop_border_fraction = manifest.get("crop_border_fraction")
    center_crop_size = manifest.get("center_crop_size", 512)
    lo_pct = float(manifest.get("min_contrast_percentile", 1.0))
    hi_pct = float(manifest.get("max_contrast_percentile", 99.0))

    mock_tiffs = [Path(p) for p in manifest["mock_tiffs"]]
    infected_tiffs = [Path(p) for p in manifest["infected_tiffs"]]
    if len(mock_tiffs) < 1 or len(infected_tiffs) < 1:
        raise ValueError("Manifest must contain at least one mock_tiff and one infected_tiff")

    model, kind = _build_model_from_checkpoint(cfg, Path(args.checkpoint), device)
    target_layer = model.backbone.layer4[-1]
    cam = GradCAM(model, target_layer)

    def forward_and_cam(x: torch.Tensor) -> Tuple[float, np.ndarray]:
        x = x.to(device)
        x.requires_grad_(True)
        if kind == "multitask":
            cls_logits, _time_pred = model(x)
            # cls_logits: [N,2] softmax convention used in multitask training
            score = cls_logits[:, 1].sum()
            prob = torch.softmax(cls_logits, dim=1)[:, 1].item()
        else:
            logits = model(x)
            score = logits[:, 0].sum()
            prob = torch.sigmoid(logits)[0, 0].item()
        model.zero_grad(set_to_none=True)
        score.backward()
        cam_map = cam(x).numpy()
        return prob, cam_map

    stacks = [(_read_stack(mock_tiffs[0]), "Mock (PBS) #1"), (_read_stack(infected_tiffs[0]), "VEEV infected (TC-83) #1")]

    n = len(hours)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6), constrained_layout=True)
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for row, (stack, row_title) in enumerate(stacks):
        for col, h in enumerate(hours):
            idx = int(round(float(h) * frames_per_hour))
            idx = max(0, min(stack.shape[0] - 1, idx))
            frame = stack[idx]
            frame = _center_crop_border(frame, crop_border_fraction)
            frame = _center_crop_pixels(frame, int(center_crop_size) if center_crop_size is not None else None)
            frame01 = _prepare_display(frame, lo_pct, hi_pct)
            img01_3 = _to_3ch(frame01)

            # Model input: [1,3,H,W] in [0,1]
            x = torch.from_numpy(np.transpose(img01_3, (2, 0, 1))).float().unsqueeze(0)
            prob, cam01 = forward_and_cam(x)
            overlay01 = _overlay_cam(img01_3, cam01, args.cmap, args.alpha)

            ax = axes[row, col]
            ax.imshow(overlay01)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{int(h)} h\nP(inf)={prob:.2f}", fontsize=10)
            if col == 0:
                ax.set_ylabel(row_title, fontsize=11)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle("Grad-CAM overlays on representative time-course frames", fontsize=13)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    meta_out = out_path.with_suffix(".json")
    meta_out.write_text(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "config": args.config,
                "manifest": str(Path(args.manifest).resolve()),
                "cmap": args.cmap,
                "alpha": args.alpha,
                "model_kind": kind,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved CAM montage to {out_path}")


if __name__ == "__main__":
    main()
