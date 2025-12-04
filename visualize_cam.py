from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import tifffile

from datasets import build_datasets
from models import build_model
from utils import build_transforms, load_config, set_seed


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

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = 0) -> torch.Tensor:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for cell classification model")
    parser.add_argument("--config", default="configs/resnet50_baseline.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt) file")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val", help="Dataset split to sample")
    parser.add_argument("--num-samples", type=int, default=8, help="Number of frames to visualize")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output-dir", default="cam_outputs", help="Directory to store CAM figures")
    parser.add_argument("--num-folds", type=int, default=1, help="Number of folds used during training (match training run)")
    parser.add_argument("--fold-index", type=int, default=0, help="Fold index to mimic when num-folds > 1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run inference on")
    parser.add_argument("--cmap", default="jet", help="Matplotlib colormap for CAM")
    parser.add_argument(
        "--weights-only",
        action="store_true",
        help=(
            "Load checkpoint with weights_only=True (defaults to False to maintain compatibility with older checkpoints). "
            "Only enable if you explicitly need PyTorch's safer weights-only mode."
        ),
    )
    return parser.parse_args()


def load_frame(path: Path, frame_index: int) -> np.ndarray:
    with tifffile.TiffFile(path) as tif:
        frame = tif.asarray(key=frame_index)
    frame = frame.astype(np.float32)
    frame -= frame.min()
    max_val = frame.max()
    if max_val > 0:
        frame /= max_val
    frame = (frame * 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = np.stack([frame] * 3, axis=-1)
    elif frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    return frame


def tensor_to_uint8(tensor: torch.Tensor, mean: List[float], std: List[float]) -> np.ndarray:
    tensor = tensor.detach().cpu().clone()
    for c in range(len(mean)):
        tensor[c] = tensor[c] * std[c] + mean[c]
    tensor = torch.clamp(tensor, 0.0, 1.0)
    array = (tensor.numpy() * 255).astype(np.uint8)
    array = np.transpose(array, (1, 2, 0))
    return array


def overlay_cam(image: np.ndarray, cam: torch.Tensor, cmap_name: str = "jet", alpha: float = 0.5) -> np.ndarray:
    cmap = cm.get_cmap(cmap_name)
    heatmap = cmap(cam.numpy())[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)
    overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
    return heatmap, overlay


def annotate_overlay(overlay: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(overlay)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    padding = 6
    try:
        text_bbox = draw.textbbox((padding, padding), text, font=font)
    except AttributeError:  # Pillow < 8.0 fallback
        text_w, text_h = draw.textsize(text, font=font)
        text_bbox = (padding, padding, padding + text_w, padding + text_h)
    rect_coords = (
        text_bbox[0] - padding,
        text_bbox[1] - padding,
        text_bbox[2] + padding,
        text_bbox[3] + padding,
    )
    draw.rectangle(rect_coords, fill=(0, 0, 0, 160))
    draw.text((rect_coords[0] + padding, rect_coords[1] + padding), text, fill=(255, 255, 255), font=font)
    return np.array(image)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device(args.device)

    data_cfg = cfg.get("data", {})
    transform_cfg = data_cfg.get("transforms", {"image_size": 512})
    transforms = build_transforms(transform_cfg)

    train_ds, val_ds, test_ds = build_datasets(
        data_cfg,
        transforms,
        fold_index=args.fold_index if args.num_folds > 1 else None,
        num_folds=args.num_folds,
    )
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}
    dataset = split_map[args.split]

    if len(dataset) == 0:
        raise RuntimeError(f"Split '{args.split}' is empty; cannot sample frames")

    mean = transform_cfg.get("mean", [0.5, 0.5, 0.5])
    std = transform_cfg.get("std", [0.25, 0.25, 0.25])

    model = build_model(cfg.get("model", {})).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=args.weights_only)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    target_layer = model.backbone.layer4[-1]
    cam_generator = GradCAM(model, target_layer)

    num_samples = min(args.num_samples, len(dataset))
    indices = random.sample(range(len(dataset)), num_samples)

    output_dir = Path(args.output_dir)
    run_name = Path(args.checkpoint).stem
    output_dir = output_dir / run_name / f"{args.split}_fold{args.fold_index}"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: List[Dict] = []

    label_names = {0: "uninfected", 1: "infected"}

    for rank, idx in enumerate(indices, start=1):
        image_tensor, label, meta = dataset[idx]
        sample_path = Path(meta["path"])
        frame_index = meta.get("frame_index", 0)
        label_id = int(label)
        label_name = label_names.get(label_id, str(label_id))

        input_tensor = image_tensor.unsqueeze(0).to(device)
        input_tensor.requires_grad_(True)
        logits = model(input_tensor)
        prob = torch.sigmoid(logits)[0, 0].item()
        pred_label = int(prob >= 0.5)
        pred_name = label_names.get(pred_label, str(pred_label))
        model.zero_grad(set_to_none=True)
        logits[:, 0].backward()
        cam = cam_generator(input_tensor)

        denorm = tensor_to_uint8(image_tensor, mean, std)
        raw_frame = load_frame(sample_path, frame_index)
        heatmap, overlay = overlay_cam(denorm, cam, cmap_name=args.cmap)
        overlay = annotate_overlay(
            overlay,
            text=f"Label: {label_name} | Pred: {pred_name} | P(infected)={prob:.2f}",
        )

        base_name = f"cam_{rank:03d}_{label_name}_prob{prob:.2f}"
        Image.fromarray(raw_frame).save(output_dir / f"{base_name}_raw.png")
        Image.fromarray(denorm).save(output_dir / f"{base_name}_input.png")
        Image.fromarray(heatmap).save(output_dir / f"{base_name}_heatmap.png")
        Image.fromarray(overlay).save(output_dir / f"{base_name}_overlay.png")

        metadata.append(
            {
                "index": idx,
                "path": str(sample_path),
                "frame_index": frame_index,
                "label": label_id,
                "label_name": label_name,
                "pred_label": pred_label,
                "pred_name": pred_name,
                "prob_infected": prob,
                "outputs": {
                    "raw": f"{base_name}_raw.png",
                    "input": f"{base_name}_input.png",
                    "heatmap": f"{base_name}_heatmap.png",
                    "overlay": f"{base_name}_overlay.png",
                },
            }
        )
        print(
            f"[{rank}/{num_samples}] Saved CAM for {sample_path.name} (frame t{frame_index}) | prob_infected={prob:.3f}"
        )

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {num_samples} CAM visualizations to {output_dir}")


if __name__ == "__main__":
    main()
