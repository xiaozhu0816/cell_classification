# Cell Classification Pipeline

This repository contains a PyTorch training pipeline that classifies live-cell imaging time-course TIFF stacks as **infected** or **uninfected**. Each TIFF file stores 95 time points (t0–t94) acquired every 30 minutes. Empirically, infected cells begin to show cytopathic effects (CPE) between 16–30 hours post-infection, so the default configuration samples frames within that window.

## Project layout

```
cell_classification/
├── configs/
│   └── resnet50_baseline.yaml   # Experiment settings
├── datasets/
│   └── timecourse_dataset.py    # TIFF dataset + split utilities
├── models/
│   └── resnet.py                # ResNet classifier wrapper
├── utils/
│   ├── config.py                # YAML loader
│   ├── logger.py                # Console/file logger
│   ├── metrics.py               # Average meters + binary metrics
│   ├── seed.py                  # Seed all RNGs
│   └── transforms.py            # Build torchvision pipelines
├── requirements.txt             # Python dependencies
└── train.py                     # Main training entrypoint
```

## Data expectations

```
DATA_ROOT/
├── infected/
│   ├── *.tiff   (multi-frame stacks containing infected wells)
├── uninfected/
│   └── *.tiff   (control stacks)
```

Update the paths within `configs/resnet50_baseline.yaml` to point at your infected/uninfected directories. The default config already references the provided UNC paths.

## Quick start

1. **Install dependencies**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

2. **Launch training**
   ```powershell
   python train.py --config configs/resnet50_baseline.yaml
   ```

Training artifacts (logs and checkpoints) are written under `outputs/` and `checkpoints/` respectively. The script automatically evaluates on the validation split every epoch and reports accuracy, precision, recall, F1, and AUC. After training, it runs a final evaluation on the held-out test split.

## Customization tips

- **Frame selection**: Adjust `data.frames.strategy` and `data.frames.window_hours` in the YAML if you want to sample earlier or later time ranges, or switch to a fixed `target_frame`.
- **Model**: Swap to another ResNet depth by changing `model.name` (`resnet18`, `resnet34`, `resnet101`, `resnet152`).
- **Augmentations**: Toggle augmentation flags inside `data.transforms` (flips, rotations, color jitter).
- **Batching**: Increase `batch_size` and `num_workers` once you confirm GPU/CPU memory headroom.

Feel free to extend the pipeline with additional datasets, metrics, or foundation models as your experiments evolve.
