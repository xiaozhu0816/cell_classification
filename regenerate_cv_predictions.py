"""
Regenerate test predictions for 5-fold CV using existing checkpoints

This script loads the trained models from each fold and generates test_predictions.npz
files so that comprehensive analysis can be run.

Usage:
    python regenerate_cv_predictions.py --result-dir outputs/multitask_resnet50/20260105-155852_5fold
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import from existing code
import sys
sys.path.insert(0, str(Path(__file__).parent))

from datasets import build_datasets
from utils.transforms import build_transforms
from utils import load_config
from models.multitask_resnet import MultiTaskResNet


def meta_batch_to_list(meta_batch):
    """Convert batched metadata to list of dicts (from train_multitask_cv.py)."""
    if isinstance(meta_batch, list):
        return meta_batch
    if isinstance(meta_batch, dict):
        keys = list(meta_batch.keys())
        if not keys:
            return []
        length = len(meta_batch[keys[0]])
        meta_list = []
        for i in range(length):
            entry = {}
            for key in keys:
                value = meta_batch[key][i]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                entry[key] = value
            meta_list.append(entry)
        return meta_list
    raise TypeError(f"Unsupported meta batch type: {type(meta_batch)}")


@torch.no_grad()
def evaluate_and_save_predictions(model, dataloader, device, output_file):
    """Evaluate model and save all predictions."""
    model.eval()
    
    all_cls_probs = []
    all_cls_preds = []
    all_cls_labels = []
    all_reg_preds = []
    all_reg_labels = []
    all_image_paths = []
    
    for batch in tqdm(dataloader, desc="Generating predictions"):
        # Dataset returns (image, label, metadata_dict)
        # DataLoader default collate converts metadata dict to dict of lists
        images = batch[0].to(device)
        labels = batch[1]
        metadata = meta_batch_to_list(batch[2])  # Convert dict of lists to list of dicts
        
        cls_labels = labels.to(device)
        
        # Extract regression labels from metadata (now list of dicts)
        reg_labels = torch.tensor([m['hours_since_start'] for m in metadata], dtype=torch.float32).to(device)
        image_paths = [m['path'] for m in metadata]
        
        # Forward pass
        cls_logits, reg_pred = model(images)
        cls_probs = torch.softmax(cls_logits, dim=1)
        cls_pred = torch.argmax(cls_probs, dim=1)
        
        # Collect results
        all_cls_probs.append(cls_probs.cpu().numpy())
        all_cls_preds.append(cls_pred.cpu().numpy())
        all_cls_labels.append(cls_labels.cpu().numpy())
        all_reg_preds.append(reg_pred.squeeze().cpu().numpy())
        all_reg_labels.append(reg_labels.cpu().numpy())
        all_image_paths.extend(image_paths)
    
    # Concatenate all batches
    cls_probs = np.concatenate(all_cls_probs, axis=0)
    cls_preds = np.concatenate(all_cls_preds, axis=0)
    cls_labels = np.concatenate(all_cls_labels, axis=0)
    reg_preds = np.concatenate(all_reg_preds, axis=0)
    reg_labels = np.concatenate(all_reg_labels, axis=0)
    
    # Save to npz
    np.savez(
        output_file,
        cls_probs=cls_probs,
        cls_preds=cls_preds,
        cls_labels=cls_labels,
        reg_preds=reg_preds,
        reg_labels=reg_labels,
        image_paths=np.array(all_image_paths)
    )
    
    print(f"✓ Saved {len(cls_preds)} predictions to {output_file}")
    
    return {
        'cls_probs': cls_probs,
        'cls_preds': cls_preds,
        'cls_labels': cls_labels,
        'reg_preds': reg_preds,
        'reg_labels': reg_labels
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=str, required=True,
                        help="Path to CV results directory")
    parser.add_argument("--config", type=str, default="configs/multitask_example.yaml",
                        help="Path to config file used for training")
    args = parser.parse_args()
    
    cv_dir = Path(args.result_dir)
    config_file = Path(args.config)
    
    print("="*80)
    print("REGENERATING TEST PREDICTIONS FOR 5-FOLD CV")
    print("="*80)
    print(f"CV Directory: {cv_dir}")
    print(f"Config File: {config_file}\n")
    
    # Check if CV summary exists
    summary_file = cv_dir / "cv_summary.json"
    if not summary_file.exists():
        print(f"❌ Error: {summary_file} not found!")
        print("Make sure this is a valid CV results directory.")
        return 1
    
    # Check if config exists
    if not config_file.exists():
        print(f"❌ Error: {config_file} not found!")
        print("Please specify the correct config file with --config")
        return 1
    
    # Load CV summary
    with open(summary_file, 'r') as f:
        cv_summary = json.load(f)
    
    num_folds = cv_summary['num_folds']
    
    # Load config
    config = load_config(str(config_file))
    
    print(f"✓ Config loaded from {config_file}")
    print(f"✓ Number of folds: {num_folds}\n")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Process each fold
    for fold_idx in range(1, num_folds + 1):
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}/{num_folds}")
        print(f"{'='*80}")
        
        fold_dir = cv_dir / f"fold_{fold_idx}"
        checkpoint_file = fold_dir / "checkpoints" / "best.pt"
        output_file = fold_dir / "test_predictions.npz"
        
        # Check if predictions already exist
        if output_file.exists():
            print(f"⚠ {output_file} already exists, skipping...")
            continue
        
        # Check checkpoint exists
        if not checkpoint_file.exists():
            print(f"❌ Error: {checkpoint_file} not found!")
            continue
        
        print(f"Loading checkpoint: {checkpoint_file}")
        
        # Create model
        model = MultiTaskResNet(
            backbone=config['model'].get('backbone', 'resnet50'),
            num_classes=config['model']['num_classes'],
            pretrained=False  # Loading from checkpoint
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state'])
        model = model.to(device)
        
        print(f"✓ Model loaded (epoch {checkpoint['epoch']})")
        
        # Create datasets (we only need test)
        print("Creating datasets...")
        data_cfg = config['data']
        transforms_dict = build_transforms(data_cfg.get("transforms", {}))
        
        _, _, test_ds = build_datasets(
            data_cfg=data_cfg,
            transforms=transforms_dict,
            fold_index=fold_idx - 1,  # 0-indexed for splitting
            num_folds=num_folds,
        )
        
        print(f"✓ Test set: {len(test_ds)} samples")
        
        # Create dataloader
        batch_size = data_cfg.get("batch_size", 32)
        eval_batch_size = batch_size * data_cfg.get("eval_batch_size_multiplier", 2)
        num_workers = data_cfg.get("num_workers", 4)
        
        test_loader = DataLoader(
            test_ds, batch_size=eval_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        # Generate and save predictions
        print("Generating predictions...")
        evaluate_and_save_predictions(model, test_loader, device, output_file)
    
    print("\n" + "="*80)
    print("✅ ALL FOLDS PROCESSED!")
    print("="*80)
    print("\nNow you can run comprehensive analysis:")
    print(f"  python analyze_cv_results_comprehensive.py --result-dir {cv_dir}")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
