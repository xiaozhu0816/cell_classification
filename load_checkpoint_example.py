"""
Example script showing how to load and use checkpoints from training-based analysis scripts.

The analysis scripts save checkpoints for the best model of each window/interval and fold.
This script demonstrates how to load these checkpoints and use them for inference.
"""

import torch
from pathlib import Path
from models import build_model


def load_checkpoint_for_inference(checkpoint_path: str):
    """
    Load a checkpoint and prepare the model for inference.
    
    Args:
        checkpoint_path: Path to the .pth checkpoint file
        
    Returns:
        model: Loaded model in eval mode
        checkpoint: Full checkpoint dictionary with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Print checkpoint metadata
    print("="*60)
    print("Checkpoint Information:")
    print("="*60)
    
    if 'window_start' in checkpoint:
        print(f"Analysis type: Sliding Window")
        print(f"Time window: [{checkpoint['window_start']:.1f}, {checkpoint['window_end']:.1f}] hours")
    elif 'mode' in checkpoint:
        print(f"Analysis type: Interval Sweep")
        print(f"Mode: {checkpoint['mode']}")
        print(f"Interval: [{checkpoint['start_hour']:.1f}, {checkpoint['upper_hour']:.1f}] hours")
    
    print(f"Fold: {checkpoint['fold']}")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Best validation score: {checkpoint['best_val_score']:.4f}")
    print(f"\nTest metrics:")
    for metric, value in checkpoint['best_metrics'].items():
        if value is not None:
            print(f"  {metric}: {value:.4f}")
    
    # Reconstruct model from config
    model_cfg = checkpoint['config'].get('model', {})
    model = build_model(model_cfg)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("="*60)
    print("Model loaded and ready for inference!")
    print("="*60)
    
    return model, checkpoint


def find_best_window(analysis_dir: str, metric: str = 'auc'):
    """
    Find the best performing window across all folds.
    
    Args:
        analysis_dir: Path to analysis output directory (e.g., outputs/sliding_window_analysis/20231210_120000)
        metric: Metric to optimize for (default: 'auc')
        
    Returns:
        best_checkpoint_path: Path to the best checkpoint
        best_score: Best metric score
    """
    analysis_path = Path(analysis_dir)
    checkpoint_dir = analysis_path / "checkpoints"
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    best_score = -float('inf')
    best_checkpoint_path = None
    
    # Search all checkpoint files
    for checkpoint_path in checkpoint_dir.rglob("*.pth"):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Get metric value
        metrics = checkpoint.get('best_metrics', {})
        score = metrics.get(metric)
        
        if score is not None and score > best_score:
            best_score = score
            best_checkpoint_path = checkpoint_path
    
    if best_checkpoint_path is None:
        raise ValueError(f"No valid checkpoints found with metric '{metric}'")
    
    print(f"\nBest checkpoint: {best_checkpoint_path}")
    print(f"Best {metric.upper()}: {best_score:.4f}\n")
    
    return str(best_checkpoint_path), best_score


def compare_windows(analysis_dir: str, metric: str = 'auc'):
    """
    Compare performance across all windows/intervals.
    
    Args:
        analysis_dir: Path to analysis output directory
        metric: Metric to compare
    """
    analysis_path = Path(analysis_dir)
    checkpoint_dir = analysis_path / "checkpoints"
    
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    # Collect results by window/interval
    results = {}
    
    for checkpoint_path in checkpoint_dir.rglob("*.pth"):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create window identifier
        if 'window_start' in checkpoint:
            window_id = f"[{checkpoint['window_start']:.0f}-{checkpoint['window_end']:.0f}]h"
        elif 'mode' in checkpoint:
            window_id = f"{checkpoint['mode']}:[{checkpoint['start_hour']:.0f}-{checkpoint['upper_hour']:.0f}]h"
        else:
            continue
        
        # Get metric value
        metrics = checkpoint.get('best_metrics', {})
        score = metrics.get(metric)
        
        if score is not None:
            if window_id not in results:
                results[window_id] = []
            results[window_id].append(score)
    
    # Print comparison
    print("="*60)
    print(f"Performance Comparison ({metric.upper()})")
    print("="*60)
    
    for window_id in sorted(results.keys()):
        scores = results[window_id]
        mean_score = sum(scores) / len(scores)
        std_score = (sum((x - mean_score)**2 for x in scores) / len(scores))**0.5
        print(f"{window_id:30s}: {mean_score:.4f} ± {std_score:.4f} (n={len(scores)})")
    
    print("="*60)


# Example usage
if __name__ == "__main__":
    # Example 1: Load a specific checkpoint
    print("\n### Example 1: Load a specific checkpoint ###\n")
    checkpoint_path = "outputs/sliding_window_analysis/20231210_120000/checkpoints/window_10-15/fold_01_best.pth"
    # Uncomment to run:
    # model, checkpoint = load_checkpoint_for_inference(checkpoint_path)
    
    # Example 2: Find and load the best window
    print("\n### Example 2: Find the best window ###\n")
    analysis_dir = "outputs/sliding_window_analysis/20231210_120000"
    # Uncomment to run:
    # best_path, best_score = find_best_window(analysis_dir, metric='auc')
    # model, checkpoint = load_checkpoint_for_inference(best_path)
    
    # Example 3: Compare all windows
    print("\n### Example 3: Compare all windows ###\n")
    # Uncomment to run:
    # compare_windows(analysis_dir, metric='auc')
    
    print("\nTo use this script:")
    print("1. Update the paths to your actual analysis output directory")
    print("2. Uncomment the example you want to run")
    print("3. Run: python load_checkpoint_example.py")
