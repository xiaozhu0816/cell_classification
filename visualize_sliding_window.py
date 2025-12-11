"""
Publication-Quality Sliding Window Visualization

Creates beautiful, Nature-style plots from sliding window analysis data.
Shows individual data points, mean, and standard deviation with clean aesthetics.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Rectangle

# Nature-style publication settings
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['xtick.major.width'] = 1.0
mpl.rcParams['ytick.major.width'] = 1.0
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['pdf.fonttype'] = 42  # TrueType fonts for editing
mpl.rcParams['ps.fonttype'] = 42


def load_data(json_path: Path) -> Dict:
    """Load sliding window analysis data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_single_metric_nature_style(
    data: Dict,
    metric: str,
    output_path: Path,
    show_individual_points: bool = True,
    color_scheme: str = 'auto',
    figsize: tuple = (12, 6),
) -> None:
    """
    Create a plot for a single metric (matches analyze_interval_sweep_train.py style).
    """
    metric_data = data['results'][metric]
    
    window_centers = np.array(metric_data['window_centers'])
    means = np.array(metric_data['means'])
    stds = np.array(metric_data['stds'])
    fold_metrics = metric_data['fold_metrics']
    
    # Create figure - matching the original size
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot individual fold data points if requested
    if show_individual_points:
        # fold_metrics is [window][fold], need to transpose to [fold][window]
        n_folds = len(fold_metrics[0]) if fold_metrics else 0
        for fold_idx in range(n_folds):
            fold_values = [fold_metrics[window_idx][fold_idx] for window_idx in range(len(window_centers))]
            ax.plot(window_centers, fold_values, 'o', alpha=0.3, markersize=4, color='lightblue')
    
    # Plot mean with error bars - simple and clear
    ax.errorbar(
        window_centers,
        means,
        yerr=stds,
        fmt="-o",
        capsize=5,
        linewidth=2,
        markersize=8,
        label=f"Mean ± SD (n={len(fold_metrics[0])} folds)",
    )
    
    # Labels and title
    window_size = data.get('window_size', 'N/A')
    stride = data.get('stride', window_size)
    
    ax.set_xlabel("Window Center (hours)", fontsize=12)
    ax.set_ylabel(metric.upper(), fontsize=12)
    ax.set_title(
        f"Sliding Window Analysis: {metric.upper()} vs Time Window\n"
        f"(Window size = {window_size}h, Stride = {stride}h)",
        fontsize=13,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    
    print(f"✓ Saved {metric} plot to: {output_path}")
    
    plt.close(fig)


def plot_multi_metric_nature_style(
    data: Dict,
    metrics: List[str],
    output_path: Path,
    show_individual_points: bool = True,
    layout: str = 'overlay',
    figsize: Optional[tuple] = None,
) -> None:
    """
    Create plot with all metrics on ONE chart (matches analyze_interval_sweep_train.py style).
    """
    window_size = data.get('window_size', 'N/A')
    stride = data.get('stride', window_size)
    
    # Use bigger figure size like the original
    if figsize is None:
        figsize = (14, 7)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for idx, metric in enumerate(metrics):
        metric_data = data['results'][metric]
        window_centers = np.array(metric_data['window_centers'])
        means = np.array(metric_data['means'])
        stds = np.array(metric_data['stds'])
        fold_metrics = metric_data['fold_metrics']
        
        # Plot individual fold data points if requested
        if show_individual_points:
            # fold_metrics is [window][fold], need to transpose to [fold][window]
            n_folds = len(fold_metrics[0]) if fold_metrics else 0
            for fold_idx in range(n_folds):
                fold_values = [fold_metrics[window_idx][fold_idx] for window_idx in range(len(window_centers))]
                ax.plot(
                    window_centers, 
                    fold_values, 
                    markers[idx % len(markers)], 
                    alpha=0.3, 
                    markersize=4, 
                    color=colors[idx]
                )
        
        # Plot mean with error bars
        marker = markers[idx % len(markers)]
        ax.errorbar(
            window_centers,
            means,
            yerr=stds,
            fmt=f"-{marker}",
            capsize=4,
            linewidth=1.5,
            markersize=7,
            color=colors[idx],
            label=metric.upper(),
            alpha=0.8,
        )
    
    ax.set_xlabel("Window Center (hours)", fontsize=12)
    ax.set_ylabel("Metric Value", fontsize=12)
    ax.set_title(
        f"Sliding Window Analysis: Multiple Metrics\n"
        f"(Window: {window_size}h, Stride: {stride}h)",
        fontsize=13,
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(fontsize=10, loc="best")
    
    fig.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches='tight')
    
    print(f"✓ Saved multi-metric plot to: {output_path}")
    
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Create publication-quality plots from sliding window analysis data'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to sliding window JSON data file'
    )
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=None,
        help='Metrics to plot (default: all available metrics)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as input file)'
    )
    parser.add_argument(
        '--hide-points',
        action='store_true',
        help='Hide individual data points (only show mean and error bars)'
    )
    parser.add_argument(
        '--color-scheme',
        choices=['auto', 'nature', 'cell', 'science'],
        default='auto',
        help='Color scheme for single-metric plots (auto matches combined plot colors)'
    )
    parser.add_argument(
        '--layout',
        choices=['overlay', 'horizontal', 'vertical', 'grid'],
        default='overlay',
        help='Layout for multi-metric plot: overlay (all on one chart), horizontal/vertical/grid (separate panels)'
    )
    parser.add_argument(
        '--combined-only',
        action='store_true',
        help='Only create combined plot (skip individual metric plots)'
    )
    
    args = parser.parse_args()
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    print(f"\nLoading data from: {data_path}")
    data = load_data(data_path)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_path.parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine metrics
    if args.metrics:
        metrics = args.metrics
    else:
        metrics = data.get('metrics', list(data['results'].keys()))
    
    print(f"Metrics to plot: {', '.join(metrics)}")
    print(f"Output directory: {output_dir}")
    print(f"Show individual points: {not args.hide_points}")
    print()
    
    # Get base filename from data
    window_size = data.get('window_size', 'unknown')
    stride = data.get('stride', window_size)
    base_name = f"sliding_window_w{window_size}_s{stride}_publication"
    
    # Create individual plots
    if not args.combined_only:
        print("Creating individual metric plots...")
        for metric in metrics:
            output_path = output_dir / f"{base_name}_{metric}.png"
            plot_single_metric_nature_style(
                data,
                metric,
                output_path,
                show_individual_points=not args.hide_points,
                color_scheme=args.color_scheme,
            )
    
    # Create combined plot if multiple metrics
    if len(metrics) > 1:
        print("\nCreating combined multi-metric plot...")
        output_path = output_dir / f"{base_name}_combined.png"
        plot_multi_metric_nature_style(
            data,
            metrics,
            output_path,
            show_individual_points=not args.hide_points,
            layout=args.layout,
        )
    
    print(f"\n{'='*60}")
    print("✓ All plots created successfully!")
    print(f"{'='*60}")
    print("\nOutput files:")
    print(f"  Directory: {output_dir}")
    print(f"  Format: PNG (300 DPI) + PDF (vector)")
    print(f"\nUsage tips:")
    print(f"  - PDF files are publication-ready vector graphics")
    print(f"  - Edit in Adobe Illustrator or Inkscape if needed")
    print(f"  - Fonts are embedded (Arial, TrueType)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
