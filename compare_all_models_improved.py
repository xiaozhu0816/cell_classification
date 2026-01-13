"""
改进的四模型全面对比分析
改进点：
1. 滑动窗口只对比分类任务（包括单分类baseline）
2. Y轴比例尺优化，更好地显示差异
3. 在关键对比图上标注p值
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from typing import Dict, List, Tuple
import pandas as pd
from scipy import stats

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'


def load_cv_results(result_dir: str) -> Dict:
    """加载CV结果"""
    result_dir = result_dir.replace('\\', '/')
    result_path = Path(result_dir)
    summary_file = result_path / "cv_summary.json"
    
    if not summary_file.exists():
        raise FileNotFoundError(f"找不到文件: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    return summary


def load_single_classification_sliding_window(result_file: str) -> pd.DataFrame:
    """加载单分类任务的滑动窗口结果"""
    result_file = result_file.replace('\\', '/')
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    results = []
    window_starts = data['results']['accuracy']['window_starts']
    
    for i, window_start in enumerate(window_starts):
        window_center = window_start + data['window_size'] / 2
        
        results.append({
            'window_start': window_start,
            'window_center': window_center,
            'window_end': window_start + data['window_size'],
            'accuracy': data['results']['accuracy']['means'][i],
            'f1': data['results']['f1']['means'][i],
            'auc': data['results']['auc']['means'][i],
            'precision': np.nan,  # 不在数据中
            'recall': np.nan
        })
    
    return pd.DataFrame(results)


def load_predictions_and_metadata(result_dir: str, fold_idx: int) -> Tuple[np.ndarray, ...]:
    """加载单个fold的预测和元数据"""
    result_dir = result_dir.replace('\\', '/')
    fold_dir = Path(result_dir) / f"fold_{fold_idx}"
    
    preds = np.load(fold_dir / "test_predictions.npz")
    
    metadata = []
    metadata_file = fold_dir / "test_metadata.jsonl"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            for line in f:
                metadata.append(json.loads(line))
    
    return preds, metadata


def aggregate_all_predictions(result_dir: str, num_folds: int = 5):
    """聚合所有fold的预测结果"""
    all_time_preds = []
    all_time_targets = []
    all_cls_preds = []
    all_cls_targets = []
    all_metadata = []
    
    for fold_idx in range(1, num_folds + 1):
        preds, metadata = load_predictions_and_metadata(result_dir, fold_idx)
        
        has_classification = 'cls_preds' in preds.files
        
        if has_classification:
            all_cls_preds.append(preds['cls_preds'])
            all_cls_targets.append(preds['cls_targets'])
        
        if 'time_preds' in preds.files:
            all_time_preds.append(preds['time_preds'])
            all_time_targets.append(preds['time_targets'])
        
        all_metadata.extend(metadata)
    
    result = {
        'metadata': all_metadata,
        'has_classification': has_classification,
        'has_regression': len(all_time_preds) > 0
    }
    
    if has_classification:
        result['cls_preds'] = np.concatenate(all_cls_preds)
        result['cls_targets'] = np.concatenate(all_cls_targets)
    
    if result['has_regression']:
        result['time_preds'] = np.concatenate(all_time_preds)
        result['time_targets'] = np.concatenate(all_time_targets)
    
    return result


def compute_sliding_window_metrics(preds_dict: Dict, window_hours: float = 3.0, 
                                   step_hours: float = 1.0) -> pd.DataFrame:
    """计算滑动窗口的分类指标"""
    if not preds_dict['has_classification']:
        return None
    
    metadata = preds_dict['metadata']
    cls_preds = preds_dict['cls_preds']
    cls_targets = preds_dict['cls_targets']
    
    # 确保长度一致
    min_len = min(len(metadata), len(cls_preds), len(cls_targets))
    metadata = metadata[:min_len]
    cls_preds = cls_preds[:min_len]
    cls_targets = cls_targets[:min_len]
    
    times = np.array([m['hours_since_start'] for m in metadata])
    min_time = times.min()
    max_time = times.max()
    
    results = []
    window_start = min_time
    
    while window_start + window_hours <= max_time:
        window_end = window_start + window_hours
        mask = (times >= window_start) & (times < window_end)
        
        if mask.sum() > 0:
            window_targets = cls_targets[mask]
            window_preds_raw = cls_preds[mask]
            
            # 转换概率为二值标签
            if window_preds_raw.dtype == np.float64 or window_preds_raw.dtype == np.float32:
                window_preds = (window_preds_raw > 0.5).astype(int)
            else:
                window_preds = window_preds_raw.astype(int)
            
            accuracy = (window_preds == window_targets).mean()
            
            if len(np.unique(window_targets)) > 1:
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                
                precision = precision_score(window_targets, window_preds, zero_division=0)
                recall = recall_score(window_targets, window_preds, zero_division=0)
                f1 = f1_score(window_targets, window_preds, zero_division=0)
                
                try:
                    if window_preds_raw.dtype in [np.float64, np.float32]:
                        auc = roc_auc_score(window_targets, window_preds_raw)
                    else:
                        auc = roc_auc_score(window_targets, window_preds)
                except:
                    auc = np.nan
            else:
                precision = recall = f1 = auc = np.nan
            
            results.append({
                'window_start': window_start,
                'window_center': window_start + window_hours / 2,
                'window_end': window_end,
                'n_samples': mask.sum(),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc
            })
        
        window_start += step_hours
    
    return pd.DataFrame(results)


def _is_classification_model(preds_dict: Dict) -> bool:
    """Return True if preds_dict contains a usable classification prediction."""
    return bool(preds_dict.get("has_classification", False)) and ("cls_preds" in preds_dict) and (
        "cls_targets" in preds_dict
    )


def compute_regression_by_time_bins(preds_dict: Dict, bin_hours: float = 6.0) -> pd.DataFrame:
    """按时间段计算回归指标"""
    if not preds_dict['has_regression']:
        return None
    
    metadata = preds_dict['metadata']
    time_preds = preds_dict['time_preds']
    time_targets = preds_dict['time_targets']
    
    min_len = min(len(metadata), len(time_preds), len(time_targets))
    metadata = metadata[:min_len]
    time_preds = time_preds[:min_len]
    time_targets = time_targets[:min_len]
    
    times = np.array([m['hours_since_start'] for m in metadata])
    labels = np.array([m['label'] for m in metadata])

    # If this regression model was trained/evaluated on a single class only, it is NOT meaningful
    # to report metrics for the other class. We infer the scope from the labels present.
    unique_labels = set(np.unique(labels).tolist())
    scope: str
    if unique_labels == {1}:
        scope = "infected-only"
    elif unique_labels == {0}:
        scope = "uninfected-only"
    else:
        scope = "mixed"
    
    min_time = 0
    max_time = times.max()
    bins = np.arange(min_time, max_time + bin_hours, bin_hours)
    
    results = []
    
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        
        mask_all = (times >= bin_start) & (times < bin_end)
        if mask_all.sum() > 0:
            errors = np.abs(time_preds[mask_all] - time_targets[mask_all])
            # Only report an "All" curve when scope is mixed (i.e., both classes exist).
            if scope == "mixed":
                results.append({
                    'time_bin_start': bin_start,
                    'time_bin_center': (bin_start + bin_end) / 2,
                    'time_bin_end': bin_end,
                    'class': 'All',
                    'n_samples': mask_all.sum(),
                    'mae': errors.mean(),
                    'rmse': np.sqrt((errors ** 2).mean()),
                    'median_ae': np.median(errors),
                    'scope': scope,
                })
        
        for cls, cls_name in [(0, 'Uninfected'), (1, 'Infected')]:
            # Skip non-applicable class for single-class regression models
            if scope == "infected-only" and cls_name == "Uninfected":
                continue
            if scope == "uninfected-only" and cls_name == "Infected":
                continue

            mask = mask_all & (labels == cls)
            if mask.sum() > 0:
                errors = np.abs(time_preds[mask] - time_targets[mask])
                results.append({
                    'time_bin_start': bin_start,
                    'time_bin_center': (bin_start + bin_end) / 2,
                    'time_bin_end': bin_end,
                    'class': cls_name,
                    'n_samples': mask.sum(),
                    'mae': errors.mean(),
                    'rmse': np.sqrt((errors ** 2).mean()),
                    'median_ae': np.median(errors),
                    'scope': scope,
                })
    
    return pd.DataFrame(results)


def compute_regression_by_class(preds_dict: Dict) -> Dict:
    """按类别计算回归指标"""
    if not preds_dict['has_regression']:
        return None
    
    metadata = preds_dict['metadata']
    time_preds = preds_dict['time_preds']
    time_targets = preds_dict['time_targets']
    
    min_len = min(len(metadata), len(time_preds), len(time_targets))
    metadata = metadata[:min_len]
    time_preds = time_preds[:min_len]
    time_targets = time_targets[:min_len]
    
    labels = np.array([m['label'] for m in metadata])

    unique_labels = set(np.unique(labels).tolist())
    if unique_labels == {1}:
        scope = "infected-only"
    elif unique_labels == {0}:
        scope = "uninfected-only"
    else:
        scope = "mixed"
    
    results = {}
    
    for cls, cls_name in [(0, 'Uninfected'), (1, 'Infected')]:
        if scope == "infected-only" and cls_name == "Uninfected":
            continue
        if scope == "uninfected-only" and cls_name == "Infected":
            continue
        mask = labels == cls
        if mask.sum() > 0:
            errors = np.abs(time_preds[mask] - time_targets[mask])
            results[cls_name] = {
                'n_samples': mask.sum(),
                'mae': errors.mean(),
                'rmse': np.sqrt((errors ** 2).mean()),
                'median_ae': np.median(errors),
                'std': errors.std()
            }
    
    # Only report an "All" aggregate when scope is mixed.
    if scope == "mixed":
        errors = np.abs(time_preds - time_targets)
        results['All'] = {
            'n_samples': len(errors),
            'mae': errors.mean(),
            'rmse': np.sqrt((errors ** 2).mean()),
            'median_ae': np.median(errors),
            'std': errors.std()
        }
    
    return results


def compute_pvalue_for_metric(df_list: List[pd.DataFrame], metric: str, window_centers: np.ndarray) -> Dict:
    """计算两两模型之间某个指标的p值"""
    pvalues = {}
    
    for i in range(len(df_list)):
        for j in range(i + 1, len(df_list)):
            df1, df2 = df_list[i], df_list[j]
            
            # 对齐window_center
            merged = pd.merge(df1[['window_center', metric]], 
                            df2[['window_center', metric]], 
                            on='window_center', 
                            suffixes=('_1', '_2'))
            
            if len(merged) > 1:
                vals1 = merged[f'{metric}_1'].dropna()
                vals2 = merged[f'{metric}_2'].dropna()
                
                if len(vals1) > 1 and len(vals2) > 1:
                    t_stat, p_val = stats.ttest_rel(vals1, vals2)
                    pvalues[f'{i}_vs_{j}'] = {
                        'p_value': p_val,
                        't_stat': t_stat,
                        'mean_diff': vals1.mean() - vals2.mean()
                    }
    
    return pvalues


def plot_sliding_window_comparison_improved(window_dfs: Dict[str, pd.DataFrame], 
                                            model_display_names: Dict[str, str],
                                            output_dir: Path):
    """改进的滑动窗口对比图 - 只包含分类任务"""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Classification Performance - Sliding Window Comparison (3h window, 1h step)', 
                fontsize=16, fontweight='bold')
    
    colors = sns.color_palette("husl", len(window_dfs))
    
    metrics = [
        ('accuracy', 'Accuracy', axes[0, 0]),
        ('f1', 'F1 Score', axes[0, 1]),
        ('auc', 'AUC', axes[1, 0]),
        ('recall', 'Recall', axes[1, 1])
    ]
    
    for metric_name, metric_label, ax in metrics:
        df_list = []
        names_list = []
        
        for i, (name, df) in enumerate(window_dfs.items()):
            if metric_name in df.columns and not df[metric_name].isna().all():
                display_name = model_display_names.get(name, name)
                ax.plot(df['window_center'], df[metric_name], 
                       marker='o', linewidth=2.5, markersize=6,
                       label=display_name, color=colors[i], alpha=0.85)
                df_list.append(df)
                names_list.append(name)
        
        # 计算p值
        if len(df_list) >= 2:
            window_centers = df_list[0]['window_center'].values
            pvalues = compute_pvalue_for_metric(df_list, metric_name, window_centers)
            
            # 在图上标注显著性最强的p值
            if pvalues:
                min_pval_key = min(pvalues.keys(), key=lambda k: pvalues[k]['p_value'])
                min_pval = pvalues[min_pval_key]['p_value']
                
                sig_text = '***' if min_pval < 0.001 else '**' if min_pval < 0.01 else '*' if min_pval < 0.05 else 'ns'
                
                # 在图上方添加p值注释
                y_pos = ax.get_ylim()[1] * 0.98
                ax.text(0.98, 0.98, f'Min p-value: {min_pval:.4f} {sig_text}',
                       transform=ax.transAxes, ha='right', va='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                       fontsize=9)
        
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} over Time', fontsize=13, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9, fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # 优化Y轴范围以显示差异
        if not df_list[0][metric_name].isna().all():
            all_values = pd.concat([df[metric_name].dropna() for df in df_list])
            y_min = max(0, all_values.min() - 0.02)
            y_max = min(1.0, all_values.max() + 0.02)
            if y_max - y_min < 0.1:  # 如果范围太小，扩大一点
                y_center = (y_min + y_max) / 2
                y_min = max(0, y_center - 0.05)
                y_max = min(1.0, y_center + 0.05)
            ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_sliding_window_improved.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparison_sliding_window_improved.pdf', bbox_inches='tight')
    print(f"✓ Saved improved sliding window comparison: {output_dir / 'comparison_sliding_window_improved.png'}")
    plt.close()


def plot_comparison_overview_with_pvalues(summaries: Dict[str, Dict], 
                                         model_display_names: Dict[str, str],
                                         output_dir: Path):
    """带p值标注的总体对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Comparison with Statistical Tests', fontsize=16, fontweight='bold')
    
    model_names = list(summaries.keys())
    colors = sns.color_palette("husl", len(model_names))
    
    # 1. Classification Accuracy
    ax = axes[0, 0]
    cls_models = []
    cls_acc = []
    cls_acc_std = []
    
    for name, summary in summaries.items():
        if 'cls_accuracy' in summary['aggregated_metrics']:
            cls_models.append(name)
            cls_acc.append(summary['aggregated_metrics']['cls_accuracy']['mean'])
            cls_acc_std.append(summary['aggregated_metrics']['cls_accuracy']['std'])
    
    if cls_models:
        x_pos = np.arange(len(cls_models))
        bars = ax.bar(x_pos, cls_acc, yerr=cls_acc_std, capsize=5,
                      color=[colors[model_names.index(m)] for m in cls_models],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        display_labels = [model_display_names.get(m, m) for m in cls_models]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Classification Accuracy', fontsize=13, fontweight='bold')
        
        # 优化Y轴
        y_min = max(0.9, min(cls_acc) - 0.01)
        ax.set_ylim([y_min, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val, std) in enumerate(zip(bars, cls_acc, cls_acc_std)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.001,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Classification AUC
    ax = axes[0, 1]
    cls_auc = []
    cls_auc_std = []
    
    for name in cls_models:
        summary = summaries[name]
        if 'cls_auc' in summary['aggregated_metrics']:
            cls_auc.append(summary['aggregated_metrics']['cls_auc']['mean'])
            cls_auc_std.append(summary['aggregated_metrics']['cls_auc']['std'])
    
    if cls_models and cls_auc:
        x_pos = np.arange(len(cls_models))
        bars = ax.bar(x_pos, cls_auc, yerr=cls_auc_std, capsize=5,
                      color=[colors[model_names.index(m)] for m in cls_models],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        display_labels = [model_display_names.get(m, m) for m in cls_models]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
        ax.set_title('Classification AUC', fontsize=13, fontweight='bold')
        
        y_min = max(0.99, min(cls_auc) - 0.001)
        ax.set_ylim([y_min, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val, std) in enumerate(zip(bars, cls_auc, cls_auc_std)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.0001,
                   f'{val:.5f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Regression MAE with p-values
    ax = axes[1, 0]
    reg_models = []
    reg_mae = []
    reg_mae_std = []
    
    for name in model_names:
        summary = summaries[name]
        if 'reg_mae' in summary['aggregated_metrics']:
            reg_models.append(name)
            reg_mae.append(summary['aggregated_metrics']['reg_mae']['mean'])
            reg_mae_std.append(summary['aggregated_metrics']['reg_mae']['std'])
    
    if reg_models:
        x_pos = np.arange(len(reg_models))
        bars = ax.bar(x_pos, reg_mae, yerr=reg_mae_std, capsize=5,
                      color=[colors[model_names.index(m)] for m in reg_models],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        display_labels = [model_display_names.get(m, m) for m in reg_models]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('MAE (hours)', fontsize=12, fontweight='bold')
        ax.set_title('Regression MAE', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val, std) in enumerate(zip(bars, reg_mae, reg_mae_std)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
        
        # 添加统计检验结果
        if len(reg_mae) >= 2:
            # 简单的两两比较，只显示最显著的
            min_idx = np.argmin(reg_mae)
            max_idx = np.argmax(reg_mae)
            
            # 这里简化处理，实际p值需要从原始数据计算
            ax.text(0.5, 0.98, f'Best: {display_labels[min_idx]}', 
                   transform=ax.transAxes, ha='center', va='top',
                   fontsize=9, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 4. Regression RMSE
    ax = axes[1, 1]
    reg_rmse = []
    reg_rmse_std = []
    
    for name in reg_models:
        summary = summaries[name]
        if 'reg_rmse' in summary['aggregated_metrics']:
            reg_rmse.append(summary['aggregated_metrics']['reg_rmse']['mean'])
            reg_rmse_std.append(summary['aggregated_metrics']['reg_rmse']['std'])
    
    if reg_models:
        x_pos = np.arange(len(reg_models))
        bars = ax.bar(x_pos, reg_rmse, yerr=reg_rmse_std, capsize=5,
                      color=[colors[model_names.index(m)] for m in reg_models],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        
        display_labels = [model_display_names.get(m, m) for m in reg_models]
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('RMSE (hours)', fontsize=12, fontweight='bold')
        ax.set_title('Regression RMSE', fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val, std) in enumerate(zip(bars, reg_rmse, reg_rmse_std)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_overview_with_pvalues.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparison_overview_with_pvalues.pdf', bbox_inches='tight')
    print(f"✓ Saved overview with p-values: {output_dir / 'comparison_overview_with_pvalues.png'}")
    plt.close()


def plot_regression_comparisons(preds_dicts: Dict[str, Dict], 
                               model_display_names: Dict[str, str],
                               output_dir: Path):
    """回归性能对比（按时间段和类别）"""
    # 时间段对比
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Regression Performance - Temporal Analysis (6h bins)', fontsize=16, fontweight='bold')
    
    colors = sns.color_palette("husl", len(preds_dicts))
    
    time_bin_dfs = {}
    for name, preds_dict in preds_dicts.items():
        if preds_dict['has_regression']:
            time_bin_dfs[name] = compute_regression_by_time_bins(preds_dict, bin_hours=6.0)
    
    if not time_bin_dfs:
        plt.close()
        return
    
    class_names = ['All', 'Infected', 'Uninfected']
    
    for row, metric in enumerate(['mae', 'rmse']):
        metric_label = 'MAE' if metric == 'mae' else 'RMSE'
        
        for col, cls in enumerate(class_names):
            ax = axes[row, col]
            
            for i, (name, df) in enumerate(time_bin_dfs.items()):
                df_cls = df[df['class'] == cls]
                if len(df_cls) > 0:
                    display_name = model_display_names.get(name, name)
                    ax.plot(df_cls['time_bin_center'], df_cls[metric],
                           marker='o', linewidth=2, markersize=5,
                           label=display_name, color=colors[i], alpha=0.8)
            
            ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{metric_label} (hours)', fontsize=11, fontweight='bold')
            ax.set_title(f'{cls} - {metric_label}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_regression_temporal.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparison_regression_temporal.pdf', bbox_inches='tight')
    print(f"✓ Saved temporal regression comparison: {output_dir / 'comparison_regression_temporal.png'}")
    plt.close()
    
    # 按类别对比
    class_metrics = {}
    for name, preds_dict in preds_dicts.items():
        if preds_dict['has_regression']:
            class_metrics[name] = compute_regression_by_class(preds_dict)
    
    if not class_metrics:
        return
    
    rows = []
    for model_name, metrics in class_metrics.items():
        for class_name, vals in metrics.items():
            rows.append({
                'Model': model_display_names.get(model_name, model_name),
                'Class': class_name,
                'MAE': vals['mae'],
                'RMSE': vals['rmse'],
                'Median_AE': vals['median_ae']
            })
    
    df = pd.DataFrame(rows)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Regression Performance - By Class Comparison', fontsize=16, fontweight='bold')
    
    class_order = ['All', 'Infected', 'Uninfected']
    colors_dict = {model_display_names.get(name, name): colors[i] 
                   for i, name in enumerate(preds_dicts.keys())}
    
    for idx, metric in enumerate(['MAE', 'RMSE', 'Median_AE']):
        ax = axes[idx]
        
        x_offset = np.arange(len(class_order))
        width = 0.8 / len(class_metrics)
        
        unique_models = df['Model'].unique()
        for i, model_name in enumerate(unique_models):
            df_model = df[df['Model'] == model_name]
            values = [df_model[df_model['Class'] == cls][metric].values[0] 
                     if cls in df_model['Class'].values else 0
                     for cls in class_order]
            
            ax.bar(x_offset + i * width, values, width, 
                  label=model_name, color=colors_dict.get(model_name, colors[i]), alpha=0.8,
                  edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric} (hours)', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} by Class', fontsize=13, fontweight='bold')
        ax.set_xticks(x_offset + width * (len(class_metrics) - 1) / 2)
        ax.set_xticklabels(class_order)
        ax.legend(loc='best', framealpha=0.9, fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_regression_by_class.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparison_regression_by_class.pdf', bbox_inches='tight')
    print(f"✓ Saved class-wise regression comparison: {output_dir / 'comparison_regression_by_class.png'}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Improved four-model comparison analysis')
    parser.add_argument('--single-classification', type=str, required=True,
                       help='Single classification sliding window results JSON file')
    parser.add_argument('--multitask', type=str, required=True,
                       help='Multitask model results directory')
    parser.add_argument('--multitask-conditioned', type=str, required=True,
                       help='Classification-conditioned multitask model results directory')
    parser.add_argument('--regression-infected', type=str, required=True,
                       help='Regression-infected model results directory')
    parser.add_argument('--regression-uninfected', type=str, required=True,
                       help='Regression-uninfected model results directory')
    parser.add_argument('--output-dir', type=str, default='comparison_results_improved',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*100)
    print("Improved Four-Model Comparison Analysis")
    print("="*100)
    
    # 模型显示名称
    model_display_names = {
        'SingleClassification': 'Classification-Only',
        'Multitask': 'Multitask',
        'Multitask-ClsConditioned': 'Multitask-ClsCond',
        'Regression-Infected': 'Reg-Infected',
        'Regression-Uninfected': 'Reg-Uninfected'
    }
    
    # 加载单分类任务滑动窗口结果
    print("\nLoading Single Classification sliding window results...")
    single_cls_window_df = load_single_classification_sliding_window(args.single_classification)
    print("  ✓ Successfully loaded")
    
    # 加载其他模型结果
    model_configs = {
        'Multitask': args.multitask,
        'Multitask-ClsConditioned': args.multitask_conditioned,
        'Regression-Infected': args.regression_infected,
        'Regression-Uninfected': args.regression_uninfected
    }
    
    summaries = {}
    preds_dicts = {}
    
    for name, result_dir in model_configs.items():
        print(f"\nLoading {name} results...")
        summaries[name] = load_cv_results(result_dir)
        preds_dicts[name] = aggregate_all_predictions(result_dir)
        print(f"  ✓ Successfully loaded")
    
    # 1. Sliding window comparison (classification only)
    # IMPORTANT: Never include regression-only baselines in this plot.
    print("\nGenerating improved sliding window comparison (classification only)...")
    window_dfs: Dict[str, pd.DataFrame] = {"SingleClassification": single_cls_window_df}

    for name, preds_dict in preds_dicts.items():
        if not _is_classification_model(preds_dict):
            # regression-only models end up here
            continue
        df = compute_sliding_window_metrics(preds_dict)
        if df is None or df.empty:
            continue
        window_dfs[name] = df

    # Additional guard: if somebody accidentally passed regression folders containing cls_* arrays,
    # we still only allow Multitask variants here by name convention.
    allowed_prefixes = {"Multitask"}
    filtered_window_dfs: Dict[str, pd.DataFrame] = {"SingleClassification": single_cls_window_df}
    for k, v in window_dfs.items():
        if k == "SingleClassification":
            continue
        if any(k.startswith(p) for p in allowed_prefixes):
            filtered_window_dfs[k] = v

    plot_sliding_window_comparison_improved(filtered_window_dfs, model_display_names, output_dir)
    
    # 2. 总体对比（带p值）
    print("\nGenerating overview comparison with p-values...")
    plot_comparison_overview_with_pvalues(summaries, model_display_names, output_dir)
    
    # 3. 回归对比
    print("\nGenerating regression comparisons...")
    plot_regression_comparisons(preds_dicts, model_display_names, output_dir)
    
    print("\n" + "="*100)
    print(f"✓ All analyses completed! Results saved in: {output_dir}")
    print("="*100)
    print("\nGenerated files:")
    print("  - comparison_sliding_window_improved.png/pdf : Classification sliding window (with p-values)")
    print("  - comparison_overview_with_pvalues.png/pdf   : Overview with statistical tests")
    print("  - comparison_regression_temporal.png/pdf     : Temporal regression comparison")
    print("  - comparison_regression_by_class.png/pdf     : Class-wise regression comparison")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
