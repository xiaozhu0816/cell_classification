"""
全面对比四个模型的性能
包括：
1. Classification性能对比（整体 + 滑动窗口）
2. Regression性能对比（整体 + 按时间段 + 按类别）
3. 综合对比和统计分析
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

# 设置样式（使用英文避免字体问题）
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['font.family'] = 'DejaVu Sans'


def load_cv_results(result_dir: str) -> Dict:
    """加载CV结果"""
    # 规范化路径，处理Windows/Linux路径分隔符
    result_dir = result_dir.replace('\\', '/')
    result_path = Path(result_dir)
    summary_file = result_path / "cv_summary.json"
    
    if not summary_file.exists():
        raise FileNotFoundError(f"找不到文件: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    return summary


def load_predictions_and_metadata(result_dir: str, fold_idx: int) -> Tuple[np.ndarray, ...]:
    """加载单个fold的预测和元数据"""
    # 规范化路径
    result_dir = result_dir.replace('\\', '/')
    fold_dir = Path(result_dir) / f"fold_{fold_idx}"
    
    # 加载预测
    preds = np.load(fold_dir / "test_predictions.npz")
    
    # 加载元数据
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
        
        # 检查是否有分类预测（multitask模型）
        has_classification = 'cls_preds' in preds.files
        
        if has_classification:
            all_cls_preds.append(preds['cls_preds'])
            all_cls_targets.append(preds['cls_targets'])
        
        # 检查是否有回归预测
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
    
    # 获取时间范围
    times = np.array([m['hours_since_start'] for m in metadata])
    min_time = times.min()
    max_time = times.max()
    
    results = []
    window_start = min_time
    
    while window_start + window_hours <= max_time:
        window_end = window_start + window_hours
        
        # 找到窗口内的样本
        mask = (times >= window_start) & (times < window_end)
        
        if mask.sum() > 0:
            window_targets = cls_targets[mask]
            window_preds_raw = cls_preds[mask]
            
            # 如果预测是概率值，转换为二值标签
            if window_preds_raw.dtype == np.float64 or window_preds_raw.dtype == np.float32:
                window_preds = (window_preds_raw > 0.5).astype(int)
            else:
                window_preds = window_preds_raw.astype(int)
            
            # 计算指标
            accuracy = (window_preds == window_targets).mean()
            
            # 如果窗口内有两个类别，计算更多指标
            if len(np.unique(window_targets)) > 1:
                from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
                
                precision = precision_score(window_targets, window_preds, zero_division=0)
                recall = recall_score(window_targets, window_preds, zero_division=0)
                f1 = f1_score(window_targets, window_preds, zero_division=0)
                
                # 如果有概率预测，计算AUC
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


def compute_regression_by_time_bins(preds_dict: Dict, bin_hours: float = 6.0) -> pd.DataFrame:
    """按时间段计算回归指标"""
    if not preds_dict['has_regression']:
        return None
    
    metadata = preds_dict['metadata']
    time_preds = preds_dict['time_preds']
    time_targets = preds_dict['time_targets']
    
    # 确保长度一致
    min_len = min(len(metadata), len(time_preds), len(time_targets))
    metadata = metadata[:min_len]
    time_preds = time_preds[:min_len]
    time_targets = time_targets[:min_len]
    
    # 获取时间和类别
    times = np.array([m['hours_since_start'] for m in metadata])
    labels = np.array([m['label'] for m in metadata])
    
    # 创建时间bins
    min_time = 0
    max_time = times.max()
    bins = np.arange(min_time, max_time + bin_hours, bin_hours)
    
    results = []
    
    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]
        
        # 总体
        mask_all = (times >= bin_start) & (times < bin_end)
        if mask_all.sum() > 0:
            errors = np.abs(time_preds[mask_all] - time_targets[mask_all])
            results.append({
                'time_bin_start': bin_start,
                'time_bin_center': (bin_start + bin_end) / 2,
                'time_bin_end': bin_end,
                'class': 'All',
                'n_samples': mask_all.sum(),
                'mae': errors.mean(),
                'rmse': np.sqrt((errors ** 2).mean()),
                'median_ae': np.median(errors)
            })
        
        # 按类别
        for cls, cls_name in [(0, 'Uninfected'), (1, 'Infected')]:
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
                    'median_ae': np.median(errors)
                })
    
    return pd.DataFrame(results)


def compute_regression_by_class(preds_dict: Dict) -> Dict:
    """按类别计算回归指标"""
    if not preds_dict['has_regression']:
        return None
    
    metadata = preds_dict['metadata']
    time_preds = preds_dict['time_preds']
    time_targets = preds_dict['time_targets']
    
    # 确保长度一致
    min_len = min(len(metadata), len(time_preds), len(time_targets))
    metadata = metadata[:min_len]
    time_preds = time_preds[:min_len]
    time_targets = time_targets[:min_len]
    
    labels = np.array([m['label'] for m in metadata])
    
    results = {}
    
    for cls, cls_name in [(0, 'Uninfected'), (1, 'Infected')]:
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
    
    # 总体
    errors = np.abs(time_preds - time_targets)
    results['All'] = {
        'n_samples': len(errors),
        'mae': errors.mean(),
        'rmse': np.sqrt((errors ** 2).mean()),
        'median_ae': np.median(errors),
        'std': errors.std()
    }
    
    return results


def plot_comparison_overview(summaries: Dict[str, Dict], output_dir: Path):
    """绘制总体对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Model Comparison (4 Models)', fontsize=16, fontweight='bold')
    
    model_names = list(summaries.keys())
    colors = sns.color_palette("husl", len(model_names))
    
    # 1. Classification Accuracy (仅multitask模型)
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
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Classification Accuracy', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cls_models, rotation=15, ha='right')
        ax.set_ylim([min(cls_acc) - 0.01, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # 添加数值标签
        for i, (bar, val, std) in enumerate(zip(bars, cls_acc, cls_acc_std)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.002,
                   f'{val:.4f}±{std:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Classification AUC (仅multitask模型)
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
        ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
        ax.set_title('Classification AUC', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(cls_models, rotation=15, ha='right')
        ax.set_ylim([min(cls_auc) - 0.001, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val, std) in enumerate(zip(bars, cls_auc, cls_auc_std)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.0002,
                   f'{val:.5f}±{std:.5f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Regression MAE (所有模型)
    ax = axes[1, 0]
    reg_mae = []
    reg_mae_std = []
    
    for name in model_names:
        summary = summaries[name]
        if 'reg_mae' in summary['aggregated_metrics']:
            reg_mae.append(summary['aggregated_metrics']['reg_mae']['mean'])
            reg_mae_std.append(summary['aggregated_metrics']['reg_mae']['std'])
        else:
            reg_mae.append(np.nan)
            reg_mae_std.append(0)
    
    valid_indices = [i for i, v in enumerate(reg_mae) if not np.isnan(v)]
    if valid_indices:
        x_pos = np.arange(len(valid_indices))
        valid_names = [model_names[i] for i in valid_indices]
        valid_mae = [reg_mae[i] for i in valid_indices]
        valid_std = [reg_mae_std[i] for i in valid_indices]
        
        bars = ax.bar(x_pos, valid_mae, yerr=valid_std, capsize=5,
                      color=[colors[i] for i in valid_indices],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('MAE (hours)', fontsize=12, fontweight='bold')
        ax.set_title('Regression MAE', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_names, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val, std) in enumerate(zip(bars, valid_mae, valid_std)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                   f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 4. Regression RMSE (所有模型)
    ax = axes[1, 1]
    reg_rmse = []
    reg_rmse_std = []
    
    for name in model_names:
        summary = summaries[name]
        if 'reg_rmse' in summary['aggregated_metrics']:
            reg_rmse.append(summary['aggregated_metrics']['reg_rmse']['mean'])
            reg_rmse_std.append(summary['aggregated_metrics']['reg_rmse']['std'])
        else:
            reg_rmse.append(np.nan)
            reg_rmse_std.append(0)
    
    valid_indices = [i for i, v in enumerate(reg_rmse) if not np.isnan(v)]
    if valid_indices:
        x_pos = np.arange(len(valid_indices))
        valid_names = [model_names[i] for i in valid_indices]
        valid_rmse = [reg_rmse[i] for i in valid_indices]
        valid_std = [reg_rmse_std[i] for i in valid_indices]
        
        bars = ax.bar(x_pos, valid_rmse, yerr=valid_std, capsize=5,
                      color=[colors[i] for i in valid_indices],
                      alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('RMSE (hours)', fontsize=12, fontweight='bold')
        ax.set_title('Regression RMSE', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(valid_names, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        for i, (bar, val, std) in enumerate(zip(bars, valid_rmse, valid_std)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.02,
                   f'{val:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparison_overview.pdf', bbox_inches='tight')
    print(f"✓ 保存总体对比图: {output_dir / 'comparison_overview.png'}")
    plt.close()


def plot_sliding_window_comparison(preds_dicts: Dict[str, Dict], output_dir: Path):
    """绘制滑动窗口分类性能对比"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Classification Performance - Sliding Window (3h window, 1h step)', fontsize=16, fontweight='bold')
    
    colors = sns.color_palette("husl", len(preds_dicts))
    
    # 计算所有模型的滑动窗口指标
    window_dfs = {}
    for name, preds_dict in preds_dicts.items():
        if preds_dict['has_classification']:
            window_dfs[name] = compute_sliding_window_metrics(preds_dict)
    
    if not window_dfs:
        print("⚠ No classification models, skipping sliding window comparison")
        plt.close()
        return
    
    metrics = [
        ('accuracy', 'Accuracy', axes[0, 0]),
        ('f1', 'F1 Score', axes[0, 1]),
        ('precision', 'Precision', axes[1, 0]),
        ('recall', 'Recall', axes[1, 1])
    ]
    
    for metric_name, metric_label, ax in metrics:
        for i, (name, df) in enumerate(window_dfs.items()):
            if metric_name in df.columns:
                ax.plot(df['window_center'], df[metric_name], 
                       marker='o', linewidth=2, markersize=4,
                       label=name, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
        ax.set_title(f'{metric_label} over Time', fontsize=13, fontweight='bold')
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_sliding_window.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparison_sliding_window.pdf', bbox_inches='tight')
    print(f"✓ 保存滑动窗口对比图: {output_dir / 'comparison_sliding_window.png'}")
    plt.close()


def plot_regression_time_bins_comparison(preds_dicts: Dict[str, Dict], output_dir: Path):
    """绘制按时间段的回归性能对比"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Regression Performance - Temporal Analysis (6h bins)', fontsize=16, fontweight='bold')
    
    colors = sns.color_palette("husl", len(preds_dicts))
    
    # 计算所有模型的时间段指标
    time_bin_dfs = {}
    for name, preds_dict in preds_dicts.items():
        if preds_dict['has_regression']:
            time_bin_dfs[name] = compute_regression_by_time_bins(preds_dict, bin_hours=6.0)
    
    if not time_bin_dfs:
        print("⚠ No regression models, skipping temporal bin comparison")
        plt.close()
        return
    
    # 为每个类别绘制MAE和RMSE
    class_names = ['All', 'Infected', 'Uninfected']
    
    for row, metric in enumerate(['mae', 'rmse']):
        metric_label = 'MAE' if metric == 'mae' else 'RMSE'
        
        for col, cls in enumerate(class_names):
            ax = axes[row, col]
            
            for i, (name, df) in enumerate(time_bin_dfs.items()):
                df_cls = df[df['class'] == cls]
                if len(df_cls) > 0:
                    ax.plot(df_cls['time_bin_center'], df_cls[metric],
                           marker='o', linewidth=2, markersize=5,
                           label=name, color=colors[i], alpha=0.8)
            
            ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{metric_label} (hours)', fontsize=11, fontweight='bold')
            ax.set_title(f'{cls} - {metric_label}', fontsize=12, fontweight='bold')
            ax.legend(loc='best', framealpha=0.9, fontsize=9)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_regression_time_bins.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparison_regression_time_bins.pdf', bbox_inches='tight')
    print(f"✓ 保存时间段回归对比图: {output_dir / 'comparison_regression_time_bins.png'}")
    plt.close()


def plot_regression_by_class_comparison(preds_dicts: Dict[str, Dict], output_dir: Path):
    """绘制按类别的回归性能对比"""
    # 计算所有模型的按类别指标
    class_metrics = {}
    for name, preds_dict in preds_dicts.items():
        if preds_dict['has_regression']:
            class_metrics[name] = compute_regression_by_class(preds_dict)
    
    if not class_metrics:
        print("⚠ No regression models, skipping class-wise comparison")
        return
    
    # 创建DataFrame
    rows = []
    for model_name, metrics in class_metrics.items():
        for class_name, vals in metrics.items():
            rows.append({
                'Model': model_name,
                'Class': class_name,
                'MAE': vals['mae'],
                'RMSE': vals['rmse'],
                'Median_AE': vals['median_ae'],
                'N_Samples': vals['n_samples']
            })
    
    df = pd.DataFrame(rows)
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Regression Performance - By Class Comparison', fontsize=16, fontweight='bold')
    
    class_order = ['All', 'Infected', 'Uninfected']
    colors = sns.color_palette("husl", len(class_metrics))
    
    for idx, metric in enumerate(['MAE', 'RMSE', 'Median_AE']):
        ax = axes[idx]
        
        # 为每个类别创建分组柱状图
        x_offset = np.arange(len(class_order))
        width = 0.8 / len(class_metrics)
        
        for i, model_name in enumerate(class_metrics.keys()):
            df_model = df[df['Model'] == model_name]
            values = [df_model[df_model['Class'] == cls][metric].values[0] 
                     if cls in df_model['Class'].values else 0
                     for cls in class_order]
            
            ax.bar(x_offset + i * width, values, width, 
                  label=model_name, color=colors[i], alpha=0.8,
                  edgecolor='black', linewidth=1)
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'{metric} (hours)', fontsize=12, fontweight='bold')
        ax.set_title(f'{metric} by Class', fontsize=13, fontweight='bold')
        ax.set_xticks(x_offset + width * (len(class_metrics) - 1) / 2)
        ax.set_xticklabels(class_order)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_regression_by_class.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'comparison_regression_by_class.pdf', bbox_inches='tight')
    print(f"✓ 保存按类别回归对比图: {output_dir / 'comparison_regression_by_class.png'}")
    plt.close()
    
    return df


def generate_summary_table(summaries: Dict[str, Dict], preds_dicts: Dict[str, Dict], 
                          output_dir: Path):
    """生成综合对比表格"""
    rows = []
    
    for name in summaries.keys():
        summary = summaries[name]
        preds = preds_dicts.get(name, {})
        
        row = {'Model': name}
        
        # Classification指标
        if 'cls_accuracy' in summary['aggregated_metrics']:
            row['Cls_Accuracy'] = f"{summary['aggregated_metrics']['cls_accuracy']['mean']:.4f}±{summary['aggregated_metrics']['cls_accuracy']['std']:.4f}"
            row['Cls_F1'] = f"{summary['aggregated_metrics']['cls_f1']['mean']:.4f}±{summary['aggregated_metrics']['cls_f1']['std']:.4f}"
            row['Cls_AUC'] = f"{summary['aggregated_metrics']['cls_auc']['mean']:.5f}±{summary['aggregated_metrics']['cls_auc']['std']:.5f}"
        else:
            row['Cls_Accuracy'] = 'N/A'
            row['Cls_F1'] = 'N/A'
            row['Cls_AUC'] = 'N/A'
        
        # Regression指标
        if 'reg_mae' in summary['aggregated_metrics']:
            row['Reg_MAE'] = f"{summary['aggregated_metrics']['reg_mae']['mean']:.3f}±{summary['aggregated_metrics']['reg_mae']['std']:.3f}"
            row['Reg_RMSE'] = f"{summary['aggregated_metrics']['reg_rmse']['mean']:.3f}±{summary['aggregated_metrics']['reg_rmse']['std']:.3f}"
            
            # 按类别的回归指标
            if preds.get('has_regression', False):
                class_metrics = compute_regression_by_class(preds)
                if 'Infected' in class_metrics:
                    row['Reg_MAE_Infected'] = f"{class_metrics['Infected']['mae']:.3f}"
                if 'Uninfected' in class_metrics:
                    row['Reg_MAE_Uninfected'] = f"{class_metrics['Uninfected']['mae']:.3f}"
        else:
            row['Reg_MAE'] = 'N/A'
            row['Reg_RMSE'] = 'N/A'
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # 保存为CSV
    csv_path = output_dir / 'comparison_summary_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ 保存对比表格: {csv_path}")
    
    # 打印到控制台
    print("\n" + "="*100)
    print("Comprehensive Comparison Table")
    print("="*100)
    print(df.to_string(index=False))
    print("="*100 + "\n")
    
    return df


def perform_statistical_tests(preds_dicts: Dict[str, Dict], output_dir: Path):
    """进行统计显著性检验"""
    print("\n" + "="*100)
    print("Statistical Significance Tests (Paired t-test)")
    print("="*100)
    
    results = []
    
    # 获取有回归的模型
    reg_models = {name: preds for name, preds in preds_dicts.items() 
                  if preds.get('has_regression', False)}
    
    if len(reg_models) < 2:
        print("⚠ Need at least two regression models for comparison")
        return
    
    # 两两比较
    model_names = list(reg_models.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name1, name2 = model_names[i], model_names[j]
            preds1, preds2 = reg_models[name1], reg_models[name2]
            
            # 确保长度一致
            min_len1 = min(len(preds1['metadata']), len(preds1['time_preds']), len(preds1['time_targets']))
            min_len2 = min(len(preds2['metadata']), len(preds2['time_preds']), len(preds2['time_targets']))
            
            # 计算误差
            errors1 = np.abs(preds1['time_preds'][:min_len1] - preds1['time_targets'][:min_len1])
            errors2 = np.abs(preds2['time_preds'][:min_len2] - preds2['time_targets'][:min_len2])
            
            # 确保样本数量相同
            min_len = min(len(errors1), len(errors2))
            errors1 = errors1[:min_len]
            errors2 = errors2[:min_len]
            
            # 配对t检验
            t_stat, p_value = stats.ttest_rel(errors1, errors2)
            
            results.append({
                'Model_1': name1,
                'Model_2': name2,
                'Mean_MAE_1': errors1.mean(),
                'Mean_MAE_2': errors2.mean(),
                'Difference': errors1.mean() - errors2.mean(),
                't_statistic': t_stat,
                'p_value': p_value,
                'Significant': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            })
            
            print(f"\n{name1} vs {name2}:")
            print(f"  MAE: {errors1.mean():.4f} vs {errors2.mean():.4f}")
            print(f"  Difference: {errors1.mean() - errors2.mean():.4f}")
            print(f"  t-statistic: {t_stat:.4f}, p-value: {p_value:.4e} {results[-1]['Significant']}")
    
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'statistical_tests.csv', index=False)
    print(f"\n✓ Saved statistical test results: {output_dir / 'statistical_tests.csv'}")
    print("="*100 + "\n")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='全面对比四个模型的性能')
    parser.add_argument('--multitask', type=str, required=True,
                       help='Multitask模型结果目录')
    parser.add_argument('--multitask-conditioned', type=str, required=True,
                       help='Classification-conditioned multitask模型结果目录')
    parser.add_argument('--regression-infected', type=str, required=True,
                       help='Regression-infected模型结果目录')
    parser.add_argument('--regression-uninfected', type=str, required=True,
                       help='Regression-uninfected模型结果目录')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("\n" + "="*100)
    print("Comprehensive Four-Model Comparison Analysis")
    print("="*100)
    
    # 加载所有模型的结果
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
    
    # 1. 绘制总体对比
    print("\nGenerating overview comparison plots...")
    plot_comparison_overview(summaries, output_dir)
    
    # 2. 绘制滑动窗口对比
    print("\nGenerating sliding window comparison plots...")
    plot_sliding_window_comparison(preds_dicts, output_dir)
    
    # 3. 绘制时间段回归对比
    print("\nGenerating temporal regression comparison plots...")
    plot_regression_time_bins_comparison(preds_dicts, output_dir)
    
    # 4. 绘制按类别回归对比
    print("\nGenerating class-wise regression comparison plots...")
    class_df = plot_regression_by_class_comparison(preds_dicts, output_dir)
    
    # 5. 生成综合表格
    print("\nGenerating comprehensive comparison table...")
    summary_df = generate_summary_table(summaries, preds_dicts, output_dir)
    
    # 6. 统计显著性检验
    print("\nPerforming statistical significance tests...")
    stats_df = perform_statistical_tests(preds_dicts, output_dir)
    
    # 保存完整的JSON报告
    full_report = {
        'model_configs': model_configs,
        'summaries': {name: summary['aggregated_metrics'] 
                     for name, summary in summaries.items()},
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    with open(output_dir / 'full_comparison_report.json', 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print("\n" + "="*100)
    print(f"✓ All comparison analyses completed! Results saved in: {output_dir}")
    print("="*100)
    print("\nGenerated files:")
    print("  - comparison_overview.png/pdf          : Overall performance comparison")
    print("  - comparison_sliding_window.png/pdf    : Sliding window classification comparison")
    print("  - comparison_regression_time_bins.png/pdf : Temporal regression comparison")
    print("  - comparison_regression_by_class.png/pdf  : Class-wise regression comparison")
    print("  - comparison_summary_table.csv         : Comprehensive comparison table")
    print("  - statistical_tests.csv                : Statistical significance test results")
    print("  - full_comparison_report.json          : Complete JSON report")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
