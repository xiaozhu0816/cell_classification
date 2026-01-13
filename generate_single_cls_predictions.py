# Generate complete test predictions for single classification model (interval 1-46)""""""

# Using existing 5-fold checkpoints


import argparse

import json

from pathlib import Path

import numpy as np""""""

import torch

import torch.nn as nn

from torch.utils.data import DataLoader

from tqdm import tqdmimport argparseimport argparse

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

import jsonimport json

from models.resnet import ResNetClassifier

from datasets import build_datasets, resolve_frame_policiesfrom pathlib import Pathfrom pathlib import Path

from utils import build_transforms, load_config, set_seed

import numpy as npimport numpy as np



def meta_batch_to_list(meta_batch):import torchimport torch

    """Convert batched metadata to list of dicts."""

    if isinstance(meta_batch, list):import torch.nn as nnimport torch.nn as nn

        return meta_batch

    from torch.utils.data import DataLoaderfrom torch.utils.data import DataLoader

    batch_size = len(meta_batch[list(meta_batch.keys())[0]])

    meta_list = []from tqdm import tqdmfrom tqdm import tqdm

    for i in range(batch_size):

        meta_dict = {k: v[i] if isinstance(v, (list, tuple, torch.Tensor)) else v from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_scorefrom sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

                     for k, v in meta_batch.items()}

        meta_list.append(meta_dict)import torchvision.models as models

    return meta_list

from models.resnet import ResNetClassifier



def evaluate_fold(model, dataloader, device):from datasets import build_datasets, resolve_frame_policiesfrom datasets import build_datasets, resolve_frame_policies

    """Evaluate a single fold"""

    model.eval()from utils import build_transforms, load_config, set_seedfrom utils import build_transforms, load_config, set_seed

    

    all_preds = []

    all_probs = []

    all_targets = []

    all_metadata = []

    def meta_batch_to_list(meta_batch):def build_classification_model(num_classes=2):

    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Evaluating"):    """Convert batched metadata to list of dicts."""    """构建ResNet50分类模型"""

            images = batch['image'].to(device)

            labels = batch['label'].to(device)    if isinstance(meta_batch, list):    model = models.resnet50(pretrained=False)

            metadata = batch.get('metadata', {})

                    return meta_batch    model.fc = nn.Linear(model.fc.in_features, num_classes)

            # Forward pass

            logits = model(images)        return model

            probs = torch.softmax(logits, dim=1)[:, 1]  # probability of class=1

            preds = (probs > 0.5).long()    batch_size = len(meta_batch[list(meta_batch.keys())[0]])

            

            all_preds.append(preds.cpu().numpy())    meta_list = []

            all_probs.append(probs.cpu().numpy())

            all_targets.append(labels.cpu().numpy())    for i in range(batch_size):def meta_batch_to_list(meta_batch):

            

            # Convert metadata        meta_dict = {k: v[i] if isinstance(v, (list, tuple, torch.Tensor)) else v     """Convert batched metadata to list of dicts."""

            meta_list = meta_batch_to_list(metadata)

            all_metadata.extend(meta_list)                     for k, v in meta_batch.items()}    if isinstance(meta_batch, list):

    

    # Concatenate results        meta_list.append(meta_dict)        return meta_batch

    cls_preds = np.concatenate(all_preds)

    cls_probs = np.concatenate(all_probs)    return meta_list    if isinstance(meta_batch, dict):

    cls_targets = np.concatenate(all_targets)

            keys = list(meta_batch.keys())

    # Compute metrics

    accuracy = accuracy_score(cls_targets, cls_preds)        if not keys:

    precision, recall, f1, _ = precision_recall_fscore_support(

        cls_targets, cls_preds, average='binary', zero_division=0def evaluate_fold(model, dataloader, device):            return []

    )

    auc = roc_auc_score(cls_targets, cls_probs)    """评估单个fold"""        length = len(meta_batch[keys[0]])

    

    metrics = {    model.eval()        meta_list = []

        'accuracy': float(accuracy),

        'precision': float(precision),            for i in range(length):

        'recall': float(recall),

        'f1': float(f1),    all_preds = []            entry = {}

        'auc': float(auc)

    }    all_probs = []            for key in keys:

    

    return cls_probs, cls_targets, all_metadata, metrics    all_targets = []                value = meta_batch[key][i]



    all_metadata = []                if isinstance(value, torch.Tensor):

def main():

    parser = argparse.ArgumentParser()                        value = value.item()

    parser.add_argument('--config', type=str, required=True)

    parser.add_argument('--checkpoint-dir', type=str, required=True)    with torch.no_grad():                entry[key] = value

    parser.add_argument('--output-dir', type=str, required=True)

    parser.add_argument('--num-folds', type=int, default=5)        for batch in tqdm(dataloader, desc="Evaluating"):            meta_list.append(entry)

    parser.add_argument('--batch-size', type=int, default=32)

    parser.add_argument('--num-workers', type=int, default=4)            images = batch['image'].to(device)        return meta_list

    

    args = parser.parse_args()            labels = batch['label'].to(device)    raise TypeError(f"Unsupported meta batch type: {type(meta_batch)}")

    

    cfg = load_config(args.config)            metadata = batch.get('metadata', {})

    set_seed(cfg.get('seed', 42))

                

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")            # 前向传播@torch.no_grad()

    

    output_dir = Path(args.output_dir)            logits = model(images)def evaluate_fold(model, loader, device):

    output_dir.mkdir(exist_ok=True, parents=True)

                probs = torch.softmax(logits, dim=1)[:, 1]  # 取class=1的概率    """评估单个fold"""

    checkpoint_dir = Path(args.checkpoint_dir)

    test_transform = build_transforms(cfg)            preds = (probs > 0.5).long()    model.eval()

    frame_policies = resolve_frame_policies(cfg)

                    

    print("\n" + "="*80)

    print("Frame Policies:", frame_policies)            all_preds.append(preds.cpu().numpy())    all_preds = []

    print("="*80 + "\n")

                all_probs.append(probs.cpu().numpy())    all_probs = []

    all_fold_results = []

                all_targets.append(labels.cpu().numpy())    all_targets = []

    for fold_idx in range(1, args.num_folds + 1):

        print(f"\n{'='*80}")                all_metadata = []

        print(f"Processing Fold {fold_idx}/{args.num_folds}")

        print(f"{'='*80}")            # 转换metadata    

        

        # Build model            meta_list = meta_batch_to_list(metadata)    for images, labels, meta_batch in tqdm(loader, desc="Evaluating"):

        print("Building ResNetClassifier...")

        model = ResNetClassifier(            all_metadata.extend(meta_list)        images = images.to(device)

            backbone=cfg.get('model', {}).get('name', 'resnet50'),

            pretrained=False,            labels = labels.to(device)

            num_classes=cfg.get('model', {}).get('num_classes', 2),

            dropout=cfg.get('model', {}).get('dropout', 0.2),    # 合并结果        

            train_backbone=True

        ).to(device)    cls_preds = np.concatenate(all_preds)        # 前向传播

        

        # Load checkpoint    cls_probs = np.concatenate(all_probs)        outputs = model(images)

        checkpoint_path = checkpoint_dir / f"fold_{fold_idx:02d}_best.pth"

        if not checkpoint_path.exists():    cls_targets = np.concatenate(all_targets)        

            print(f"Warning: Checkpoint not found: {checkpoint_path}")

            continue            # 获取概率和预测

        

        print(f"Loading checkpoint: {checkpoint_path}")    # 计算指标        probs = torch.softmax(outputs, dim=1)[:, 1]  # infected的概率

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint['model_state_dict'])    accuracy = accuracy_score(cls_targets, cls_preds)        preds = (probs > 0.5).long()

        model.eval()

        print("Model loaded successfully")    precision, recall, f1, _ = precision_recall_fscore_support(        

        

        # Build test dataset        cls_targets, cls_preds, average='binary', zero_division=0        all_preds.extend(preds.cpu().numpy())

        print("Building test dataset...")

        _, _, test_ds = build_datasets(    )        all_probs.extend(probs.cpu().numpy())

            cfg, 

            fold_index=fold_idx - 1,    auc = roc_auc_score(cls_targets, cls_probs)        all_targets.extend(labels.cpu().numpy())

            frame_policies=frame_policies,

            test_transform=test_transform            

        )

            metrics = {        # 保存元数据

        print(f"Test dataset size: {len(test_ds)}")

                'accuracy': float(accuracy),        meta_list = meta_batch_to_list(meta_batch)

        test_loader = DataLoader(

            test_ds,        'precision': float(precision),        all_metadata.extend(meta_list)

            batch_size=args.batch_size,

            shuffle=False,        'recall': float(recall),    

            num_workers=args.num_workers,

            pin_memory=True        'f1': float(f1),    all_preds = np.array(all_preds)

        )

                'auc': float(auc)    all_probs = np.array(all_probs)

        # Evaluate

        print("Evaluating...")    }    all_targets = np.array(all_targets)

        cls_probs, cls_targets, metadata, metrics = evaluate_fold(model, test_loader, device)

                

        print(f"\nTest Metrics:")

        for k, v in metrics.items():    return cls_probs, cls_targets, all_metadata, metrics    # 计算指标

            print(f"  {k}: {v:.4f}")

            accuracy = accuracy_score(all_targets, all_preds)

        # Save results

        fold_dir = output_dir / f"fold_{fold_idx}"    precision, recall, f1, _ = precision_recall_fscore_support(

        fold_dir.mkdir(exist_ok=True, parents=True)

        def main():        all_targets, all_preds, average='binary', zero_division=0

        np.savez(

            fold_dir / "test_predictions.npz",    parser = argparse.ArgumentParser()    )

            cls_preds=cls_probs,

            cls_targets=cls_targets    parser.add_argument('--config', type=str, required=True,    auc = roc_auc_score(all_targets, all_probs)

        )

                               help='Path to config file')    

        # Save metadata

        with open(fold_dir / "test_metadata.jsonl", 'w') as f:    parser.add_argument('--checkpoint-dir', type=str, required=True,    metrics = {

            for meta in metadata:

                meta_dict = {}                       help='Directory containing fold_XX_best.pth checkpoints')        'accuracy': float(accuracy),

                for k, v in meta.items():

                    if isinstance(v, torch.Tensor):    parser.add_argument('--output-dir', type=str, required=True,        'precision': float(precision),

                        meta_dict[k] = v.item() if v.numel() == 1 else v.tolist()

                    elif isinstance(v, np.ndarray):                       help='Output directory for predictions')        'recall': float(recall),

                        meta_dict[k] = v.item() if v.size == 1 else v.tolist()

                    elif isinstance(v, (np.int64, np.int32, np.float32, np.float64)):    parser.add_argument('--num-folds', type=int, default=5,        'f1': float(f1),

                        meta_dict[k] = v.item()

                    else:                       help='Number of folds')        'auc': float(auc)

                        meta_dict[k] = v

                f.write(json.dumps(meta_dict) + '\n')    parser.add_argument('--batch-size', type=int, default=32,    }

        

        with open(fold_dir / "results.json", 'w') as f:                       help='Batch size for evaluation')    

            json.dump({'test_metrics': metrics}, f, indent=2)

            parser.add_argument('--num-workers', type=int, default=4,    return all_probs, all_preds, all_targets, all_metadata, metrics

        all_fold_results.append(metrics)

        print(f"Fold {fold_idx} completed and saved to {fold_dir}")                       help='Number of dataloader workers')

    

    # Compute CV summary    

    if all_fold_results:

        print(f"\n{'='*80}")    args = parser.parse_args()def main():

        print("Cross-Validation Summary")

        print(f"{'='*80}")        parser = argparse.ArgumentParser()

        

        summary = {}    # 加载配置    parser.add_argument('--config', type=str, required=True,

        for metric_name in all_fold_results[0].keys():

            values = [r[metric_name] for r in all_fold_results]    cfg = load_config(args.config)                       help='Path to config file')

            summary[metric_name] = {

                'mean': float(np.mean(values)),    set_seed(cfg.get('seed', 42))    parser.add_argument('--checkpoint-dir', type=str, required=True,

                'std': float(np.std(values)),

                'min': float(np.min(values)),                           help='Directory containing fold_XX_best.pth files')

                'max': float(np.max(values)),

                'values': values    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    parser.add_argument('--output-dir', type=str, required=True,

            }

            print(f"{metric_name}: {summary[metric_name]['mean']:.4f} +/- {summary[metric_name]['std']:.4f}")    print(f"Using device: {device}")                       help='Output directory for predictions')

        

        with open(output_dir / "cv_summary.json", 'w') as f:        parser.add_argument('--num-folds', type=int, default=5,

            json.dump({

                'num_folds': len(all_fold_results),    # 创建输出目录                       help='Number of folds')

                'fold_results': all_fold_results,

                'aggregated_metrics': summary    output_dir = Path(args.output_dir)    parser.add_argument('--batch-size', type=int, default=32,

            }, f, indent=2)

            output_dir.mkdir(exist_ok=True, parents=True)                       help='Batch size')

        print(f"\nCV summary saved to {output_dir / 'cv_summary.json'}")

            

    print(f"\n{'='*80}")

    print(f"All done! Results saved to: {output_dir}")    checkpoint_dir = Path(args.checkpoint_dir)    args = parser.parse_args()

    print(f"{'='*80}\n")

        



if __name__ == '__main__':    # 构建transforms    # 加载配置

    main()

    test_transform = build_transforms(cfg)    cfg = load_config(args.config)

        set_seed(cfg.get('seed', 42))

    # 解析frame policies    

    frame_policies = resolve_frame_policies(cfg)    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "="*80)    print(f"Using device: {device}")

    print("Frame Extraction Policies:")    

    print(f"  {frame_policies}")    # 创建输出目录

    print("="*80 + "\n")    output_dir = Path(args.output_dir)

        output_dir.mkdir(exist_ok=True, parents=True)

    all_fold_results = []    

        checkpoint_dir = Path(args.checkpoint_dir)

    # 对每个fold进行评估    

    for fold_idx in range(1, args.num_folds + 1):    # 构建transforms

        print(f"\n{'='*80}")    test_transform = build_transforms(cfg)

        print(f"Processing Fold {fold_idx}/{args.num_folds}")    

        print(f"{'='*80}")    # 解析frame policies

            frame_policies = resolve_frame_policies(cfg)

        # 构建模型    print("\n" + "="*80)

        print("Building ResNetClassifier model...")    print("Frame Extraction Policies:")

        model = ResNetClassifier(    print("="*80)

            backbone=cfg.get('model', {}).get('name', 'resnet50'),    print(f"  {frame_policies}")

            pretrained=False,    print("="*80 + "\n")

            num_classes=cfg.get('model', {}).get('num_classes', 2),    

            dropout=cfg.get('model', {}).get('dropout', 0.2),    all_fold_results = []

            train_backbone=True    

        ).to(device)    # 对每个fold进行评估

            for fold_idx in range(1, args.num_folds + 1):

        # 加载checkpoint        print(f"\n{'='*80}")

        checkpoint_path = checkpoint_dir / f"fold_{fold_idx:02d}_best.pth"        print(f"Processing Fold {fold_idx}/{args.num_folds}")

        if not checkpoint_path.exists():        print(f"{'='*80}")

            print(f"⚠ Checkpoint not found: {checkpoint_path}")        

            continue        # 加载checkpoint

                checkpoint_path = checkpoint_dir / f"fold_{fold_idx:02d}_best.pth"

        print(f"Loading checkpoint: {checkpoint_path}")        if not checkpoint_path.exists():

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)            print(f"⚠ Checkpoint not found: {checkpoint_path}")

        model.load_state_dict(checkpoint['model_state_dict'])            continue

        model.eval()        

        print(f"✓ Model loaded successfully")        print(f"Loading checkpoint: {checkpoint_path}")

                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # 构建测试数据集        

        print("Building test dataset...")        # 构建模型

        _, _, test_ds = build_datasets(        model = build_classification_model(num_classes=2)

            cfg,         model.load_state_dict(checkpoint['model_state_dict'])

            fold_index=fold_idx - 1,  # 0-indexed        model = model.to(device)

            frame_policies=frame_policies,        

            test_transform=test_transform        # 构建数据集（使用相同的fold split）

        )        _, _, test_ds = build_datasets(

                    cfg,

        print(f"Test dataset size: {len(test_ds)}")            train_transform=None,

                    test_transform=test_transform,

        # 创建dataloader            frame_policies=frame_policies,

        test_loader = DataLoader(            fold_index=fold_idx - 1,  # 0-indexed

            test_ds,            num_folds=args.num_folds

            batch_size=args.batch_size,        )

            shuffle=False,        

            num_workers=args.num_workers,        test_loader = DataLoader(

            pin_memory=True            test_ds,

        )            batch_size=args.batch_size,

                    shuffle=False,

        # 评估            num_workers=cfg.get('num_workers', 4),

        print("Evaluating...")            pin_memory=True

        cls_probs, cls_targets, metadata, metrics = evaluate_fold(model, test_loader, device)        )

                

        print(f"\nTest Metrics:")        print(f"Test set size: {len(test_ds)}")

        for k, v in metrics.items():        

            print(f"  {k}: {v:.4f}")        # 评估

                probs, preds, targets, metadata, metrics = evaluate_fold(model, test_loader, device)

        # 保存结果到fold目录        

        fold_dir = output_dir / f"fold_{fold_idx}"        print(f"\nFold {fold_idx} Test Metrics:")

        fold_dir.mkdir(exist_ok=True, parents=True)        for k, v in metrics.items():

                    print(f"  {k}: {v:.4f}")

        # 保存预测        

        np.savez(        # 保存结果到fold目录

            fold_dir / "test_predictions.npz",        fold_dir = output_dir / f"fold_{fold_idx}"

            cls_preds=cls_probs,  # 保存概率        fold_dir.mkdir(exist_ok=True, parents=True)

            cls_targets=cls_targets        

        )        # 保存预测（与multitask格式一致，但只有分类部分）

                np.savez(

        # 保存metadata            fold_dir / "test_predictions.npz",

        with open(fold_dir / "test_metadata.jsonl", 'w') as f:            cls_preds=probs,  # 保存概率

            for meta in metadata:            cls_targets=targets

                # 转换为可序列化的格式        )

                meta_dict = {}        

                for k, v in meta.items():        # 保存元数据

                    if isinstance(v, torch.Tensor):        with open(fold_dir / "test_metadata.jsonl", 'w') as f:

                        meta_dict[k] = v.item() if v.numel() == 1 else v.tolist()            for meta in metadata:

                    elif isinstance(v, np.ndarray):                f.write(json.dumps(meta) + '\n')

                        meta_dict[k] = v.item() if v.size == 1 else v.tolist()        

                    elif isinstance(v, (np.int64, np.int32, np.float32, np.float64)):        # 保存指标

                        meta_dict[k] = v.item()        with open(fold_dir / "results.json", 'w') as f:

                    else:            json.dump(metrics, f, indent=2)

                        meta_dict[k] = v        

                f.write(json.dumps(meta_dict) + '\n')        all_fold_results.append({

                    'fold_index': fold_idx - 1,

        # 保存指标            'test_metrics': metrics

        with open(fold_dir / "results.json", 'w') as f:        })

            json.dump({'test_metrics': metrics}, f, indent=2)        

                print(f"✓ Saved fold {fold_idx} results to {fold_dir}")

        all_fold_results.append(metrics)    

            # 计算平均指标

        print(f"✓ Fold {fold_idx} completed and saved to {fold_dir}")    if all_fold_results:

            metric_names = list(all_fold_results[0]['test_metrics'].keys())

    # 计算平均指标        aggregated = {}

    if all_fold_results:        

        print(f"\n{'='*80}")        for metric in metric_names:

        print("Cross-Validation Summary")            values = [fold['test_metrics'][metric] for fold in all_fold_results]

        print(f"{'='*80}")            aggregated[f'cls_{metric}'] = {  # 添加cls_前缀

                        'mean': float(np.mean(values)),

        summary = {}                'std': float(np.std(values)),

        for metric_name in all_fold_results[0].keys():                'min': float(np.min(values)),

            values = [r[metric_name] for r in all_fold_results]                'max': float(np.max(values)),

            summary[metric_name] = {                'values': values

                'mean': float(np.mean(values)),            }

                'std': float(np.std(values)),        

                'min': float(np.min(values)),        # 保存CV总结

                'max': float(np.max(values)),        cv_summary = {

                'values': values            'num_folds': len(all_fold_results),

            }            'fold_results': all_fold_results,

            print(f"{metric_name}: {summary[metric_name]['mean']:.4f} ± {summary[metric_name]['std']:.4f}")            'aggregated_metrics': aggregated

                }

        # 保存CV汇总        

        with open(output_dir / "cv_summary.json", 'w') as f:        with open(output_dir / "cv_summary.json", 'w') as f:

            json.dump({            json.dump(cv_summary, f, indent=2)

                'num_folds': len(all_fold_results),        

                'fold_results': all_fold_results,        print(f"\n{'='*80}")

                'aggregated_metrics': summary        print("Cross-Validation Summary:")

            }, f, indent=2)        print(f"{'='*80}")

                for metric, stats in aggregated.items():

        print(f"\n✓ CV summary saved to {output_dir / 'cv_summary.json'}")            print(f"{metric:>15}: {stats['mean']:.4f} ± {stats['std']:.4f}")

            print(f"{'='*80}")

    print(f"\n{'='*80}")        

    print(f"All done! Results saved to: {output_dir}")        print(f"\n✓ All results saved to: {output_dir}")

    print(f"{'='*80}\n")    else:

        print("\n⚠ No fold results generated")



if __name__ == '__main__':

    main()if __name__ == '__main__':

    main()
