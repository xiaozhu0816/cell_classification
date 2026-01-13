import numpy as np
import json

fold_dir = "outputs/multitask_resnet50/20260108-124513_5fold/fold_1"
preds = np.load(f"{fold_dir}/test_predictions.npz")

print("NPZ files:", preds.files)
print("\nShapes:")
for key in preds.files:
    print(f"  {key}: {preds[key].shape}, dtype: {preds[key].dtype}")

print("\nSample cls_preds:", preds['cls_preds'][:10])
print("Sample cls_targets:", preds['cls_targets'][:10])

# 读取metadata
metadata = []
with open(f"{fold_dir}/test_metadata.jsonl", 'r') as f:
    for line in f:
        metadata.append(json.loads(line))

print(f"\nMetadata count: {len(metadata)}")
print(f"First metadata: {metadata[0]}")
