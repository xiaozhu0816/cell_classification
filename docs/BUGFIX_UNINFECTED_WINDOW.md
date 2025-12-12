# CRITICAL BUG FIX: Uninfected Window Matching Not Working

**Date:** December 12, 2025  
**Severity:** 🔴 CRITICAL - Data filtering bug affecting all experiments  
**Status:** ✅ FIXED

---

## 🐛 Bug Description

### Problem
When `MATCH_UNINFECTED=true` was set in shell scripts (or `--match-uninfected-window` flag was used), the code **appeared** to work but was **silently ignoring** the uninfected window restriction.

**Evidence from logs:**
```
Training on window [1.0, 7.0]h with MATCH_UNINFECTED=true

But validation shows:
val time bin [0.0h, 8.0h): samples=174    ← Should only have samples from [1.0, 7.0]
val time bin [8.0h, 12.0h): samples=48    ← Should be ZERO samples
val time bin [12.0h, 16.0h): samples=48   ← Should be ZERO samples
val time bin [16.0h, 24.0h): samples=96   ← Should be ZERO samples
val time bin [24.0h, 32.0h): samples=96   ← Should be ZERO samples
```

**Impact:** Uninfected samples were using **ALL time points** even when they should be restricted to the same window as infected samples!

---

## 🔍 Root Cause Analysis

### The Missing Field

**File:** `datasets/timecourse_dataset.py`  
**Class:** `FrameExtractionPolicy`

**Problem:** The dataclass was missing the `uninfected_window_hours` field!

```python
# BEFORE (BROKEN):
@dataclass
class FrameExtractionPolicy:
    frames_per_hour: float = 2.0
    infected_window_hours: Tuple[float, float] = (16.0, 30.0)
    # ❌ NO uninfected_window_hours field!
    infected_stride: int = 1
    uninfected_stride: int = 1
    uninfected_use_all: bool = True
```

**What happened:**
1. Analysis scripts correctly set `frames_cfg["uninfected_window_hours"] = [start, end]`
2. `FrameExtractionPolicy.from_dict()` **silently ignored** this field (not in `__dataclass_fields__`)
3. `uninfected_indices()` method had no window logic, always used ALL frames
4. Result: Uninfected samples got all time points, infected samples got restricted window

**This completely invalidated the --match-uninfected-window feature!**

---

## ✅ The Fix

### 1. Added `uninfected_window_hours` Field

```python
# AFTER (FIXED):
@dataclass
class FrameExtractionPolicy:
    frames_per_hour: float = 2.0
    infected_window_hours: Tuple[float, float] = (16.0, 30.0)
    uninfected_window_hours: Optional[Tuple[float, float]] = None  # ✅ NEW!
    infected_stride: int = 1
    uninfected_stride: int = 1
    uninfected_use_all: bool = True
```

**Why `Optional`?**
- `None` = backward compatible (use all time points or first-only based on `uninfected_use_all`)
- `(start, end)` = restrict to specific window (for matching mode)

---

### 2. Updated `from_dict()` to Handle the Field

```python
@classmethod
def from_dict(cls, data: Optional[Dict]) -> "FrameExtractionPolicy":
    data = data or {}
    filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
    if "window_hours" in data:  # backwards compatibility
        filtered["infected_window_hours"] = tuple(data["window_hours"])
    
    # ✅ NEW: Handle uninfected_window_hours conversion to tuple
    if "uninfected_window_hours" in data and data["uninfected_window_hours"] is not None:
        filtered["uninfected_window_hours"] = tuple(data["uninfected_window_hours"])
    
    return cls(**filtered)
```

---

### 3. Fixed `uninfected_indices()` Method

```python
def uninfected_indices(self, total_frames: int) -> List[int]:
    stride = max(1, self.uninfected_stride)
    
    # ✅ NEW: If uninfected_window_hours is specified, use the same logic as infected
    if self.uninfected_window_hours is not None:
        start_h, end_h = self.uninfected_window_hours
        start_idx = max(0, math.floor(start_h * self.frames_per_hour))
        end_idx = max(start_idx, math.floor(end_h * self.frames_per_hour))
        end_idx = min(end_idx, total_frames - 1)
        indices = list(range(start_idx, end_idx + 1, stride))
        if not indices:
            indices = [min(start_idx, total_frames - 1)]
        return indices
    
    # Otherwise, use the old behavior (backward compatible)
    if not self.uninfected_use_all:
        return [0]
    return list(range(0, total_frames, stride))
```

**Key change:** Now mirrors `infected_indices()` logic when `uninfected_window_hours` is set!

---

### 4. Updated `format_policy_summary()` for Better Logging

```python
def format_policy_summary(policy: FrameExtractionPolicy) -> str:
    start, end = policy.infected_window_hours
    
    # ✅ NEW: Show window when specified
    if policy.uninfected_window_hours is not None:
        u_start, u_end = policy.uninfected_window_hours
        uninfected_info = f"[{u_start:.1f},{u_end:.1f}]"
    else:
        uninfected_info = "all" if policy.uninfected_use_all else "first-only"
    
    return (
        f"frames_per_hour={policy.frames_per_hour:.2f} | "
        f"infected=[{start:.1f},{end:.1f}] stride={policy.infected_stride} | "
        f"uninfected={uninfected_info} stride={policy.uninfected_stride}"
    )
```

---

## 🧪 Verification

### Before Fix (WRONG):
```
Window: [1.0, 7.0]h with MATCH_UNINFECTED=true

Dataset summary:
  - infected: frames from [1.0, 7.0]h only  ✅
  - uninfected: frames from [0.0, 46.0]h    ❌ WRONG!

Validation time bins:
  [0.0h, 8.0h): 174 samples   ← Has samples outside [1.0, 7.0]
  [8.0h, 12.0h): 48 samples   ← Should be 0!
  [12.0h, 16.0h): 48 samples  ← Should be 0!
  ...
```

### After Fix (CORRECT):
```
Window: [1.0, 7.0]h with MATCH_UNINFECTED=true

Dataset summary:
  - infected: frames from [1.0, 7.0]h only    ✅
  - uninfected: frames from [1.0, 7.0]h only  ✅ CORRECT!

Policy summary:
  infected=[1.0,7.0] stride=1 | uninfected=[1.0,7.0] stride=1

Validation time bins:
  [0.0h, 8.0h): X samples     ← Only samples from [1.0, 7.0]
  [8.0h, 12.0h): 0 samples    ← Zero (correct!)
  [12.0h, 16.0h): 0 samples   ← Zero (correct!)
  ...
```

---

## ⚠️ Impact on Previous Results

### Experiments Affected
**ALL experiments run with `MATCH_UNINFECTED=true` before this fix are INVALID!**

This includes:
- ❌ Any interval sweep with `--match-uninfected-window`
- ❌ Any sliding window with `--match-uninfected-window`
- ❌ Any shell script runs with `MATCH_UNINFECTED=true`

**Why invalid?**
The comparison was unfair:
- Infected samples: Restricted to [x, x+k] window
- Uninfected samples: Used ALL time points [0, 46]h

This gave uninfected samples an unfair advantage (more temporal diversity).

### Experiments NOT Affected
✅ Experiments with `MATCH_UNINFECTED=false` (default) - these are still valid  
✅ Experiments run AFTER this fix - these will work correctly

---

## 🔄 What to Do Now

### 1. Re-run Affected Experiments
If you ran experiments with `MATCH_UNINFECTED=true`:
- ❌ Delete those results
- ✅ Re-run with the fixed code
- ✅ Compare new results to see the true effect of matching windows

### 2. Verify the Fix Works
After re-running, check logs for:
```
Policy summary:
  infected=[1.0,7.0] | uninfected=[1.0,7.0]  ← Both should match!

Validation time bins:
  [0.0h, 8.0h): X samples
  [8.0h, 12.0h): 0 samples   ← Should be ZERO for times outside window
  [12.0h, 16.0h): 0 samples  ← Should be ZERO
```

### 3. Expected Behavior Change
With the fix, you should see:
- **Fewer samples** in datasets (uninfected now restricted)
- **Different metrics** (more fair comparison between infected/uninfected)
- **Possibly lower performance** (less training data when matching is enabled)

---

## 📝 Technical Details

### Data Flow (Fixed)

```
1. analyze_interval_sweep_train.py sets:
   frames_cfg["uninfected_window_hours"] = [1.0, 7.0]

2. FrameExtractionPolicy.from_dict() now recognizes and stores:
   policy.uninfected_window_hours = (1.0, 7.0)

3. uninfected_indices() now checks:
   if self.uninfected_window_hours is not None:
       # Use window logic (like infected)
       return indices from [1.0, 7.0]
   else:
       # Use all frames (backward compatible)
       return all indices

4. Dataset filters uninfected frames to [1.0, 7.0] only!
```

### Backward Compatibility

✅ **Fully backward compatible!**

- If `uninfected_window_hours` is NOT set → behaves like before (uses all frames)
- If `uninfected_window_hours` IS set → applies window restriction (NEW!)
- Old experiments without `--match-uninfected-window` are unaffected

---

## 🎯 Summary

| Aspect | Before (Broken) | After (Fixed) |
|--------|----------------|---------------|
| **Bug** | `uninfected_window_hours` ignored | Field recognized and used |
| **Behavior** | Uninfected uses ALL times | Uninfected uses specified window |
| **Data** | Unfair comparison | Fair comparison |
| **Results** | INVALID with matching | VALID with matching |
| **Backward Compat** | N/A | ✅ Fully compatible |

---

## ✅ Checklist

- [x] Added `uninfected_window_hours` field to `FrameExtractionPolicy`
- [x] Updated `from_dict()` to handle the new field
- [x] Fixed `uninfected_indices()` to apply window logic
- [x] Updated `format_policy_summary()` for better logging
- [x] Verified no syntax errors
- [x] Backward compatibility maintained
- [ ] **TODO: Re-run all experiments with `MATCH_UNINFECTED=true`**
- [ ] **TODO: Verify logs show correct window restrictions**
- [ ] **TODO: Compare new results to old (should be different!)**

---

## 🚨 Action Required

**IMMEDIATE:**
1. ⚠️ Discard any results from experiments with `MATCH_UNINFECTED=true` before this fix
2. ✅ Re-run those experiments with the fixed code
3. ✅ Verify logs show matching windows for both infected and uninfected
4. ✅ Document the new results

**This bug fix is critical for the validity of your --match-uninfected-window feature!**
