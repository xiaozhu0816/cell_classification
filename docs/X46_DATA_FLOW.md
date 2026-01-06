# Data Flow Comparison: Blue vs Orange Lines at x=46h

## Question
Why do blue and orange lines show different performance at x=46h when they both test on [1-46h]?

## Answer in One Sentence
**They test on the same data, but the orange line model was trained on a DIFFERENT uninfected distribution.**

---

## Visual Data Flow

### BLUE LINE (Train-Test Mode)

```
Step 1: TRAINING
┌──────────────────────────────────────────────────┐
│  Infected Cell Images                            │
│  ████████████████████████████████████            │
│  Time: 1h  5h  10h  15h  20h  30h  40h  46h     │
│                                                   │
│  Uninfected Cell Images                          │
│  ████████████████████████████████████            │
│  Time: 1h  5h  10h  15h  20h  30h  40h  46h     │
│        ↑────────────────────────────────↑        │
│        Same time range as infected!              │
└──────────────────────────────────────────────────┘
           ↓
    [Train Model]
           ↓
    Model learns:
    "Need to distinguish infected vs uninfected
     based on MORPHOLOGY (cell shape, etc.)
     within the 1-46h time window"
           ↓
Step 2: TESTING
┌──────────────────────────────────────────────────┐
│  Test Infected:   1h ────────────────────── 46h  │
│  Test Uninfected: 1h ────────────────────── 46h  │
│                   ↑                          ↑   │
│                   Matches training exactly!      │
└──────────────────────────────────────────────────┘
           ↓
    Result: AUC = 0.99 ✓
    (Model sees the same distribution it trained on)
```

### ORANGE LINE (Test-Only Mode)

```
Step 1: TRAINING
┌──────────────────────────────────────────────────┐
│  Infected Cell Images                            │
│  ████████████████████████████████████            │
│  Time: 1h  5h  10h  15h  20h  30h  40h  46h     │
│                                                   │
│  Uninfected Cell Images                          │
│  ██████████████████████████████████████████      │
│  Time: 0h 1h 5h 10h 15h 20h 30h 40h 46h 48h     │
│        ↑                                  ↑      │
│     Extra!                              Extra!   │
└──────────────────────────────────────────────────┘
           ↓
    [Train Model]
           ↓
    Model learns:
    "Cells at 0-1h usually uninfected (shortcut!)
     Cells at 46-48h usually uninfected (shortcut!)
     Cells at 1-46h need morphology check"
           ↓
Step 2: TESTING
┌──────────────────────────────────────────────────┐
│  Test Infected:   1h ────────────────────── 46h  │
│  Test Uninfected: 1h ────────────────────── 46h  │
│                   ↑                          ↑   │
│              Missing 0-1h!            Missing 46-48h!
│              (Model expected these!)             │
└──────────────────────────────────────────────────┘
           ↓
    Result: AUC = 0.90 ✗
    (Model's shortcuts don't work → confusion → lower performance)
```

---

## The Critical Difference

| What | Blue Line | Orange Line |
|------|-----------|-------------|
| **Uninfected TRAINING range** | [1 - 46h] | [**0 - 48h**] ⚠️ |
| **Uninfected TESTING range** | [1 - 46h] | [1 - 46h] |
| **Distribution match?** | ✅ YES | ❌ NO |
| **Performance** | ~0.99 | ~0.90 |

---

## Why This Proves Temporal Confounding

### Hypothesis 1: Model learns TRUE infection features
- Prediction: Performance should be IDENTICAL (same test data!)
- Observation: 9% difference
- **Conclusion: REJECTED** ❌

### Hypothesis 2: Model uses temporal shortcuts
- Prediction: Performance should DROP when temporal distribution shifts
- Observation: 9% drop when uninfected distribution changes
- **Conclusion: CONFIRMED** ✅

---

## Real Numbers from Your Graph

At x=46h (rightmost point):

**Blue Line:**
- AUC: ~0.99
- Accuracy: ~0.95
- F1: ~0.70

**Orange Line:**  
- AUC: ~0.90-0.92 (7-9% lower!)
- Accuracy: ~0.78-0.80 (15-17% lower!)
- F1: ~0.60-0.65 (5-10% lower!)

Even though both test on **identical data**, the orange line is much worse because it was trained on a different uninfected temporal distribution!

---

## Key Takeaway

**Training distribution matters MORE than test distribution!**

The orange model's poor performance at x=46h proves it learned to rely on temporal features (when during the experiment was this cell imaged?) rather than morphological features (does this cell show infection signs?).

This is why matched temporal sampling is critical - it forces the model to learn robust, biology-based features instead of temporal shortcuts!
