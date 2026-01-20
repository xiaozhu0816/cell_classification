# Figure plan (6-figure set)

This document maps manuscript figures to existing output files in:

- `outputs/multitask_resnet50/20260109-164300_5fold/`

It also suggests panel layouts and a “minimum submission set”.

## Figure 1 — Experimental + modeling overview (schematic)

**Goal:** give readers the experiment story (HBMVEC, PBS mock vs VEEV TC-83, 30-min imaging) and the model story (multi-task ResNet with two heads + CV split by position).

**Panels (recommended):**
- **(A)** Experimental design timeline:
  - Mock-infected (PBS) well C4: T0–T96 mock
  - Infected well B2 (MOI=5): T0–T1 treated uninfected, T2–T96 infected
  - Imaging interval: 30 min/frame
- **(B)** Model architecture cartoon:
  - ResNet50 backbone
  - Head 1: infection probability (infected vs mock)
  - Head 2: time regression (hours)
- **(C)** Evaluation scheme:
  - 5-fold CV
  - split unit: field-of-view / position

**Source files:** none (needs a drawn schematic). You can assemble this in PowerPoint/Illustrator/Inkscape.

---

## Figure 2 — Temporal reliability profiling of infection classification

**Core message:** early time windows are harder; later windows are near-perfect.

**Panel layout:**
- **(A)** Single panel plot (already contains multiple curves).

**Use file:**
- `classification_by_time_window.png`

**Notes:**
- If you want the early-stage dip to be more visible, keep the y-axis zoomed (as you already adjusted).

---

## Figure 3 — Time prediction accuracy + residual bias

**Core message:** regression is accurate overall (scatter close to diagonal) and residual plot indicates bias patterns over time.

**Panel layout:**
- **(A)** Scatter plot (All + Infected + Mock) — already in one image.
- **(B)** Residual over time (two curves + shaded SD regions).

**Use files:**
- `prediction_scatter.png`  → Figure 3A
- `regression_residual_over_time.png` → Figure 3B

**Notes:**
- Keep regression error reported in hours in the main text; optionally add “~2.3 frames” in parentheses (30 min/frame).

---

## Figure 4 — Stage-dependent error distributions by time range

**Core message:** errors vary by time range; infected vs mock may have different error profiles.

**Panel layout:**
- **(A)** infected boxplots (top-left) + infected trend (bottom-left)
- **(B)** mock/uninfected boxplots (top-right) + mock trend (bottom-right)

**Use file:**
- `error_distribution_by_time_range.png`

**Notes:**
- This is already a 2×2 multi-panel figure; you can label quadrants as A–D in a figure editor if the journal requires.

---

## Figure 5 — Valley-period focused analysis (13–19 h)

**Core message:** a focused window analysis shows whether a specific stage range exhibits increased uncertainty.

**Panel layout:**
- **(A)** mean error bars by time range (infected vs mock)
- **(B)** valley vs non-valley boxplots
- **(C)** statistical test summary panel
- **(D)** histogram comparison

**Use file:**
- `valley_period_analysis.png`

**Notes:**
- Only keep this figure if you can justify “valley period” biologically, or present it as a data-driven difficult stage interval.

---

## Figure 6 — Confidence–error coupling + temporal generalization

**Core message:** confidence can be used for quality control; model robustness across time shifts.

**Panel layout:**
- **(A)** confidence vs error plot
- **(B)** temporal generalization plot

**Use files:**
- `error_vs_classification_confidence.png` → Figure 6A
- `cv_temporal_generalization.png` → Figure 6B

---

## Minimum submission set (recommended)

If you need to shorten:

- Keep: Figures **1–4** and **6**
- Optional: Figure **5** (valley analysis), depending on how strongly you can motivate the 13–19 h window.

## File checklist for final submission

- [ ] Export each figure as 300 dpi TIFF or high-quality PDF (MDPI usually accepts both).
- [ ] Ensure consistent font sizes and line widths across figures.
- [ ] Add panel letters (A, B, C...) where applicable.
- [ ] Use consistent terminology across all plots: “mock-infected” vs “infected”.
- [ ] Verify axis labels:
  - time in hours
  - residual = predicted − true (hours)

## Quick cross-reference to manuscript

- Intro + Methods: Figure 1
- Temporal reliability: Figure 2
- Regression accuracy + bias: Figure 3
- Error stratification: Figure 4
- Valley (optional): Figure 5
- Quality control + robustness: Figure 6
