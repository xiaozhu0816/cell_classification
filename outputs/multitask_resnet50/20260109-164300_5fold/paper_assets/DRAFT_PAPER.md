# Draft manuscript (MDPI Cells style)

> Working draft centered on Track **A**: *state + progression time* with *temporal reliability profiling*.
> Fill in the brackets `[LIKE THIS]` with your experiment-specific details.

## Title (pick one)

1. **Joint prediction of infection state and progression time from microscopy using a multi-task ResNet with temporal reliability profiling**
2. **Microscopy-based infection phenotyping by multi-task learning of infection state and progression time**
3. **From state to stage: multi-task deep learning for infection classification and temporal staging in live-cell microscopy**

## Short running title

Multi-task infection phenotyping and staging from microscopy

## Abstract (near-final draft)

**Background:** Quantifying infection phenotypes from microscopy often requires both accurate discrimination of infected cells and estimation of infection progression stage. However, model performance may vary over time due to stage-dependent morphological changes and temporal distribution shift.

**Methods:** We developed a multi-task convolutional neural network (ResNet50 backbone) to jointly predict (i) infection state (infected vs. mock-infected) and (ii) progression time from microscopy image patches of human brain microvascular endothelial cells (HBMVEC). Imaging included mock-infected controls (PBS) and cells infected with Venezuelan equine encephalitis virus (VEEV) vaccine strain **TC-83** (MOI = 5). We evaluated performance using 5-fold cross-validation and performed temporal reliability profiling using sliding time windows (window = 6 h; stride = 3 h). For infected samples, progression time was defined as **hours since infection**; for mock-infected samples, time was defined as **hours since experiment start**.

**Results:** Across five folds, the model achieved classification accuracy **0.9932 ± 0.0015**, F1-score **0.9931 ± 0.0015**, and AUC **0.9999 ± 0.0000**. We report time prediction error in **hours** (MAE **1.144 ± 0.101 h**; RMSE **1.472 ± 0.106 h**), corresponding to ~2.3 frames at a 30 min imaging interval (reported as a secondary, intuitive scale). Temporal profiling showed lower performance in early-stage windows (e.g., window center 3 h: accuracy 0.950; recall 0.890; AUC 0.994) with rapid improvement thereafter (from window center 12 h onward, accuracy/F1/AUC ≈ 1.0).

**Conclusions:** Joint modeling of infection state and progression time enables accurate infection phenotyping and hour-level temporal staging from microscopy. Temporal reliability profiling reveals stage-dependent uncertainty, which can inform downstream experimental interpretation and quality control.

**Keywords:** microscopy; infection; multi-task learning; temporal staging; deep learning; ResNet; reliability profiling

## Data and outputs in this repo

- Results directory: `outputs/multitask_resnet50/20260109-164300_5fold`
- Paper assets directory: `outputs/multitask_resnet50/20260109-164300_5fold/paper_assets`
- Tables:
	- `paper_table_overall_metrics.csv`
	- `paper_table_temporal_metrics.csv`
- Figures already generated:
	- `prediction_scatter.png`
	- `classification_by_time_window.png`
	- `cv_temporal_generalization.png`
	- `error_distribution_by_time_range.png`
	- `regression_residual_over_time.png`
	- `valley_period_analysis.png`
	- `error_vs_classification_confidence.png`

## 1. Introduction (draft)

Microscopy provides a direct, high-content view of infection-related cellular phenotypes, including changes in morphology, cytoplasmic texture, and subcellular organization over time. In many experimental workflows, a central need is to identify **whether** a cell is infected (infection state) and **how far** infection has progressed (infection stage). These questions are particularly relevant in time-course imaging where downstream conclusions often depend on correctly placing observations along a progression timeline.

Human brain microvascular endothelial cells (HBMVEC) are a key cellular component of the blood–brain barrier and are widely used for studying neurotropic viral infection and host responses. In this study, we consider imaging data containing mock-infected controls (PBS) and infection with Venezuelan equine encephalitis virus (VEEV) vaccine strain TC-83. Reliable, automated inference of infection state and stage from imaging could support high-throughput phenotyping, screening, and quantitative characterization of progression dynamics.

Deep learning has achieved strong performance in classification of cellular phenotypes from image patches. However, infection is inherently dynamic: early-stage phenotypes can be subtle, heterogeneous, or visually confounded with baseline variability, while later stages may show clearer and more uniform changes. As a result, models trained and evaluated only on pooled time points may mask **stage-dependent performance variation**, which can limit practical trust and interpretability.

To address these challenges, we propose a multi-task learning framework that jointly predicts infection state and progression time from microscopy patches. Joint prediction encourages shared representations that align with biologically meaningful progression. Importantly, we complement overall accuracy reporting with **temporal reliability profiling** via sliding time windows, enabling explicit quantification of when the model is reliable and when uncertainty is expected.

**Contributions:**
1. A multi-task ResNet model that jointly predicts infection state and progression time from microscopy patches.
2. A 5-fold cross-validation evaluation with both classification and regression metrics.
3. Temporal reliability profiling that characterizes stage-dependent performance and highlights early-stage difficulty.
4. Error analyses (time-stratified errors, residual bias trends, confidence-error relationships) to support interpretability and quality control.

## 2. Results (draft)

### 2.1 Overall multi-task performance (5-fold cross-validation)

Across five CV folds, we observed highly consistent classification performance (accuracy **0.9932 ± 0.0015**, F1 **0.9931 ± 0.0015**, AUC **0.9999 ± 0.0000**) and hour-level time prediction accuracy (MAE **1.144 ± 0.101 h**, RMSE **1.472 ± 0.106 h**) (Table 1). The regression scatter plot (Figure 3A) demonstrates close alignment between predicted and ground-truth times for both infected and uninfected samples.

**Table 1.** Overall performance (mean ± std across 5 folds). See `paper_table_overall_metrics.csv`.

### 2.2 Temporal reliability profiling reveals early-stage difficulty

We next evaluated classification performance across time using sliding windows (6 h window; 3 h stride). In early-stage windows (e.g., center = 3 h), performance was lower (accuracy 0.950; recall 0.890; AUC 0.994), indicating that early infection phenotypes may be subtle or heterogeneous. Performance improved rapidly with time (center = 6–9 h) and reached near-perfect metrics from center = 12 h onward (accuracy/F1/AUC ≈ 1.0) (Table 2; Figure 2).

Biologically, this pattern is consistent with an early phase in which infection-related morphological signatures are not yet dominant at the single-cell level, followed by a period where infection-associated changes become increasingly pronounced and more consistently detectable.

**Table 2.** Temporal window performance (mean ± std across folds). See `paper_table_temporal_metrics.csv`.

### 2.3 Error distributions and bias patterns across time

We analyzed regression errors stratified by time range and infection state (Figure 4). The time-stratified boxplots summarize absolute error distributions per time bin, while trend plots visualize how error evolves over continuous time. Additionally, residual analysis (prediction − true time; Figure 3B) highlights potential systematic over- or under-estimation in specific time regions.

If a “valley period” (13–19 h) is biologically meaningful in this system, we provide a focused comparison of valley vs. non-valley errors with statistical testing (Figure 4C–D).

### 2.4 Confidence-error coupling supports quality control

We examined how regression error varies as a function of classification confidence (Figure 5). A monotonic relationship (higher confidence, lower error) would support using confidence thresholds to filter unreliable predictions in downstream analyses.

## 3. Discussion (bullet draft)

### 3.1 Summary of findings

Our multi-task model achieved highly accurate infection state discrimination and hour-level staging on HBMVEC microscopy patches under 5-fold cross-validation. Beyond aggregate metrics, temporal reliability profiling revealed a clear stage dependence: early-stage windows exhibited reduced recall, while later windows achieved near-perfect classification performance.

### 3.2 Why multi-task learning is a good fit for infection imaging

Infection progression is not only a categorical state change but a continuous biological process. Multi-task learning provides an inductive bias that encourages learned representations to be informative for both infection state and temporal staging. This is particularly useful when subtle, shared features (e.g., early texture changes) may be insufficient for perfect classification but still predictive of temporal progression.

### 3.3 Biological interpretation and practical implications

The reduced early-stage recall suggests that shortly after infection, single-cell image patches may not yet display strong morphological signatures distinguishable from baseline variability in mock-infected controls. Practically, this implies that downstream biological conclusions relying on early-stage classification should incorporate uncertainty-aware filtering. The observed coupling between classification confidence and regression error further supports the use of confidence as an internal quality-control signal.

### 3.4 Limitations

- **Label semantics for time:** infected samples use hours since infection, whereas mock-infected samples use hours since experiment start. This is appropriate for each condition but should be considered when interpreting “time error” across mixed conditions.
- **Generalization:** results are derived from a specific imaging setup (10× objective, 0.862 μm/pixel) and specific experimental conditions (VEEV TC-83, MOI = 5). External validity to new replicates, microscopes, or viral strains should be evaluated.
- **Potential correlation structure:** patch overlap (5%) and within-position correlations can inflate effective sample size; position-level splitting mitigates leakage, but future analyses could report position-level confidence intervals.

### 3.5 Future work

- Perform explicit **leave-one-position** or **leave-one-time-block** evaluations and report bootstrap confidence intervals.
- Add interpretable visualizations (e.g., CAM/Grad-CAM) contrasting early vs. late-stage regions to relate model features to plausible cellular phenotypes.
- Extend evaluation across additional batches/replicates and, if available, additional cell types relevant to neurotropic infection.

## 4. Materials and Methods (template)

### 4.1 Dataset and annotations

Human brain microvascular endothelial cells (HBMVEC) were imaged under two conditions: mock infection (PBS) and infection with Venezuelan equine encephalitis virus (VEEV) vaccine strain TC-83 using a single viral stock (MOI = 5). Data were obtained from two wells: **B2** (infected) and **C4** (mock-infected). In the infected well, time points T0–T1 were treated as uninfected and time points T2–T96 were treated as infected. In the mock-infected well, time points T0–T96 were treated as mock-infected.

The dataset contained approximately 45,000 cells per well. Image patches were extracted with 5% spatial overlap.

Imaging was performed at 10× objective with spatial sampling 0.862 μm/pixel. The field of view was 0.8 × 0.7 mm (0.0056 cm²). The total well area was 3.8 cm², corresponding to ~0.14% coverage per image. Time-lapse imaging was acquired every 30 minutes.

Each patch was assigned (i) a binary infection state label (infected vs mock-infected) and (ii) a continuous time target in hours. For infected samples, the regression target represents **hours since infection**; for mock-infected samples, the target represents **hours since experiment start**.

To reduce information leakage due to within-field correlations and patch overlap, we used 5-fold cross-validation with splitting at the **field-of-view/position** level; i.e., all patches originating from the same imaging position were assigned to a single fold.

### 4.2 Model architecture

We used a ResNet50 convolutional backbone with two task-specific heads. The classification head outputs a scalar probability for infection (infected vs mock-infected). The regression head outputs a continuous prediction for time (in hours). The network was trained end-to-end with a joint objective combining a classification loss and a regression loss (see training script configuration for exact loss definitions and weights).

### 4.3 Training procedure

Models were trained using 5-fold cross-validation with per-fold train/validation/test splits. For each fold, a ResNet50 multitask model initialized from ImageNet weights was trained for 100 epochs. The best checkpoint per fold was selected by a combined validation score defined as $0.6\times\mathrm{F1}+0.4\times(1-\mathrm{MAE}/48)$.

Optimization used AdamW (learning rate $1\times10^{-4}$; weight decay $1\times10^{-4}$) with a cosine annealing learning-rate schedule (CosineAnnealingLR; $T_{\max}=100$; $\eta_{\min}=1\times10^{-6}$). The multitask objective combined cross-entropy loss for infection classification and Smooth L1 loss for time regression with equal weights ($\lambda_{cls}=1.0$, $\lambda_{reg}=1.0$). Training used mixed precision (AMP) and global gradient-norm clipping (1.0).

Training used mini-batches of 128 image patches (evaluation used 256). Input patches were resized to $512\times512$ pixels and augmented with random flips and random rotations; color jitter was not used. Unless otherwise noted, data loading used 4 worker processes, and we did not apply class-balanced sampling.

**Reproducibility:** All experiments were run with a fixed random seed (seed = 42) and a fixed 5-fold split. Hyperparameters were defined in a YAML configuration file (e.g., `configs/multitask_example.yaml`) and the cross-validation training script (`train_multitask_cv.py`) wrote all fold outputs (checkpoints, predictions, and per-fold metrics) to a timestamped results directory.

### 4.4 Evaluation metrics

We report classification performance using accuracy, precision, recall, F1-score, and ROC-AUC. Regression performance is reported using mean absolute error (MAE) and root mean squared error (RMSE), both in hours.

Temporal reliability profiling was performed by binning test-set predictions into overlapping sliding time windows (window width = 6 h; stride = 3 h). For each window, metrics were computed on the subset of predictions whose ground-truth times fell within the window and then aggregated across folds (mean ± SD). These windowed metrics are reported in `paper_table_temporal_metrics.csv` and visualized in `classification_by_time_window.png`.

For window assignment, the ground-truth time for each sample followed the same condition-specific definition used for training: infected samples were indexed by hours since infection onset, whereas mock-infected samples were indexed by hours since experiment start.

## Figure captions (draft, 6-figure set)

> Note: filenames refer to outputs in `outputs/multitask_resnet50/20260109-164300_5fold/`.

**Figure 1. Experimental design and multi-task modeling overview.**
Schematic of the microscopy time-course experiment in HBMVEC and the proposed multi-task learning framework. Mock-infected controls (PBS) and VEEV TC-83–infected cells (MOI = 5) were imaged every 30 minutes. A ResNet50 backbone with two task heads jointly predicts infection state (infected vs mock-infected) and progression time (hours since infection for infected samples; hours since experiment start for mock-infected samples). Data are split by field-of-view/position for 5-fold cross-validation to reduce leakage across folds.

**Figure 2. Temporal reliability profiling of infection classification.** (`classification_by_time_window.png`)
Sliding-window classification performance across time (window = 6 h; stride = 3 h), reporting mean ± SD across CV folds for accuracy, F1-score, and ROC-AUC. Performance is lower in early windows (e.g., center 3 h) and becomes near-perfect after later windows (≥ 12 h), consistent with stage-dependent detectability.

**Figure 3. Time prediction accuracy and residual bias over time.** (`prediction_scatter.png`, `regression_residual_over_time.png`)
(A) Scatter plot comparing predicted vs ground-truth progression time across all CV test predictions, stratified by infection state. The diagonal line indicates perfect agreement (y = x). (B) Mean residual (prediction − truth) over time for infected and mock-infected samples, computed in 1-hour bins and visualized as mean ± 1 SD, highlighting potential systematic over- or under-estimation trends.

**Figure 4. Stage-dependent regression error distributions by time range and infection state.** (`error_distribution_by_time_range.png`)
Absolute time prediction errors stratified by infection state and time range. Top panels show boxplots of absolute error across pre-defined time bins for infected and mock-infected samples. Bottom panels show time-resolved trends of mean absolute error with variability (±1 SD), illustrating how regression accuracy changes across the time course.

**Figure 5. Valley-period focused error analysis.** (`valley_period_analysis.png`)
Targeted evaluation of a pre-defined “valley” period (13–19 h) compared with non-valley time ranges, stratified by infection state. Panels summarize mean errors by time range, distributional comparisons (boxplots), and statistical testing of valley vs non-valley error distributions, highlighting time windows with increased uncertainty.

**Figure 6. Confidence–error coupling and temporal generalization.** (`error_vs_classification_confidence.png`, `cv_temporal_generalization.png`)
(A) Relationship between classification confidence (probability of the predicted class) and regression error (absolute time error). Points show individual predictions, and the trend line summarizes binned mean error, supporting confidence-based quality control. (B) Temporal generalization analysis summarizing how model performance changes across temporal shifts, providing an additional perspective on robustness under time-dependent distribution changes.

## Checklist before submission

- [ ] Fill dataset/imaging details.
- [x] Confirm time definition for infected vs uninfected (hours since infection vs hours since experiment start).
- [ ] Confirm CV split strategy avoids leakage (same cell/position/well across folds).
- [ ] Decide whether to include (or omit) p-values; if included, specify test and multiple-comparison handling.
- [ ] Add CAM/interpretability examples (optional but helpful for Cells).
