# Research Progress Summary - Outline for Group Meeting
## Zhengjie Zhu | Cell Classification Project

---

## 📋 PROPOSED STRUCTURE (Simple Version)

Please review and tell me which sections to **keep**, **remove**, or **expand**:

---

### **PART 1: PROJECT OVERVIEW** (1-2 slides)
- **Research Goal**: Classify infected vs uninfected cells in time-lapse microscopy
- **Dataset**: 
  - HBMVEC cells, 0-48h imaging, infection onset at 2h
  - 95 time points, 2 frames/hour
  - Phase-contrast microscopy
- **Model**: ResNet-50 with 5-fold cross-validation

---

### **PART 2: TEMPORAL CONFOUNDING DISCOVERY** (2-3 slides)
*This is your major finding!*

#### What We Found:
- Models trained with all timepoints achieve 99.9% AUC → TOO GOOD!
- Problem: Models learn temporal shortcuts, not biological features
- **"The x=46h Paradox"**:
  - Same infected data → Different uninfected distributions
  - Performance drops 7-9% → Proves temporal confounding

#### Two Key Experiments:

**A. Interval Sweep Analysis** (~70 models)
- Train on progressively longer windows: [1h, 7h], [1h, 10h], ..., [1h, 46h]
- Test on same windows
- **Result**: Performance valley at 13-19h when temporal shortcuts removed

**B. Sliding Window Analysis** (~70 models)  
- Train on 6-hour windows sliding across 0-48h
- **Result**: Wave pattern reveals true difficulty at 7-10h and 16-19h

---

### **PART 3: MULTITASK LEARNING APPROACH** (2-3 slides)
*Your solution to the problem!*

#### Why Multitask?
- **Problem**: Need to preserve temporal information WITHOUT learning shortcuts
- **Solution**: Predict BOTH class AND time simultaneously

#### Model Architecture:
- Shared ResNet-50 backbone
- Two heads:
  1. **Classification head**: Infected vs uninfected
  2. **Regression head**: Time prediction
  
#### Key Innovation - Class-Conditional Time Targets:
- **Infected cells**: Predict time since infection onset
- **Uninfected cells**: Predict experiment elapsed time
- Single regression head learns both!

#### Training Improvements:
- **Problem**: AUC saturated at 1.0 (can't select best model)
- **Solution**: Combined metric = 0.6×F1 + 0.4×(1-MAE/48)
- More sensitive to performance differences

---

### **PART 4: CROSS-VALIDATION & ROBUSTNESS** (1-2 slides)
*Ensuring reliable results*

#### What We Did:
- 5-fold cross-validation for all approaches
- Matched data splits across experiments (same random seed)
- **Result**: Mean ± Std statistics for robust evaluation

#### Temporal Generalization Analysis:
- Sliding window evaluation (6h windows, 3h stride)
- Tests model performance across infection timeline
- **Result**: Aggregated plot with mean ± std across 5 folds
- Y-axis auto-scaled to show differences clearly

---

### **PART 5: TECHNICAL CONTRIBUTIONS** (1 slide, optional)
*Can remove if time-limited*

- Fixed bugs in training pipeline
- Automated visualization generation
- Integrated temporal analysis into CV workflow
- Auto-scaled plots for better visibility

---

### **PART 6: RESULTS SUMMARY** (2-3 slides)
*The numbers!*

#### Classification Performance:
- Baseline (single-task): AUC = X.XX ± X.XX
- Multitask: AUC = X.XX ± X.XX
- (Need to fill in actual numbers from your runs)

#### Regression Performance:
- Time prediction MAE = X.X ± X.X hours
- Shows model captures temporal information

#### Combined Performance:
- Combined metric = X.XX ± X.XX
- Balances both tasks effectively

#### Temporal Generalization:
- Model stable across infection timeline
- Performance valleys at 13-19h (as expected)

---

### **PART 7: KEY VISUALIZATIONS** (2-3 slides)
*Show the graphs!*

Suggested plots to include:
1. **Interval Sweep Plot**: Shows temporal confounding effect
2. **Sliding Window Plot**: Shows performance valleys
3. **Multitask Training Curves**: Loss convergence
4. **Temporal Generalization (CV)**: Mean ± std across timeline
5. **Prediction Scatter Plot**: Time regression performance

---

### **PART 8: CONCLUSIONS** (1 slide)

#### What We Learned:
1. Temporal confounding is real and significant (7-9% drop)
2. Matched temporal sampling reveals true classification difficulty
3. Multitask learning preserves temporal info without shortcuts
4. 5-fold CV provides robust evaluation

#### Biological Insights:
- 13-19h is genuinely difficult (transitional CPE period)
- Early (<7h) and late (>30h) stages easier to classify

---

### **PART 9: FUTURE WORK** (1 slide, optional)

- Compare multitask vs single-task on same test set
- Statistical significance testing (paired t-test)
- Attention visualization (where model looks)
- Deployment considerations

---

## 📊 TOTAL SLIDE ESTIMATE: 12-18 slides

---

## ❓ YOUR FEEDBACK NEEDED:

Please tell me:

1. **Which PARTS to keep?** (1-9)
   - Example: "Keep 1, 2, 3, 6, 7, 8. Remove 4, 5, 9"

2. **What's MOST important to emphasize?**
   - Temporal confounding discovery?
   - Multitask solution?
   - Results/numbers?

3. **Audience background?**
   - Do they know machine learning basics?
   - Need to explain CV, AUC, etc.?

4. **Time limit?**
   - How many minutes for presentation?
   - How many slides max?

5. **Specific focus?**
   - More on experiments or more on results?
   - Show code/technical details or just outcomes?

6. **Comparison with baseline?**
   - Do you have single-task baseline results to compare?
   - Should we emphasize improvement over baseline?

---

After you review, I'll expand the selected sections into detailed slide content with:
- Clear bullet points
- Numbers/statistics
- Figure captions
- Talking points

Let me know what you think! 🎯
