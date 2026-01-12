# MASTER PROMPT: CRC Prediction Feature Selection Methodology Improvement

## Project Overview

This project is improving the **feature selection methodology** for a **colorectal cancer (CRC) risk prediction model** with highly imbalanced data (250:1 negative:positive ratio). The model predicts CRC diagnosis within 6 months for unscreened patients.

**Current Status**: All notebooks complete. Ready to run pipeline in Databricks.

## Directory Structure

```
METHODOLOGY_IMBALANCED/
├── 2nd_Dataset_Creation/           # TRANSFORMED notebooks (working versions)
│   ├── V2_Book0_Cohort_Creation.py # Base cohort with SGKF splits (COMPLETE)
│   ├── V2_Book1_Vitals.py          # Train-only feature selection (COMPLETE)
│   ├── V2_Book2_ICD10.py           # Train-only feature selection (COMPLETE)
│   ├── V2_Book3_Social_Factors.py  # No changes needed (all features excluded)
│   ├── V2_Book4_Labs_Combined.py   # Train-only feature selection (COMPLETE)
│   ├── V2_Book5_1_Medications_Outpatient.py  # Train-only feature selection (COMPLETE)
│   ├── V2_Book5_2_Medications_Inpatient.py   # Train-only feature selection (COMPLETE)
│   ├── V2_Book6_Visit_History.py   # Train-only feature selection (COMPLETE)
│   ├── V2_Book7_Procedures.py      # Train-only feature selection (COMPLETE)
│   ├── V2_Book8_Compilation.py     # No changes needed (just joins tables)
│   └── V2_Book9_Feature_Selection.py  # Feature selection pipeline (COMPLETE)
├── Original_2nd_Dataset_Creation/  # ORIGINAL notebooks (reference/backup)
│   └── (same file structure)
├── Original_Methodology/           # Original clustering/SHAP notebooks (analyzed)
│   ├── CORRELATION_HIERARCHICAL_FEATURE_CLUSTERING.py
│   └── CRC_ITER1_MODEL-PREVALENCE.py
└── Prompts/                        # Additional prompts/documentation
```

## Completed Work Summary

### Phase 1: Book 0 - Cohort Creation with SGKF Splits (COMPLETE)

Added train/val/test split assignments using StratifiedGroupKFold:
- **TEST (Q6)**: Last quarter of observations (temporal holdout)
- **TRAIN (~67%)**: Q0-Q5 patients via SGKF
- **VAL (~33%)**: Q0-Q5 patients via SGKF

Key guarantees:
- NO patient appears in multiple splits (grouped by PAT_ID)
- Class balance maintained across splits (stratified by FUTURE_CRC_EVENT)
- Random seed 217 for reproducibility

Bug fixed: Changed `F.first("FUTURE_CRC_EVENT")` to `F.max("FUTURE_CRC_EVENT")` for correct patient-level stratification.

### Phase 2: Books 1-8 - Train-Only Feature Selection (COMPLETE)

All feature engineering notebooks now filter on `SPLIT='train'` for feature selection metrics:

| Book | Status | Changes |
|------|--------|---------|
| Book 1 (Vitals) | COMPLETE | df_train for risk ratios & MI |
| Book 2 (ICD10) | COMPLETE | df_train for risk ratios & MI |
| Book 3 (Social) | NO CHANGES | All features excluded due to data quality |
| Book 4 (Labs) | COMPLETE | df_train for risk ratios & MI |
| Book 5.1 (Outpatient Meds) | COMPLETE | df_train for risk ratios & MI |
| Book 5.2 (Inpatient Meds) | COMPLETE | df_train for risk ratios & MI |
| Book 6 (Visit History) | COMPLETE | df_train for risk ratios & MI |
| Book 7 (Procedures) | COMPLETE | df_train for risk ratios & MI |
| Book 8 (Compilation) | NO CHANGES | Just joins reduced tables |

### Phase 3: Original Methodology Analysis (COMPLETE)

Analyzed two notebooks in Original_Methodology folder:

**1. CORRELATION_HIERARCHICAL_FEATURE_CLUSTERING.py** (465 lines)
- Spearman correlation with distance = 1 - |correlation|
- Average linkage hierarchical clustering
- Fixed threshold 0.7 (correlation > 0.3)
- Outputs cluster assignments to CSV

**2. CRC_ITER1_MODEL-PREVALENCE.py** (1873 lines)
- SHAP calculation separate for positive/negative cases
- Combined importance: `(importance_pos * 2 + importance_neg) / 3`
- Removal criteria: near-zero (<0.0002), neg-biased ratio (<0.15), bottom 8%
- Features must meet 2+ criteria for removal
- Iterative removal with cluster preservation rules

### Issues Identified in Original Methodology

| Issue | Problem | Impact |
|-------|---------|--------|
| **SHAP Weighting** | 2:1 positive weight for 250:1 imbalance | Negatives dominate ~99% of combined importance |
| **Fixed Threshold** | 0.7 hardcoded for all features | May be suboptimal; no data-driven justification |
| **No Train-Only Split** | Clustering on full dataset | Potential data leakage (now fixed in Books 1-8) |

---

## Phase 4: New Feature Selection Pipeline (COMPLETE)

### Pipeline Overview

Create `V2_Book9_Feature_Selection.py` using a **Hybrid Two-Phase Approach**:

| Phase | Method | Features | Purpose |
|-------|--------|----------|---------|
| **Phase 1** | Cluster-Based Reduction | 171 → ~70-80 | Remove redundant/correlated features |
| **Phase 2** | Iterative SHAP Winnowing | ~70-80 → Final | Fine-tune with 20-25 removals per iteration |

### Input
- Wide feature table from Book 8 compilation (~171 features)
- Train/val/test split assignments from Book 0

### Output
- Final reduced feature set (determined by validation gate, not arbitrary target)
- Cluster assignments with justification
- SHAP importance rankings per iteration
- Iteration tracking CSV with overfitting metrics

---

### PHASE 1: Cluster-Based Reduction (171 → ~70-80 features)

**Goal**: Remove redundant features by keeping one representative per cluster.

**Rationale**: Correlated features provide redundant information. Keeping the best representative per cluster removes redundancy without losing predictive signal.

```
Step 1.1: Load Data
├── Load wide feature table from Book 8
├── Filter to SPLIT='train' only for correlation computation
└── Verify ~171 features loaded

Step 1.2: Compute Correlation Matrix
├── Spearman correlation (handles non-linear relationships)
├── Distance matrix: distance = 1 - |correlation|
└── Compute on TRAINING DATA ONLY

Step 1.3: Dynamic Threshold Selection
├── Test thresholds: 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
├── For each threshold:
│   ├── Run hierarchical clustering (average linkage)
│   ├── Count resulting clusters
│   └── Compute silhouette score (cluster quality)
├── Select threshold that balances:
│   ├── Reasonable cluster count (not too few, not all singletons)
│   └── Good silhouette score
├── Document chosen threshold with rationale
└── CHECKPOINT: Save clusters.pkl (correlation matrix, linkage, assignments)

Step 1.4: Train Baseline Model (All Features)
├── XGBoost with conservative hyperparameters
├── Early stopping on validation set
├── Record: baseline_auprc_train, baseline_auprc_val, baseline_auprc_test
├── Record: baseline_train_val_gap (overfitting indicator)
└── CHECKPOINT: Save baseline_model.pkl

Step 1.5: Compute SHAP with 2:1 Positive Weighting
├── TreeExplainer on baseline model
├── Compute SHAP separately for positive and negative cases
├── 2:1 weighting (model already uses scale_pos_weight for imbalance):
│   importance_combined = (importance_pos * 2 + importance_neg) / 3
├── Compute importance_ratio = importance_pos / importance_neg
└── This ratio identifies features most predictive of POSITIVE cases
├── CHECKPOINT: Save shap_phase1.pkl

Step 1.6: Select Cluster Representatives
├── For each cluster:
│   ├── Rank features by importance_ratio (descending)
│   ├── Cluster size 1-7: Keep top 1 feature
│   └── Cluster size 8+: Keep top 2 features
├── Singletons: Keep if importance_ratio > median, else flag for Phase 2
└── Output: ~70-80 features (depends on cluster count)

Step 1.7: Phase 1 Validation Gate
├── Train model with reduced feature set
├── Compare to baseline:
│   ├── val_auprc_drop = (baseline_val - reduced_val) / baseline_val
│   └── train_val_gap_change = reduced_gap - baseline_gap
├── PASS if: val_auprc_drop < 10% AND gap didn't increase significantly
├── If FAIL: Adjust (keep top 2 per cluster, or raise threshold)
└── CHECKPOINT: Save phase1_complete.pkl (feature list, model, metrics)
```

---

### PHASE 2: Iterative SHAP Winnowing (~70-80 → Final)

**Goal**: Carefully remove low-value features 20-25 at a time until validation performance degrades.

**Rationale**: After redundancy removal, remaining features may still include low-value ones. Iterative removal with validation gates prevents over-reduction.

```
For each iteration (starting from Phase 1 output):

Step 2.1: Train Model on Current Feature Set
├── XGBoost with same hyperparameters
├── Early stopping on validation set
├── Record: iter_auprc_train, iter_auprc_val, iter_auprc_test, iter_gap
└── CHECKPOINT: Save iter{N}_model.pkl

Step 2.2: Compute SHAP with 2:1 Positive Weighting
├── Same method as Phase 1
├── Rank all features by importance_combined
└── CHECKPOINT: Save iter{N}_shap.pkl

Step 2.3: Identify Removal Candidates (max 20-25)
├── Criteria for removal (must meet 2+ of 3):
│   ├── Near-zero importance: < 0.0002
│   ├── Negative-biased ratio: importance_ratio < 0.2
│   └── Bottom percentile: bottom 15% by importance_combined
├── Cap removals at 20-25 per iteration
└── Never remove features in top 50% by importance_ratio

Step 2.4: Iteration Validation Gate
├── Compute metrics:
│   ├── val_auprc_drop = (baseline_val - iter_val) / baseline_val
│   ├── gap_increase = iter_gap - baseline_gap
│   └── test_auprc_drop (monitor only, don't optimize on it)
├── STOP CONDITIONS (any triggers stop):
│   ├── val_auprc_drop > 5% from BASELINE (not previous iteration)
│   ├── gap_increase > 0.02 (overfitting getting worse)
│   ├── Feature count < 30 (minimum threshold)
│   └── No features meet removal criteria
└── If STOP: Revert to previous iteration's feature set

Step 2.5: Log & Checkpoint
├── Append to iteration_tracking.csv
├── CHECKPOINT: Save iter{N}_complete.pkl (feature list, metrics, removal list)
└── Continue to next iteration OR stop if gate triggered
```

---

### Overfitting Detection Strategy

| Metric | Formula | Threshold | Action |
|--------|---------|-----------|--------|
| **Val AUPRC Drop** | (baseline - current) / baseline | > 5% | STOP |
| **Train-Val Gap** | train_auprc - val_auprc | Increasing > 0.02 | STOP |
| **Test Monitoring** | Track but don't optimize | N/A | Report only |

**Key Principle**: We optimize on validation, monitor test, and never touch test for decisions. Final test evaluation happens once at the very end.

---

### Checkpoint System (Resumable Iterations)

**Goal**: Stop anytime and resume without starting over.

```
checkpoints/
├── step1_2_correlation.pkl      # After correlation matrix computed
├── step1_3_clusters.pkl         # After clustering complete
├── step1_4_baseline_model.pkl   # After baseline model trained
├── step1_5_shap_phase1.pkl      # After Phase 1 SHAP computed
├── step1_7_phase1_complete.pkl  # After Phase 1 validation gate
├── step2_iter1_model.pkl        # After iteration 1 model
├── step2_iter1_shap.pkl         # After iteration 1 SHAP
├── step2_iter1_complete.pkl     # After iteration 1 complete
├── step2_iter2_model.pkl        # ...and so on
└── iteration_tracking.csv       # Running log (always updated)
```

**On notebook startup:**
1. Scan for existing checkpoints
2. Display: "Found checkpoint at Phase 2, Iteration 3. Resume? [Y/n]"
3. If resume: Load checkpoint and continue
4. If fresh: Clear checkpoints directory and start over

**You can kill the notebook anytime** - just re-run and it picks up from the last checkpoint.

---

### Key Improvements Over Original

| Aspect | Original | New |
|--------|----------|-----|
| **SHAP Weighting** | 2:1 | 2:1 (same; model handles imbalance via scale_pos_weight) |
| **Clustering Threshold** | Fixed 0.7 | Dynamic via silhouette score |
| **Removal Cap** | 30 per iteration | 20-25 per iteration (more conservative) |
| **Stop Criterion** | Manual | Automatic validation gate |
| **Overfitting Detection** | Train-val gap tracking | Gap + absolute drop thresholds |
| **Two-Phase Approach** | Single iterative loop | Cluster reduction THEN iterative winnowing |
| **Resumability** | None (start over) | Granular checkpoints after each step |

---

### Expected Outcome

```
Starting:     171 features
After Phase 1: ~70-80 features (cluster representatives)
After Phase 2: ~40-60 features (validation-gated)

Iterations breakdown:
  Phase 2, Iter 1: 75 → 55 features (-20)
  Phase 2, Iter 2: 55 → 35 features (-20)
  Phase 2, Iter 3: STOP (val_auprc_drop > 5%)
  Final: Revert to 55 features
```

The actual stopping point depends on the data - we let the validation gate decide.

---

## Important Technical Context

- **Platform**: Databricks with PySpark
- **Data**: Can't query directly (siloed environment) - work from code
- **Table**: `{trgt_cat}.clncl_ds.herald_eda_train_final_cohort`
- **Class Imbalance**: 250:1 (0.41% positive rate)
- **Prediction Window**: 6 months
- **Cohort Period**: Jan 2023 - Sept 2024
- **Random Seed**: 217 (for all random operations)

## User Preferences

- Linear, readable code (no nested functions for now)
- Discuss changes before implementing
- Keep all logic in single notebooks (no separate utility files)
- Add progress print statements for long-running operations
- Preserve all original functionality unless explicitly changing
- Add "What This Cell Does" / "Conclusion" markdown cells

## Key Technical Decisions

1. **Temporal + Group Separation**: Q6 as temporal test set + SGKF for train/val
2. **3-Fold CV**: For computational efficiency with ~171 features
3. **Linear Code Style**: No nested functions - keep readable for debugging
4. **Documentation**: "What This Cell Does" + "Conclusion" markdown cells
5. **Dynamic Clustering Threshold**: Silhouette-based instead of fixed 0.7
6. **SHAP Weighting**: 2:1 for positive cases (model handles imbalance via scale_pos_weight)
7. **End Goal**: predict_proba -> isotonic calibration -> 0-100 risk score
8. **Resumable Pipeline**: Granular checkpoints allow stopping/resuming at any step

---

## Commands to Resume

```bash
# Navigate to project
cd /Users/michaeljoyce/Desktop/CLAUDE_CODE/METHODOLOGY_IMBALANCED

# Start Claude Code
claude

# Paste this prompt to resume:
"I'm continuing work on the CRC feature selection methodology improvement.
Please read MASTER_PROMPT.md for full context. All notebooks are complete.
V2_Book9_Feature_Selection.py implements the hybrid two-phase feature
selection pipeline with dynamic clustering and iterative SHAP winnowing."
```

## Running the Pipeline

1. **Upload notebooks to Databricks** (Books 0-9 in order)
2. **Run Books 0-8** to create the wide feature table with SPLIT column
3. **Run Book 9** for feature selection:
   - Phase 1: Cluster-based reduction (~171 → ~70-80 features)
   - Phase 2: Iterative SHAP winnowing (~70-80 → final)
   - Checkpoints saved after each step (kill anytime, resume on re-run)
4. **Outputs** in `feature_selection_outputs/`:
   - `final_features.txt` - Feature list
   - `final_features.py` - Importable Python list
   - `final_model.pkl` - Trained model
   - `iteration_tracking.csv` - Metrics per iteration

---

## Reference: Pattern Applied to Books 1-8

```python
# 1. Add SPLIT to the cohort join
df_cohort = spark.sql("""
    SELECT PAT_ID, END_DTTM, FUTURE_CRC_EVENT, SPLIT
    FROM dev.clncl_ds.herald_eda_train_final_cohort
""")

# 2. Create training-only dataframe for feature selection
df_train = df_spark.filter(F.col("SPLIT") == "train")
df_train.cache()

# 3. Use df_train for all feature selection metrics
baseline_crc_rate = df_train.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]
# Risk ratios calculated on df_train
# MI scores calculated on df_train sample
```

## Reference: SGKF Implementation (Book 0)

```python
from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=217)
# Takes first fold split: ~67% train, ~33% val
train_idx, val_idx = list(sgkf.split(X_dummy, y, groups))[0]
```
