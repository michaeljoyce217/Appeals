# Quick Reference for Sonnet 4.5

## Key Changes Summary

### 1. Multi-Class Stratification by Cancer Type (Book 0)

The SGKF now stratifies by cancer type, not binary positive/negative:
- **Class 0**: Negative (no CRC)
- **Class 1**: C18 (colon cancer)
- **Class 2**: C19 (rectosigmoid junction)
- **Class 3**: C20 (rectal cancer)
- **Class 4**: C21 (anal cancer) - only if `include_anus = True`

**Config flag** (line 156 in Book 0):
```python
include_anus = False  # Set to True to include C21
```

### 2. Train-Only Feature Selection (Books 1-7)

All feature selection metrics are computed on training data only:
```python
df_train = df_spark.filter(F.col("SPLIT") == "train")
baseline_crc_rate = df_train.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]
```

### 3. Excluded Columns (Books 8 & 9)

These columns are NOT features - exclude from all feature operations:
```python
exclude_cols = ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT', 'SPLIT', 'ICD10_CODE', 'ICD10_GROUP']
```

### 4. Expected Cancer Type Distribution

When running Book 0, verify these approximate distributions are preserved across splits:
- **C18 (Colon)**: ~70-75%
- **C20 (Rectum)**: ~15-20%
- **C19 (Rectosigmoid)**: ~3-5%
- **C21 (Anus)**: ~5-8% (only if included)

---

## Error Correction Prompt

Use this prompt when errors occur during notebook execution:

```
I'm running the CRC prediction pipeline notebooks and encountered an error.

**Notebook**: [Book number and name]
**Cell**: [Cell number or description]
**Error message**:
```
[paste error here]
```

**Context**:
- Running in Databricks
- Table catalog: {trgt_cat}.clncl_ds
- Random seed: 217

Please help me:
1. Diagnose the root cause
2. Provide the exact code fix
3. Explain if this affects downstream notebooks

Key constraints:
- Maintain train-only feature selection (SPLIT='train' for metrics)
- Preserve multi-class stratification by cancer type (C18/C19/C20)
- Never include ICD10_CODE or ICD10_GROUP as features (data leakage)
- Keep linear code style (no nested functions)
```

---

## Notebook Execution Order

```
Book 0  → Cohort Creation (creates SPLIT column, ICD10_GROUP)
Book 1  → Vitals
Book 2  → ICD10 Diagnoses
Book 3  → Social Factors (all features excluded - skip or run for documentation)
Book 4  → Labs
Book 5.1 → Outpatient Medications
Book 5.2 → Inpatient Medications
Book 6  → Visit History
Book 7  → Procedures
Book 8  → Compilation (joins all features, excludes ICD10 columns)
Book 9  → Feature Selection (hybrid clustering + SHAP)
```

---

## Common Issues & Fixes

### Issue: "Column SPLIT not found"
**Cause**: Book 0 didn't complete or table wasn't saved
**Fix**: Re-run Book 0 Section 10 (SPLIT assignment)

### Issue: Cancer type distribution unbalanced
**Cause**: SGKF not using multi-class stratification
**Fix**: Verify `strat_label` is used (not binary `label`) in SGKF

### Issue: ICD10_GROUP in feature columns
**Cause**: Book 8/9 not excluding diagnosis columns
**Fix**: Add to exclude_cols: `'ICD10_CODE', 'ICD10_GROUP'`

### Issue: "No positive cases in training data"
**Cause**: SPLIT filter applied incorrectly or data issue
**Fix**: Verify SPLIT column values: `df.groupBy("SPLIT").count().show()`

---

## Verification Checkpoints

After each book, verify:

**Book 0**:
```python
# Check SPLIT distribution
df.groupBy("SPLIT").agg(F.count("*"), F.mean("FUTURE_CRC_EVENT")).show()

# Check cancer type distribution
df.filter(F.col("FUTURE_CRC_EVENT") == 1).groupBy("SPLIT", "ICD10_GROUP").count().show()
```

**Book 8**:
```python
# Verify ICD10 columns NOT in features
assert 'ICD10_CODE' not in feature_cols
assert 'ICD10_GROUP' not in feature_cols
```

**Book 9**:
```python
# Verify exclusions
print(f"Excluded: {exclude_cols}")
print(f"Feature count: {len(feature_cols)}")
```
