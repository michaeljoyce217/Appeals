# Databricks notebook source
# MAGIC %md
# MAGIC # Book 9: Feature Selection Pipeline
# MAGIC
# MAGIC ## Hybrid Two-Phase Approach
# MAGIC
# MAGIC | Phase | Method | Features | Purpose |
# MAGIC |-------|--------|----------|---------|
# MAGIC | **Phase 1** | Cluster-Based Reduction | All → ~50% | Remove redundant/correlated features |
# MAGIC | **Phase 2** | Iterative SHAP Winnowing | ~50% → Final | Fine-tune with up to 25 removals per iteration |
# MAGIC | **Phase 3** | CV Stability Analysis | Final → Stable | Validate selection across N_CV_FOLDS folds |
# MAGIC
# MAGIC ## Key Features
# MAGIC - **Dynamic clustering threshold** via silhouette score (not fixed 0.7)
# MAGIC - **2:1 SHAP weighting** for positive cases (model handles imbalance via scale_pos_weight)
# MAGIC - **Clinical must-keep features** - preserve interpretable CRC signals regardless of ranking
# MAGIC - **3-fold CV stability analysis** - identify features robust across different train/val splits
# MAGIC - **Test set holdout** - test metrics evaluated only once at final model (no peeking)
# MAGIC - **Granular checkpoints** - stop anytime and resume without starting over
# MAGIC - **Automatic validation gates** - stops when performance degrades
# MAGIC
# MAGIC ## Checkpoint System
# MAGIC Checkpoints saved after each step using portable formats (JSON for models, Parquet for DataFrames).
# MAGIC Kill notebook anytime, re-run to resume. Backward compatible with legacy pickle checkpoints.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# ============================================================================
# CONFIGURATION - Modify these as needed
# ============================================================================

# Resume behavior: Set to True to check for and resume from checkpoints
# Set to False to start fresh (will prompt to confirm clearing checkpoints)
AUTO_RESUME = True

# Phase 2 iteration limits
MAX_REMOVALS_PER_ITERATION = 25  # Cap at 20-25 features removed per iteration
MIN_FEATURES_THRESHOLD = 30      # Never go below this many features

# Validation gate thresholds
MAX_VAL_AUPRC_DROP = 0.05        # Stop if validation AUPRC drops > 5% from baseline
MAX_GAP_INCREASE = 0.02          # Stop if train-val gap increases > 0.02

# Phase 1 validation gate (more lenient since we're removing redundancy)
PHASE1_MAX_VAL_DROP = 0.10       # Allow up to 10% drop in Phase 1

# Clustering threshold search range
THRESHOLD_MIN = 0.50
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.05

# Random seed for reproducibility
RANDOM_SEED = 217

# Output directories
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "feature_selection_outputs"

# =============================================================================
# CLINICAL MUST-KEEP FEATURES
# =============================================================================
# These features are preserved regardless of their statistical ranking.
# They represent clinically important or interpretable signals that should
# always be available in the final model for clinical decision-making.
#
# To add a feature: Include the exact column name as it appears in the data.
# To disable: Set CLINICAL_MUST_KEEP_FEATURES = [] (empty list)
# =============================================================================
CLINICAL_MUST_KEEP_FEATURES = [
    # Weight loss indicators - cardinal sign of CRC
    'HAS_RAPID_WEIGHT_LOSS',
    'MAX_WEIGHT_LOSS_PCT_60D',
    'WEIGHT_LOSS_FLAG_6M',

    # GI bleeding - high-specificity CRC symptom
    'HAS_GI_BLEEDING',
    'HAS_RECTAL_BLEEDING',

    # Anemia - common CRC presentation
    'HAS_IRON_DEFICIENCY_ANEMIA',
    'HEMOGLOBIN_LATEST',

    # Age - fundamental risk factor
    'AGE',

    # Change in bowel habits - classic CRC symptom
    'HAS_BOWEL_HABIT_CHANGE',
]

# Number of CV folds for feature selection stability
N_CV_FOLDS = 3
CV_FEATURE_THRESHOLD = 0.67  # Keep features appearing in >= 67% of folds (2/3)

print("="*70)
print("CONFIGURATION")
print("="*70)
print(f"AUTO_RESUME: {AUTO_RESUME}")
print(f"MAX_REMOVALS_PER_ITERATION: {MAX_REMOVALS_PER_ITERATION}")
print(f"MIN_FEATURES_THRESHOLD: {MIN_FEATURES_THRESHOLD}")
print(f"MAX_VAL_AUPRC_DROP: {MAX_VAL_AUPRC_DROP}")
print(f"MAX_GAP_INCREASE: {MAX_GAP_INCREASE}")
print(f"RANDOM_SEED: {RANDOM_SEED}")
print(f"N_CV_FOLDS: {N_CV_FOLDS}")
print(f"CV_FEATURE_THRESHOLD: {CV_FEATURE_THRESHOLD}")
print(f"CLINICAL_MUST_KEEP_FEATURES: {len(CLINICAL_MUST_KEEP_FEATURES)} features")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imports and Environment Setup

# COMMAND ----------

import os
import pickle
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    silhouette_score, average_precision_score, roc_auc_score, brier_score_loss
)

from xgboost import XGBClassifier
import shap

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Initialize Spark
spark = SparkSession.builder.getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "America/Chicago")

# Get catalog from environment
trgt_cat = os.environ.get('trgt_cat', 'dev')

print("="*70)
print("ENVIRONMENT INITIALIZED")
print("="*70)
print(f"Timestamp: {datetime.now()}")
print(f"Spark version: {spark.version}")
print(f"Target catalog: {trgt_cat}")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Checkpoint Management Functions
# MAGIC
# MAGIC Checkpoints use portable, version-stable formats:
# MAGIC - **XGBoost models**: JSON format via `model.save_model()` (portable across XGBoost versions)
# MAGIC - **DataFrames**: Parquet format (efficient, schema-preserving)
# MAGIC - **NumPy arrays**: `.npy` format
# MAGIC - **Simple types**: JSON format
# MAGIC - **Complex objects**: Pickle fallback (scipy linkage matrices, etc.)
# MAGIC
# MAGIC Each checkpoint is stored as a directory with typed files + metadata.
# MAGIC Legacy pickle checkpoints (`.pkl` files) are still supported for backward compatibility.

# COMMAND ----------

def ensure_directories():
    """Create checkpoint and output directories if they don't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Directories verified: {CHECKPOINT_DIR}/, {OUTPUT_DIR}/")

def get_checkpoint_dir(name):
    """Get directory path for a checkpoint (each checkpoint is a directory)."""
    return os.path.join(CHECKPOINT_DIR, name)

def save_checkpoint(name, data):
    """
    Save a checkpoint with format-appropriate storage:
    - XGBoost models: JSON format (portable across versions)
    - DataFrames: Parquet format (efficient, portable)
    - Other data: JSON for simple types, pickle for complex types
    """
    checkpoint_dir = get_checkpoint_dir(name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    metadata = {'keys': []}

    for key, value in data.items():
        metadata['keys'].append(key)

        if isinstance(value, XGBClassifier):
            # Save XGBoost model as JSON
            model_path = os.path.join(checkpoint_dir, f"{key}.json")
            value.save_model(model_path)

        elif isinstance(value, pd.DataFrame):
            # Save DataFrame as Parquet
            parquet_path = os.path.join(checkpoint_dir, f"{key}.parquet")
            value.to_parquet(parquet_path, index=False)

        elif isinstance(value, np.ndarray):
            # Save numpy arrays as .npy
            npy_path = os.path.join(checkpoint_dir, f"{key}.npy")
            np.save(npy_path, value)

        elif isinstance(value, (list, dict, str, int, float, bool, type(None))):
            # Save simple types as JSON
            json_path = os.path.join(checkpoint_dir, f"{key}.json")
            with open(json_path, 'w') as f:
                json.dump(value, f, indent=2, default=str)

        else:
            # Fallback to pickle for complex types (scipy linkage, etc.)
            pkl_path = os.path.join(checkpoint_dir, f"{key}.pkl")
            with open(pkl_path, 'wb') as f:
                pickle.dump(value, f)

    # Save metadata
    with open(os.path.join(checkpoint_dir, "_metadata.json"), 'w') as f:
        json.dump(metadata, f)

    print(f"✓ CHECKPOINT SAVED: {name}")

def load_checkpoint(name):
    """Load a checkpoint by name. Returns None if not found."""
    checkpoint_dir = get_checkpoint_dir(name)

    # Check for new format (directory-based)
    if os.path.isdir(checkpoint_dir):
        metadata_path = os.path.join(checkpoint_dir, "_metadata.json")
        if not os.path.exists(metadata_path):
            return None

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        data = {}
        for key in metadata['keys']:
            # Try each format in order
            json_path = os.path.join(checkpoint_dir, f"{key}.json")
            parquet_path = os.path.join(checkpoint_dir, f"{key}.parquet")
            npy_path = os.path.join(checkpoint_dir, f"{key}.npy")
            pkl_path = os.path.join(checkpoint_dir, f"{key}.pkl")

            if key == 'model' and os.path.exists(json_path):
                # Load XGBoost model
                model = XGBClassifier()
                model.load_model(json_path)
                data[key] = model
            elif os.path.exists(parquet_path):
                # Load DataFrame
                data[key] = pd.read_parquet(parquet_path)
            elif os.path.exists(npy_path):
                # Load numpy array
                data[key] = np.load(npy_path)
            elif os.path.exists(json_path):
                # Load JSON
                with open(json_path, 'r') as f:
                    data[key] = json.load(f)
            elif os.path.exists(pkl_path):
                # Load pickle
                with open(pkl_path, 'rb') as f:
                    data[key] = pickle.load(f)

        print(f"✓ CHECKPOINT LOADED: {name}")
        return data

    # Fallback: check for legacy pickle format
    legacy_path = os.path.join(CHECKPOINT_DIR, f"{name}.pkl")
    if os.path.exists(legacy_path):
        print(f"⚠ Loading legacy pickle checkpoint: {name}")
        with open(legacy_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ CHECKPOINT LOADED: {name} (legacy format)")
        return data

    return None

def checkpoint_exists(name):
    """Check if a checkpoint exists (new or legacy format)."""
    checkpoint_dir = get_checkpoint_dir(name)
    legacy_path = os.path.join(CHECKPOINT_DIR, f"{name}.pkl")
    return os.path.isdir(checkpoint_dir) or os.path.exists(legacy_path)

def list_checkpoints():
    """List all existing checkpoints (new and legacy formats)."""
    if not os.path.exists(CHECKPOINT_DIR):
        return []

    checkpoints = set()

    # New format: directories
    for item in os.listdir(CHECKPOINT_DIR):
        item_path = os.path.join(CHECKPOINT_DIR, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "_metadata.json")):
            checkpoints.add(item)

    # Legacy format: .pkl files
    for f in os.listdir(CHECKPOINT_DIR):
        if f.endswith('.pkl'):
            checkpoints.add(f.replace('.pkl', ''))

    return sorted(checkpoints)

def get_latest_checkpoint():
    """Determine the latest checkpoint and what step to resume from."""
    checkpoints = list_checkpoints()
    if not checkpoints:
        return None, "start"

    # Define checkpoint order
    checkpoint_order = [
        "step1_1_data",
        "step1_2_correlation",
        "step1_3_clusters",
        "step1_4_baseline_model",
        "step1_5_shap_phase1",
        "step1_6_cluster_representatives",
        "step1_7_phase1_complete"
    ]

    # Check Phase 1 checkpoints
    latest_phase1 = None
    for cp in checkpoint_order:
        if cp in checkpoints:
            latest_phase1 = cp

    # Check Phase 2 iteration checkpoints
    phase2_iters = [cp for cp in checkpoints if cp.startswith("step2_iter")]
    if phase2_iters:
        # Find highest iteration number
        iter_nums = []
        for cp in phase2_iters:
            try:
                # Extract iteration number from names like "step2_iter3_complete"
                parts = cp.split("_")
                iter_num = int(parts[1].replace("iter", ""))
                iter_nums.append((iter_num, cp))
            except:
                pass
        if iter_nums:
            max_iter = max(iter_nums, key=lambda x: x[0])
            return max_iter[1], f"phase2_iter{max_iter[0]}"

    if latest_phase1:
        return latest_phase1, latest_phase1

    return None, "start"

def clear_checkpoints():
    """Clear all checkpoints to start fresh (handles both new and legacy formats)."""
    import shutil

    if os.path.exists(CHECKPOINT_DIR):
        for item in os.listdir(CHECKPOINT_DIR):
            item_path = os.path.join(CHECKPOINT_DIR, item)
            if os.path.isdir(item_path):
                # New format: remove entire checkpoint directory
                shutil.rmtree(item_path)
            elif item.endswith('.pkl'):
                # Legacy format: remove pickle file
                os.remove(item_path)
        print("✓ All checkpoints cleared")

def display_checkpoint_status():
    """Display current checkpoint status."""
    checkpoints = list_checkpoints()
    latest, resume_point = get_latest_checkpoint()

    print("="*70)
    print("CHECKPOINT STATUS")
    print("="*70)

    if not checkpoints:
        print("No checkpoints found. Starting fresh.")
    else:
        print(f"Found {len(checkpoints)} checkpoint(s):")
        for cp in checkpoints:
            print(f"  - {cp}")
        print(f"\nLatest: {latest}")
        print(f"Resume point: {resume_point}")

    print("="*70)
    return latest, resume_point

# Initialize directories
ensure_directories()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check for Existing Checkpoints and Resume Point

# COMMAND ----------

# Display checkpoint status and determine resume point
latest_checkpoint, resume_point = display_checkpoint_status()

# Determine whether to resume or start fresh
if latest_checkpoint and AUTO_RESUME:
    print(f"\n>>> AUTO_RESUME is ON. Resuming from: {resume_point}")
    START_FRESH = False
elif latest_checkpoint and not AUTO_RESUME:
    print(f"\n>>> Checkpoints exist but AUTO_RESUME is OFF.")
    print(">>> Set AUTO_RESUME = True to resume, or run clear_checkpoints() to start fresh.")
    START_FRESH = True
else:
    print("\n>>> No checkpoints found. Starting fresh.")
    START_FRESH = True

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utility Functions

# COMMAND ----------

def train_xgboost_model(X_train, y_train, X_val, y_val, feature_cols, scale_pos_weight):
    """
    Train an XGBoost model with conservative hyperparameters.
    Returns the trained model.
    """
    params = {
        'max_depth': 4,
        'min_child_weight': 50,
        'gamma': 1.0,
        'subsample': 0.5,
        'colsample_bytree': 0.5,
        'reg_alpha': 1.0,
        'reg_lambda': 10.0,
        'learning_rate': 0.01,
        'n_estimators': 2000,
        'scale_pos_weight': scale_pos_weight,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'random_state': RANDOM_SEED,
        'early_stopping_rounds': 100
    }

    model = XGBClassifier(**params)
    model.fit(
        X_train[feature_cols], y_train,
        eval_set=[(X_val[feature_cols], y_val)],
        verbose=False
    )

    return model

def evaluate_model(model, X, y, feature_cols, split_name=""):
    """
    Evaluate model and return metrics dictionary.
    """
    y_pred = model.predict_proba(X[feature_cols])[:, 1]

    auprc = average_precision_score(y, y_pred)
    auroc = roc_auc_score(y, y_pred)
    brier = brier_score_loss(y, y_pred)
    baseline_rate = y.mean()

    metrics = {
        'auprc': auprc,
        'auroc': auroc,
        'brier': brier,
        'baseline_rate': baseline_rate,
        'lift': auprc / baseline_rate if baseline_rate > 0 else 0,
        'n_samples': len(y),
        'n_events': int(y.sum())
    }

    if split_name:
        print(f"  {split_name}: AUPRC={auprc:.4f} ({metrics['lift']:.1f}x lift), AUROC={auroc:.4f}")

    return metrics

def compute_shap_values(model, X_val, y_val, feature_cols):
    """
    Compute SHAP values separately for positive and negative cases.
    Returns importance_pos, importance_neg, importance_combined, importance_ratio.
    """
    print("  Computing SHAP values...")

    # Get indices for positive and negative cases
    pos_mask = y_val == 1
    neg_mask = y_val == 0

    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()

    print(f"    Positive cases: {n_pos:,}")
    print(f"    Negative cases: {n_neg:,}")

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Compute SHAP for positive cases
    print("    Computing SHAP for positive cases...")
    start = time.time()
    X_pos = X_val.loc[pos_mask, feature_cols]
    shap_values_pos = explainer.shap_values(X_pos)
    print(f"    Completed in {time.time() - start:.1f}s")

    # Compute SHAP for negative cases (sample if too large)
    print("    Computing SHAP for negative cases...")
    start = time.time()
    X_neg = X_val.loc[neg_mask, feature_cols]

    # Sample negative cases if there are too many (>50k)
    if len(X_neg) > 50000:
        print(f"    Sampling 50,000 from {len(X_neg):,} negative cases...")
        X_neg_sample = X_neg.sample(n=50000, random_state=RANDOM_SEED)
    else:
        X_neg_sample = X_neg

    shap_values_neg = explainer.shap_values(X_neg_sample)
    print(f"    Completed in {time.time() - start:.1f}s")

    # Calculate importance metrics
    importance_pos = np.abs(shap_values_pos).mean(axis=0)
    importance_neg = np.abs(shap_values_neg).mean(axis=0)

    # 2:1 weighting for positive cases
    importance_combined = (importance_pos * 2 + importance_neg) / 3

    # Ratio: higher = more predictive of positive cases
    importance_ratio = importance_pos / (importance_neg + 1e-10)

    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'SHAP_Positive': importance_pos,
        'SHAP_Negative': importance_neg,
        'SHAP_Combined': importance_combined,
        'SHAP_Ratio': importance_ratio
    }).sort_values('SHAP_Combined', ascending=False)

    importance_df['Rank'] = range(1, len(importance_df) + 1)

    print(f"  ✓ SHAP computation complete")

    return importance_df

def update_tracking_csv(iteration_data):
    """
    Append iteration results to tracking CSV.
    """
    csv_path = os.path.join(OUTPUT_DIR, "iteration_tracking.csv")

    df_new = pd.DataFrame([iteration_data])

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        # Remove existing row for this iteration if re-running
        phase = iteration_data.get('phase', '')
        iteration = iteration_data.get('iteration', '')
        df_existing = df_existing[~((df_existing['phase'] == phase) & (df_existing['iteration'] == iteration))]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)
    print(f"  ✓ Tracking CSV updated: {csv_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # PHASE 1: Cluster-Based Reduction
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.1: Load Data

# COMMAND ----------

# Check if we can skip this step
if checkpoint_exists("step1_1_data") and not START_FRESH:
    print(">>> Loading from checkpoint: step1_1_data")
    data_checkpoint = load_checkpoint("step1_1_data")
    df_pandas = data_checkpoint['df_pandas']
    feature_cols = data_checkpoint['feature_cols']
    scale_pos_weight = data_checkpoint['scale_pos_weight']
else:
    print("="*70)
    print("STEP 1.1: LOAD DATA")
    print("="*70)

    # Load wide feature table with SPLIT column
    print("Loading data from Spark...")

    df_spark = spark.sql(f'''
    SELECT *
    FROM {trgt_cat}.clncl_ds.herald_eda_train_wide_cleaned
    ''')

    # Convert to Pandas
    print("Converting to Pandas...")
    df_pandas = df_spark.toPandas()

    # Convert datetime
    df_pandas['END_DTTM'] = pd.to_datetime(df_pandas['END_DTTM'])

    # Identify feature columns (exclude identifiers and target)
    exclude_cols = ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT', 'SPLIT']
    feature_cols = [c for c in df_pandas.columns if c not in exclude_cols]

    # Calculate scale_pos_weight from training data
    train_mask = df_pandas['SPLIT'] == 'train'
    n_neg = (df_pandas.loc[train_mask, 'FUTURE_CRC_EVENT'] == 0).sum()
    n_pos = (df_pandas.loc[train_mask, 'FUTURE_CRC_EVENT'] == 1).sum()
    scale_pos_weight = n_neg / n_pos

    print(f"\nData loaded:")
    print(f"  Total observations: {len(df_pandas):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Train: {train_mask.sum():,}")
    print(f"  Val: {(df_pandas['SPLIT'] == 'val').sum():,}")
    print(f"  Test: {(df_pandas['SPLIT'] == 'test').sum():,}")
    print(f"  Scale pos weight: {scale_pos_weight:.1f}")

    # Save checkpoint
    save_checkpoint("step1_1_data", {
        'df_pandas': df_pandas,
        'feature_cols': feature_cols,
        'scale_pos_weight': scale_pos_weight
    })

print(f"\n✓ Data ready: {len(df_pandas):,} observations, {len(feature_cols)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.1b: Create CV Folds for Feature Selection Stability
# MAGIC
# MAGIC To ensure feature selection stability, we run the selection process across multiple
# MAGIC cross-validation folds. Features that are consistently selected across folds are more
# MAGIC likely to be genuinely important rather than artifacts of a particular train/val split.
# MAGIC
# MAGIC **Approach:**
# MAGIC - Test set (Q6) remains fixed as temporal holdout
# MAGIC - Q0-Q5 data is split into 3 folds using StratifiedGroupKFold
# MAGIC - Feature selection runs on each fold
# MAGIC - Final features = those selected in ≥2/3 folds (67% threshold)

# COMMAND ----------

from sklearn.model_selection import StratifiedGroupKFold

# Check if we can skip this step
if checkpoint_exists("step1_1b_cv_folds") and not START_FRESH:
    print(">>> Loading from checkpoint: step1_1b_cv_folds")
    cv_checkpoint = load_checkpoint("step1_1b_cv_folds")
    cv_fold_assignments = cv_checkpoint['cv_fold_assignments']
else:
    print("="*70)
    print("STEP 1.1b: CREATE CV FOLDS FOR FEATURE SELECTION")
    print("="*70)

    # Separate test (Q6) from train/val pool (existing split)
    test_mask = df_pandas['SPLIT'] == 'test'
    trainval_mask = df_pandas['SPLIT'].isin(['train', 'val'])

    print(f"Test set (fixed): {test_mask.sum():,} observations")
    print(f"Train/Val pool: {trainval_mask.sum():,} observations")

    # Get patient-level labels for SGKF
    trainval_df = df_pandas[trainval_mask].copy()
    patient_labels = trainval_df.groupby('PAT_ID')['FUTURE_CRC_EVENT'].max().reset_index()
    patient_labels.columns = ['PAT_ID', 'label']

    print(f"Unique patients in train/val: {len(patient_labels):,}")
    print(f"Positive patients: {patient_labels['label'].sum():,}")

    # Create 3-fold SGKF
    sgkf = StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    X_dummy = np.zeros(len(patient_labels))
    y = patient_labels['label'].values
    groups = patient_labels['PAT_ID'].values

    # Store fold assignments for each patient
    cv_fold_assignments = {}
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X_dummy, y, groups)):
        train_patients = set(patient_labels.iloc[train_idx]['PAT_ID'].values)
        val_patients = set(patient_labels.iloc[val_idx]['PAT_ID'].values)

        cv_fold_assignments[fold_idx] = {
            'train_patients': train_patients,
            'val_patients': val_patients
        }

        # Calculate stats
        train_obs = trainval_df[trainval_df['PAT_ID'].isin(train_patients)]
        val_obs = trainval_df[trainval_df['PAT_ID'].isin(val_patients)]

        print(f"\nFold {fold_idx + 1}:")
        print(f"  Train: {len(train_patients):,} patients, {len(train_obs):,} observations")
        print(f"  Val:   {len(val_patients):,} patients, {len(val_obs):,} observations")
        print(f"  Train event rate: {train_obs['FUTURE_CRC_EVENT'].mean()*100:.4f}%")
        print(f"  Val event rate:   {val_obs['FUTURE_CRC_EVENT'].mean()*100:.4f}%")

    # Save checkpoint
    save_checkpoint("step1_1b_cv_folds", {
        'cv_fold_assignments': cv_fold_assignments
    })

print(f"\n✓ CV folds created: {N_CV_FOLDS} folds for feature selection stability")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.2: Compute Correlation Matrix (Training Data Only)

# COMMAND ----------

# Check if we can skip this step
if checkpoint_exists("step1_2_correlation") and not START_FRESH:
    print(">>> Loading from checkpoint: step1_2_correlation")
    corr_checkpoint = load_checkpoint("step1_2_correlation")
    corr_matrix = corr_checkpoint['corr_matrix']
    dist_matrix = corr_checkpoint['dist_matrix']
else:
    print("="*70)
    print("STEP 1.2: COMPUTE CORRELATION MATRIX")
    print("="*70)

    # Filter to training data only
    train_mask = df_pandas['SPLIT'] == 'train'
    df_train = df_pandas.loc[train_mask, feature_cols]

    print(f"Computing Spearman correlation on {len(df_train):,} training observations...")
    print(f"Features: {len(feature_cols)}")

    start = time.time()
    corr_matrix = df_train.corr(method='spearman')
    elapsed = time.time() - start
    print(f"✓ Correlation matrix computed in {elapsed:.1f}s")

    # Convert to distance matrix: distance = 1 - |correlation|
    dist_matrix = 1 - corr_matrix.abs()

    # Save checkpoint
    save_checkpoint("step1_2_correlation", {
        'corr_matrix': corr_matrix,
        'dist_matrix': dist_matrix
    })

print(f"\n✓ Correlation matrix ready: {corr_matrix.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.3: Dynamic Threshold Selection via Silhouette Score

# COMMAND ----------

# Check if we can skip this step
if checkpoint_exists("step1_3_clusters") and not START_FRESH:
    print(">>> Loading from checkpoint: step1_3_clusters")
    cluster_checkpoint = load_checkpoint("step1_3_clusters")
    linkage_matrix = cluster_checkpoint['linkage_matrix']
    chosen_threshold = cluster_checkpoint['chosen_threshold']
    cluster_labels = cluster_checkpoint['cluster_labels']
    cluster_df = cluster_checkpoint['cluster_df']
    threshold_results = cluster_checkpoint['threshold_results']
else:
    print("="*70)
    print("STEP 1.3: DYNAMIC THRESHOLD SELECTION")
    print("="*70)

    # Convert distance matrix to condensed form for hierarchical clustering
    # Ensure diagonal is 0 and matrix is symmetric
    dist_values = dist_matrix.values
    np.fill_diagonal(dist_values, 0)
    condensed_dist = squareform(dist_values)

    # Perform hierarchical clustering
    print("Computing hierarchical clustering (average linkage)...")
    linkage_matrix = linkage(condensed_dist, method='average')

    # Test different thresholds
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
    threshold_results = []

    print(f"\nTesting thresholds from {THRESHOLD_MIN} to {THRESHOLD_MAX}:")
    print("-"*60)

    for thresh in thresholds:
        # Get cluster labels at this threshold
        labels = fcluster(linkage_matrix, t=thresh, criterion='distance')
        n_clusters = len(np.unique(labels))

        # Count singletons
        cluster_sizes = pd.Series(labels).value_counts()
        n_singletons = (cluster_sizes == 1).sum()

        # Compute silhouette score (only if we have 2+ clusters and not all singletons)
        if n_clusters > 1 and n_clusters < len(feature_cols):
            try:
                sil_score = silhouette_score(dist_values, labels, metric='precomputed')
            except:
                sil_score = -1
        else:
            sil_score = -1

        threshold_results.append({
            'threshold': thresh,
            'n_clusters': n_clusters,
            'n_singletons': n_singletons,
            'pct_singletons': n_singletons / n_clusters * 100,
            'silhouette': sil_score
        })

        print(f"  Threshold {thresh:.2f}: {n_clusters} clusters, {n_singletons} singletons ({n_singletons/n_clusters*100:.0f}%), silhouette={sil_score:.3f}")

    threshold_df = pd.DataFrame(threshold_results)

    # Select best threshold:
    # - Silhouette score > 0
    # - Not too many singletons (< 80%)
    # - Reasonable cluster count (not all singletons, not one giant cluster)
    valid_thresholds = threshold_df[
        (threshold_df['silhouette'] > 0) &
        (threshold_df['pct_singletons'] < 80) &
        (threshold_df['n_clusters'] > 10)
    ]

    if len(valid_thresholds) > 0:
        # Choose threshold with best silhouette among valid options
        best_idx = valid_thresholds['silhouette'].idxmax()
        chosen_threshold = valid_thresholds.loc[best_idx, 'threshold']
    else:
        # Fallback to 0.7 if no valid threshold found
        print("\n⚠ No threshold met all criteria. Falling back to 0.7")
        chosen_threshold = 0.7

    print(f"\n>>> CHOSEN THRESHOLD: {chosen_threshold}")

    # Get final cluster assignments
    cluster_labels = fcluster(linkage_matrix, t=chosen_threshold, criterion='distance')

    # Create cluster DataFrame
    cluster_df = pd.DataFrame({
        'Feature': feature_cols,
        'Cluster': cluster_labels
    })

    n_clusters = len(np.unique(cluster_labels))
    cluster_sizes = cluster_df['Cluster'].value_counts()

    print(f"\nCluster summary at threshold {chosen_threshold}:")
    print(f"  Total clusters: {n_clusters}")
    print(f"  Singletons: {(cluster_sizes == 1).sum()}")
    print(f"  Largest cluster: {cluster_sizes.max()} features")
    print(f"  Mean cluster size: {cluster_sizes.mean():.1f}")

    # Save checkpoint
    save_checkpoint("step1_3_clusters", {
        'linkage_matrix': linkage_matrix,
        'chosen_threshold': chosen_threshold,
        'cluster_labels': cluster_labels,
        'cluster_df': cluster_df,
        'threshold_results': threshold_results
    })

print(f"\n✓ Clustering complete: {len(cluster_df['Cluster'].unique())} clusters at threshold {chosen_threshold}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Clustering Results

# COMMAND ----------

# Plot threshold analysis
threshold_df = pd.DataFrame(threshold_results)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Silhouette scores
axes[0].plot(threshold_df['threshold'], threshold_df['silhouette'], 'bo-')
axes[0].axvline(x=chosen_threshold, color='r', linestyle='--', label=f'Chosen: {chosen_threshold}')
axes[0].set_xlabel('Threshold')
axes[0].set_ylabel('Silhouette Score')
axes[0].set_title('Silhouette Score vs Threshold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Number of clusters
axes[1].plot(threshold_df['threshold'], threshold_df['n_clusters'], 'go-')
axes[1].axvline(x=chosen_threshold, color='r', linestyle='--', label=f'Chosen: {chosen_threshold}')
axes[1].set_xlabel('Threshold')
axes[1].set_ylabel('Number of Clusters')
axes[1].set_title('Cluster Count vs Threshold')
axes[1].legend()
axes[1].grid(alpha=0.3)

# Singleton percentage
axes[2].plot(threshold_df['threshold'], threshold_df['pct_singletons'], 'mo-')
axes[2].axvline(x=chosen_threshold, color='r', linestyle='--', label=f'Chosen: {chosen_threshold}')
axes[2].set_xlabel('Threshold')
axes[2].set_ylabel('% Singletons')
axes[2].set_title('Singleton Percentage vs Threshold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'threshold_analysis.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"✓ Saved: {OUTPUT_DIR}/threshold_analysis.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.4: Train Baseline Model (All Features)
# MAGIC
# MAGIC Train XGBoost model with all features to establish baseline performance.
# MAGIC Evaluation is performed on **train and validation sets only** - test set is held out
# MAGIC until final model evaluation to prevent information leakage during feature selection.

# COMMAND ----------

# Check if we can skip this step
if checkpoint_exists("step1_4_baseline_model") and not START_FRESH:
    print(">>> Loading from checkpoint: step1_4_baseline_model")
    baseline_checkpoint = load_checkpoint("step1_4_baseline_model")
    baseline_model = baseline_checkpoint['model']
    baseline_metrics = baseline_checkpoint['metrics']
else:
    print("="*70)
    print("STEP 1.4: TRAIN BASELINE MODEL (ALL FEATURES)")
    print("="*70)

    # Prepare data splits
    train_mask = df_pandas['SPLIT'] == 'train'
    val_mask = df_pandas['SPLIT'] == 'val'
    test_mask = df_pandas['SPLIT'] == 'test'

    X_train = df_pandas.loc[train_mask].copy()
    y_train = df_pandas.loc[train_mask, 'FUTURE_CRC_EVENT'].copy()

    X_val = df_pandas.loc[val_mask].copy()
    y_val = df_pandas.loc[val_mask, 'FUTURE_CRC_EVENT'].copy()

    X_test = df_pandas.loc[test_mask].copy()
    y_test = df_pandas.loc[test_mask, 'FUTURE_CRC_EVENT'].copy()

    print(f"Training baseline model with {len(feature_cols)} features...")
    print(f"  Train: {len(y_train):,} obs, {y_train.sum():,} events")
    print(f"  Val: {len(y_val):,} obs, {y_val.sum():,} events")
    print(f"  Test: {len(y_test):,} obs, {y_test.sum():,} events")

    start = time.time()
    baseline_model = train_xgboost_model(
        X_train, y_train, X_val, y_val,
        feature_cols, scale_pos_weight
    )
    elapsed = time.time() - start
    print(f"✓ Model trained in {elapsed:.1f}s (best iteration: {baseline_model.best_iteration})")

    # Evaluate on train/val only (test is held out until final evaluation)
    print("\nBaseline performance:")
    baseline_metrics = {
        'train': evaluate_model(baseline_model, X_train, y_train, feature_cols, "Train"),
        'val': evaluate_model(baseline_model, X_val, y_val, feature_cols, "Val")
    }

    # Calculate train-val gap
    baseline_metrics['train_val_gap'] = baseline_metrics['train']['auprc'] - baseline_metrics['val']['auprc']
    print(f"  Train-Val Gap: {baseline_metrics['train_val_gap']:.4f}")
    print("  (Test set held out - evaluated only at final model)")

    # Save checkpoint
    save_checkpoint("step1_4_baseline_model", {
        'model': baseline_model,
        'metrics': baseline_metrics
    })

    # Update tracking CSV
    update_tracking_csv({
        'phase': 'phase1',
        'iteration': 'baseline',
        'n_features': len(feature_cols),
        'n_removed': 0,
        'train_auprc': baseline_metrics['train']['auprc'],
        'val_auprc': baseline_metrics['val']['auprc'],
        'train_val_gap': baseline_metrics['train_val_gap'],
        'val_drop_from_baseline': 0.0,
        'timestamp': datetime.now().isoformat()
    })

print(f"\n✓ Baseline model ready: Val AUPRC = {baseline_metrics['val']['auprc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.5: Compute SHAP with 2:1 Positive Weighting
# MAGIC
# MAGIC ### Why SHAP is Computed on Validation Data
# MAGIC
# MAGIC We compute SHAP values on the validation set rather than the training set for two reasons:
# MAGIC
# MAGIC 1. **Avoid overfitting to training quirks**: SHAP on training data would reflect feature
# MAGIC    contributions to training patterns, including any noise the model memorized. Validation
# MAGIC    data provides a cleaner signal of generalizable feature importance.
# MAGIC
# MAGIC 2. **Consistent with early stopping**: The model was trained with early stopping on validation
# MAGIC    performance, so its learned structure is already optimized for validation. SHAP on
# MAGIC    validation reflects the model's actual generalization behavior.
# MAGIC
# MAGIC **Tradeoff**: The validation set has fewer positive cases (~1/3 of training), so importance
# MAGIC estimates have higher variance for rare features. We mitigate this by keeping all positive
# MAGIC cases in the SHAP computation.

# COMMAND ----------

# Check if we can skip this step
if checkpoint_exists("step1_5_shap_phase1") and not START_FRESH:
    print(">>> Loading from checkpoint: step1_5_shap_phase1")
    shap_checkpoint = load_checkpoint("step1_5_shap_phase1")
    importance_df = shap_checkpoint['importance_df']
else:
    print("="*70)
    print("STEP 1.5: COMPUTE SHAP VALUES (2:1 POSITIVE WEIGHTING)")
    print("="*70)

    # Use validation set for SHAP
    val_mask = df_pandas['SPLIT'] == 'val'
    X_val = df_pandas.loc[val_mask].copy()
    y_val = df_pandas.loc[val_mask, 'FUTURE_CRC_EVENT'].copy()

    importance_df = compute_shap_values(baseline_model, X_val, y_val, feature_cols)

    # Display top features
    print("\nTop 20 features by SHAP_Combined:")
    print(importance_df[['Rank', 'Feature', 'SHAP_Combined', 'SHAP_Ratio']].head(20).to_string(index=False))

    # Save checkpoint
    save_checkpoint("step1_5_shap_phase1", {
        'importance_df': importance_df
    })

print(f"\n✓ SHAP values computed for {len(importance_df)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.6: Select Cluster Representatives

# COMMAND ----------

# Check if we can skip this step
if checkpoint_exists("step1_6_cluster_representatives") and not START_FRESH:
    print(">>> Loading from checkpoint: step1_6_cluster_representatives")
    rep_checkpoint = load_checkpoint("step1_6_cluster_representatives")
    selected_features = rep_checkpoint['selected_features']
    selection_df = rep_checkpoint['selection_df']
else:
    print("="*70)
    print("STEP 1.6: SELECT CLUSTER REPRESENTATIVES")
    print("="*70)

    # Identify which clinical must-keep features exist in the data
    valid_must_keep = [f for f in CLINICAL_MUST_KEEP_FEATURES if f in feature_cols]
    missing_must_keep = [f for f in CLINICAL_MUST_KEEP_FEATURES if f not in feature_cols]

    if missing_must_keep:
        print(f"⚠ Clinical must-keep features not found in data: {missing_must_keep}")
    print(f"Clinical must-keep features to preserve: {len(valid_must_keep)}")

    # Merge cluster assignments with SHAP importance
    cluster_importance = cluster_df.merge(importance_df, on='Feature')

    selected_features = []
    selection_records = []

    # For each cluster, select representative(s)
    for cluster_id in sorted(cluster_importance['Cluster'].unique()):
        cluster_features = cluster_importance[cluster_importance['Cluster'] == cluster_id].copy()
        cluster_size = len(cluster_features)

        # Sort by importance_ratio (descending) - prefer features predictive of positives
        cluster_features = cluster_features.sort_values('SHAP_Ratio', ascending=False)

        # Determine how many to keep
        if cluster_size >= 8:
            # Large cluster: keep top 2
            n_keep = 2
        else:
            # Small/medium cluster: keep top 1
            n_keep = 1

        # Select top feature(s)
        kept = cluster_features.head(n_keep)

        for _, row in kept.iterrows():
            selected_features.append(row['Feature'])
            selection_records.append({
                'Feature': row['Feature'],
                'Cluster': cluster_id,
                'Cluster_Size': cluster_size,
                'SHAP_Combined': row['SHAP_Combined'],
                'SHAP_Ratio': row['SHAP_Ratio'],
                'Selection_Reason': f"Top by SHAP_Ratio in cluster of {cluster_size}"
            })

    # Add clinical must-keep features that weren't already selected
    for feat in valid_must_keep:
        if feat not in selected_features:
            # Get the feature's importance info
            feat_info = importance_df[importance_df['Feature'] == feat]
            feat_cluster = cluster_df[cluster_df['Feature'] == feat]['Cluster'].values[0]

            selected_features.append(feat)
            selection_records.append({
                'Feature': feat,
                'Cluster': feat_cluster,
                'Cluster_Size': len(cluster_df[cluster_df['Cluster'] == feat_cluster]),
                'SHAP_Combined': feat_info['SHAP_Combined'].values[0] if len(feat_info) > 0 else 0,
                'SHAP_Ratio': feat_info['SHAP_Ratio'].values[0] if len(feat_info) > 0 else 0,
                'Selection_Reason': "Clinical must-keep feature"
            })
            print(f"  + Added clinical must-keep: {feat}")

    selection_df = pd.DataFrame(selection_records)

    print(f"\nCluster representative selection:")
    print(f"  Original features: {len(feature_cols)}")
    print(f"  Clusters: {len(cluster_importance['Cluster'].unique())}")
    print(f"  Selected features: {len(selected_features)}")
    print(f"  Reduction: {len(feature_cols) - len(selected_features)} features removed ({(len(feature_cols) - len(selected_features))/len(feature_cols)*100:.1f}%)")

    # Show selection summary by cluster size
    print("\nSelection by cluster size:")
    for size_cat in ['1', '2-3', '4-7', '8+']:
        if size_cat == '1':
            mask = selection_df['Cluster_Size'] == 1
        elif size_cat == '2-3':
            mask = (selection_df['Cluster_Size'] >= 2) & (selection_df['Cluster_Size'] <= 3)
        elif size_cat == '4-7':
            mask = (selection_df['Cluster_Size'] >= 4) & (selection_df['Cluster_Size'] <= 7)
        else:
            mask = selection_df['Cluster_Size'] >= 8

        n_selected = mask.sum()
        if n_selected > 0:
            print(f"  Cluster size {size_cat}: {n_selected} features selected")

    # Save checkpoint
    save_checkpoint("step1_6_cluster_representatives", {
        'selected_features': selected_features,
        'selection_df': selection_df
    })

print(f"\n✓ Selected {len(selected_features)} cluster representatives")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.7: Phase 1 Validation Gate

# COMMAND ----------

# Check if we can skip this step
if checkpoint_exists("step1_7_phase1_complete") and not START_FRESH:
    print(">>> Loading from checkpoint: step1_7_phase1_complete")
    phase1_checkpoint = load_checkpoint("step1_7_phase1_complete")
    phase1_features = phase1_checkpoint['phase1_features']
    phase1_metrics = phase1_checkpoint['phase1_metrics']
    phase1_passed = phase1_checkpoint['phase1_passed']
else:
    print("="*70)
    print("STEP 1.7: PHASE 1 VALIDATION GATE")
    print("="*70)

    # Prepare data
    train_mask = df_pandas['SPLIT'] == 'train'
    val_mask = df_pandas['SPLIT'] == 'val'
    test_mask = df_pandas['SPLIT'] == 'test'

    X_train = df_pandas.loc[train_mask].copy()
    y_train = df_pandas.loc[train_mask, 'FUTURE_CRC_EVENT'].copy()

    X_val = df_pandas.loc[val_mask].copy()
    y_val = df_pandas.loc[val_mask, 'FUTURE_CRC_EVENT'].copy()

    X_test = df_pandas.loc[test_mask].copy()
    y_test = df_pandas.loc[test_mask, 'FUTURE_CRC_EVENT'].copy()

    print(f"Training model with {len(selected_features)} selected features...")

    # Train model with selected features
    phase1_model = train_xgboost_model(
        X_train, y_train, X_val, y_val,
        selected_features, scale_pos_weight
    )

    # Evaluate on train/val only (test is held out until final evaluation)
    print("\nPhase 1 reduced model performance:")
    phase1_metrics = {
        'train': evaluate_model(phase1_model, X_train, y_train, selected_features, "Train"),
        'val': evaluate_model(phase1_model, X_val, y_val, selected_features, "Val")
    }

    phase1_metrics['train_val_gap'] = phase1_metrics['train']['auprc'] - phase1_metrics['val']['auprc']

    # Calculate drops from baseline
    val_drop = (baseline_metrics['val']['auprc'] - phase1_metrics['val']['auprc']) / baseline_metrics['val']['auprc']
    gap_change = phase1_metrics['train_val_gap'] - baseline_metrics['train_val_gap']

    print(f"\nValidation Gate Check:")
    print(f"  Baseline Val AUPRC: {baseline_metrics['val']['auprc']:.4f}")
    print(f"  Phase 1 Val AUPRC:  {phase1_metrics['val']['auprc']:.4f}")
    print(f"  Val AUPRC Drop:     {val_drop*100:.2f}% (threshold: {PHASE1_MAX_VAL_DROP*100}%)")
    print(f"  Train-Val Gap Change: {gap_change:.4f}")

    # Check if passed
    phase1_passed = val_drop <= PHASE1_MAX_VAL_DROP

    if phase1_passed:
        print(f"\n✓ PHASE 1 VALIDATION GATE: PASSED")
        phase1_features = selected_features
    else:
        print(f"\n⚠ PHASE 1 VALIDATION GATE: FAILED")
        print("  Consider: keeping top 2 per cluster, or raising threshold")
        # For now, proceed with selected features but flag the issue
        phase1_features = selected_features

    # Save checkpoint
    save_checkpoint("step1_7_phase1_complete", {
        'phase1_features': phase1_features,
        'phase1_metrics': phase1_metrics,
        'phase1_passed': phase1_passed,
        'val_drop': val_drop,
        'gap_change': gap_change
    })

    # Update tracking CSV
    update_tracking_csv({
        'phase': 'phase1',
        'iteration': 'cluster_reduction',
        'n_features': len(phase1_features),
        'n_removed': len(feature_cols) - len(phase1_features),
        'train_auprc': phase1_metrics['train']['auprc'],
        'val_auprc': phase1_metrics['val']['auprc'],
        'train_val_gap': phase1_metrics['train_val_gap'],
        'val_drop_from_baseline': val_drop,
        'timestamp': datetime.now().isoformat()
    })

print(f"\n✓ Phase 1 complete: {len(phase1_features)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # PHASE 2: Iterative SHAP Winnowing
# MAGIC ---
# MAGIC
# MAGIC Iteratively remove low-importance features while monitoring validation performance.
# MAGIC
# MAGIC **Key controls:**
# MAGIC - **Validation gates**: Stop if val AUPRC drops >5% or train-val gap increases >0.02
# MAGIC - **Clinical must-keep**: Never remove clinically important features (weight loss, GI bleeding, etc.)
# MAGIC - **Protected features**: Never remove features in top 50% by SHAP ratio
# MAGIC - **Test holdout**: All decisions based on validation metrics only; test evaluated at final model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phase 2 Iteration Loop

# COMMAND ----------

print("="*70)
print("PHASE 2: ITERATIVE SHAP WINNOWING")
print("="*70)

# Initialize Phase 2
current_features = phase1_features.copy()
iteration = 0
stop_reason = None

# Check for existing Phase 2 checkpoints to resume
phase2_checkpoints = [cp for cp in list_checkpoints() if cp.startswith("step2_iter") and cp.endswith("_complete")]
if phase2_checkpoints and not START_FRESH:
    # Find the latest complete iteration
    iter_nums = []
    for cp in phase2_checkpoints:
        try:
            iter_num = int(cp.split("_")[1].replace("iter", ""))
            iter_nums.append(iter_num)
        except:
            pass

    if iter_nums:
        last_iter = max(iter_nums)
        print(f">>> Found Phase 2 checkpoint at iteration {last_iter}. Resuming...")

        last_checkpoint = load_checkpoint(f"step2_iter{last_iter}_complete")
        current_features = last_checkpoint['current_features']
        iteration = last_iter

        # Check if we should stop
        if last_checkpoint.get('stop_triggered', False):
            stop_reason = last_checkpoint.get('stop_reason', 'Unknown')
            print(f">>> Previous iteration triggered stop: {stop_reason}")

print(f"\nStarting Phase 2 from iteration {iteration}")
print(f"Current features: {len(current_features)}")
print(f"Max removals per iteration: {MAX_REMOVALS_PER_ITERATION}")
print(f"Min features threshold: {MIN_FEATURES_THRESHOLD}")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run Phase 2 Iterations

# COMMAND ----------

# Prepare data splits (needed for iterations)
train_mask = df_pandas['SPLIT'] == 'train'
val_mask = df_pandas['SPLIT'] == 'val'
test_mask = df_pandas['SPLIT'] == 'test'

X_train = df_pandas.loc[train_mask].copy()
y_train = df_pandas.loc[train_mask, 'FUTURE_CRC_EVENT'].copy()

X_val = df_pandas.loc[val_mask].copy()
y_val = df_pandas.loc[val_mask, 'FUTURE_CRC_EVENT'].copy()

X_test = df_pandas.loc[test_mask].copy()
y_test = df_pandas.loc[test_mask, 'FUTURE_CRC_EVENT'].copy()

# Main iteration loop
while stop_reason is None:
    iteration += 1

    print(f"\n{'='*70}")
    print(f"PHASE 2 - ITERATION {iteration}")
    print(f"{'='*70}")
    print(f"Current features: {len(current_features)}")

    # =========================================================================
    # Step 2.1: Train Model
    # =========================================================================
    checkpoint_name = f"step2_iter{iteration}_model"

    if checkpoint_exists(checkpoint_name) and not START_FRESH:
        print(f">>> Loading from checkpoint: {checkpoint_name}")
        model_checkpoint = load_checkpoint(checkpoint_name)
        iter_model = model_checkpoint['model']
        iter_metrics = model_checkpoint['metrics']
    else:
        print(f"\nStep 2.{iteration}.1: Training model...")

        iter_model = train_xgboost_model(
            X_train, y_train, X_val, y_val,
            current_features, scale_pos_weight
        )

        iter_metrics = {
            'train': evaluate_model(iter_model, X_train, y_train, current_features, "Train"),
            'val': evaluate_model(iter_model, X_val, y_val, current_features, "Val")
        }
        iter_metrics['train_val_gap'] = iter_metrics['train']['auprc'] - iter_metrics['val']['auprc']

        save_checkpoint(checkpoint_name, {
            'model': iter_model,
            'metrics': iter_metrics
        })

    # =========================================================================
    # Step 2.2: Compute SHAP
    # =========================================================================
    checkpoint_name = f"step2_iter{iteration}_shap"

    if checkpoint_exists(checkpoint_name) and not START_FRESH:
        print(f">>> Loading from checkpoint: {checkpoint_name}")
        shap_checkpoint = load_checkpoint(checkpoint_name)
        iter_importance_df = shap_checkpoint['importance_df']
    else:
        print(f"\nStep 2.{iteration}.2: Computing SHAP values...")

        iter_importance_df = compute_shap_values(iter_model, X_val, y_val, current_features)

        save_checkpoint(checkpoint_name, {
            'importance_df': iter_importance_df
        })

    # =========================================================================
    # Step 2.3: Identify Removal Candidates
    # =========================================================================
    print(f"\nStep 2.{iteration}.3: Identifying removal candidates...")

    # Calculate thresholds
    zero_threshold = 0.0002
    ratio_threshold = 0.2
    bottom_pct = 0.15

    # Identify features meeting each criterion
    zero_importance = set(iter_importance_df[iter_importance_df['SHAP_Combined'] < zero_threshold]['Feature'])
    neg_biased = set(iter_importance_df[iter_importance_df['SHAP_Ratio'] < ratio_threshold]['Feature'])

    importance_cutoff = iter_importance_df['SHAP_Combined'].quantile(bottom_pct)
    bottom_features = set(iter_importance_df[iter_importance_df['SHAP_Combined'] < importance_cutoff]['Feature'])

    # Features meeting 2+ criteria
    candidates = (
        (zero_importance & neg_biased) |
        (zero_importance & bottom_features) |
        (neg_biased & bottom_features)
    )

    # Never remove features in top 50% by importance_ratio
    median_ratio = iter_importance_df['SHAP_Ratio'].median()
    protected = set(iter_importance_df[iter_importance_df['SHAP_Ratio'] >= median_ratio]['Feature'])
    candidates = candidates - protected

    # Never remove clinical must-keep features
    clinical_protected = set(f for f in CLINICAL_MUST_KEEP_FEATURES if f in current_features)
    candidates = candidates - clinical_protected

    # Sort candidates by importance (remove lowest first)
    candidate_df = iter_importance_df[iter_importance_df['Feature'].isin(candidates)].sort_values('SHAP_Combined')

    # Cap at MAX_REMOVALS_PER_ITERATION
    features_to_remove = candidate_df.head(MAX_REMOVALS_PER_ITERATION)['Feature'].tolist()

    print(f"  Near-zero importance: {len(zero_importance)}")
    print(f"  Negative-biased ratio: {len(neg_biased)}")
    print(f"  Bottom {bottom_pct*100:.0f}%: {len(bottom_features)}")
    print(f"  Meeting 2+ criteria (after protections): {len(candidates)}")
    print(f"  Protected (top 50% ratio): {len(protected)}")
    print(f"  Protected (clinical must-keep): {len(clinical_protected)}")
    print(f"  Final candidates: {len(features_to_remove)}")

    # =========================================================================
    # Step 2.4: Validation Gate
    # =========================================================================
    print(f"\nStep 2.{iteration}.4: Validation gate check...")

    # Calculate metrics relative to baseline
    val_drop = (baseline_metrics['val']['auprc'] - iter_metrics['val']['auprc']) / baseline_metrics['val']['auprc']
    gap_increase = iter_metrics['train_val_gap'] - baseline_metrics['train_val_gap']

    print(f"  Val AUPRC: {iter_metrics['val']['auprc']:.4f} (drop from baseline: {val_drop*100:.2f}%)")
    print(f"  Train-Val Gap: {iter_metrics['train_val_gap']:.4f} (change: {gap_increase:+.4f})")
    print(f"  Features remaining: {len(current_features)}")

    # Check stop conditions
    if val_drop > MAX_VAL_AUPRC_DROP:
        stop_reason = f"Val AUPRC drop {val_drop*100:.2f}% > {MAX_VAL_AUPRC_DROP*100}% threshold"
    elif gap_increase > MAX_GAP_INCREASE:
        stop_reason = f"Train-Val gap increase {gap_increase:.4f} > {MAX_GAP_INCREASE} threshold"
    elif len(current_features) - len(features_to_remove) < MIN_FEATURES_THRESHOLD:
        stop_reason = f"Would go below {MIN_FEATURES_THRESHOLD} features"
    elif len(features_to_remove) == 0:
        stop_reason = "No features meet removal criteria"

    # =========================================================================
    # Step 2.5: Log & Checkpoint
    # =========================================================================

    # Update tracking CSV
    update_tracking_csv({
        'phase': 'phase2',
        'iteration': iteration,
        'n_features': len(current_features),
        'n_removed': len(features_to_remove) if stop_reason is None else 0,
        'train_auprc': iter_metrics['train']['auprc'],
        'val_auprc': iter_metrics['val']['auprc'],
        'train_val_gap': iter_metrics['train_val_gap'],
        'val_drop_from_baseline': val_drop,
        'gap_increase': gap_increase,
        'stop_triggered': stop_reason is not None,
        'stop_reason': stop_reason if stop_reason else '',
        'timestamp': datetime.now().isoformat()
    })

    if stop_reason:
        print(f"\n>>> STOP CONDITION TRIGGERED: {stop_reason}")
        print(f">>> Keeping features from iteration {iteration}")

        # Save final checkpoint
        save_checkpoint(f"step2_iter{iteration}_complete", {
            'current_features': current_features,
            'metrics': iter_metrics,
            'features_removed': [],
            'stop_triggered': True,
            'stop_reason': stop_reason
        })
        break
    else:
        # Remove features and continue
        print(f"\n>>> Removing {len(features_to_remove)} features:")
        for feat in features_to_remove[:10]:
            shap_val = iter_importance_df[iter_importance_df['Feature'] == feat]['SHAP_Combined'].values[0]
            ratio_val = iter_importance_df[iter_importance_df['Feature'] == feat]['SHAP_Ratio'].values[0]
            print(f"    - {feat} (SHAP={shap_val:.6f}, Ratio={ratio_val:.3f})")
        if len(features_to_remove) > 10:
            print(f"    ... and {len(features_to_remove) - 10} more")

        # Update feature list
        previous_features = current_features.copy()
        current_features = [f for f in current_features if f not in features_to_remove]

        # Save checkpoint
        save_checkpoint(f"step2_iter{iteration}_complete", {
            'current_features': current_features,
            'previous_features': previous_features,
            'metrics': iter_metrics,
            'features_removed': features_to_remove,
            'stop_triggered': False,
            'stop_reason': None
        })

        print(f"\n✓ Iteration {iteration} complete. Features: {len(previous_features)} → {len(current_features)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Final Results
# MAGIC ---
# MAGIC
# MAGIC Train final model with selected features and evaluate performance.
# MAGIC **This is the first and only time the test set is evaluated** - all feature selection
# MAGIC decisions were made using validation metrics only.

# COMMAND ----------

print("="*70)
print("FINAL RESULTS")
print("="*70)

final_features = current_features

print(f"\nFeature reduction summary:")
print(f"  Initial features: {len(feature_cols)}")
print(f"  After Phase 1 (clustering): {len(phase1_features)}")
print(f"  After Phase 2 (winnowing): {len(final_features)}")
print(f"  Total reduction: {len(feature_cols) - len(final_features)} features ({(len(feature_cols) - len(final_features))/len(feature_cols)*100:.1f}%)")

if stop_reason:
    print(f"\nStop reason: {stop_reason}")

# Train final model
print("\n" + "-"*70)
print("FINAL MODEL PERFORMANCE")
print("-"*70)

final_model = train_xgboost_model(
    X_train, y_train, X_val, y_val,
    final_features, scale_pos_weight
)

print("\nFinal model metrics (train/val):")
final_metrics = {
    'train': evaluate_model(final_model, X_train, y_train, final_features, "Train"),
    'val': evaluate_model(final_model, X_val, y_val, final_features, "Val")
}

# TEST SET EVALUATION - First and only evaluation on held-out test set
print("\n" + "="*70)
print("TEST SET EVALUATION (First and Only)")
print("="*70)
print("This is the first time the test set has been evaluated.")
print("All feature selection decisions were made using validation data only.")
final_metrics['test'] = evaluate_model(final_model, X_test, y_test, final_features, "Test")

print("\n" + "-"*70)
print("COMPARISON: BASELINE vs FINAL (Validation)")
print("-"*70)
print(f"{'Metric':<20} {'Baseline':>12} {'Final':>12} {'Change':>12}")
print("-"*56)
print(f"{'Features':<20} {len(feature_cols):>12} {len(final_features):>12} {len(final_features) - len(feature_cols):>+12}")
print(f"{'Val AUPRC':<20} {baseline_metrics['val']['auprc']:>12.4f} {final_metrics['val']['auprc']:>12.4f} {final_metrics['val']['auprc'] - baseline_metrics['val']['auprc']:>+12.4f}")
print(f"{'Train-Val Gap':<20} {baseline_metrics['train_val_gap']:>12.4f} {final_metrics['train']['auprc'] - final_metrics['val']['auprc']:>12.4f}")
print(f"\n{'Test AUPRC (held out)':<20} {final_metrics['test']['auprc']:>12.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Final Outputs
# MAGIC
# MAGIC Outputs saved to `feature_selection_outputs/`:
# MAGIC - `final_features.txt` - Feature names, one per line
# MAGIC - `final_features.py` - Python list for easy import
# MAGIC - `final_model.pkl` - Trained XGBoost model
# MAGIC - `feature_selection_summary.json` - Summary metrics and feature list
# MAGIC - `iteration_tracking.csv` - Metrics at each iteration
# MAGIC - `cv_stability_report.json` - Cross-validation stability analysis (added after CV section runs)

# COMMAND ----------

# Save final feature list
final_features_path = os.path.join(OUTPUT_DIR, "final_features.txt")
with open(final_features_path, 'w') as f:
    f.write(f"# Final Feature List\n")
    f.write(f"# Generated: {datetime.now()}\n")
    f.write(f"# Initial features: {len(feature_cols)}\n")
    f.write(f"# Final features: {len(final_features)}\n")
    f.write(f"# Stop reason: {stop_reason}\n\n")
    for feat in sorted(final_features):
        f.write(f"{feat}\n")

print(f"✓ Saved: {final_features_path}")

# Save final feature list as Python list
final_features_py_path = os.path.join(OUTPUT_DIR, "final_features.py")
with open(final_features_py_path, 'w') as f:
    f.write(f"# Final Feature List for CRC Prediction Model\n")
    f.write(f"# Generated: {datetime.now()}\n\n")
    f.write(f"FINAL_FEATURES = [\n")
    for feat in sorted(final_features):
        f.write(f"    '{feat}',\n")
    f.write(f"]\n")

print(f"✓ Saved: {final_features_py_path}")

# Save model
final_model_path = os.path.join(OUTPUT_DIR, "final_model.pkl")
with open(final_model_path, 'wb') as f:
    pickle.dump(final_model, f)

print(f"✓ Saved: {final_model_path}")

# Save summary JSON
summary = {
    'timestamp': datetime.now().isoformat(),
    'initial_features': len(feature_cols),
    'phase1_features': len(phase1_features),
    'final_features': len(final_features),
    'stop_reason': stop_reason,
    'baseline_val_auprc': baseline_metrics['val']['auprc'],
    'final_val_auprc': final_metrics['val']['auprc'],
    'final_test_auprc': final_metrics['test']['auprc'],  # Test evaluated only at final model
    'feature_list': final_features
}

summary_path = os.path.join(OUTPUT_DIR, "feature_selection_summary.json")
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"✓ Saved: {summary_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Visualize Final Results

# COMMAND ----------

# Load tracking data
tracking_path = os.path.join(OUTPUT_DIR, "iteration_tracking.csv")
if os.path.exists(tracking_path):
    tracking_df = pd.read_csv(tracking_path)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Feature count over iterations
    axes[0, 0].plot(range(len(tracking_df)), tracking_df['n_features'], 'bo-', linewidth=2, markersize=8)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Number of Features')
    axes[0, 0].set_title('Feature Count Over Iterations')
    axes[0, 0].grid(alpha=0.3)

    # Validation AUPRC (test held out until final evaluation)
    axes[0, 1].plot(range(len(tracking_df)), tracking_df['val_auprc'], 'go-', linewidth=2, markersize=8, label='Val')
    axes[0, 1].plot(range(len(tracking_df)), tracking_df['train_auprc'], 'bo-', linewidth=2, markersize=8, label='Train', alpha=0.5)
    axes[0, 1].axhline(y=baseline_metrics['val']['auprc'] * (1 - MAX_VAL_AUPRC_DROP), color='g', linestyle='--', alpha=0.5, label='5% drop threshold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('AUPRC')
    axes[0, 1].set_title('AUPRC Over Iterations (Test Held Out)')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)

    # Train-Val Gap
    axes[1, 0].plot(range(len(tracking_df)), tracking_df['train_val_gap'], 'mo-', linewidth=2, markersize=8)
    axes[1, 0].axhline(y=baseline_metrics['train_val_gap'] + MAX_GAP_INCREASE, color='r', linestyle='--', alpha=0.5, label='Gap threshold')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Train-Val Gap')
    axes[1, 0].set_title('Overfitting Indicator (Train-Val Gap)')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    # Val drop from baseline
    axes[1, 1].plot(range(len(tracking_df)), tracking_df['val_drop_from_baseline'] * 100, 'co-', linewidth=2, markersize=8)
    axes[1, 1].axhline(y=MAX_VAL_AUPRC_DROP * 100, color='r', linestyle='--', alpha=0.5, label='5% threshold')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Val AUPRC Drop (%)')
    axes[1, 1].set_title('Validation Performance Drop from Baseline')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'iteration_progress.png'), dpi=150, bbox_inches='tight')
    plt.show()

    print(f"✓ Saved: {OUTPUT_DIR}/iteration_progress.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Final Summary

# COMMAND ----------

print("="*70)
print("FEATURE SELECTION COMPLETE")
print("="*70)
print(f"""
Summary:
  - Initial features: {len(feature_cols)}
  - After Phase 1 (clustering): {len(phase1_features)}
  - Final features: {len(final_features)}
  - Reduction: {(len(feature_cols) - len(final_features))/len(feature_cols)*100:.1f}%

Performance:
  - Baseline Val AUPRC: {baseline_metrics['val']['auprc']:.4f}
  - Final Val AUPRC: {final_metrics['val']['auprc']:.4f}
  - Final Test AUPRC: {final_metrics['test']['auprc']:.4f}

Stop Reason: {stop_reason}

Outputs saved to: {OUTPUT_DIR}/
  - final_features.txt
  - final_features.py
  - final_model.pkl
  - feature_selection_summary.json
  - iteration_tracking.csv
  - iteration_progress.png
  - threshold_analysis.png

Checkpoints saved to: {CHECKPOINT_DIR}/
  (Can be used to resume if notebook is interrupted)
""")
print("="*70)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC # Cross-Validation Stability Analysis
# MAGIC ---
# MAGIC
# MAGIC This section runs a simplified feature selection on the remaining CV folds to assess
# MAGIC stability. Features that are consistently selected across folds are more robust.

# COMMAND ----------

print("="*70)
print("CV STABILITY ANALYSIS")
print("="*70)
print(f"Running feature selection on {N_CV_FOLDS - 1} additional folds...")
print(f"Stability threshold: features must appear in ≥{CV_FEATURE_THRESHOLD*100:.0f}% of folds")

# Track features selected in each fold
# Fold 0 is the main pipeline result
fold_selected_features = {0: set(final_features)}

# Test set indices (fixed across folds)
test_mask = df_pandas['SPLIT'] == 'test'
X_test = df_pandas.loc[test_mask].copy()
y_test = df_pandas.loc[test_mask, 'FUTURE_CRC_EVENT'].copy()

# Run on remaining folds (1 and 2)
for fold_idx in range(1, N_CV_FOLDS):
    print(f"\n{'='*50}")
    print(f"FOLD {fold_idx + 1} / {N_CV_FOLDS}")
    print(f"{'='*50}")

    # Get train/val split for this fold
    fold_train_patients = cv_fold_assignments[fold_idx]['train_patients']
    fold_val_patients = cv_fold_assignments[fold_idx]['val_patients']

    # Create masks
    trainval_mask = df_pandas['SPLIT'].isin(['train', 'val'])
    fold_train_mask = trainval_mask & df_pandas['PAT_ID'].isin(fold_train_patients)
    fold_val_mask = trainval_mask & df_pandas['PAT_ID'].isin(fold_val_patients)

    X_fold_train = df_pandas.loc[fold_train_mask].copy()
    y_fold_train = df_pandas.loc[fold_train_mask, 'FUTURE_CRC_EVENT'].copy()
    X_fold_val = df_pandas.loc[fold_val_mask].copy()
    y_fold_val = df_pandas.loc[fold_val_mask, 'FUTURE_CRC_EVENT'].copy()

    print(f"  Train: {len(y_fold_train):,} obs, {y_fold_train.sum():,} events")
    print(f"  Val: {len(y_fold_val):,} obs, {y_fold_val.sum():,} events")

    # Calculate scale_pos_weight for this fold
    fold_scale_pos_weight = (y_fold_train == 0).sum() / (y_fold_train == 1).sum()

    # Train baseline model on this fold (using clustering features from main pipeline)
    print("\n  Training model on cluster representatives...")
    fold_model = train_xgboost_model(
        X_fold_train, y_fold_train, X_fold_val, y_fold_val,
        phase1_features, fold_scale_pos_weight
    )

    # Compute SHAP on this fold's validation set
    print("  Computing SHAP values...")
    fold_importance_df = compute_shap_values(fold_model, X_fold_val, y_fold_val, phase1_features)

    # Identify removal candidates (same logic as main pipeline)
    zero_threshold = 0.0002
    ratio_threshold = 0.2
    bottom_pct = 0.15

    zero_importance = set(fold_importance_df[fold_importance_df['SHAP_Combined'] < zero_threshold]['Feature'])
    neg_biased = set(fold_importance_df[fold_importance_df['SHAP_Ratio'] < ratio_threshold]['Feature'])
    importance_cutoff = fold_importance_df['SHAP_Combined'].quantile(bottom_pct)
    bottom_features = set(fold_importance_df[fold_importance_df['SHAP_Combined'] < importance_cutoff]['Feature'])

    # Features meeting 2+ criteria
    candidates = (
        (zero_importance & neg_biased) |
        (zero_importance & bottom_features) |
        (neg_biased & bottom_features)
    )

    # Never remove protected features
    median_ratio = fold_importance_df['SHAP_Ratio'].median()
    protected = set(fold_importance_df[fold_importance_df['SHAP_Ratio'] >= median_ratio]['Feature'])
    clinical_protected = set(f for f in CLINICAL_MUST_KEEP_FEATURES if f in phase1_features)
    candidates = candidates - protected - clinical_protected

    # Remove candidates from phase1_features to get this fold's selection
    fold_features = [f for f in phase1_features if f not in candidates]

    print(f"  Features selected: {len(fold_features)}")
    fold_selected_features[fold_idx] = set(fold_features)

    # Quick validation
    fold_val_metrics = evaluate_model(fold_model, X_fold_val, y_fold_val, phase1_features, f"  Fold {fold_idx + 1} Val")

# Compute stability statistics
print("\n" + "="*70)
print("STABILITY ANALYSIS RESULTS")
print("="*70)

# Count how many folds each feature appears in
feature_fold_counts = {}
all_features = set()
for fold_idx, features in fold_selected_features.items():
    all_features.update(features)

for feat in all_features:
    count = sum(1 for fold_features in fold_selected_features.values() if feat in fold_features)
    feature_fold_counts[feat] = count

# Categorize by stability
stable_features = [f for f, count in feature_fold_counts.items() if count >= N_CV_FOLDS * CV_FEATURE_THRESHOLD]
unstable_features = [f for f, count in feature_fold_counts.items() if count < N_CV_FOLDS * CV_FEATURE_THRESHOLD]

print(f"\nFeature stability summary:")
print(f"  Total unique features selected across folds: {len(all_features)}")
print(f"  Stable features (≥{int(N_CV_FOLDS * CV_FEATURE_THRESHOLD)}/{N_CV_FOLDS} folds): {len(stable_features)}")
print(f"  Unstable features (<{int(N_CV_FOLDS * CV_FEATURE_THRESHOLD)}/{N_CV_FOLDS} folds): {len(unstable_features)}")

# Show unstable features
if unstable_features:
    print(f"\n  Unstable features (may be overfitting artifacts):")
    for feat in sorted(unstable_features)[:20]:
        count = feature_fold_counts[feat]
        in_final = "✓" if feat in final_features else " "
        print(f"    [{in_final}] {feat}: {count}/{N_CV_FOLDS} folds")
    if len(unstable_features) > 20:
        print(f"    ... and {len(unstable_features) - 20} more")

# Check overlap with final features from main pipeline
final_stable = [f for f in final_features if f in stable_features]
final_unstable = [f for f in final_features if f in unstable_features]

print(f"\n  Final features from main pipeline: {len(final_features)}")
print(f"    - Stable: {len(final_stable)} ({len(final_stable)/len(final_features)*100:.1f}%)")
print(f"    - Unstable: {len(final_unstable)} ({len(final_unstable)/len(final_features)*100:.1f}%)")

# Save stability report
stability_report = {
    'n_folds': N_CV_FOLDS,
    'threshold': CV_FEATURE_THRESHOLD,
    'stable_features': sorted(stable_features),
    'unstable_features': sorted(unstable_features),
    'feature_fold_counts': feature_fold_counts,
    'final_stable': sorted(final_stable),
    'final_unstable': sorted(final_unstable)
}

stability_path = os.path.join(OUTPUT_DIR, "cv_stability_report.json")
with open(stability_path, 'w') as f:
    json.dump(stability_report, f, indent=2)
print(f"\n✓ Saved stability report: {stability_path}")

# Optionally update final features to only include stable ones
print(f"\n{'='*70}")
print("RECOMMENDATION")
print("="*70)
if len(final_unstable) > 0:
    print(f"Consider reviewing the {len(final_unstable)} unstable features in the final set.")
    print("These may be overfitting to the specific train/val split.")
    print(f"For a more robust model, use only the {len(final_stable)} stable features.")
else:
    print("All final features are stable across CV folds. Feature selection is robust.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Utility: Clear Checkpoints (Run Manually if Needed)

# COMMAND ----------

# # Uncomment and run this cell to clear all checkpoints and start fresh
# # clear_checkpoints()
# # print("Checkpoints cleared. Re-run notebook to start fresh.")
