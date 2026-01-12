# Databricks notebook source
# MAGIC %md
# MAGIC # Book 9: Feature Selection Pipeline
# MAGIC
# MAGIC ## Hybrid Two-Phase Approach
# MAGIC
# MAGIC | Phase | Method | Features | Purpose |
# MAGIC |-------|--------|----------|---------|
# MAGIC | **Phase 1** | Cluster-Based Reduction | 171 → ~70-80 | Remove redundant/correlated features |
# MAGIC | **Phase 2** | Iterative SHAP Winnowing | ~70-80 → Final | Fine-tune with 20-25 removals per iteration |
# MAGIC
# MAGIC ## Key Features
# MAGIC - **Dynamic clustering threshold** via silhouette score (not fixed 0.7)
# MAGIC - **2:1 SHAP weighting** for positive cases (model handles imbalance via scale_pos_weight)
# MAGIC - **Granular checkpoints** - stop anytime and resume without starting over
# MAGIC - **Automatic validation gates** - stops when performance degrades
# MAGIC
# MAGIC ## Checkpoint System
# MAGIC Checkpoints saved after each step. Kill notebook anytime, re-run to resume.

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

print("="*70)
print("CONFIGURATION")
print("="*70)
print(f"AUTO_RESUME: {AUTO_RESUME}")
print(f"MAX_REMOVALS_PER_ITERATION: {MAX_REMOVALS_PER_ITERATION}")
print(f"MIN_FEATURES_THRESHOLD: {MIN_FEATURES_THRESHOLD}")
print(f"MAX_VAL_AUPRC_DROP: {MAX_VAL_AUPRC_DROP}")
print(f"MAX_GAP_INCREASE: {MAX_GAP_INCREASE}")
print(f"RANDOM_SEED: {RANDOM_SEED}")
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

# COMMAND ----------

def ensure_directories():
    """Create checkpoint and output directories if they don't exist."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"✓ Directories verified: {CHECKPOINT_DIR}/, {OUTPUT_DIR}/")

def get_checkpoint_path(name):
    """Get full path for a checkpoint file."""
    return os.path.join(CHECKPOINT_DIR, f"{name}.pkl")

def save_checkpoint(name, data):
    """Save a checkpoint with the given name."""
    path = get_checkpoint_path(name)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ CHECKPOINT SAVED: {name}")

def load_checkpoint(name):
    """Load a checkpoint by name. Returns None if not found."""
    path = get_checkpoint_path(name)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        print(f"✓ CHECKPOINT LOADED: {name}")
        return data
    return None

def checkpoint_exists(name):
    """Check if a checkpoint exists."""
    return os.path.exists(get_checkpoint_path(name))

def list_checkpoints():
    """List all existing checkpoints."""
    if not os.path.exists(CHECKPOINT_DIR):
        return []
    checkpoints = [f.replace('.pkl', '') for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pkl')]
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
    """Clear all checkpoints to start fresh."""
    if os.path.exists(CHECKPOINT_DIR):
        for f in os.listdir(CHECKPOINT_DIR):
            if f.endswith('.pkl'):
                os.remove(os.path.join(CHECKPOINT_DIR, f))
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

    # Evaluate on all splits
    print("\nBaseline performance:")
    baseline_metrics = {
        'train': evaluate_model(baseline_model, X_train, y_train, feature_cols, "Train"),
        'val': evaluate_model(baseline_model, X_val, y_val, feature_cols, "Val"),
        'test': evaluate_model(baseline_model, X_test, y_test, feature_cols, "Test")
    }

    # Calculate train-val gap
    baseline_metrics['train_val_gap'] = baseline_metrics['train']['auprc'] - baseline_metrics['val']['auprc']
    print(f"  Train-Val Gap: {baseline_metrics['train_val_gap']:.4f}")

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
        'test_auprc': baseline_metrics['test']['auprc'],
        'train_val_gap': baseline_metrics['train_val_gap'],
        'val_drop_from_baseline': 0.0,
        'timestamp': datetime.now().isoformat()
    })

print(f"\n✓ Baseline model ready: Val AUPRC = {baseline_metrics['val']['auprc']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1.5: Compute SHAP with 2:1 Positive Weighting

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

    # Evaluate
    print("\nPhase 1 reduced model performance:")
    phase1_metrics = {
        'train': evaluate_model(phase1_model, X_train, y_train, selected_features, "Train"),
        'val': evaluate_model(phase1_model, X_val, y_val, selected_features, "Val"),
        'test': evaluate_model(phase1_model, X_test, y_test, selected_features, "Test")
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
        'test_auprc': phase1_metrics['test']['auprc'],
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
            'val': evaluate_model(iter_model, X_val, y_val, current_features, "Val"),
            'test': evaluate_model(iter_model, X_test, y_test, current_features, "Test")
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

    # Sort candidates by importance (remove lowest first)
    candidate_df = iter_importance_df[iter_importance_df['Feature'].isin(candidates)].sort_values('SHAP_Combined')

    # Cap at MAX_REMOVALS_PER_ITERATION
    features_to_remove = candidate_df.head(MAX_REMOVALS_PER_ITERATION)['Feature'].tolist()

    print(f"  Near-zero importance: {len(zero_importance)}")
    print(f"  Negative-biased ratio: {len(neg_biased)}")
    print(f"  Bottom {bottom_pct*100:.0f}%: {len(bottom_features)}")
    print(f"  Meeting 2+ criteria: {len(candidates)}")
    print(f"  Protected (top 50% ratio): {len(protected)}")
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
        'test_auprc': iter_metrics['test']['auprc'],
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

print("\nFinal model metrics:")
final_metrics = {
    'train': evaluate_model(final_model, X_train, y_train, final_features, "Train"),
    'val': evaluate_model(final_model, X_val, y_val, final_features, "Val"),
    'test': evaluate_model(final_model, X_test, y_test, final_features, "Test")
}

print("\n" + "-"*70)
print("COMPARISON: BASELINE vs FINAL")
print("-"*70)
print(f"{'Metric':<20} {'Baseline':>12} {'Final':>12} {'Change':>12}")
print("-"*56)
print(f"{'Features':<20} {len(feature_cols):>12} {len(final_features):>12} {len(final_features) - len(feature_cols):>+12}")
print(f"{'Val AUPRC':<20} {baseline_metrics['val']['auprc']:>12.4f} {final_metrics['val']['auprc']:>12.4f} {final_metrics['val']['auprc'] - baseline_metrics['val']['auprc']:>+12.4f}")
print(f"{'Test AUPRC':<20} {baseline_metrics['test']['auprc']:>12.4f} {final_metrics['test']['auprc']:>12.4f} {final_metrics['test']['auprc'] - baseline_metrics['test']['auprc']:>+12.4f}")
print(f"{'Train-Val Gap':<20} {baseline_metrics['train_val_gap']:>12.4f} {final_metrics['train']['auprc'] - final_metrics['val']['auprc']:>12.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Final Outputs

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
    'baseline_test_auprc': baseline_metrics['test']['auprc'],
    'final_test_auprc': final_metrics['test']['auprc'],
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

    # Validation AUPRC
    axes[0, 1].plot(range(len(tracking_df)), tracking_df['val_auprc'], 'go-', linewidth=2, markersize=8, label='Val')
    axes[0, 1].plot(range(len(tracking_df)), tracking_df['test_auprc'], 'ro-', linewidth=2, markersize=8, label='Test')
    axes[0, 1].axhline(y=baseline_metrics['val']['auprc'] * (1 - MAX_VAL_AUPRC_DROP), color='g', linestyle='--', alpha=0.5, label='5% drop threshold')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('AUPRC')
    axes[0, 1].set_title('AUPRC Over Iterations')
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
# MAGIC ## Utility: Clear Checkpoints (Run Manually if Needed)

# COMMAND ----------

# # Uncomment and run this cell to clear all checkpoints and start fresh
# # clear_checkpoints()
# # print("Checkpoints cleared. Re-run notebook to start fresh.")
