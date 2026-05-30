"""
AI3013 Machine Learning Course Project - Group 12
Speed-Dating Matching Prediction

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
import sys
from datetime import datetime

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

warnings.filterwarnings('ignore')

# ── Font & style settings (macaron palette) ───────────────────────────
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = '#FFF5F5'
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['axes.edgecolor'] = '#F3BBB1'
plt.rcParams['grid.color'] = '#F3BBB1'
plt.rcParams['grid.alpha'] = 0.4

# ── Output directory ─────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "data_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXPERIMENT_LOG = os.path.join(_HERE, "experiment_run_log.md")

def append_experiment_log(title, lines):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n## {timestamp} - {title}\n")
        for line in lines:
            f.write(f"- {line}\n")

def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [The chart has been saved.] {path}")
    plt.close()

def clear_existing_figures(output_dir=OUTPUT_DIR):
    removed = 0
    for name in os.listdir(output_dir):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".svg", ".pdf")):
            os.remove(os.path.join(output_dir, name))
            removed += 1
    print(f"  [Old visualization figures deleted] {removed} files")

def plot_raw_match_distribution(df_raw: pd.DataFrame):
    counts = df_raw['match'].value_counts()
    neg = int(counts.get(0, 0))
    pos = int(counts.get(1, 0))
    colors = ['#F7A6AC', '#B8E5FA']
    labels = ['No Match (0)', 'Match (1)']

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        [neg, pos],
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0, 0.1),
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )

    ax.set_title(f'Raw Class Distribution\n(Total: {neg + pos:,} samples)',
                 fontsize=13, fontweight='bold', pad=15)
    fig.suptitle('Original Data Match Distribution',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig('raw_match_distribution.png')


def normalize_raw_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Unify the short field names of the original CSV of Speed Dating to the long field names used in the project.
    In this way, the subsequent GPB, interaction features, and candidate_features can normally identify the core variables.
    """
    df = df.copy()

    rename_map = {
        # Ratings for partners
        'attr':    'attractive',
        'sinc':    'sincere',
        'intel':   'intelligence',
        'fun':     'funny',
        'amb':     'ambition',

        # Partner's rating
        # keep spelling mistakes: sinsere_o / ambitous_o
        'attr_o':  'attractive_o',
        'sinc_o':  'sinsere_o',
        'intel_o': 'intelligence_o',
        'fun_o':   'funny_o',
        'amb_o':   'ambitous_o',
        'shar_o':  'shared_interests_o',

        # Overall feeling
        'prob':    'guess_prob_liked',

        # Original interest relevance
        'int_corr': 'interests_correlate',

        # The weight of partner's preference for a spouse
        'pf_o_att': 'attractive_partner',
        'pf_o_sin': 'sincere_partner',
        'pf_o_int': 'intelligence_partner',
        'pf_o_fun': 'funny_partner',
        'pf_o_amb': 'ambition_partner',
        'pf_o_sha': 'shared_interests_partner',
    }

    existing_map = {old: new for old, new in rename_map.items() if old in df.columns}
    df = df.rename(columns=existing_map)

    print("\n[Unify the feature names]")
    print(f"  Renamed {len(existing_map)} original features")
    for old, new in existing_map.items():
        print(f"  {old:10s} -> {new}")

    return df


# ============================================================
# STEP 1: Data loading and exploratory analysis (EDA)
# ============================================================

def load_and_explore(file_path: str) -> pd.DataFrame:
    """
    Load data and conduct exploratory analysis, print key statistical information and generate visualization.
    """
    print("\n" + "="*60)
    print("STEP 1: EDA")
    print("="*60)

    # latin-1 maps every byte to a valid character, so this read never
    # fails regardless of the OS or Python build (some Linux Python
    # builds don't ship the gbk codec). Non-ASCII bytes in the original
    # file (only in a few school-name fields) are not used as features.
    df_raw = pd.read_csv(file_path, encoding='latin-1')
    raw_match_counts = df_raw['match'].value_counts()
    raw_match_ratio = df_raw['match'].mean()
    plot_raw_match_distribution(df_raw)
    df = df_raw.copy()

    # New: unify original CSV field names
    df = normalize_raw_column_names(df)

    # ── 1.1 Basic information ──────────────────────────────────────────
    print(f"\n[Dataset size]")
    print(f"  Rows (number of dating records): {df.shape[0]}")
    print(f"  Columns (number of features)    : {df.shape[1]}")
    print(f"\n[Target variable match distribution]")
    match_counts = raw_match_counts
    match_ratio  = raw_match_ratio
    print(f"  match=1 (matched): {raw_match_counts.get(1, 0)} records ({raw_match_ratio:.2%})")
    print(f"  match=0 (not matched): {raw_match_counts.get(0, 0)} records ({1-raw_match_ratio:.2%})")
    print(f"  ⚠️  Sample imbalance ratio approx 1:{int((1-raw_match_ratio)/raw_match_ratio):.0f}, must handle during modeling")
    print(f"\n✅ STEP 1 completed: only raw match distribution pie chart generated")
    

    # ── 1.2 Missing value heatmap ────────────────────────────────────────
    print(f"\n[Missing values Top-20 columns]")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'Missing count': missing, 'Missing rate(%)': missing_pct})
    missing_df = missing_df[missing_df['Missing count'] > 0].sort_values('Missing rate(%)', ascending=False)
    print(missing_df.head(20).to_string())

    # Missing rate bar chart (Top 25 columns)
    top_missing = missing_df.head(25)
    if len(top_missing) > 0:
        fig, ax = plt.subplots(figsize=(14, 5))
        colors = ['#F7A6AC' if v > 30 else '#EEC78A' if v > 10 else '#B8E5FA'
                  for v in top_missing['Missing rate(%)']]
        ax.bar(range(len(top_missing)), top_missing['Missing rate(%)'], color=colors)
        ax.axhline(30, color='red',    linestyle='--', alpha=0.6, label='30% threshold (consider dropping column)')
        ax.axhline(10, color='orange', linestyle='--', alpha=0.6, label='10% threshold (need focused imputation)')
        ax.set_xticks(range(len(top_missing)))
        ax.set_xticklabels(top_missing.index, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Missing rate (%)')
        ax.set_title('Missing rate per column (Top 25)')
        ax.legend()
        plt.tight_layout()
        save_fig('01_missing_rate.png')

    # ── 1.3 Target variable pie chart ──────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].pie([raw_match_counts.get(1, 0), raw_match_counts.get(0, 0)],
                labels=['Match (match=1)', 'No Match (match=0)'],
                colors=['#F7A6AC', '#B8E5FA'], autopct='%1.1f%%',
                startangle=140, explode=(0.05, 0))
    axes[0].set_title('Raw data target variable distribution (unprocessed)')

    # Match rate by gender
    if 'gender' in df.columns:
        gender_match = df.groupby('gender')['match'].mean()
        gender_labels = {0: 'Female (0)', 1: 'Male (1)'}
        labels = [gender_labels.get(g, str(g)) for g in gender_match.index]
        axes[1].bar(labels, gender_match.values, color=['#F7B2C7', '#B8E5FA'], alpha=0.8)
        axes[1].set_ylabel('Match rate')
        axes[1].set_title('Match rate by gender')
        axes[1].set_ylim(0, 1)
        for i, v in enumerate(gender_match.values):
            axes[1].text(i, v + 0.01, f'{v:.2%}', ha='center', fontsize=11)
    plt.tight_layout()
    save_fig('02_target_distribution.png')

    # ── 1.4 Key numeric feature distributions ──────────────────────────────────
    key_numeric = ['age', 'age_o', 'attractive', 'sincere', 'intelligence',
                   'funny', 'ambition', 'like', 'guess_prob_liked']
    key_numeric = [c for c in key_numeric if c in df.columns]

    n = len(key_numeric)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()

    for i, col in enumerate(key_numeric):
        data_no_nan = df[col].dropna()
        axes[i].hist(data_no_nan[df['match'] == 0], bins=30, alpha=0.6,
                     color='#B8E5FA', label='No match', density=True)
        axes[i].hist(data_no_nan[df['match'] == 1], bins=30, alpha=0.6,
                     color='#F7A6AC', label='Match', density=True)
        axes[i].set_title(col)
        axes[i].legend(fontsize=8)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Density')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Key feature distributions (grouped by match outcome)', fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_fig('03_feature_distributions.png')

    # ── 1.5 Interest feature correlation matrix ───────────────────────────────
    interest_cols = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
                     'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
                     'movies', 'concerts', 'music', 'shopping', 'yoga']
    interest_cols = [c for c in interest_cols if c in df.columns]

    if len(interest_cols) >= 5:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df[interest_cols].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, linewidths=0.5, ax=ax, annot_kws={'size': 7})
        ax.set_title('17 interest features correlation matrix (raw scores)', fontsize=13)
        plt.tight_layout()
        save_fig('04_interest_correlation_raw.png')

    print(f"\n✅ STEP 1 completed: generated 4 exploratory analysis charts")
    return df


# ============================================================
# STEP 2: Data cleaning
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Data cleaning:
      - Remove rows with missing target variable
      - Remove rows where core feature missing rate > 30%
      - Handle numeric outliers with 3σ rule (clip rather than delete)
      - Fill numeric features with median, categorical features with mode
    """
    print("\n" + "="*60)
    print("STEP 2: Data cleaning")
    print("="*60)

    n_before = len(df)

    # New: replace empty strings/spaces with NaN
    df = df.replace(['', ' ', '  '], np.nan)

    # 2.1 Remove rows with missing target variable
    df = df.dropna(subset=['match']).copy()
    df['match'] = df['match'].astype(int)
    print(f"\n  After removing rows with missing match: {n_before} → {len(df)} rows")

    # 2.2 Remove rows where core feature missing rate > 30%
    core_features = ['attractive', 'sincere', 'intelligence', 'funny',
                     'attractive_o', 'like', 'guess_prob_liked']
    core_features = [c for c in core_features if c in df.columns]
    row_missing_pct = df[core_features].isnull().mean(axis=1)
    df = df[row_missing_pct <= 0.30].copy()
    print(f"  After removing rows with core features missing >30%: {len(df)} rows")

    # 2.3 Leakage-safe note
    # Do not fit clipping / imputation statistics here, because this step
    # happens before train/test split. Missing-value imputation and 3-sigma
    # clipping are fitted later inside build_feature_matrix() using train only.
    print("\n  [leakage-safe] defer imputation and outlier clipping to train-only Step 6")

    # 2.5 Sample imbalance visualization (after cleaning)
    match_counts_clean = df['match'].value_counts()
    print(f"\n  [Target variable distribution after cleaning]")
    print(f"  match=1: {match_counts_clean.get(1,0)}  |  match=0: {match_counts_clean.get(0,0)}")
    print(f"  ⚠️  Imbalance ratio: 1 : {match_counts_clean.get(0,0)/max(match_counts_clean.get(1,1),1):.1f}")
    print(f"  → Need to handle during modeling (recommend setting class_weight in loss function)")

    print(f"\n✅ STEP 2 completed")
    df = df.reset_index(drop=True)
    return df


# ============================================================
# STEP 3: Within-person standardization (GPB) + Interest feature engineering (core innovation)
# ============================================================

INTEREST_COLS = ['sports', 'tvsports', 'exercise', 'dining', 'museums', 'art',
                 'hiking', 'gaming', 'clubbing', 'reading', 'tv', 'theater',
                 'movies', 'concerts', 'music', 'shopping', 'yoga']

# All possible score columns for GPB (self ratings, partner requirements, interests)
GPB_SCORE_COLS = [
    'attractive', 'sincere', 'intelligence', 'funny', 'ambition',
    'attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o', 'ambitous_o',
    'attractive_partner', 'sincere_partner', 'intelligence_partner',
    'funny_partner', 'ambition_partner', 'shared_interests_partner',
    'attractive_important', 'sincere_important', 'intellicence_important',
    'funny_important', 'ambtition_important', 'shared_interests_important',
] + INTEREST_COLS


def compute_gpb(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 1: Compute Global Persona Baseline (GPB)
      μ_i = mean of all rating columns (rating baseline: lenient/strict)
      σ_i = standard deviation of all rating columns (rating sensitivity)
    """
    available_gpb_cols = [c for c in GPB_SCORE_COLS if c in df.columns]
    print(f"\n  GPB uses {len(available_gpb_cols)} rating columns: {available_gpb_cols[:8]}...")

    df['gpb_mu']    = df[available_gpb_cols].mean(axis=1)
    df['gpb_sigma'] = df[available_gpb_cols].std(axis=1)
    df['gpb_sigma'] = df['gpb_sigma'].replace(0, 1e-6)  # prevent division by zero

    print(f"  μ_i   stats: mean={df['gpb_mu'].mean():.3f}, std={df['gpb_mu'].std():.3f}")
    print(f"  σ_i   stats: mean={df['gpb_sigma'].mean():.3f}, std={df['gpb_sigma'].std():.3f}")
    return df


def compute_interest_zscores(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Step 2: Map the 17-dimensional raw interest scores to relative strength scores that eliminate rating bias
      z_{i,k} = (RawScore_{i,k} - μ_i) / σ_i
    Return dataframe and list of z-score column names
    """
    available_interest = [c for c in INTEREST_COLS if c in df.columns]
    z_cols = []
    for col in available_interest:
        z_col = f'z_{col}'
        df[z_col] = (df[col] - df['gpb_mu']) / df['gpb_sigma']
        z_cols.append(z_col)

    print(f"\n  Generated {len(z_cols)} z-score interest feature columns")

    # Visualization: comparison before and after standardization (using sports as example)
    if 'sports' in df.columns and 'z_sports' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].hist(df['sports'].dropna(), bins=30, color='#B8E5FA', alpha=0.7)
        axes[0].set_title('Raw sports (with rating baseline bias)')
        axes[0].set_xlabel('Raw score')
        axes[1].hist(df['z_sports'].dropna(), bins=30, color='#F7A6AC', alpha=0.7)
        axes[1].set_title('After z-score z_sports (bias removed)')
        axes[1].set_xlabel('Standardized score')
        plt.suptitle('GPB within-person standardization effect example (sports feature)', fontweight='bold')
        plt.tight_layout()
        save_fig('05_gpb_normalization_demo.png')

    return df, z_cols


def build_partner_interest_lookup(df: pd.DataFrame, z_cols: list) -> dict:
    """
    Build a lookup table iid → average z-score vector.
    In speed-dating data, the same iid may appear multiple times across different sessions, take the mean.
    """
    if 'iid' not in df.columns:
        return {}

    lookup = {}
    for iid, group in df.groupby('iid'):
        vec = group[z_cols].mean().values  # (17,)
        lookup[iid] = vec
    print(f"  Built partner z-score lookup table: {len(lookup)} unique participants")
    return lookup


def compute_pair_similarity_features(df: pd.DataFrame, z_cols: list,
                                      lookup: dict,
                                      train_idx=None,
                                      reference_df=None) -> pd.DataFrame:
    """
    Step 3 & 4: Compute directional similarity (cosine) and magnitude similarity (Euclidean distance) between two persons,
    and SIS (Standardized Interest Similarity).

    SIS   = 1 - (1/17) * Σ|z_m,k - z_f,k|, clipped to [0,1]
    Cosine = V_A · V_B / (|V_A| |V_B|), measures taste direction consistency
    Euclidean = sqrt(Σ w_k (z_A,k - z_B,k)²), measures energy/intensity difference
    """
    if 'iid' not in df.columns or 'pid' not in df.columns or len(lookup) == 0:
        print("  ⚠️  iid/pid columns not found, skip building pair similarity features")
        return df

    n = len(z_cols)

    # Compute feature weights based on variance (higher variance → higher discriminative power → higher weight)
    # Use train-only reference statistics to avoid test distribution leakage.
    if reference_df is None:
        reference_df = df

    z_data = reference_df[z_cols].values
    variances = np.nanvar(z_data, axis=0)
    variances[variances == 0] = 1e-9
    weights = variances / variances.sum()  # normalized weights

    sis_list       = []
    cosine_list    = []
    euclidean_list = []

    for _, row in df.iterrows():
        iid = row['iid']
        pid = row['pid'] if 'pid' in row.index else None

        vec_self    = np.array([row[c] for c in z_cols], dtype=float)
        vec_partner = lookup.get(pid, None) if pid is not None else None

        if vec_partner is None or np.all(np.isnan(vec_self)) or np.all(np.isnan(vec_partner)):
            sis_list.append(np.nan)
            cosine_list.append(np.nan)
            euclidean_list.append(np.nan)
            continue

        # Handle NaN: fill with 0 (neutral)
        vs = np.where(np.isnan(vec_self),    0, vec_self)
        vp = np.where(np.isnan(vec_partner), 0, vec_partner)

        # SIS
        sis_val = 1 - (1 / n) * np.sum(np.abs(vs - vp))
        sis_val = float(np.clip(sis_val, 0, 1))

        # Cosine similarity
        norm_s = np.linalg.norm(vs)
        norm_p = np.linalg.norm(vp)
        if norm_s < 1e-9 or norm_p < 1e-9:
            cos_val = 0.0
        else:
            cos_val = float(np.dot(vs, vp) / (norm_s * norm_p))

        # Weighted Euclidean distance
        euc_val = float(np.sqrt(np.sum(weights * (vs - vp) ** 2)))

        sis_list.append(sis_val)
        cosine_list.append(cos_val)
        euclidean_list.append(euc_val)

    df['SIS']              = sis_list    # core innovation feature: standardized interest similarity
    df['interest_cosine']  = cosine_list # directional similarity (taste structure alignment)
    df['interest_euclidean'] = euclidean_list  # magnitude distance (smaller = more similar)

    # Convert Euclidean distance to similarity (larger = more similar)
    # Normalize Euclidean distance with train-only max when available.
    if train_idx is not None:
        max_euc = df.iloc[train_idx]['interest_euclidean'].max()
    else:
        max_euc = df['interest_euclidean'].max()

    if max_euc > 0:
        df['interest_euclidean_sim'] = 1 - df['interest_euclidean'] / max_euc
    else:
        df['interest_euclidean_sim'] = 0.0

    # Statistics and validation
    print(f"\n  [SIS feature validation]")
    for label, val in [('SIS', df['SIS']),
                        ('interest_cosine', df['interest_cosine']),
                        ('interest_euclidean_sim', df['interest_euclidean_sim'])]:
        corr = val.corr(df['match'])
        print(f"  {label:25s} → correlation with match: {corr:.4f}")

    # Compare with original interests_correlate
    if 'interests_correlate' in df.columns:
        orig_corr = df['interests_correlate'].corr(df['match'])
        sis_corr  = df['SIS'].corr(df['match'])
        print(f"\n  Original interests_correlate correlation with match: {orig_corr:.4f}")
        print(f"  Optimized SIS               correlation with match: {sis_corr:.4f}")
        improvement = (abs(sis_corr) - abs(orig_corr)) / abs(orig_corr) * 100 if orig_corr != 0 else 0
        print(f"  Correlation improvement: {improvement:+.1f}%")

    return df


def visualize_similarity_features(df: pd.DataFrame):
    """Visualize distributions of SIS, cosine similarity, Euclidean similarity (grouped by match outcome)"""
    sim_cols = ['SIS', 'interest_cosine', 'interest_euclidean_sim']
    sim_cols = [c for c in sim_cols if c in df.columns]
    if not sim_cols:
        return

    fig, axes = plt.subplots(1, len(sim_cols), figsize=(5 * len(sim_cols), 4))
    if len(sim_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, sim_cols):
        for label, color in [(0, '#B8E5FA'), (1, '#F7A6AC')]:
            data = df[df['match'] == label][col].dropna()
            ax.hist(data, bins=30, alpha=0.6, color=color,
                    label=f'match={label}', density=True)
        ax.set_title(col)
        ax.set_xlabel('Similarity value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)

    plt.suptitle('Interest similarity feature distributions (grouped by match outcome)', fontweight='bold')
    plt.tight_layout()
    save_fig('06_similarity_features_dist.png')


# ============================================================
# STEP 4a: Within-Person Relative Rating Features
# DEPRECATED: uses full-data groupby — leakage risk!
# Use add_leakage_safe_centered_features() instead.
# ============================================================

# DEPRECATED: leakage risk, do not use
# This function computes group means on the full dataset (train+test),
# which leaks test-set information into the features.
# Replaced by add_leakage_safe_centered_features() in build_feature_matrix().
def add_within_person_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    raise RuntimeError(
        "DEPRECATED: add_within_person_relative_features() uses full-data groupby "
        "and causes data leakage. Use add_leakage_safe_centered_features() instead."
    )


# ============================================================
# STEP 4: Other feature construction
# ============================================================

def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct interaction features to capture non-linear effects of gender on mate preferences:
      - gender × attractive (gender difference in physical appearance preference)
      - gender × sincere
      - gender × intelligence
      - gender × income (gender difference in socioeconomic condition preference)
      - age_diff (age difference)
      - avg_trait_score (average trait score)
      - partner_avg_trait (partner's average trait score)
     Note:
      Originally planned to construct gender × income, but income has high missing rate and unstable data quality,
      so this income-derived interaction feature is removed to improve model stability.
    """
    print("\n[Construct interaction features & difference features]")

    # Gender encoding
    if 'gender' in df.columns:
        if df['gender'].dtype == object:
            unique_vals = df['gender'].dropna().unique()
            mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
            df['gender'] = df['gender'].map(mapping).fillna(0).astype(int)

        # female indicator: raw data gender=1 means male, gender=0 means female
        df['female'] = 1 - df['gender']

    # gender × trait score interactions
    trait_cols = ['attractive', 'sincere', 'intelligence', 'funny', 'ambition']
    for col in trait_cols:
        if col in df.columns and 'gender' in df.columns:
            df[f'gender_{col}'] = df['gender'] * df[col]

    # Paper-guided feature: female × samerace
    # Fisman et al. found that females have a stronger preference for same-race partners
    if 'female' in df.columns and 'samerace' in df.columns:
        df['female_samerace'] = df['female'] * df['samerace']

    # Paper-guided feature: male × SIS
    # gender=1 indicates male, capturing the additional effect of interest similarity on male samples
    if 'gender' in df.columns and 'SIS' in df.columns:
        df['male_SIS'] = df['gender'] * df['SIS']


    # Age difference
    if 'age' in df.columns and 'age_o' in df.columns:
        df['age_diff'] = np.abs(df['age'] - df['age_o'])
        print(f"  age_diff: mean={df['age_diff'].mean():.2f}, max={df['age_diff'].max():.2f}")

    # Average trait score (self)
    self_trait_cols = [c for c in ['attractive', 'sincere', 'intelligence', 'funny', 'ambition']
                       if c in df.columns]
    if self_trait_cols:
        df['avg_trait_score'] = df[self_trait_cols].mean(axis=1)

    # Partner average trait score
    partner_trait_cols_map = {
        'attractive_o': 'attractive_o',
        'sinsere_o': 'sinsere_o',
        'intelligence_o': 'intelligence_o',
        'funny_o': 'funny_o',
        'ambitous_o': 'ambitous_o'
    }
    available_partner = [v for v in partner_trait_cols_map.values() if v in df.columns]
    if available_partner:
        df['partner_avg_trait'] = df[available_partner].mean(axis=1)

    # =========================
    # Strong interaction features v3
    # Constructed based on the currently highest-ranked features
    # =========================

    # 1. like × guess_prob_liked: you like the partner, and also think the partner might like you
    if 'like' in df.columns and 'guess_prob_liked' in df.columns:
        df['like_x_guess_prob_liked'] = df['like'] * df['guess_prob_liked']

    # 2. like × shared_interests_o: combination of liking degree and shared interests rating
    if 'like' in df.columns and 'shared_interests_o' in df.columns:
        df['like_x_shared_interests_o'] = df['like'] * df['shared_interests_o']

    # 3. attractive × attractive_o: combination of appearance evaluations from both sides
    if 'attractive' in df.columns and 'attractive_o' in df.columns:
        df['attractive_pair'] = df['attractive'] * df['attractive_o']

    # 4. funny x funny_o: pairwise humor evaluation interaction
    if 'funny' in df.columns and 'funny_o' in df.columns:
        df['funny_pair'] = df['funny'] * df['funny_o']

    # 5. avg_trait_score × partner_avg_trait: combination of overall traits from both sides
    if 'avg_trait_score' in df.columns and 'partner_avg_trait' in df.columns:
        df['trait_pair'] = df['avg_trait_score'] * df['partner_avg_trait']

    # =========================
    # Gap features v1
    # Capture the degree of asymmetry between evaluations/expectations of the two sides
    # =========================

    # 1. attractive_gap: difference in appearance evaluation between the two sides
    if 'attractive' in df.columns and 'attractive_o' in df.columns:
        df['attractive_gap'] = np.abs(df['attractive'] - df['attractive_o'])


    # 2. funny_gap: difference in sense of humor evaluation between the two sides
    if 'funny' in df.columns and 'funny_o' in df.columns:
        df['funny_gap'] = np.abs(df['funny'] - df['funny_o'])

    # 3. trait_gap: difference in average trait score between the two sides
    if 'avg_trait_score' in df.columns and 'partner_avg_trait' in df.columns:
        df['trait_gap'] = np.abs(df['avg_trait_score'] - df['partner_avg_trait'])

    # 4. like_guess_gap: difference between liking the partner vs thinking the partner likes you
    if 'like' in df.columns and 'guess_prob_liked' in df.columns:
        df['like_guess_gap'] = np.abs(df['like'] - df['guess_prob_liked'])


    created = [c for c in df.columns if c.startswith('gender_') or
               c in ['age_diff', 'avg_trait_score', 'partner_avg_trait',
                     'female_samerace', 'male_SIS',
                     'like_x_guess_prob_liked',
                     'like_x_shared_interests_o',
                     'attractive_pair',
                     'funny_pair',
                     'trait_pair',
                     'attractive_gap',
                     'funny_gap',
                     'trait_gap',
                     'like_guess_gap']]
    print(f"  Constructed {len(created)} interaction/difference features: {created}")
    return df


def visualize_interaction_features(df: pd.DataFrame):
    """Visualize the impact of key interaction features on match outcome"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    # age_diff vs match
    plot_pairs = [
        ('age_diff',          'Age difference'),
        ('avg_trait_score',   'Self average trait score'),
        ('partner_avg_trait', 'Partner average trait score'),
        ('like',              'Liking degree (like)'),
        ('guess_prob_liked',  'Estimated partner liking of self'),
        ('gender_attractive', 'Gender × appearance interaction'),
    ]

    for ax, (col, label) in zip(axes, plot_pairs):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        for match_val, color in [(0, '#B8E5FA'), (1, '#F7A6AC')]:
            data = df[df['match'] == match_val][col].dropna()
            ax.hist(data, bins=25, alpha=0.6, color=color,
                    label=f'match={match_val}', density=True)
        ax.set_title(label)
        ax.legend(fontsize=8)
        ax.set_xlabel('Value')

    plt.suptitle('Key feature distribution comparison (match vs no match)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_fig('07_interaction_features_dist.png')


# ============================================================
# STEP 5: Feature correlation analysis (EDA only, not for final feature selection)
# ============================================================

def analyze_feature_correlation(df: pd.DataFrame, feature_list: list):
    """
    Calculate correlation between each feature and the target variable match, and visualize.
    The output of this step is used directly as a reference for Random Forest / ElasticNet feature selection.
    """
    print("\n" + "="*60)
    print("STEP 5: Feature correlation analysis (EDA only, not for final feature selection)")
    print("="*60)

    # 🔥 Core fix: keep only numeric columns + replace empty strings with NaN
    df_clean = df.copy()
    # Replace empty strings with NaN
    df_clean = df_clean.replace('', np.nan)
    # Filter only numeric columns
    numeric_df = df_clean.select_dtypes(include=[np.number])

    available = [f for f in feature_list if f in numeric_df.columns]
    if not available:
        print("⚠️ No valid numeric features, skip correlation analysis")
        return None

    # Calculate correlation
    corr_with_match = numeric_df[available + ['match']].corr()['match'].drop('match')
    corr_df = corr_with_match.abs().sort_values(ascending=False)

    print(f"\n  [Correlation with match Top-20 (absolute value)]")
    print(corr_df.head(20).to_string())

    # Correlation bar chart
    top_corr = corr_df.head(20)
    raw_corr = corr_with_match[top_corr.index]

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ['#F7A6AC' if v > 0 else '#B8E5FA' for v in raw_corr.values]
    ax.barh(range(len(top_corr)), raw_corr.values, color=colors)
    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(top_corr.index, fontsize=10)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Pearson correlation coefficient')
    ax.set_title('Correlation of features with match (Top 20)\nGreen=positive | Red=negative', fontsize=12)
    plt.tight_layout()
    save_fig('08_feature_correlation_with_match.png')

    # Feature inter-correlation heatmap
    top_features = corr_df.head(15).index.tolist()
    if len(top_features) >= 5:
        fig, ax = plt.subplots(figsize=(12, 10))
        inter_corr = numeric_df[top_features].corr()
        mask = np.triu(np.ones_like(inter_corr, dtype=bool))
        sns.heatmap(inter_corr, mask=mask, annot=True, fmt='.2f',
                    cmap='RdBu_r', center=0, linewidths=0.5, ax=ax,
                    annot_kws={'size': 8})
        ax.set_title('Top-15 feature internal correlation (collinearity analysis)', fontsize=12)
        plt.tight_layout()
        save_fig('09_top_features_inter_correlation.png')

    print(f"\n  💡 Suggestion: feature pairs with |correlation| > 0.6 have collinearity risk, ElasticNet can handle automatically")
    return corr_with_match

# ============================================================
# STEP 6: Feature matrix construction and standardization
# ============================================================

def add_leakage_safe_centered_features(df, train_idx, test_idx):
    """
    Leakage-safe Within-Person Relative Rating Features v1

    Correct logic:
    1. First split train/test
    2. Use only the train set to compute average rating for each iid / pid
    3. Both train and test use train group mean for centering
    4. For iid / pid not seen in test, fall back to train global mean
    """
    df = df.copy()
    train_df = df.iloc[train_idx].copy()

    specs = [
        # subject / iid dimension
        ('iid', 'like',             'like_iid_centered'),
        ('iid', 'guess_prob_liked', 'guess_prob_liked_iid_centered'),
        ('iid', 'attractive',       'attractive_iid_centered'),
        ('iid', 'funny',            'funny_iid_centered'),
        # partner / pid dimension
        ('pid', 'attractive_o',     'attractive_o_pid_centered'),
        ('pid', 'funny_o',          'funny_o_pid_centered'),
    ]

    created = []

    for group_col, value_col, new_col in specs:
        if group_col not in df.columns or value_col not in df.columns:
            print(f"  [SKIP] {new_col}: missing {group_col} or {value_col}")
            continue

        # Use only train set to compute group mean
        train_group_mean  = train_df.groupby(group_col)[value_col].mean()
        train_global_mean = train_df[value_col].mean()

        mapped_mean = df[group_col].map(train_group_mean)
        mapped_mean = mapped_mean.fillna(train_global_mean)

        df[new_col] = df[value_col] - mapped_mean
        created.append(new_col)

    print("\n[Leakage-free centered features]")
    print(f"  Constructed {len(created)} leakage-safe within-person relative rating features:")
    for c in created:
        print(f"    - {c}")

    return df

def build_feature_matrix(df: pd.DataFrame, z_cols: list,
                         train_idx=None, test_idx=None) -> tuple:
    """
    Assemble final feature matrix, perform z-score standardization, and stratified split (without sklearn).
    Returns: X_train, X_test, y_train, y_test, feature_names, scaler_params
    """
    print("\n" + "="*60)
    print("STEP 6: Feature matrix construction and standardization")
    print("="*60)

    # 6.0 First split train/test indices
    # Note: leakage-safe centered features must use only train set statistics
    y = df['match'].values.astype(int)
    if train_idx is None or test_idx is None:
        train_idx, test_idx = leakage_safe_train_test_indices(
            df, test_size=0.3, seed=42
        )

    df = add_leakage_safe_centered_features(df, train_idx, test_idx)

    # 6.1 Candidate feature list
    candidate_features = (
    # Basic demographic attributes
    ['gender', 'female', 'age', 'age_o', 'age_diff', 'samerace', 'female_samerace']
    # Partner's rating of self
    + ['attractive', 'sincere', 'intelligence', 'funny', 'ambition',
       'avg_trait_score']
    # Self rating of partner
    + ['attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o',
       'ambitous_o', 'shared_interests_o', 'partner_avg_trait']
    # Importance weights
    + ['attractive_partner', 'sincere_partner', 'intelligence_partner',
       'funny_partner', 'ambition_partner', 'shared_interests_partner']
    # Overall feelings
    + ['like', 'guess_prob_liked', 'met']
    # Core innovation interest similarity features
    + ['SIS', 'interest_cosine', 'interest_euclidean_sim']
    # Original interest correlation for comparison
    + ['interests_correlate']
    # Interaction features: paper-guided + strong interactions + gap features
    + ['gender_attractive', 'gender_sincere', 'gender_intelligence',
       'gender_funny', 'gender_ambition',
       'male_SIS',
       'like_x_guess_prob_liked',
       'like_x_shared_interests_o',
       'attractive_pair',
       'funny_pair',
       'trait_pair',
       'attractive_gap',
       'funny_gap',
       'trait_gap',
       'like_guess_gap',
       'like_iid_centered',
       'guess_prob_liked_iid_centered',
       'attractive_iid_centered',
       'funny_iid_centered',
       'attractive_o_pid_centered',
       'funny_o_pid_centered']
    )

    existing = [f for f in candidate_features if f in df.columns]
    print(f"\n  Final included features: {len(existing)}")
    print(f"  Feature list: {existing}")

    # ========== Critical modifications: clean empty strings + convert to numeric ==========
    # 1. Select feature columns and copy to avoid modifying original data
    df_features = df[existing].copy()

    # 2. Replace empty strings, space strings with NaN
    df_features = df_features.replace(['', ' ', '  '], np.nan)

    # 3. Force conversion to numeric type (non-convertible become NaN)
    for col in df_features.columns:
        df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

    # 4. Leakage-safe missing-value imputation + outlier clipping
    #    Fit statistics on train_idx only, then apply to both train/test rows.
    for col in df_features.columns:
        train_col = df_features.iloc[train_idx][col]

        if df_features[col].isnull().sum() > 0:
            median_val = train_col.median()

            if pd.isna(median_val):
                median_val = 0.0

            df_features[col] = df_features[col].fillna(median_val)
            print(f"  [train-only] fill {col}: median = {median_val:.4f}")

        train_col_filled = df_features.iloc[train_idx][col]
        mu = train_col_filled.mean()
        sigma = train_col_filled.std()

        if pd.notna(sigma) and sigma > 0:
            lower = mu - 3 * sigma
            upper = mu + 3 * sigma
            df_features[col] = df_features[col].clip(lower=lower, upper=upper)

    # 5. Verify all columns are numeric
    non_numeric_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(f"Still have non-numeric columns: {non_numeric_cols}, please check data")

    # ========== Build feature matrix ==========
    X = df_features.values.astype(float)

    print(f"\n  Feature matrix shape: X={X.shape}, y={y.shape}")
    print(f"  Positive samples (match=1): {y.sum()} | Negative samples: {(y==0).sum()}")

    # 6.2 Use previously generated train_idx / test_idx for split
    # No longer call stratified_train_test_split to avoid leakage of centered features before split
    X_train = X[train_idx]
    X_test  = X[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]

    X_mean = np.mean(X_train, axis=0)
    X_std  = np.std(X_train,  axis=0)
    X_std[X_std == 0] = 1  # prevent division by zero

    X_train_scaled = (X_train - X_mean) / X_std
    X_test_scaled  = (X_test  - X_mean) / X_std

    scaler_params = {'mean': X_mean, 'std': X_std}

    print(f"\n  Training set: {X_train_scaled.shape[0]} rows, test set: {X_test_scaled.shape[0]} rows")
    print(f"  Standardization completed (using training set μ/σ to prevent data leakage)")

    # 6.3 Visualization: distribution of standardized features
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col_name, data in [
        (axes[0], 'like (standardized)',       X_train_scaled[:, existing.index('like')] if 'like' in existing else None),
        (axes[1], 'SIS (standardized)',         X_train_scaled[:, existing.index('SIS')]  if 'SIS'  in existing else None),
        (axes[2], 'age_diff (standardized)',    X_train_scaled[:, existing.index('age_diff')] if 'age_diff' in existing else None),
    ]:
        if data is None:
            ax.set_visible(False)
            continue
        ax.hist(data, bins=30, color='#B8E5FA', alpha=0.8)
        ax.set_title(col_name)
        ax.set_xlabel('Standardized value')
        ax.set_ylabel('Frequency')
    plt.suptitle('Distribution of selected features after standardization (training set)', fontweight='bold')
    plt.tight_layout()
    save_fig('10_scaled_feature_dist.png')

    return X_train_scaled, X_test_scaled, y_train, y_test, existing, scaler_params


def leakage_safe_train_test_indices(df, test_size=0.3, seed=42):
    """
    Strict leakage-safe split.

    Priority 1: split by wave if available. In the Speed Dating dataset,
    each wave contains a separate set of participants, so wave-level split
    prevents the same participant from appearing in both train and test.

    Fallback: split participants by iid/pid and keep only rows where both
    participants belong to the same side. Cross-side rows are discarded.
    This is stricter but may reduce sample size.
    """
    rng = np.random.default_rng(seed)
    y = df['match'].values.astype(int)

    if 'wave' in df.columns:
        wave_rows = []
        for wave, group in df.groupby('wave', sort=True):
            idx = df.index.get_indexer(group.index)
            wave_rows.append({
                'id': wave,
                'idx': idx,
                'n': len(idx),
                'pos': int(group['match'].sum()),
                'rate': float(group['match'].mean())
            })

        target_n = int(round(len(df) * test_size))
        target_rate = float(np.mean(y))
        best_score = None
        best_test_waves = None
        n_trials = 5000

        for _ in range(n_trials):
            order = rng.permutation(len(wave_rows))
            selected = []
            selected_n = 0
            selected_pos = 0
            for j in order:
                row = wave_rows[j]
                if selected and selected_n >= target_n:
                    break
                selected.append(j)
                selected_n += row['n']
                selected_pos += row['pos']

            selected_rate = selected_pos / selected_n
            size_penalty = abs(selected_n - target_n) / len(df)
            rate_penalty = abs(selected_rate - target_rate)
            score = size_penalty + 2.0 * rate_penalty

            if best_score is None or score < best_score:
                best_score = score
                best_test_waves = selected

        test_waves = {wave_rows[j]['id'] for j in best_test_waves}
        train_waves = {row['id'] for row in wave_rows if row['id'] not in test_waves}
        test_idx = np.where(df['wave'].isin(test_waves).to_numpy())[0]
        train_idx = np.where(df['wave'].isin(train_waves).to_numpy())[0]

        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        train_iid = set(df.iloc[train_idx]['iid'].dropna().astype(int)) if 'iid' in df.columns else set()
        test_iid = set(df.iloc[test_idx]['iid'].dropna().astype(int)) if 'iid' in df.columns else set()
        train_pid = set(df.iloc[train_idx]['pid'].dropna().astype(int)) if 'pid' in df.columns else set()
        test_pid = set(df.iloc[test_idx]['pid'].dropna().astype(int)) if 'pid' in df.columns else set()
        iid_overlap = train_iid & test_iid
        pid_overlap = train_pid & test_pid
        all_overlap = (train_iid | train_pid) & (test_iid | test_pid)

        split_meta_path = os.path.join(OUTPUT_DIR, "split_metadata.txt")
        with open(split_meta_path, "w", encoding="utf-8") as f:
            f.write("Leakage-safe wave-level split\n")
            f.write(f"seed={seed}\n")
            f.write(f"test_size_target={test_size}\n")
            f.write(f"train_rows={len(train_idx)}\n")
            f.write(f"test_rows={len(test_idx)}\n")
            f.write(f"train_positive_rate={df.iloc[train_idx]['match'].mean():.6f}\n")
            f.write(f"test_positive_rate={df.iloc[test_idx]['match'].mean():.6f}\n")
            f.write(f"train_waves={sorted(train_waves)}\n")
            f.write(f"test_waves={sorted(test_waves)}\n")
            f.write(f"iid_overlap={len(iid_overlap)}\n")
            f.write(f"pid_overlap={len(pid_overlap)}\n")
            f.write(f"any_participant_overlap={len(all_overlap)}\n")

        print("\n[Leakage-safe split]")
        print("  Method: wave-level split")
        print(f"  Train waves: {sorted(train_waves)}")
        print(f"  Test waves : {sorted(test_waves)}")
        print(f"  Train rows: {len(train_idx)} | Test rows: {len(test_idx)}")
        print(f"  Train positive rate: {df.iloc[train_idx]['match'].mean():.4f}")
        print(f"  Test positive rate : {df.iloc[test_idx]['match'].mean():.4f}")
        print(f"  Participant overlap by iid: {len(iid_overlap)}")
        print(f"  Participant overlap by pid: {len(pid_overlap)}")
        print(f"  Participant overlap by iid/pid union: {len(all_overlap)}")
        print(f"  Split metadata saved: {split_meta_path}")

        append_experiment_log("Preprocessing split design updated", [
            "Implemented leakage-safe wave-level split from ML_preprocessing_design.docx.",
            f"Train rows={len(train_idx)}, test rows={len(test_idx)}.",
            f"Train waves={sorted(train_waves)}.",
            f"Test waves={sorted(test_waves)}.",
            f"Train positive rate={df.iloc[train_idx]['match'].mean():.4f}, test positive rate={df.iloc[test_idx]['match'].mean():.4f}.",
            f"Participant overlap union={len(all_overlap)}.",
            f"Split metadata={split_meta_path}."
        ])

        return train_idx, test_idx

    if 'iid' not in df.columns or 'pid' not in df.columns:
        print("\n[Leakage-safe split] iid/pid unavailable; falling back to row split.")
        return stratified_train_test_indices(y, test_size=test_size, seed=seed)

    participants = np.unique(
        np.concatenate([
            df['iid'].dropna().to_numpy(),
            df['pid'].dropna().to_numpy()
        ])
    )
    rng.shuffle(participants)
    n_test = int(len(participants) * test_size)
    test_people = set(participants[:n_test])
    train_people = set(participants[n_test:])

    iid = df['iid'].to_numpy()
    pid = df['pid'].to_numpy()
    train_mask = np.array([
        a in train_people and b in train_people
        for a, b in zip(iid, pid)
    ])
    test_mask = np.array([
        a in test_people and b in test_people
        for a, b in zip(iid, pid)
    ])

    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    discarded = len(df) - len(train_idx) - len(test_idx)
    print("\n[Leakage-safe split]")
    print("  Method: participant-disjoint iid/pid split")
    print(f"  Train rows: {len(train_idx)} | Test rows: {len(test_idx)} | Discarded cross-side rows: {discarded}")
    print(f"  Train positive rate: {df.iloc[train_idx]['match'].mean():.4f}")
    print(f"  Test positive rate : {df.iloc[test_idx]['match'].mean():.4f}")

    return train_idx, test_idx

def stratified_train_test_indices(y, test_size=0.3, seed=42):
    """
    Stratified split indices, without directly slicing X.
    This allows first obtaining train_idx / test_idx,
    then using only the train set to compute leakage-safe iid/pid means.
    """
    np.random.seed(seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    pos_test_size = int(len(pos_idx) * test_size)
    neg_test_size = int(len(neg_idx) * test_size)

    pos_test  = pos_idx[:pos_test_size]
    pos_train = pos_idx[pos_test_size:]

    neg_test  = neg_idx[:neg_test_size]
    neg_train = neg_idx[neg_test_size:]

    train_idx = np.concatenate([pos_train, neg_train])
    test_idx  = np.concatenate([pos_test,  neg_test])

    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    return train_idx, test_idx

def stratified_train_test_split(X, y, test_size=0.3, seed=42):
    """
    Stratified split (manual implementation, does not rely on sklearn).
    Ensure that the proportion of match=1 and match=0 in training/test sets is exactly the same as in the original data.
    """
    np.random.seed(seed)
    # Separate indices of positive and negative samples
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # Shuffle positive and negative samples separately
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)

    # Split test set by proportion
    pos_test_size = int(len(pos_idx) * test_size)
    neg_test_size = int(len(neg_idx) * test_size)

    # Build test and training set indices
    pos_test = pos_idx[:pos_test_size]
    pos_train = pos_idx[pos_test_size:]
    neg_test = neg_idx[:neg_test_size]
    neg_train = neg_idx[neg_test_size:]

    # Merge indices
    train_idx = np.concatenate([pos_train, neg_train])
    test_idx = np.concatenate([pos_test, neg_test])

    # Shuffle final indices
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)

    # Return split data
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ============================================================
# STEP 7: Save preprocessing results (for subsequent modeling)
# ============================================================

def save_preprocessed_data(X_train, X_test, y_train, y_test,
                            feature_names, scaler_params, output_dir=OUTPUT_DIR):
    """Save preprocessing results as .npy and .txt files for direct loading by modeling scripts"""
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_test.npy'),  X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_test.npy'),  y_test)
    np.save(os.path.join(output_dir, 'scaler_mean.npy'), scaler_params['mean'])
    np.save(os.path.join(output_dir, 'scaler_std.npy'),  scaler_params['std'])

    with open(os.path.join(output_dir, 'feature_names.txt'), 'w', encoding='utf-8') as f:
        for name in feature_names:
            f.write(name + '\n')

    print(f"\n  Preprocessing results saved to {output_dir}/")
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"  y_train positive rate: {y_train.mean():.2%}, y_test positive rate: {y_test.mean():.2%}")


# ============================================================
# Main function
# ============================================================

def main():
    # ── Please modify to local CSV path ─────────────────────────────────
    FILE_PATH = os.path.join(_HERE, "Speed Dating Data.csv")
    # ─────────────────────────────────────────────────────────

    print("=" * 60)
    print("Speed-Dating Match Prediction - Data Preprocessing & Feature Engineering")
    print("Group 12 | AI3013 Machine Learning")
    print("=" * 60)
    clear_existing_figures()

    # Step 1: Load and explore
    df = load_and_explore(FILE_PATH)

    # Step 2: Data cleaning
    df = clean_data(df)

    # Step 3: Within-person standardization (GPB) + Interest feature engineering (core innovation)
    print("\n" + "="*60)
    print("STEP 3: Within-person standardization (GPB) + Interest feature engineering (core innovation)")
    print("="*60)

    df = compute_gpb(df)
    df, z_cols = compute_interest_zscores(df)

    # Fix train/test split before group-stat features.
    # Partner lookup and distance normalization are fit on train rows only.
    split_y = df['match'].values.astype(int)
    train_idx, test_idx = leakage_safe_train_test_indices(
        df, test_size=0.3, seed=42
    )

    lookup = build_partner_interest_lookup(df.iloc[train_idx], z_cols)
    df = compute_pair_similarity_features(
        df,
        z_cols,
        lookup,
        train_idx=train_idx,
        reference_df=df.iloc[train_idx]
    )
    visualize_similarity_features(df)

    # Step 4: Interaction feature construction
    print("\n" + "="*60)
    print("STEP 4: Interaction feature construction")
    print("="*60)
    df = build_interaction_features(df)
    visualize_interaction_features(df)

    # Step 5: Feature correlation analysis
    all_candidate = (
        ['gender', 'female', 'age', 'age_o', 'age_diff', 'samerace', 'female_samerace',
         'attractive', 'sincere', 'intelligence', 'funny', 'ambition', 'avg_trait_score',
         'attractive_o', 'sinsere_o', 'intelligence_o', 'funny_o', 'ambitous_o',
         'shared_interests_o', 'partner_avg_trait',
         'like', 'guess_prob_liked', 'met',
         'SIS', 'interest_cosine', 'interest_euclidean_sim', 'interests_correlate',
         'gender_attractive', 'gender_sincere', 'gender_intelligence',
         'gender_funny', 'gender_ambition',
         'female_samerace', 'male_SIS',
         'like_x_guess_prob_liked',
         'like_x_shared_interests_o',
         'attractive_pair',
         'funny_pair',
         'trait_pair',
         'attractive_gap',
         'funny_gap',
         'trait_gap',
         'like_guess_gap']
    )
    corr_series = analyze_feature_correlation(df, all_candidate)

    # Step 6: Build feature matrix
    X_train, X_test, y_train, y_test, feature_names, scaler_params = \
        build_feature_matrix(df, z_cols, train_idx=train_idx, test_idx=test_idx)

    # Step 7: Save
    print("\n" + "="*60)
    print("STEP 7: Save preprocessing results")
    print("="*60)
    save_preprocessed_data(X_train, X_test, y_train, y_test,
                            feature_names, scaler_params)

    # ── Save train-only feature correlation coefficients ──────────────────────────
    print("\n[Feature correlation - based on training set]")
    corrs = np.array([
        abs(np.corrcoef(X_train[:, j], y_train)[0, 1])
        for j in range(X_train.shape[1])
    ])
    order = np.argsort(corrs)[::-1]
    corr_save_path = os.path.join(OUTPUT_DIR, "feature_correlations.txt")
    with open(corr_save_path, "w", encoding="utf-8") as f:
        f.write("rank,feature,pearson_r\n")
        for rank, idx in enumerate(order, 1):
            f.write(f"{rank},{feature_names[idx]},{corrs[idx]:.6f}\n")
            print(f"  {rank:2d}. {feature_names[idx]:35s} |r|={corrs[idx]:.4f}")
    print(f"[Saved] {corr_save_path}")

    print("\n" + "="*60)
    print("✅ Data Preprocessing & Feature Engineering completed!")
    print(f"   Generated 10 analysis charts, saved in {OUTPUT_DIR}/")
    print(f"   X_train shape: {X_train.shape}")
    print(f"   Feature list: {feature_names}")
    print("="*60)
    print("\nNext step modeling can directly load:")
    print("  X_train = np.load('data_outputs/X_train.npy')")
    print("  feature_names = open('data_outputs/feature_names.txt').read().splitlines()")

    return X_train, X_test, y_train, y_test, feature_names, scaler_params


if __name__ == "__main__":
    main()