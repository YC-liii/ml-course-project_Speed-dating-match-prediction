# ============================================================
# lr.py  --  Speed Dating LR From Scratch
#   Uses ML_data.py pipeline + Speed Dating Data.csv
#   Caches data_outputs/*.npy; auto-runs preprocessing if missing
# ============================================================
import os
import sys
import warnings
import importlib.util
import time
import hashlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'

# ── Fix random seed for reproducibility ──────────────────────────────
SEED = 42
np.random.seed(SEED)

# ============================================================
# Path config (all relative to script directory)
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV     = os.path.join(_HERE, "Speed Dating Data.csv")
PREPROC_PY   = os.path.join(_HERE, "ML_data.py")
CACHE_DIR    = os.path.join(_HERE, "data_outputs")
CACHE_FILES  = {
    'X_train':       os.path.join(CACHE_DIR, 'X_train.npy'),
    'X_test':        os.path.join(CACHE_DIR, 'X_test.npy'),
    'y_train':       os.path.join(CACHE_DIR, 'y_train.npy'),
    'y_test':        os.path.join(CACHE_DIR, 'y_test.npy'),
    'feature_names': os.path.join(CACHE_DIR, 'feature_names.txt'),
}
EXPERIMENT_LOG = os.path.join(_HERE, "experiment_run_log.md")

def append_experiment_log(title, lines):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n## {timestamp} - {title}\n")
        for line in lines:
            f.write(f"- {line}\n")

# ============================================================
# Old Fixed Correlation 18 Features
#   Source: core features consistently selected by correlation_selection in nested CV
#   Used for Pipeline C (baseline comparison with new features)
# ============================================================
CORR_18_FEATURES = [
    'like', 'funny', 'funny_o', 'avg_trait_score', 'partner_avg_trait',
    'attractive', 'attractive_o', 'shared_interests_o', 'guess_prob_liked',
    'intelligence', 'intelligence_o', 'sincere', 'sinsere_o',
    'ambition', 'ambitous_o', 'age_diff', 'gender_attractive', 'gender_funny',
]


# ============================================================
# 0. Data loading (cache first, otherwise run preprocessing pipeline)
# ============================================================
def _load_from_cache():
    X_train = np.load(CACHE_FILES['X_train'])
    X_test  = np.load(CACHE_FILES['X_test'])
    y_train = np.load(CACHE_FILES['y_train'])
    y_test  = np.load(CACHE_FILES['y_test'])
    with open(CACHE_FILES['feature_names'], encoding='utf-8') as f:
        feature_names = [ln.strip() for ln in f if ln.strip()]
    return X_train, X_test, y_train, y_test, feature_names


#  Speed Dating Data.csv abbreviated column names -> cleaned names expected by preprocessing
#  Note: ML_data.py has typos (sinsere_o / ambitous_o / intellicence /
#  ambtition); we keep these spellings to match upstream code.
COLUMN_RENAME = {
    # Self-rating of partner
    'attr': 'attractive', 'sinc': 'sincere', 'intel': 'intelligence',
    'fun':  'funny',       'amb':  'ambition', 'shar': 'shared_interests',
    # Partner's rating of self
    'attr_o':  'attractive_o',  'sinc_o':  'sinsere_o',
    'intel_o': 'intelligence_o', 'fun_o':  'funny_o',
    'amb_o':   'ambitous_o',    'shar_o': 'shared_interests_o',
    # Partner's stated ideal-partner preferences
    'pf_o_att': 'attractive_partner', 'pf_o_sin': 'sincere_partner',
    'pf_o_int': 'intelligence_partner','pf_o_fun': 'funny_partner',
    'pf_o_amb': 'ambition_partner',   'pf_o_sha': 'shared_interests_partner',
    # Self-reported importance weights (100-point allocation)
    'attr1_1':  'attractive_important',  'sinc1_1':  'sincere_important',
    'intel1_1': 'intellicence_important','fun1_1':   'funny_important',
    'amb1_1':   'ambtition_important',   'shar1_1':  'shared_interests_important',
    # Misc
    'prob':     'guess_prob_liked',
    'int_corr': 'interests_correlate',
}


def _load_and_rename(csv_path):
    """Load raw Speed Dating Data.csv, rename columns, clean income commas."""
    import pandas as pd
    # latin-1 is portable across OS / Python builds (gbk codec is not
    # always available on Linux). Non-ASCII bytes only appear in a few
    # school-name fields that are not used as features.
    df = pd.read_csv(csv_path, encoding='latin-1')
    df = df.rename(columns=COLUMN_RENAME)

    # income is string "69,487.00", convert to float
    if 'income' in df.columns and df['income'].dtype == object:
        df['income'] = (df['income'].astype(str)
                        .str.replace(',', '', regex=False)
                        .str.strip()
                        .replace({'': None, 'nan': None}))
        df['income'] = pd.to_numeric(df['income'], errors='coerce')

    # gender is already 0/1 int (female=0, male=1 per codebook)
    # ML_data.build_interaction_features skips encoding for non-object dtype, OK

    print(f"  [Rename] Renamed {len(COLUMN_RENAME)} abbreviated columns -> cleaned names")
    print(f"  [Data]   原始形状: {df.shape}, income dtype={df['income'].dtype if 'income' in df.columns else 'N/A'}")
    return df


def _run_preprocess_pipeline():
    """动态Loading ML_data.py 并执行全管道."""
    print("\n[Preprocess] Cache missing, running ML_data.py full pipeline...")
    spec = importlib.util.spec_from_file_location("preprocess2", PREPROC_PY)
    pp = importlib.util.module_from_spec(spec)

    # Set script dir as CWD so ML_data can find CSV and output dirs
    old_cwd = os.getcwd()
    os.chdir(_HERE)
    try:
        spec.loader.exec_module(pp)

        # ⚠️ 不走 pp.load_and_explore (它期望旧列名); 自己 load + rename 再传给后续步骤
        df = _load_and_rename(DATA_CSV)

        # 保留 pp.load_and_explore 里的缺失率/分布图? 用 rename 后的 df 跑一遍即可
        try:
            # Print key info only, skip redundant plots (iid=1 同 id=1 会覆盖, 无害)
            print(f"  [EDA]   match 分布: {df['match'].value_counts().to_dict()}")
            print(f"  [EDA]   特征总数: {df.shape[1]}")
        except Exception:
            pass

        df = pp.clean_data(df)
        df = pp.compute_gpb(df)
        df, z_cols = pp.compute_interest_zscores(df)
        lookup = pp.build_partner_interest_lookup(df, z_cols)
        df = pp.compute_pair_similarity_features(df, z_cols, lookup)
        df = pp.build_interaction_features(df)

        X_train, X_test, y_train, y_test, feature_names, scaler_params = \
            pp.build_feature_matrix(df, z_cols)

        pp.save_preprocessed_data(X_train, X_test, y_train, y_test,
                                   feature_names, scaler_params)
    finally:
        os.chdir(old_cwd)

    return X_train, X_test, y_train, y_test, feature_names


def load_data_smart(force_rebuild=False):
    """Cache first; if all cached -> load directly; otherwise run full preprocessing pipeline."""
    have_cache = all(os.path.exists(p) for p in CACHE_FILES.values())
    if have_cache and not force_rebuild:
        print(f"[Cache] Loading {CACHE_DIR}/*.npy")
        return _load_from_cache()
    return _run_preprocess_pipeline()


# ============================================================
# 0b. Split / Cache Fingerprint Utilities
# ============================================================
def array_fingerprint(arr):
    arr = np.ascontiguousarray(arr)
    h = hashlib.md5()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def print_split_fingerprint(X_train, X_test, y_train, y_test, feature_names, tag=""):
    print("\n" + "=" * 60)
    print(f"  SPLIT / CACHE FINGERPRINT CHECK {tag}")
    print("=" * 60)
    print("X_train shape:", X_train.shape)
    print("X_test shape :", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape :", y_test.shape)
    print("Train positive ratio:", f"{np.mean(y_train):.4f}")
    print("Test positive ratio :", f"{np.mean(y_test):.4f}")
    print("Train positive count:", int(np.sum(y_train)))
    print("Test positive count :", int(np.sum(y_test)))
    print("X_train fp:", array_fingerprint(X_train))
    print("X_test fp :", array_fingerprint(X_test))
    print("y_train fp:", array_fingerprint(y_train))
    print("y_test fp :", array_fingerprint(y_test))
    feature_text = "|".join(feature_names)
    feature_fp = hashlib.md5(feature_text.encode()).hexdigest()
    print("feature_names fp:", feature_fp)
    print("=" * 60)


# ============================================================
# 1. Data Audit
# ============================================================
def check_data_quality(X, y, feature_names):
    print("\n" + "=" * 55)
    print("  DATA AUDIT")
    print("=" * 55)
    issues = []

    if np.isnan(X).sum() > 0:
        issues.append(f"  [WARN] {np.isnan(X).sum()} NaN values in X")
    else:
        print("  [OK]   No missing values")

    bad_mean = int(np.sum(np.abs(np.mean(X, axis=0)) > 0.05))
    bad_std  = int(np.sum(np.abs(np.std(X, axis=0) - 1.0) > 0.1))
    print("  [OK]   Mean normalisation passed" if bad_mean == 0
          else f"  [WARN] {bad_mean} cols: |mean| > 0.05")
    print("  [OK]   Std normalisation passed" if bad_std == 0
          else f"  [WARN] {bad_std} cols: |std-1| > 0.1")

    # 新预处理的 gender 交互特征命名: gender_attractive / gender_sincere / gender_intelligence
    expected = ['gender_attractive', 'gender_sincere', 'gender_intelligence']
    missing = set(expected) - set(feature_names)
    if missing:
        issues.append(f"  [WARN] missing interaction features: {missing}")
    else:
        print("  [OK]   All 3 gender interaction features present")

    # 新预处理的核心创新特征
    innovation = ['SIS', 'interest_cosine', 'interest_euclidean_sim']
    have_inn = [f for f in innovation if f in feature_names]
    if len(have_inn) == len(innovation):
        print(f"  [OK]   Innovation features present: {have_inn}")
    else:
        print(f"  [WARN] Some innovation features missing: "
              f"{set(innovation) - set(have_inn)}")

    pos_r = float(np.mean(y))
    print(f"  [INFO] Positive-class ratio: {pos_r:.2%}  "
          f"(imbalance ~1:{int(round((1 - pos_r) / max(pos_r, 1e-9)))})")
    for iss in issues:
        print(iss)
    if not issues:
        print("  [PASS] All checks passed")
    print("=" * 55)
    return len(issues) == 0


def regularization_redundancy_check(X, y, feature_names, lambdas=None):
    """L1/L2 penalty path using scratch LR — checks feature redundancy."""
    if lambdas is None:
        lambdas = [0.001, 0.01, 0.1, 1.0, 10.0]
    print("\n[Regularization Redundancy Check]")
    print(f"  {'lambda':>8}  {'L1 nonzero':>12}  {'L2 ||w||':>10}")
    for lam in lambdas:
        m1 = LogisticRegressionScratch(lr=0.05, lambda_=lam, n_epochs=150,
                                        batch_size=128, momentum=0.9,
                                        penalty='l1')
        m1.fit(X, y)
        m2 = LogisticRegressionScratch(lr=0.05, lambda_=lam, n_epochs=150,
                                        batch_size=128, momentum=0.9,
                                        penalty='l2')
        m2.fit(X, y)
        nz  = int(np.sum(np.abs(m1.coef_) > 1e-4))
        l2n = float(np.linalg.norm(m2.coef_))
        print(f"  {lam:>8.3f}  {nz:>12d}  {l2n:>10.4f}")


# ============================================================
# 2a. Feature Selection - L1+L2 Regularization Fusion (from scratch)
# ============================================================
def l1_l2_fusion_selection(X, y, feature_names, plot=True):
    """
    Fusion strategy (pure NumPy, no sklearn):
      - L1 as hard gate: coef==0 features are eliminated (subgradient descent)
      - L2 as soft ranking: coefficient magnitude reflects true importance
      - Fusion score = L1 survival mask x |L2 coefficient|
    Auto-search lambda to keep 50-75% features via L1.
    """
    n_feat = X.shape[1]
    target_lo, target_hi = 0.50, 0.75

    lam_vals = np.logspace(-2, 2, 30)
    chosen_lam = lam_vals[0]
    for lam in lam_vals:
        m = LogisticRegressionScratch(lr=0.05, lambda_=lam, n_epochs=150,
                                       batch_size=128, momentum=0.9,
                                       penalty='l1')
        m.fit(X, y)
        ratio = np.mean(np.abs(m.coef_) > 1e-4)
        if target_lo <= ratio <= target_hi:
            chosen_lam = lam
            break

    m_l1 = LogisticRegressionScratch(lr=0.05, lambda_=chosen_lam, n_epochs=200,
                                      batch_size=128, momentum=0.9,
                                      penalty='l1')
    m_l1.fit(X, y)
    m_l2 = LogisticRegressionScratch(lr=0.05, lambda_=chosen_lam, n_epochs=200,
                                      batch_size=128, momentum=0.9,
                                      penalty='l2')
    m_l2.fit(X, y)

    l1_mask  = np.abs(m_l1.coef_) > 1e-4
    l2_score = np.abs(m_l2.coef_)
    fusion   = l1_mask.astype(float) * l2_score

    order     = np.argsort(fusion)[::-1]
    sel_idx   = order[fusion[order] > 0]
    sel_names = [feature_names[i] for i in sel_idx]

    print(f"\n[L1+L2 Fusion]  lambda={chosen_lam:.4f}  "
          f"L1 kept {l1_mask.sum()}/{n_feat} -> fusion kept {len(sel_idx)}")
    print(f"  Selected: {sel_names}")

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        nz_counts = []
        for lam in lam_vals:
            m = LogisticRegressionScratch(lr=0.05, lambda_=lam, n_epochs=100,
                                           batch_size=128, momentum=0.9,
                                           penalty='l1')
            m.fit(X, y)
            nz_counts.append(int(np.sum(np.abs(m.coef_) > 1e-4)))
        axes[0].semilogx(lam_vals, nz_counts, marker='o', ms=4,
                         color='#8e44ad', lw=2)
        axes[0].axvline(chosen_lam, color='red', ls='--',
                        label=f'chosen lambda={chosen_lam:.3f}')
        axes[0].set_xlabel('lambda (regularisation strength)')
        axes[0].set_ylabel('L1 non-zero features')
        axes[0].set_title('L1 Sparsity Path')
        axes[0].legend(); axes[0].grid(alpha=0.3)

        top20 = order[:min(20, len(feature_names))]
        colors = ['#e74c3c' if i in sel_idx else '#bdc3c7' for i in top20]
        axes[1].bar(range(len(top20)), fusion[top20], color=colors)
        axes[1].set_xticks(range(len(top20)))
        axes[1].set_xticklabels([feature_names[i] for i in top20],
                                rotation=90, fontsize=7)
        axes[1].set_ylabel('Fusion Score (L1 gate x |L2 coef|)')
        axes[1].set_title('L1+L2 Fusion Feature Score (top 20)')

        plt.tight_layout()
        plt.savefig(os.path.join(CACHE_DIR, '11_l1_l2_fusion_selection.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    return sel_idx, sel_names


# ============================================================
# 2b. Feature Selection - Correlation Filter (from scratch)
# ============================================================
def correlation_selection(X, y, feature_names, plot=True):
    """Pearson correlation filter — ranks features by |correlation| with target.
    Keeps features with |r| above the median of all |r| values (adaptive threshold).
    Pure NumPy, no sklearn.
    """
    n_feat = X.shape[1]
    correlations = np.zeros(n_feat)
    for j in range(n_feat):
        xj = X[:, j]
        # Pearson correlation = cov(x,y) / (std(x)*std(y))
        xm, ym = xj - xj.mean(), y - y.mean()
        num = np.dot(xm, ym)
        den = np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))
        correlations[j] = np.abs(num / den) if den > 1e-12 else 0.0

    order = np.argsort(correlations)[::-1]
    sorted_corr = correlations[order]
    sorted_fn   = [feature_names[i] for i in order]

    # Adaptive threshold: keep features with |r| > median of all |r|
    threshold = np.median(sorted_corr)
    keep_mask = sorted_corr > threshold
    n_keep    = int(np.sum(keep_mask))
    # Fallback: keep at least 3 features
    if n_keep < 3:
        n_keep = min(3, n_feat)
    sel_idx   = order[:n_keep]
    sel_names = [feature_names[i] for i in sel_idx]

    print(f"\n[Correlation]  {n_feat} features -> {len(sel_names)} selected  "
          f"(|r| > {threshold:.4f})")
    print(f"  Selected: {sel_names}")

    if plot:
        fig, ax = plt.subplots(figsize=(12, 5))
        colors = ['#27ae60' if i < n_keep else '#bdc3c7'
                  for i in range(len(sorted_fn))]
        ax.bar(range(len(sorted_fn)), sorted_corr, color=colors)
        ax.axhline(threshold, color='black', ls='--', lw=1,
                   label=f'threshold = {threshold:.4f}')
        ax.set_xticks(range(len(sorted_fn)))
        ax.set_xticklabels(sorted_fn, rotation=90, fontsize=7)
        ax.set_ylabel('|Pearson r| with match')
        ax.set_title('Correlation Filter Feature Ranking')
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(CACHE_DIR, '16_correlation_filter.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    return sel_idx, sel_names


# ============================================================
# 3. LR From Scratch  (NumPy only)
#    Mini-batch SGD + Momentum + L2 reg + Class Weight
#    theta = [bias, w1, w2, ...]   bias excluded from L2
# ============================================================
class LogisticRegressionScratch:

    def __init__(self, lr=0.01, lambda_=0.01, n_epochs=200,
                 batch_size=64, momentum=0.9, class_weight='balanced',
                 penalty='l2', pos_weight_scale=1.0):
        self.lr               = lr
        self.lambda_          = lambda_
        self.n_epochs         = n_epochs
        self.batch_size       = batch_size
        self.momentum         = momentum
        self.class_weight     = class_weight
        self.penalty          = penalty  # 'l1' or 'l2'
        self.pos_weight_scale = pos_weight_scale
        self.theta            = None
        self.train_losses     = []
        self.val_losses       = []
        self.train_time       = 0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def _sample_weights(self, y):
        if self.class_weight == 'balanced':
            n     = len(y)
            n_pos = float(np.sum(y == 1)) + 1e-9
            n_neg = float(np.sum(y == 0)) + 1e-9
            w_pos = n / (2.0 * n_pos) * self.pos_weight_scale
            w_neg = n / (2.0 * n_neg)
            return np.where(y == 1, w_pos, w_neg)
        return np.ones(len(y))

    def _logloss(self, Xb, y, theta):
        """Regularized log-loss; bias (theta[0]) excluded from penalty."""
        n   = len(y)
        p   = self._sigmoid(Xb @ theta)
        eps = 1e-12
        wts = self._sample_weights(y)
        ll  = -np.mean(wts * (y * np.log(p + eps) +
                               (1 - y) * np.log(1 - p + eps)))
        if self.penalty == 'l1':
            reg = (self.lambda_ / n) * float(np.sum(np.abs(theta[1:])))
        else:
            reg = (self.lambda_ / (2.0 * n)) * float(np.sum(theta[1:] ** 2))
        return ll + reg

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        start_time = time.time()  # 计时开始
        n, d       = X_train.shape
        Xb         = np.c_[np.ones(n), X_train]
        self.theta = np.zeros(d + 1)
        velocity   = np.zeros(d + 1)
        Xbv        = (None if X_val is None
                      else np.c_[np.ones(len(X_val)), X_val])

        for _ in range(self.n_epochs):
            perm = np.random.permutation(n)
            Xbs  = Xb[perm];  ys = y_train[perm]

            for s in range(0, n, self.batch_size):
                Xbi = Xbs[s:s + self.batch_size]
                ybi = ys[s:s + self.batch_size]
                nb  = len(ybi)
                p   = self._sigmoid(Xbi @ self.theta)
                err = self._sample_weights(ybi) * (p - ybi)
                grad      = Xbi.T @ err / nb
                if self.penalty == 'l1':
                    grad[1:] += (self.lambda_ / n) * np.sign(self.theta[1:])
                else:
                    grad[1:] += (self.lambda_ / n) * self.theta[1:]
                velocity    = self.momentum * velocity + self.lr * grad
                self.theta -= velocity

            self.train_losses.append(
                self._logloss(Xb, y_train, self.theta))
            if Xbv is not None:
                self.val_losses.append(
                    self._logloss(Xbv, y_val, self.theta))
        
        self.train_time = time.time() - start_time  # 计时结束
        return self

    def predict_proba(self, X):
        return self._sigmoid(np.c_[np.ones(len(X)), X] @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def coef_(self):      return self.theta[1:]

    @property
    def intercept_(self): return self.theta[0]


# ============================================================
# 4. Evaluation Suite  (no sklearn metrics allowed)
# ============================================================
def _accuracy(yt, yp):
    return float(np.mean(yt == yp))

def _precision(yt, yp):
    tp = float(np.sum((yp == 1) & (yt == 1)))
    fp = float(np.sum((yp == 1) & (yt == 0)))
    return tp / (tp + fp + 1e-12)

def _recall(yt, yp):
    tp = float(np.sum((yp == 1) & (yt == 1)))
    fn = float(np.sum((yp == 0) & (yt == 1)))
    return tp / (tp + fn + 1e-12)

def _f1(yt, yp):
    p = _precision(yt, yp);  r = _recall(yt, yp)
    return 2 * p * r / (p + r + 1e-12)


def find_best_threshold(y_true, y_prob):
    """
    Search for optimal classification threshold on validation set.
    Objective: maximize F1-score.
    Pure NumPy implementation, no sklearn.
    """
    thresholds = np.linspace(0.05, 0.95, 181)

    best_t  = 0.5
    best_f1 = -1.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        score  = _f1(y_true, y_pred)

        if score > best_f1:
            best_f1 = score
            best_t  = t

    return best_t, best_f1

def _auc_trapezoid(yt, y_prob):
    """ROC-AUC via trapezoidal rule."""
    thresholds = np.linspace(0, 1, 300)[::-1]
    fprs, tprs = [], []
    for t in thresholds:
        yp  = (y_prob >= t).astype(int)
        tp  = float(np.sum((yp == 1) & (yt == 1)))
        fp  = float(np.sum((yp == 1) & (yt == 0)))
        fn  = float(np.sum((yp == 0) & (yt == 1)))
        tn  = float(np.sum((yp == 0) & (yt == 0)))
        fprs.append(fp / (fp + tn + 1e-12))
        tprs.append(tp / (tp + fn + 1e-12))
    fprs = np.array(fprs);  tprs = np.array(tprs)
    order = np.argsort(fprs)
    return float(np.trapz(tprs[order], fprs[order]))

def _confusion_matrix(yt, yp):
    tn = int(np.sum((yp == 0) & (yt == 0)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    tp = int(np.sum((yp == 1) & (yt == 1)))
    return np.array([[tn, fp], [fn, tp]])

def full_eval(yt, yp, y_prob, label=''):
    acc = _accuracy(yt, yp);   pre = _precision(yt, yp)
    rec = _recall(yt, yp);     f1s = _f1(yt, yp)
    auc = _auc_trapezoid(yt, y_prob)
    print(f"\n{'─' * 45}")
    if label:
        print(f"  {label}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {pre:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1s:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"{'─' * 45}")
    return dict(acc=acc, pre=pre, rec=rec, f1=f1s, auc=auc)


# ============================================================
# 5. Hand-written Stratified 5-Fold CV + Grid Search
# ============================================================
def stratified_kfold_indices(y, k=5, seed=42):
    """
    Hand-written Stratified K-Fold.
    Ensures each fold has similar positive/negative ratio as the full dataset.
    """
    rng = np.random.default_rng(seed)
    y   = np.asarray(y)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)

    folds = []
    for i in range(k):
        val_idx = np.concatenate([pos_folds[i], neg_folds[i]])

        train_idx = np.concatenate([
            np.concatenate([pos_folds[j] for j in range(k) if j != i]),
            np.concatenate([neg_folds[j] for j in range(k) if j != i])
        ])

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)

        folds.append((train_idx, val_idx))

    return folds


def kfold_cv(X, y, params, k=5, seed=42):
    """
    Stratified K-Fold CV + threshold tuning。
    每一折：
      1. 用训练折训练模型
      2. 在验证折上预测概率
      3. 在验证折上搜索最佳 threshold
      4. 用该 threshold 计算 F1
    返回：
      平均 F1、平均最佳 threshold
    """
    folds = stratified_kfold_indices(y, k=k, seed=seed)

    scores     = []
    thresholds = []

    for train_idx, val_idx in folds:
        model = LogisticRegressionScratch(**params)
        model.fit(X[train_idx], y[train_idx])

        val_prob        = model.predict_proba(X[val_idx])
        best_t, best_f1 = find_best_threshold(y[val_idx], val_prob)

        scores.append(best_f1)
        thresholds.append(best_t)

    return float(np.mean(scores)), float(np.mean(thresholds))


def grid_search(X, y, param_grid, k=5, seed=42):
    lrs     = param_grid.get('lr',         [0.01])
    lams    = param_grid.get('lambda_',    [0.01])
    batches = param_grid.get('batch_size', [64])

    total = len(lrs) * len(lams) * len(batches)

    folds              = stratified_kfold_indices(y, k=k, seed=seed)
    overall_pos_ratio  = float(np.mean(y))
    fold_ratios        = [float(np.mean(y[val_idx])) for _, val_idx in folds]

    print(f"\n[Grid Search]  {total} combos x {k}-Fold Stratified CV + Threshold Tuning")
    print(f"  Overall positive ratio: {overall_pos_ratio:.2%}")
    print("  Per-fold positive ratio: " +
          "  ".join([f"fold{i+1}={r:.2%}" for i, r in enumerate(fold_ratios)]))

    best_f1       = -1.0
    best_params   = {}
    best_threshold = 0.5
    done          = 0

    for lr in lrs:
        for lam in lams:
            for bs in batches:
                params = dict(
                    lr=lr,
                    lambda_=lam,
                    batch_size=bs,
                    n_epochs=100,
                    momentum=0.9
                )

                score, threshold = kfold_cv(X, y, params, k=k, seed=seed)

                done += 1
                print(
                    f"  [{done:2d}/{total}] "
                    f"lr={lr}  lam={lam}  bs={bs}"
                    f"  -> F1={score:.4f}  threshold={threshold:.3f}"
                )

                if score > best_f1:
                    best_f1        = score
                    best_params    = params.copy()
                    best_threshold = threshold

    print(f"\n  Best F1={best_f1:.4f}")
    print(f"  Best params={best_params}")
    print(f"  Best CV threshold={best_threshold:.3f}")

    return best_params, best_threshold, best_f1


def grid_search_nested_fs(X, y, feature_names, sel_fn, param_grid, k=5, seed=42):
    """
    Nested feature selection inside Stratified K-Fold CV.

    每个 CV fold 内部：
      1. 只用 train fold 做 feature selection
      2. 用选出来的特征训练 LR
      3. 在 val fold 上预测
      4. 在 val fold 上调 threshold
      5. 记录 F1 和 threshold

    Validation fold does not participate in feature selection, making CV scores more rigorous.
    """
    lrs     = param_grid.get('lr',         [0.01])
    lams    = param_grid.get('lambda_',    [0.01])
    batches = param_grid.get('batch_size', [64])

    total = len(lrs) * len(lams) * len(batches)

    folds = stratified_kfold_indices(y, k=k, seed=seed)

    overall_pos_ratio = float(np.mean(y))
    fold_ratios = [float(np.mean(y[val_idx])) for _, val_idx in folds]

    print(f"\n[Nested Grid Search]  {total} combos x {k}-Fold Stratified CV")
    print("  Feature selection is performed inside each CV fold")
    print(f"  Overall positive ratio: {overall_pos_ratio:.2%}")
    print("  Per-fold positive ratio: " +
          "  ".join([f"fold{i+1}={r:.2%}" for i, r in enumerate(fold_ratios)]))

    best_score     = -1.0
    best_params    = None
    best_threshold = 0.5

    count = 0

    for lr in lrs:
        for lam in lams:
            for bs in batches:
                count += 1

                params = {
                    'lr':           lr,
                    'lambda_':      lam,
                    'batch_size':   bs,
                    'n_epochs':     param_grid.get('n_epochs', [100])[0],
                    'momentum':     param_grid.get('momentum', [0.9])[0],
                    'penalty':      'l2',
                    'class_weight': 'balanced'
                }

                fold_f1s        = []
                fold_thresholds = []
                fold_n_features = []

                for train_idx, val_idx in folds:
                    X_fold_train = X[train_idx]
                    y_fold_train = y[train_idx]
                    X_fold_val   = X[val_idx]
                    y_fold_val   = y[val_idx]

                    # 核心：只在 train fold 上做特征选择，val fold 不参与
                    sel_idx, sel_names = sel_fn(
                        X_fold_train,
                        y_fold_train,
                        feature_names,
                        plot=False
                    )

                    X_fold_train_sel = X_fold_train[:, sel_idx]
                    X_fold_val_sel   = X_fold_val[:, sel_idx]

                    model = LogisticRegressionScratch(**params)
                    model.fit(X_fold_train_sel, y_fold_train)

                    val_prob        = model.predict_proba(X_fold_val_sel)
                    best_t, best_f1 = find_best_threshold(y_fold_val, val_prob)

                    fold_f1s.append(best_f1)
                    fold_thresholds.append(best_t)
                    fold_n_features.append(len(sel_idx))

                mean_f1         = float(np.mean(fold_f1s))
                mean_threshold  = float(np.mean(fold_thresholds))
                mean_n_features = float(np.mean(fold_n_features))

                print(
                    f"  [{count:>3}/{total}] "
                    f"lr={lr:<7} lambda={lam:<7} batch={bs:<4} "
                    f"CV-F1={mean_f1:.4f}  "
                    f"threshold={mean_threshold:.3f}  "
                    f"avg_features={mean_n_features:.1f}"
                )

                if mean_f1 > best_score:
                    best_score     = mean_f1
                    best_params    = params.copy()
                    best_threshold = mean_threshold

    print("\n[Nested Grid Search Best]")
    print(f"  Best CV F1      = {best_score:.4f}")
    print(f"  Best params     = {best_params}")
    print(f"  Best threshold  = {best_threshold:.3f}")

    return best_params, best_threshold, best_score


# ============================================================
# 6. Visualizations
# ============================================================
def plot_learning_curves(model, title='Learning Curves', fname='learning_curves.png'):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(model.train_losses, lw=2, color='#2980b9', label='Train Loss')
    if model.val_losses:
        ax.plot(model.val_losses, lw=2, ls='--',
                color='#e74c3c', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Weighted Log-Loss (L2 reg)')
    ax.set_title(title)
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()


def plot_3d_loss_surface(X, y, model, feat_idx=(0, 1), fname='loss_surface_3d.png'):
    """Vary two weight dimensions; fix the rest at their fitted values."""
    i, j    = feat_idx[0] + 1, feat_idx[1] + 1
    theta0  = model.theta.copy()
    grid    = np.linspace(-3, 3, 35)
    Z       = np.zeros((35, 35))
    Xb      = np.c_[np.ones(len(X)), X]
    eps     = 1e-12
    wts     = model._sample_weights(y)

    for ii, wi in enumerate(grid):
        for jj, wj in enumerate(grid):
            t = theta0.copy(); t[i] = wi; t[j] = wj
            p = model._sigmoid(Xb @ t)
            Z[ii, jj] = float(-np.mean(
                wts * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))))

    W1, W2 = np.meshgrid(grid, grid)
    fig    = plt.figure(figsize=(10, 7))
    ax     = fig.add_subplot(111, projection='3d')
    surf   = ax.plot_surface(W1, W2, Z.T, cmap='viridis', alpha=0.85)
    ax.set_xlabel(f'w[feat {feat_idx[0]}]')
    ax.set_ylabel(f'w[feat {feat_idx[1]}]')
    ax.set_zlabel('Log-Loss')
    ax.set_title('3D Loss Surface (two-weight slice)')
    fig.colorbar(surf, shrink=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()


def plot_l2_regularization_path(X, y, feature_names, top_k=8, fname='l2_reg_path.png'):
    """Coefficient trajectories as lambda increases."""
    lambdas    = np.logspace(-3, 2, 25)
    coef_paths = []
    for lam in lambdas:
        m = LogisticRegressionScratch(lr=0.05, lambda_=lam, n_epochs=150,
                                      batch_size=128, momentum=0.9)
        m.fit(X, y)
        coef_paths.append(m.coef_.copy())
    coef_paths = np.array(coef_paths)

    top_idx = np.argsort(np.mean(np.abs(coef_paths), axis=0))[::-1][:top_k]
    cmap    = cm.get_cmap('tab10', top_k)

    fig, ax = plt.subplots(figsize=(9, 5))
    for rank, fi in enumerate(top_idx):
        ax.semilogx(lambdas, coef_paths[:, fi],
                    label=feature_names[fi], color=cmap(rank), lw=2)
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel('lambda (regularisation strength)')
    ax.set_ylabel('Coefficient value')
    ax.set_title('L2 Regularisation Path')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(yt, yp, title='Confusion Matrix', fname='confusion_matrix.png'):
    cm_arr = _confusion_matrix(yt, yp)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap='Blues', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    labels  = ['No Match (0)', 'Match (1)']
    ticks   = [0, 1]
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=20)
    ax.set_yticks(ticks); ax.set_yticklabels(labels)
    thresh = cm_arr.max() / 2.0
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm_arr[r, c]), ha='center', va='center',
                    fontsize=14,
                    color='white' if cm_arr[r, c] > thresh else 'black')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# 7. Main  (nested feature selection)
# ============================================================
def main():
    print("=" * 55)
    print("  Speed Dating -- LR From Scratch  (lr.py)")
    print("  Using ML_data.py + Speed Dating Data.csv")
    print("=" * 55)

    # ── Load data (cache preferred, else run preprocessing) ─
    X_train, X_test, y_train, y_test, feature_names = load_data_smart()
    print(f"\n[Data]  X_train={X_train.shape}  X_test={X_test.shape}  "
          f"features={len(feature_names)}")

    # ── 1. Data audit (on training set) ─────────────────────
    check_data_quality(X_train, y_train, feature_names)

    # 固定超参数搜索空间
    param_grid = {
        'lr':         [0.005, 0.01, 0.05],
        'lambda_':    [0.001, 0.01, 0.1],
        'batch_size': [64, 128],
    }

    pipelines = [
        ('L1+L2 Fusion', l1_l2_fusion_selection),
        ('Correlation Filter', correlation_selection),
    ]
    results = {}

    for pipe_name, sel_fn in pipelines:
        print("\n" + "=" * 55)
        print(f"  PIPELINE: {pipe_name}")
        print("=" * 55)

        # ── nested grid search: feature selection inside CV ──────────
        best_params, best_threshold, best_cv_f1 = grid_search_nested_fs(
            X_train,
            y_train,
            feature_names,
            sel_fn,
            param_grid,
            k=5,
            seed=42
        )

        # ── final feature selection on full training set ──────────────
        # 注意：这里用完整 X_train 做最终特征选择
        # 因为这是最终模型训练阶段，不再用于估计 CV 分数
        sel_idx, sel_names = sel_fn(
            X_train,
            y_train,
            feature_names,
            plot=True
        )

        Xtr = X_train[:, sel_idx]
        Xte = X_test[:, sel_idx]
        ytr, yte = y_train, y_test

        print(f"  Final selected features: {len(sel_names)}")
        print(f"  Train: {Xtr.shape}   Test: {Xte.shape}")

        regularization_redundancy_check(Xtr, ytr, sel_names)

        # ── final model ────────────────────────────────────
        best_params['n_epochs'] = 300
        best_params['momentum'] = 0.9
        model = LogisticRegressionScratch(**best_params)
        # 最终模型使用全部训练集训练，不再额外切 10% 验证集
        # threshold 已由 CV 选好，让模型吃满训练数据
        model.fit(Xtr, ytr)

        # ── evaluation ─────────────────────────────────────
        y_prob = model.predict_proba(Xte)
        y_pred = model.predict(Xte, threshold=best_threshold)

        print(f"  CV-selected threshold: {best_threshold:.3f}")
        print(f"  Best CV F1 with threshold tuning: {best_cv_f1:.4f}")

        m = full_eval(yte, y_pred, y_prob, label=f'{pipe_name} -- Test Set')
        
        # 打印训练耗时
        print(f"  Training Time: {model.train_time:.4f} seconds")
        
        results[pipe_name] = dict(metrics=m, model=model,
                                  sel_names=sel_names,
                                  Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte,
                                  train_time=model.train_time)

        # ── visualizations (pipeline-specific filenames, numbered 11-20) ───
        safe = pipe_name.replace('+', '').replace(' ', '_').lower()
        if pipe_name == 'L1+L2 Fusion':
            n_base = 12
        else:
            n_base = 17
        plot_learning_curves(model,
                             title=f'Learning Curves -- {pipe_name}',
                             fname=f'{n_base}_learning_curves_{safe}.png')
        plot_confusion_matrix(yte, y_pred,
                              title=f'Confusion Matrix -- {pipe_name}',
                              fname=f'{n_base+1}_confusion_matrix_{safe}.png')
        plot_l2_regularization_path(Xtr, ytr, sel_names,
                                     fname=f'{n_base+2}_l2_reg_path_{safe}.png')
        top2 = list(np.argsort(np.abs(model.coef_))[::-1][:2])
        plot_3d_loss_surface(Xte, yte, model, feat_idx=tuple(top2),
                              fname=f'{n_base+3}_loss_surface_3d_{safe}.png')

    # ── side-by-side comparison ────────────────────────────
    print("\n" + "=" * 55)
    print("  FINAL COMPARISON")
    print("=" * 55)
    keys = list(results.keys())
    print(f"  {'Metric':<12}" + ''.join(f"{k:>20}" for k in keys))
    print("  " + "─" * (12 + 20 * len(keys)))
    for metric in ['acc', 'pre', 'rec', 'f1', 'auc']:
        row = f"  {metric:<12}" + ''.join(
            f"{results[k]['metrics'][metric]:>20.4f}" for k in keys)
        print(row)
    
    # 添加训练时间对比
    print(f"  {'Train Time':<12}" + ''.join(
        f"{results[k]['train_time']:>20.4f}s" for k in keys))
    print("=" * 55)

    print("\nAll plots saved. Done.")
    return results


# ============================================================
# 8. Correlation Top-K Selection  (pure NumPy)
# ============================================================
def correlation_top_k_selection(X, y, feature_names, top_k=18):
    """
    按 |Pearson r| 从当前特征里选 Top K。
    用于 fixed-feature pipeline（不做 nested FS，直接固定特征）。
    """
    n_feat = X.shape[1]
    correlations = np.zeros(n_feat)

    for j in range(n_feat):
        xj  = X[:, j]
        xm  = xj - xj.mean()
        ym  = y  - y.mean()
        num = np.dot(xm, ym)
        den = np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))
        correlations[j] = abs(num / den) if den > 1e-12 else 0.0

    order     = np.argsort(correlations)[::-1]
    sel_idx   = order[:min(top_k, n_feat)]
    sel_names = [feature_names[i] for i in sel_idx]

    print(f"\n[Correlation Top-{top_k} Selection]")
    print(f"  Selected {len(sel_names)} / {n_feat} features")
    print("  Selected features:")
    for rank, idx in enumerate(sel_idx, 1):
        print(f"    {rank:2d}. {feature_names[idx]:30s} |r|={correlations[idx]:.4f}")

    return sel_idx, sel_names


# ============================================================
# 9. Grid Search — Fixed Features + pos_weight_scale
# ============================================================
def grid_search_fixed_features_weight(X, y, param_grid_fixed, k=5, seed=42):
    """
    Fixed-feature Grid Search: no feature selection, only tune hyperparams + pos_weight_scale.

    pos_weight_scale 含义：
      1.0  -> 原始 balanced weight
      >1.0 -> 更激进预测正类（提高 Recall）
      <1.0 -> 更保守（提高 Precision）
    """
    lrs    = param_grid_fixed.get('lr',               [0.01])
    lams   = param_grid_fixed.get('lambda_',          [0.01])
    bss    = param_grid_fixed.get('batch_size',        [64])
    epochs = param_grid_fixed.get('n_epochs',          [100])
    moms   = param_grid_fixed.get('momentum',          [0.9])
    pws    = param_grid_fixed.get('pos_weight_scale',  [1.0])

    total = (len(lrs) * len(lams) * len(bss)
             * len(epochs) * len(moms) * len(pws))
    folds = stratified_kfold_indices(y, k=k, seed=seed)

    overall_pos = float(np.mean(y))
    fold_ratios = [float(np.mean(y[vi])) for _, vi in folds]

    print(f"\n[Grid Search Fixed Features]  {total} combos x {k}-Fold Stratified CV")
    print(f"  Features={X.shape[1]}  Samples={X.shape[0]}")
    print(f"  Overall positive ratio: {overall_pos:.2%}")
    print("  Per-fold positive ratio: " +
          "  ".join([f"fold{i+1}={r:.2%}" for i, r in enumerate(fold_ratios)]))
    print(f"  pos_weight_scale search: {pws}")

    best_score     = -1.0
    best_params    = {}
    best_threshold = 0.5
    count          = 0

    for lr in lrs:
        for lam in lams:
            for bs in bss:
                for n_ep in epochs:
                    for mom in moms:
                        for pws_val in pws:
                            count += 1
                            params = dict(
                                lr=lr, lambda_=lam,
                                batch_size=bs, n_epochs=n_ep,
                                momentum=mom, penalty='l2',
                                class_weight='balanced',
                                pos_weight_scale=pws_val,
                            )
                            fold_f1s, fold_thrs = [], []
                            for tr_idx, va_idx in folds:
                                m = LogisticRegressionScratch(**params)
                                m.fit(X[tr_idx], y[tr_idx])
                                vp     = m.predict_proba(X[va_idx])
                                bt, bf = find_best_threshold(y[va_idx], vp)
                                fold_f1s.append(bf)
                                fold_thrs.append(bt)

                            mean_f1  = float(np.mean(fold_f1s))
                            mean_thr = float(np.mean(fold_thrs))

                            print(
                                f"  [{count:>3}/{total}] "
                                f"lr={lr:<6} lam={lam:<6} bs={bs:<4} "
                                f"ep={n_ep}  pws={pws_val:<4} "
                                f"-> CV-F1={mean_f1:.4f}  thr={mean_thr:.3f}"
                            )

                            if mean_f1 > best_score:
                                best_score     = mean_f1
                                best_params    = params.copy()
                                best_threshold = mean_thr

    print(f"\n[Grid Search Fixed Features Best]")
    print(f"  Best CV F1        = {best_score:.4f}")
    print(f"  Best threshold    = {best_threshold:.3f}")
    print(f"  Best params       = {best_params}")
    return best_params, best_threshold, best_score


# ============================================================
# 10. Main — Fixed Feature + pos_weight_scale Tuning
# ============================================================
def main_fixed_feature_weight_tuning():
    print("=" * 65)
    print("  lr.py  --  Fixed-Feature + pos_weight_scale Tuning")
    print("  Pipeline A: Full All Features")
    print("  Pipeline B: Re-selected Correlation Top18 (from latest cache)")
    print("  Pipeline C: Old Fixed Correlation 18 (lr_5.py baseline)")
    print("=" * 65)

    # ── Loading数据 ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = load_data_smart()

    print_split_fingerprint(
        X_train, X_test, y_train, y_test, feature_names,
        tag="[LR]"
    )

    print(f"\n[Data]  X_train={X_train.shape}  X_test={X_test.shape}  "
          f"features={len(feature_names)}")
    check_data_quality(X_train, y_train, feature_names)

    # ── Pipeline A: Full All Features ────────────────────────
    full_idx   = np.arange(X_train.shape[1])
    full_names = list(feature_names)

    # ── Pipeline B: Re-selected Correlation Top18 ────────────
    top18_idx, top18_names = correlation_top_k_selection(
        X_train, y_train, feature_names, top_k=18)

    # ── Pipeline C: Old Fixed Correlation 18 ─────────────────
    missing_old = [f for f in CORR_18_FEATURES if f not in feature_names]
    if missing_old:
        print(f"\n[WARN]  Old Fixed 18 中以下特征不在当前缓存: {missing_old}")
        old_use = [f for f in CORR_18_FEATURES if f in feature_names]
    else:
        old_use = CORR_18_FEATURES
    old_idx   = np.array([feature_names.index(f) for f in old_use])
    old_names = old_use

    pipelines = [
        ('Full All Features',        full_idx,  full_names),
        ('Re-selected Corr Top18',   top18_idx, top18_names),
        ('Old Fixed Corr18',         old_idx,   old_names),
    ]

    # ── Unified search space ──────────────────────────────────────────
    param_grid_fixed = {
        'lr':               [0.005, 0.01, 0.03],
        'lambda_':          [0.003, 0.01, 0.03, 0.1],
        'batch_size':       [64, 128],
        'n_epochs':         [100],
        'momentum':         [0.9],
        'pos_weight_scale': [0.7, 0.8, 0.9, 1.0],
    }
    n_combos = (len(param_grid_fixed['lr'])
                * len(param_grid_fixed['lambda_'])
                * len(param_grid_fixed['batch_size'])
                * len(param_grid_fixed['n_epochs'])
                * len(param_grid_fixed['momentum'])
                * len(param_grid_fixed['pos_weight_scale']))
    print(f"\n[Grid]  {n_combos} combos x 5-Fold CV = {n_combos * 5} trainings/pipeline")

    # ── Main loop ───────────────────────────────────────────────
    all_results = {}

    for pipe_name, sel_idx, sel_names in pipelines:
        print("\n" + "=" * 65)
        print(f"  PIPELINE: {pipe_name}  ({len(sel_names)} features)")
        print("=" * 65)

        Xtr = X_train[:, sel_idx]
        Xte = X_test[:, sel_idx]
        print(f"  Train: {Xtr.shape}   Test: {Xte.shape}")

        best_params, best_thr, best_cv_f1 = grid_search_fixed_features_weight(
            Xtr, y_train, param_grid_fixed, k=5, seed=42)

        # Final model
        print(f"\n[Final Model]  Training on full X_train ...")
        model = LogisticRegressionScratch(**best_params)
        model.fit(Xtr, y_train)
        print(f"  Training time: {model.train_time:.2f}s")

        # Test evaluation
        y_prob = model.predict_proba(Xte)
        y_pred = model.predict(Xte, threshold=best_thr)
        m = full_eval(y_test, y_pred, y_prob,
                      label=f'{pipe_name} -- Test Set')

        all_results[pipe_name] = dict(
            metrics        = m,
            model          = model,
            sel_names      = sel_names,
            n_features     = len(sel_names),
            best_params    = best_params,
            best_threshold = best_thr,
            best_cv_f1     = best_cv_f1,
            train_time     = model.train_time,
            y_prob         = y_prob,
            y_pred         = y_pred,
            y_test         = y_test,
        )

        # Plots (prefix lr_fixed_ to avoid overwriting nested version)
        safe = pipe_name.replace(' ', '_').lower()
        plot_learning_curves(
            model,
            title=f'Learning Curves — {pipe_name}',
            fname=f'lr_fixed_{safe}_lc.png')
        plot_confusion_matrix(
            y_test, y_pred,
            title=f'Confusion Matrix — {pipe_name}',
            fname=f'lr_fixed_{safe}_cm.png')
        plot_l2_regularization_path(
            Xtr, y_train, sel_names,
            fname=f'lr_fixed_{safe}_l2_reg_path.png')

    # ── Final comparison table ───────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  FIXED FEATURE WEIGHT TUNING — FINAL COMPARISON")
    print("=" * 75)
    hdr = (f"  {'Pipeline':<26} {'N':>3}  {'CV-F1':>6}  {'pws':>4}  "
           f"{'thr':>5}  {'Acc':>6}  {'Pre':>6}  {'Rec':>6}  "
           f"{'F1':>6}  {'AUC':>6}  {'Time':>6}")
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    for name, r in all_results.items():
        m   = r['metrics']
        pws = r['best_params']['pos_weight_scale']
        thr = r['best_threshold']
        print(
            f"  {name:<26} {r['n_features']:>3}  {r['best_cv_f1']:>6.4f}  "
            f"{pws:>4.2f}  {thr:>5.3f}  "
            f"{m['acc']:>6.4f}  {m['pre']:>6.4f}  "
            f"{m['rec']:>6.4f}  {m['f1']:>6.4f}  {m['auc']:>6.4f}  "
            f"{r['train_time']:>5.1f}s"
        )

    print()
    print("  Previous LR Results Before Strong Interaction v3:")
    print("    Old Fixed Corr18:        F1=0.5199  AUC=0.8384")
    print("    Re-selected Corr Top18:  F1=0.5383  AUC=0.8445")
    print("=" * 75)

    print()
    for name, r in all_results.items():
        print(f"  Best params — {name}:")
        for k, v in r['best_params'].items():
            print(f"    {k:<20} = {v}")
        print()

    best_name = max(all_results.keys(), key=lambda k: all_results[k]['metrics']['f1'])
    best = all_results[best_name]
    model_output_dir = os.path.join(_HERE, "model_outputs")
    os.makedirs(model_output_dir, exist_ok=True)
    np.savez(
        os.path.join(model_output_dir, "lr_results.npz"),
        model_name=np.array("LR"),
        source_file=np.array("lr.py"),
        selected_pipeline=np.array(best_name),
        y_prob=best['y_prob'],
        y_pred=best['y_pred'],
        y_test=best['y_test'],
        threshold=np.array(best['best_threshold']),
        train_time=np.array(best['train_time']),
        accuracy=np.array(best['metrics']['acc']),
        precision=np.array(best['metrics']['pre']),
        recall=np.array(best['metrics']['rec']),
        f1=np.array(best['metrics']['f1']),
        auc=np.array(best['metrics']['auc']),
        cv_f1=np.array(best['best_cv_f1']),
        n_features=np.array(best['n_features']),
    )
    print(f"[Saved comparison output] {os.path.join(model_output_dir, 'lr_results.npz')}")
    append_experiment_log("LR standalone model completed", [
        "Source script: lr.py",
        f"Selected pipeline={best_name}",
        f"Threshold={best['best_threshold']:.4f}",
        f"Accuracy={best['metrics']['acc']:.4f}",
        f"Precision={best['metrics']['pre']:.4f}",
        f"Recall={best['metrics']['rec']:.4f}",
        f"F1={best['metrics']['f1']:.4f}",
        f"AUC={best['metrics']['auc']:.4f}",
        f"Train time={best['train_time']:.4f}s",
        f"Output={os.path.join(model_output_dir, 'lr_results.npz')}",
    ])

    print(f"[Done]  Plots saved to {CACHE_DIR}/  (prefix: lr_fixed_)")
    return all_results


# ── 运行模式选择 ──────────────────────────────────────────────
RUN_MODE = "fixed_weight"
# 可选：
# RUN_MODE = "nested"

if __name__ == "__main__":
    if RUN_MODE == "nested":
        main()
    elif RUN_MODE == "fixed_weight":
        main_fixed_feature_weight_tuning()
    else:
        raise ValueError("RUN_MODE must be 'nested' or 'fixed_weight'")
