# ============================================================
# adaboost_grid.py  --  AdaBoost Stump Strict TopK Grid Search
#   Speed Dating Match Prediction (from scratch, no sklearn)
#
#   Key improvement: TopK is treated as a hyperparameter.
#   Feature selection is fitted inside each CV fold to avoid leakage.
#   This justifies why Top28 is selected based on CV performance.
# ============================================================
import os
import warnings
import time
import hashlib
import csv
from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'

SEED = 42
np.random.seed(SEED)

# ============================================================
# Paths
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))

# Multi-seed experiment support: default seed=42
RUN_SEED = int(os.environ.get("SPLIT_SEED", "42"))

# Load seed-specific cache; fall back to unified data_outputs if not found
CACHE_DIR = os.path.join(_HERE, f"output_figures_seed{RUN_SEED}")
if not os.path.isdir(CACHE_DIR):
    CACHE_DIR = os.path.join(_HERE, "data_outputs")
CACHE_FILES = {
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
# 0. Data Loading
# ============================================================
def load_data_smart():
    """Load from .npy cache. Run preprocessing first if cache missing."""
    have_cache = all(os.path.exists(p) for p in CACHE_FILES.values())
    if not have_cache:
        raise FileNotFoundError(
            "[ERROR] Cache files not found. Please run ML_data.py first.")
    print(f"[Cache] Loading from {CACHE_DIR}/*.npy")
    X_train = np.load(CACHE_FILES['X_train'])
    X_test  = np.load(CACHE_FILES['X_test'])
    y_train = np.load(CACHE_FILES['y_train'])
    y_test  = np.load(CACHE_FILES['y_test'])
    with open(CACHE_FILES['feature_names'], encoding='utf-8') as f:
        feature_names = [ln.strip() for ln in f if ln.strip()]
    return X_train, X_test, y_train, y_test, feature_names


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
# 1. Evaluation Suite  (pure NumPy, no sklearn)
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
    """Scan thresholds 0.05-0.95 to maximise F1. Pure NumPy."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        score = _f1(y_true, (y_prob >= t).astype(int))
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t, best_f1

def _auc_trapezoid(yt, y_prob):
    """ROC-AUC via trapezoidal rule."""
    thresholds = np.linspace(0, 1, 300)[::-1]
    fprs, tprs = [], []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        tp = float(np.sum((yp == 1) & (yt == 1)))
        fp = float(np.sum((yp == 1) & (yt == 0)))
        fn = float(np.sum((yp == 0) & (yt == 1)))
        tn = float(np.sum((yp == 0) & (yt == 0)))
        fprs.append(fp / (fp + tn + 1e-12))
        tprs.append(tp / (tp + fn + 1e-12))
    fprs  = np.array(fprs);  tprs = np.array(tprs)
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
    print(f"\n{'─' * 50}")
    if label:
        print(f"  {label}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {pre:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1s:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"{'─' * 50}")
    return dict(acc=acc, pre=pre, rec=rec, f1=f1s, auc=auc)


# ============================================================
# 2. Stratified K-Fold  (pure NumPy)
# ============================================================
def stratified_kfold_indices(y, k=5, seed=42):
    rng = np.random.default_rng(seed)
    y   = np.asarray(y)
    pos_idx = np.where(y == 1)[0];  rng.shuffle(pos_idx)
    neg_idx = np.where(y == 0)[0];  rng.shuffle(neg_idx)
    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)
    folds = []
    for i in range(k):
        val_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        train_idx = np.concatenate([
            np.concatenate([pos_folds[j] for j in range(k) if j != i]),
            np.concatenate([neg_folds[j] for j in range(k) if j != i])
        ])
        rng.shuffle(train_idx);  rng.shuffle(val_idx)
        folds.append((train_idx, val_idx))
    return folds


# ============================================================
# 2b. Correlation Selection  (pure NumPy) — flexible top_k
# ============================================================
def correlation_selection(X, y, feature_names, keep_mode='top30'):
    """
    Pearson correlation filter from scratch.
    Selects features by |correlation with y|.
    Supports any 'topN' string (e.g. 'top25', 'top28', 'top32').
    """
    n_feat = X.shape[1]
    correlations = np.zeros(n_feat)

    for j in range(n_feat):
        xj = X[:, j]
        xm = xj - xj.mean()
        ym = y  - y.mean()
        num = np.dot(xm, ym)
        den = np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))
        correlations[j] = abs(num / den) if den > 1e-12 else 0.0

    order = np.argsort(correlations)[::-1]

    # Support flexible modes like top25, top28, top30, top32, top35
    if isinstance(keep_mode, str) and keep_mode.startswith("top"):
        try:
            top_k = int(keep_mode.replace("top", ""))
        except ValueError:
            raise ValueError("keep_mode must look like 'top25', 'top30', etc.")
    else:
        raise ValueError("keep_mode must look like 'top25', 'top30', etc.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    sel_idx   = order[:min(top_k, n_feat)]
    sel_names = [feature_names[i] for i in sel_idx]

    print(f"\n[Correlation Selection] mode={keep_mode}  (top_k={top_k})")
    print(f"  Selected {len(sel_names)} / {n_feat} features")
    print("  Selected features:")
    for rank, idx in enumerate(sel_idx, 1):
        print(f"    {rank:2d}. {feature_names[idx]:30s} |r|={correlations[idx]:.4f}")

    return sel_idx, sel_names


# ============================================================
# 2c. Quiet Correlation Selection (for nested CV)
# ============================================================
def parse_topk_mode(keep_mode):
    """Convert 'top28' to 28."""
    if isinstance(keep_mode, str) and keep_mode.startswith("top"):
        return int(keep_mode.replace("top", ""))
    raise ValueError("keep_mode must look like 'top25', 'top28', etc.")


def correlation_selection_quiet(X, y, feature_names, keep_mode='top28'):
    """
    Quiet Pearson correlation feature selection.
    Used inside CV to avoid printing hundreds of feature lists.
    """
    top_k = parse_topk_mode(keep_mode)
    n_feat = X.shape[1]
    correlations = np.zeros(n_feat)

    y_centered = y - y.mean()

    for j in range(n_feat):
        xj = X[:, j]
        x_centered = xj - xj.mean()
        num = np.dot(x_centered, y_centered)
        den = np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
        correlations[j] = abs(num / den) if den > 1e-12 else 0.0

    order = np.argsort(correlations)[::-1]
    sel_idx = order[:min(top_k, n_feat)]
    sel_names = [feature_names[i] for i in sel_idx]

    return sel_idx, sel_names, correlations


# ============================================================
# 3. Weighted Decision Stump
# ============================================================
class WeightedDecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold     = None
        self.polarity      = None

    def fit(self, X, y_signed, sample_weight):
        n, d = X.shape
        best_error = np.inf
        for feat in range(d):
            vals = X[:, feat]
            candidates = np.unique(np.percentile(vals, np.linspace(5, 95, 19)))
            for thr in candidates:
                for polarity in (+1, -1):
                    pred = np.where(vals < thr, -polarity, +polarity)
                    err  = float(np.sum(sample_weight[pred != y_signed]))
                    if err < best_error:
                        best_error         = err
                        self.feature_index = feat
                        self.threshold     = thr
                        self.polarity      = polarity
        return self

    def predict(self, X):
        vals = X[:, self.feature_index]
        return np.where(vals < self.threshold,
                        -self.polarity, +self.polarity).astype(float)


# ============================================================
# 4. AdaBoost (from scratch)
# ============================================================
class AdaBoostScratch:
    def __init__(self, n_estimators=50, learning_rate=0.5,
                 weight_mode='balanced', base_learner='stump',
                 max_depth=2, min_samples_leaf=20, random_state=42):
        self.n_estimators     = n_estimators
        self.learning_rate    = learning_rate
        self.weight_mode      = weight_mode
        self.base_learner     = base_learner
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state
        self.learners_        = []
        self.alphas_          = []
        self.train_time       = 0.0

    @property
    def stumps_(self):
        return self.learners_

    def _init_weights(self, y):
        n = len(y)
        if self.weight_mode == 'balanced':
            n_pos = float(np.sum(y == 1)) + 1e-9
            n_neg = float(np.sum(y == 0)) + 1e-9
            w = np.where(y == 1, n / (2.0 * n_pos), n / (2.0 * n_neg)).astype(float)
        else:
            w = np.ones(n, dtype=float)
        return w / w.sum()

    def fit(self, X, y):
        np.random.seed(self.random_state)
        t0 = time.time()
        y_signed      = np.where(y == 1, +1.0, -1.0)
        sample_weight = self._init_weights(y)
        self.learners_ = []
        self.alphas_   = []
        for m in range(self.n_estimators):
            if self.base_learner == 'stump':
                learner = WeightedDecisionStump()
            else:
                raise ValueError(f"base_learner must be 'stump', got '{self.base_learner}'")
            learner.fit(X, y_signed, sample_weight)
            pred = learner.predict(X)
            err  = float(np.clip(np.sum(sample_weight[pred != y_signed]), 1e-10, 1.0 - 1e-10))
            if err >= 0.5:
                continue
            alpha         = self.learning_rate * 0.5 * np.log((1.0 - err) / err)
            sample_weight = sample_weight * np.exp(-alpha * y_signed * pred)
            sample_weight = sample_weight / sample_weight.sum()
            self.learners_.append(learner)
            self.alphas_.append(alpha)
            if err <= 1e-10:
                break
        self.train_time = time.time() - t0
        return self

    def decision_function(self, X):
        if not self.learners_:
            return np.zeros(len(X))
        scores = np.zeros(len(X))
        for alpha, learner in zip(self.alphas_, self.learners_):
            scores += alpha * learner.predict(X)
        return scores

    def predict_proba(self, X):
        score = self.decision_function(X)
        return 1.0 / (1.0 + np.exp(-2.0 * np.clip(score, -250, 250)))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ============================================================
# 5. Repeated Strict TopK + AdaBoost Grid Search (Multi-Seed CV)
# ============================================================
def repeated_grid_search_adaboost_topk(
    X, y, feature_names,
    topk_modes,
    param_grid,
    k=5,
    cv_seeds=(0, 1, 2, 3, 4, 42, 77, 123, 2024, 3407),
    model_seed=42
):
    """
    Repeated strict TopK + AdaBoost grid search.

    Selection rule:
    Choose the configuration with the highest mean CV-F1 across multiple
    stratified K-fold random seeds.

    Important:
    Feature selection is fitted only on the fold-training part in each CV fold.
    The held-out test set is never used during model selection.
    """
    n_estimators_list = param_grid.get('n_estimators', [90])
    learning_rate_list = param_grid.get('learning_rate', [0.3])
    weight_mode_list = param_grid.get('weight_mode', ['balanced'])
    base_learner_list = param_grid.get('base_learner', ['stump'])
    max_depth_list = param_grid.get('max_depth', [1])
    min_leaf_list = param_grid.get('min_samples_leaf', [1])

    all_combos = list(product(
        topk_modes,
        n_estimators_list,
        learning_rate_list,
        weight_mode_list,
        base_learner_list,
        max_depth_list,
        min_leaf_list
    ))

    print("\n" + "=" * 95)
    print("  REPEATED STRICT TOPK + ADABOOST GRID SEARCH")
    print("=" * 95)
    print(f"  Total combos: {len(all_combos)}")
    print(f"  CV setting   : {len(cv_seeds)} seeds x {k}-Fold Stratified CV")
    print(f"  CV seeds     : {list(cv_seeds)}")
    print(f"  Overall positive ratio: {np.mean(y):.2%}")

    records = []

    for count, combo in enumerate(all_combos, 1):
        keep_mode, n_est, lr, wm, bl, md, ml = combo

        params = dict(
            n_estimators=n_est,
            learning_rate=lr,
            weight_mode=wm,
            base_learner=bl,
            max_depth=md,
            min_samples_leaf=ml,
            random_state=model_seed,
        )

        all_fold_f1s = []
        all_fold_thresholds = []

        for cv_seed in cv_seeds:
            folds = stratified_kfold_indices(y, k=k, seed=cv_seed)

            for train_idx, val_idx in folds:
                # Train-only feature selection inside each fold
                sel_idx, _, _ = correlation_selection_quiet(
                    X[train_idx],
                    y[train_idx],
                    feature_names,
                    keep_mode=keep_mode
                )

                X_fold_train = X[train_idx][:, sel_idx]
                X_fold_val = X[val_idx][:, sel_idx]

                model = AdaBoostScratch(**params)
                model.fit(X_fold_train, y[train_idx])

                val_prob = model.predict_proba(X_fold_val)
                best_t, best_f1 = find_best_threshold(y[val_idx], val_prob)

                all_fold_f1s.append(best_f1)
                all_fold_thresholds.append(best_t)

        mean_cv_f1 = float(np.mean(all_fold_f1s))
        std_cv_f1 = float(np.std(all_fold_f1s))
        mean_threshold = float(np.mean(all_fold_thresholds))

        record = {
            'keep_mode': keep_mode,
            'mean_cv_f1': mean_cv_f1,
            'std_cv_f1': std_cv_f1,
            'threshold': mean_threshold,
            'params': params,
            'n_cv_scores': len(all_fold_f1s),
        }
        records.append(record)

        print(
            f"  [{count:>3}/{len(all_combos)}] "
            f"{keep_mode:<6} bl={bl:<5} n={n_est:<4} lr={lr:<4} "
            f"Mean-CV-F1={mean_cv_f1:.4f} ± {std_cv_f1:.4f} "
            f"thr={mean_threshold:.3f}"
        )

    best_record = max(records, key=lambda r: r['mean_cv_f1'])

    print("\n" + "=" * 95)
    print("  REPEATED GRID SEARCH BEST BY MEAN CV-F1")
    print("=" * 95)
    print(f"  Best TopK        = {best_record['keep_mode']}")
    print(f"  Best Mean CV-F1  = {best_record['mean_cv_f1']:.4f}")
    print(f"  CV-F1 Std        = {best_record['std_cv_f1']:.4f}")
    print(f"  Best threshold   = {best_record['threshold']:.3f}")
    print(f"  Best params      = {best_record['params']}")
    print("=" * 95)

    return best_record, records


# ============================================================
# 6. Visualisations
# ============================================================
def plot_confusion_matrix_ada(yt, yp, title='Confusion Matrix', fname='ada_cm.png'):
    cm_arr = _confusion_matrix(yt, yp)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_arr, cmap='Oranges', interpolation='nearest')
    plt.colorbar(im, ax=ax)
    labels = ['No Match (0)', 'Match (1)']
    ax.set_xticks([0, 1]);  ax.set_xticklabels(labels, rotation=20)
    ax.set_yticks([0, 1]);  ax.set_yticklabels(labels)
    thresh = cm_arr.max() / 2.0
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm_arr[r, c]), ha='center', va='center',
                    fontsize=14, color='white' if cm_arr[r, c] > thresh else 'black')
    ax.set_ylabel('True label');  ax.set_xlabel('Predicted label')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()


def plot_alpha_trajectory(alphas, fname='ada_alpha_trajectory.png'):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(alphas) + 1), alphas, marker='o', ms=4, lw=2, color='#e67e22')
    ax.set_xlabel('Boosting round')
    ax.set_ylabel('Alpha (stump importance)')
    ax.set_title('AdaBoost – Stump Importance per Round')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, fname), dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_usage(stumps, feature_names, fname='ada_feature_usage.png'):
    counts = np.zeros(len(feature_names), dtype=int)
    for s in stumps:
        counts[s.feature_index] += 1
    order = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#FFF5F5')
    ax.set_facecolor('#FFFFFF')
    ax.bar(range(len(feature_names)), counts[order], color='#F7A6AC',
           edgecolor='white', linewidth=0.6, alpha=0.9)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in order], rotation=90, fontsize=8)
    ax.set_ylabel('Times selected as split feature')
    ax.set_title('AdaBoost – Feature Usage Across Stumps')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(CACHE_DIR, fname), dpi=150,
                bbox_inches='tight', facecolor='#FFF5F5')
    plt.close()


# ============================================================
# 7. Main — Strict TopK Grid Search Pipeline
# ============================================================
def main():
    print("=" * 70)
    print(f"  adaboost_grid.py  --  Strict TopK Grid Search | seed={RUN_SEED}")
    print("  Speed Dating Match Prediction  (no sklearn)")
    print("=" * 70)

    PREV_BEST_F1  = 0.5437
    PREV_BEST_AUC = 0.8539

    # ── Load data ──────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = load_data_smart()

    print_split_fingerprint(
        X_train, X_test, y_train, y_test, feature_names,
        tag="[AdaBoost Grid Search]"
    )

    print(f"\n[Data]  X_train={X_train.shape}  X_test={X_test.shape}")
    print(f"  Positive ratio (train): {np.mean(y_train):.2%}")
    print(f"  Positive ratio (test) : {np.mean(y_test):.2%}")

    # ============================================================
    # Stage 1: Repeated Strict TopK + AdaBoost Grid Search
    # ============================================================
    print("\n" + "=" * 90)
    print("  STAGE 1: REPEATED STRICT TOPK + ADABOOST GRID SEARCH")
    print("=" * 90)

    # TopK is treated as a model-selection hyperparameter.
    # This is the key change needed to justify why Top28 is selected.
    stump_topk_modes = [f"top{k}" for k in range(25, 36)]  # top25 ~ top35

    param_grid_stump_topk = {
        'n_estimators':     [50, 70, 90, 110],
        'learning_rate':    [0.1, 0.3, 0.5],
        'weight_mode':      ['balanced'],
        'base_learner':     ['stump'],
        'max_depth':        [1],
        'min_samples_leaf': [1],
    }

    cv_seeds = [0, 1, 2, 3, 4, 42, 77, 123, 2024, 3407]

    best_record, grid_records = repeated_grid_search_adaboost_topk(
        X_train,
        y_train,
        feature_names,
        topk_modes=stump_topk_modes,
        param_grid=param_grid_stump_topk,
        k=5,
        cv_seeds=cv_seeds,
        model_seed=42
    )

    # Save grid search results for report / appendix
    os.makedirs(os.path.join(_HERE, "model_outputs"), exist_ok=True)
    grid_csv_path = os.path.join(_HERE, "model_outputs", "adaboost_topk_grid_search_results.csv")

    with open(grid_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "TopK", "Mean_CV_F1", "Std_CV_F1", "Threshold",
            "n_estimators", "learning_rate", "weight_mode",
            "base_learner", "max_depth", "min_samples_leaf"
        ])
        for r in sorted(grid_records, key=lambda x: x['mean_cv_f1'], reverse=True):
            p = r['params']
            writer.writerow([
                r['keep_mode'],
                f"{r['mean_cv_f1']:.4f}",
                f"{r['std_cv_f1']:.4f}",
                f"{r['threshold']:.3f}",
                p['n_estimators'],
                p['learning_rate'],
                p['weight_mode'],
                p['base_learner'],
                p['max_depth'],
                p['min_samples_leaf'],
            ])

    print(f"\n[Saved grid search results] {grid_csv_path}")

    # ============================================================
    # Stage 2: Final model using the best CV-selected configuration
    # ============================================================
    print("\n" + "=" * 90)
    print("  STAGE 2: FINAL ADABOOST MODEL FROM BEST CV CONFIGURATION")
    print("=" * 90)

    best_keep_mode = best_record['keep_mode']
    best_params = best_record['params']
    best_threshold = best_record['threshold']
    best_cv_f1 = best_record['mean_cv_f1']
    best_cv_std = best_record['std_cv_f1']

    # Final feature selection on full training data only
    corr_idx, corr_names = correlation_selection(
        X_train, y_train, feature_names, keep_mode=best_keep_mode
    )

    X_train_corr = X_train[:, corr_idx]
    X_test_corr = X_test[:, corr_idx]

    print(f"\n[Final Features] Using Re-selected Correlation {best_keep_mode}")
    print(f"  Train shape: {X_train_corr.shape}   Test shape: {X_test_corr.shape}")

    final_model = AdaBoostScratch(**best_params)
    final_model.fit(X_train_corr, y_train)

    print(f"\n[Final Model]")
    print(f"  Best TopK        = {best_keep_mode}")
    print(f"  Mean CV-F1       = {best_cv_f1:.4f} ± {best_cv_std:.4f}")
    print(f"  CV threshold     = {best_threshold:.3f}")
    print(f"  Best params      = {best_params}")
    print(f"  Learners used    = {len(final_model.learners_)} / {best_params['n_estimators']}")
    print(f"  Training time    = {final_model.train_time:.2f}s")

    y_prob = final_model.predict_proba(X_test_corr)
    y_pred = final_model.predict(X_test_corr, threshold=best_threshold)

    m = full_eval(
        y_test,
        y_pred,
        y_prob,
        label=f"Final Strict AdaBoost Stump -- {best_keep_mode} -- Test Set"
    )

    os.makedirs(CACHE_DIR, exist_ok=True)

    plot_confusion_matrix_ada(
        y_test,
        y_pred,
        title=f"AdaBoost Stump – Confusion Matrix ({best_keep_mode})",
        fname=f"ada_stump_confusion_matrix_{best_keep_mode}.png"
    )

    plot_alpha_trajectory(
        final_model.alphas_,
        fname=f"ada_stump_alpha_trajectory_{best_keep_mode}.png"
    )

    plot_feature_usage(
        final_model.stumps_,
        corr_names,
        fname=f"ada_stump_feature_usage_{best_keep_mode}.png"
    )

    print("\n" + "=" * 90)
    print("  FINAL STRICT ADABOOST RESULT")
    print("=" * 90)
    print(f"  Best TopK        = {best_keep_mode}")
    print(f"  Mean CV-F1       = {best_cv_f1:.4f} ± {best_cv_std:.4f}")
    print(f"  CV threshold     = {best_threshold:.3f}")
    print(f"  Test Accuracy    = {m['acc']:.4f}")
    print(f"  Test Precision   = {m['pre']:.4f}")
    print(f"  Test Recall      = {m['rec']:.4f}")
    print(f"  Test F1          = {m['f1']:.4f}")
    print(f"  Test AUC         = {m['auc']:.4f}")
    print(f"  Best params      = {best_params}")
    print("=" * 90)

    # Comparison with previous best
    print("\n[Comparison with Previous Best]")
    print(f"  Previous best F1  = {PREV_BEST_F1:.4f}")
    print(f"  Current  best F1  = {m['f1']:.4f}")
    print(f"  ΔF1               = {m['f1'] - PREV_BEST_F1:+.4f}")
    print(f"  Previous best AUC = {PREV_BEST_AUC:.4f}")
    print(f"  Current  best AUC = {m['auc']:.4f}")
    print(f"  ΔAUC              = {m['auc'] - PREV_BEST_AUC:+.4f}")

    # Save model outputs
    np.savez(
        os.path.join(_HERE, "model_outputs", "adaboost_results.npz"),
        y_test=y_test,
        y_prob=y_prob,
        y_pred=y_pred,
        selected_features=np.array(corr_names, dtype=object),
        best_topk=best_keep_mode,
        best_cv_f1=best_cv_f1,
        best_threshold=best_threshold,
        test_acc=m['acc'],
        test_precision=m['pre'],
        test_recall=m['rec'],
        test_f1=m['f1'],
        test_auc=m['auc'],
    )

    append_experiment_log("AdaBoost Grid Search completed", [
        "Source script: adaboost_grid.py",
        f"CV seeds: {cv_seeds}",
        f"Selected TopK={best_keep_mode}",
        f"Mean CV-F1={best_cv_f1:.4f} ± {best_cv_std:.4f}",
        f"Threshold={best_threshold:.4f}",
        f"Test Accuracy={m['acc']:.4f}",
        f"Test Precision={m['pre']:.4f}",
        f"Test Recall={m['rec']:.4f}",
        f"Test F1={m['f1']:.4f}",
        f"Test AUC={m['auc']:.4f}",
        f"Train time={final_model.train_time:.4f}s",
        f"Grid search results saved to: {grid_csv_path}",
    ])

    return final_model, m


if __name__ == '__main__':
    main()
