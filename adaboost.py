# ============================================================
# adaboost.py  --  Final AdaBoost Stump Model
#   Speed Dating Match Prediction (from scratch, no sklearn)
#
#   Focus: Final AdaBoost Stump model using repeated-CV selected best configuration
#          Final setting: Top35 features, 110 stumps, learning_rate=0.5
#          Threshold: 0.660
#
#   Model selection was done in adaboost_grid.py using repeated stratified CV.
# ============================================================
import os
import warnings
import time
import hashlib
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
# 4. Weighted Depth-2 Decision Tree  (retained, not used this run)
# ============================================================
class WeightedDepth2Tree:
    def __init__(self, max_depth=2, min_samples_leaf=20):
        self.max_depth        = max_depth
        self.min_samples_leaf = min_samples_leaf
        self._tree            = None

    @staticmethod
    def _weighted_leaf_value(y_sub, w_sub):
        pos_w = float(np.sum(w_sub[y_sub == +1]))
        neg_w = float(np.sum(w_sub[y_sub == -1]))
        return +1.0 if pos_w >= neg_w else -1.0

    @staticmethod
    def _best_split(X_sub, y_sub, w_sub, min_samples_leaf):
        n, d = X_sub.shape
        best_feat, best_thr, best_err = None, None, np.inf
        for feat in range(d):
            vals       = X_sub[:, feat]
            candidates = np.unique(np.percentile(vals, np.linspace(5, 95, 19)))
            for thr in candidates:
                left_mask  = vals < thr
                right_mask = ~left_mask
                if left_mask.sum() < min_samples_leaf or right_mask.sum() < min_samples_leaf:
                    continue
                lv   = WeightedDepth2Tree._weighted_leaf_value(y_sub[left_mask],  w_sub[left_mask])
                rv   = WeightedDepth2Tree._weighted_leaf_value(y_sub[right_mask], w_sub[right_mask])
                pred = np.where(left_mask, lv, rv)
                err  = float(np.sum(w_sub[pred != y_sub]))
                if err < best_err:
                    best_feat, best_thr, best_err = feat, thr, err
        return (best_feat, best_thr, best_err) if best_feat is not None else None

    def _build_node(self, X_sub, y_sub, w_sub, depth):
        if depth >= self.max_depth or len(y_sub) < 2 * self.min_samples_leaf:
            return {'is_leaf': True, 'value': self._weighted_leaf_value(y_sub, w_sub)}
        if len(np.unique(y_sub)) == 1:
            return {'is_leaf': True, 'value': float(y_sub[0])}
        result = self._best_split(X_sub, y_sub, w_sub, self.min_samples_leaf)
        if result is None:
            return {'is_leaf': True, 'value': self._weighted_leaf_value(y_sub, w_sub)}
        feat, thr, _ = result
        left_mask    = X_sub[:, feat] < thr
        right_mask   = ~left_mask
        return {
            'is_leaf': False, 'feature_index': feat, 'threshold': thr,
            'left':  self._build_node(X_sub[left_mask],  y_sub[left_mask],  w_sub[left_mask],  depth+1),
            'right': self._build_node(X_sub[right_mask], y_sub[right_mask], w_sub[right_mask], depth+1),
        }

    @staticmethod
    def _predict_one(node, x):
        while not node['is_leaf']:
            node = node['left'] if x[node['feature_index']] < node['threshold'] else node['right']
        return node['value']

    def fit(self, X, y_signed, sample_weight):
        self._tree = self._build_node(X, y_signed, sample_weight, depth=0)
        return self

    def predict(self, X):
        return np.array([self._predict_one(self._tree, x) for x in X], dtype=float)


# ============================================================
# 5. AdaBoost (from scratch)
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
            elif self.base_learner == 'depth2':
                learner = WeightedDepth2Tree(self.max_depth, self.min_samples_leaf)
            else:
                raise ValueError(f"base_learner must be 'stump' or 'depth2', got '{self.base_learner}'")
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
# 6. Grid Search for AdaBoost
# ============================================================
def grid_search_adaboost(X, y, param_grid, k=5, seed=42):
    n_estimators_list  = param_grid.get('n_estimators',    [50])
    learning_rate_list = param_grid.get('learning_rate',   [0.5])
    weight_mode_list   = param_grid.get('weight_mode',     ['balanced'])
    base_learner_list  = param_grid.get('base_learner',    ['stump'])
    max_depth_list     = param_grid.get('max_depth',       [1])
    min_leaf_list      = param_grid.get('min_samples_leaf',[20])

    total = (len(n_estimators_list) * len(learning_rate_list)
             * len(weight_mode_list) * len(base_learner_list)
             * len(max_depth_list)   * len(min_leaf_list))

    folds       = stratified_kfold_indices(y, k=k, seed=seed)
    overall_pos = float(np.mean(y))
    fold_ratios = [float(np.mean(y[val_idx])) for _, val_idx in folds]

    print(f"\n[AdaBoost Grid Search]  {total} combos x {k}-Fold Stratified CV")
    print(f"  Overall positive ratio: {overall_pos:.2%}")
    print("  Per-fold positive ratio: " +
          "  ".join(f"fold{i+1}={r:.2%}" for i, r in enumerate(fold_ratios)))

    best_cv_f1     = -1.0
    best_params    = {}
    best_threshold = 0.5
    count          = 0

    for n_est in n_estimators_list:
        for lr in learning_rate_list:
            for wm in weight_mode_list:
                for bl in base_learner_list:
                    for md in max_depth_list:
                        for ml in min_leaf_list:
                            count += 1
                            params = dict(
                                n_estimators=n_est, learning_rate=lr,
                                weight_mode=wm, base_learner=bl,
                                max_depth=md, min_samples_leaf=ml,
                                random_state=seed,
                            )
                            fold_f1s, fold_thresholds = [], []
                            for train_idx, val_idx in folds:
                                model = AdaBoostScratch(**params)
                                model.fit(X[train_idx], y[train_idx])
                                val_prob        = model.predict_proba(X[val_idx])
                                best_t, best_f1 = find_best_threshold(y[val_idx], val_prob)
                                fold_f1s.append(best_f1)
                                fold_thresholds.append(best_t)
                            mean_f1        = float(np.mean(fold_f1s))
                            mean_threshold = float(np.mean(fold_thresholds))
                            print(
                                f"  [{count:>3}/{total}]  "
                                f"bl={bl:<7} n={n_est:<4} lr={lr:<5} "
                                f"md={md} ml={ml:<4}  "
                                f"CV-F1={mean_f1:.4f}  thr={mean_threshold:.3f}"
                            )
                            if mean_f1 > best_cv_f1:
                                best_cv_f1     = mean_f1
                                best_params    = params.copy()
                                best_threshold = mean_threshold

    print(f"\n[AdaBoost Grid Search Best]")
    print(f"  Best CV F1     = {best_cv_f1:.4f}")
    print(f"  Best threshold = {best_threshold:.3f}")
    print(f"  Best params    = {best_params}")
    return best_params, best_threshold, best_cv_f1


# ============================================================
# 7. Visualisations
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
# 8. Main — Strict Stump TopK Search
# ============================================================
def main():
    print("=" * 70)
    print(f"  adaboost.py  --  Final Top35 AdaBoost Stump | seed={RUN_SEED}")
    print("  Speed Dating Match Prediction  (no sklearn)")
    print("=" * 70)

    # Final best configuration selected by repeated stratified 5-fold CV
    FINAL_KEEP_MODE = "top35"
    FINAL_THRESHOLD = 0.660
    FINAL_PARAMS = {
        "n_estimators": 110,
        "learning_rate": 0.5,
        "weight_mode": "balanced",
        "base_learner": "stump",
        "max_depth": 1,
        "min_samples_leaf": 1,
        "random_state": 42,
    }

    # For reporting only: repeated CV result from grid-search script
    FINAL_MEAN_CV_F1 = 0.5557
    FINAL_STD_CV_F1 = 0.0203

    # ── Load data ──────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = load_data_smart()

    print_split_fingerprint(
        X_train, X_test, y_train, y_test, feature_names,
        tag="[Final AdaBoost Top35]"
    )

    print(f"\n[Data]  X_train={X_train.shape}  X_test={X_test.shape}")
    print(f"  Positive ratio (train): {np.mean(y_train):.2%}")
    print(f"  Positive ratio (test) : {np.mean(y_test):.2%}")

    # ============================================================
    # Final Feature Selection: Top35 on full training data only
    # ============================================================
    print("\n" + "=" * 85)
    print("  FINAL FEATURE SELECTION: RE-SELECTED CORRELATION TOP35")
    print("=" * 85)

    corr_idx, corr_names = correlation_selection(
        X_train, y_train, feature_names, keep_mode=FINAL_KEEP_MODE
    )

    X_train_corr = X_train[:, corr_idx]
    X_test_corr = X_test[:, corr_idx]

    print(f"\n[Final Features] Using Re-selected Correlation {FINAL_KEEP_MODE}")
    print(f"  Train shape: {X_train_corr.shape}   Test shape: {X_test_corr.shape}")

    # ============================================================
    # Final Model: Top35 + 110 stumps + lr=0.5
    # ============================================================
    print("\n" + "=" * 85)
    print("  FINAL ADABOOST MODEL USING REPEATED-CV BEST CONFIGURATION")
    print("=" * 85)

    print(f"  Best TopK        = {FINAL_KEEP_MODE}")
    print(f"  Mean CV-F1       = {FINAL_MEAN_CV_F1:.4f} ± {FINAL_STD_CV_F1:.4f}")
    print(f"  CV threshold     = {FINAL_THRESHOLD:.3f}")
    print(f"  Best params      = {FINAL_PARAMS}")

    final_model = AdaBoostScratch(**FINAL_PARAMS)
    final_model.fit(X_train_corr, y_train)

    print(f"  Learners used    = {len(final_model.learners_)} / {FINAL_PARAMS['n_estimators']}")
    print(f"  Training time    = {final_model.train_time:.2f}s")

    y_prob = final_model.predict_proba(X_test_corr)
    y_pred = final_model.predict(X_test_corr, threshold=FINAL_THRESHOLD)

    final_res = full_eval(
        y_test, y_pred, y_prob,
        label=f"Final Strict AdaBoost Stump -- {FINAL_KEEP_MODE} -- Test Set"
    )

    # ============================================================
    # Visualisations — disabled in final-model script.
    # Feature usage data is saved into adaboost_results.npz
    # (key: feature_usage_counts) and re-rendered by
    # experimental_analysis.py (exp_12_feature_importance.png).
    # All other diagnostics are produced by experimental_analysis.py
    # from the .npz alone, so we do not write working PNGs here.
    # ============================================================

    # ============================================================
    # Final Summary
    # ============================================================
    print("\n" + "=" * 85)
    print("  FINAL STRICT ADABOOST RESULT")
    print("=" * 85)
    print(f"  Best TopK        = {FINAL_KEEP_MODE}")
    print(f"  Mean CV-F1       = {FINAL_MEAN_CV_F1:.4f} ± {FINAL_STD_CV_F1:.4f}")
    print(f"  CV threshold     = {FINAL_THRESHOLD:.3f}")
    print(f"  Test Accuracy    = {final_res['acc']:.4f}")
    print(f"  Test Precision   = {final_res['pre']:.4f}")
    print(f"  Test Recall      = {final_res['rec']:.4f}")
    print(f"  Test F1          = {final_res['f1']:.4f}")
    print(f"  Test AUC         = {final_res['auc']:.4f}")
    print(f"  Best params      = {FINAL_PARAMS}")
    print("=" * 85)

    # ============================================================
    # Save outputs for comparison script / report figures
    # ============================================================
    model_output_dir = os.path.join(_HERE, "model_outputs")
    os.makedirs(model_output_dir, exist_ok=True)

    # Stump feature-usage counts (for plot_12 in experimental_analysis.py)
    feature_usage_counts = np.zeros(len(corr_names), dtype=int)
    for stump in final_model.stumps_:
        feature_usage_counts[stump.feature_index] += 1

    np.savez(
        os.path.join(model_output_dir, "adaboost_results.npz"),
        model_name=np.array("AdaBoost"),
        source_file=np.array("adaboost.py"),
        selected_pipeline=np.array(FINAL_KEEP_MODE),
        selected_features=np.array(corr_names, dtype=object),
        feature_usage_counts=feature_usage_counts,
        y_prob=y_prob,
        y_pred=y_pred,
        y_test=y_test,
        threshold=np.array(FINAL_THRESHOLD),
        train_time=np.array(final_model.train_time),
        accuracy=np.array(final_res["acc"]),
        precision=np.array(final_res["pre"]),
        recall=np.array(final_res["rec"]),
        f1=np.array(final_res["f1"]),
        auc=np.array(final_res["auc"]),
        cv_f1=np.array(FINAL_MEAN_CV_F1),
        cv_f1_std=np.array(FINAL_STD_CV_F1),
        n_features=np.array(len(corr_names)),
        n_estimators=np.array(FINAL_PARAMS["n_estimators"]),
        learning_rate=np.array(FINAL_PARAMS["learning_rate"]),
    )

    print(f"[Saved comparison output] {os.path.join(model_output_dir, 'adaboost_results.npz')}")

    result_path = os.path.join(CACHE_DIR, "adaboost_top35_final_results.csv")
    write_header = not os.path.exists(result_path)

    with open(result_path, "a", encoding="utf-8") as f:
        if write_header:
            f.write(
                "seed,topk,n_estimators,learning_rate,threshold,"
                "accuracy,precision,recall,f1,auc,train_time\n"
            )

        f.write(
            f"{RUN_SEED},{FINAL_KEEP_MODE},"
            f"{FINAL_PARAMS['n_estimators']},{FINAL_PARAMS['learning_rate']},"
            f"{FINAL_THRESHOLD:.4f},"
            f"{final_res['acc']:.4f},{final_res['pre']:.4f},"
            f"{final_res['rec']:.4f},{final_res['f1']:.4f},{final_res['auc']:.4f},"
            f"{final_model.train_time:.4f}\n"
        )

    print(f"[Saved final result] {result_path}")

    append_experiment_log("AdaBoost final Top35 model completed", [
        "Source script: adaboost.py",
        f"Selected pipeline={FINAL_KEEP_MODE}",
        f"Mean CV-F1={FINAL_MEAN_CV_F1:.4f} ± {FINAL_STD_CV_F1:.4f}",
        f"Threshold={FINAL_THRESHOLD:.4f}",
        f"n_estimators={FINAL_PARAMS['n_estimators']}",
        f"learning_rate={FINAL_PARAMS['learning_rate']}",
        f"Accuracy={final_res['acc']:.4f}",
        f"Precision={final_res['pre']:.4f}",
        f"Recall={final_res['rec']:.4f}",
        f"F1={final_res['f1']:.4f}",
        f"AUC={final_res['auc']:.4f}",
        f"Train time={final_model.train_time:.4f}s",
        f"Output={os.path.join(model_output_dir, 'adaboost_results.npz')}",
    ])

    return final_model, final_res


if __name__ == '__main__':
    main()
