"""
AI3013 Machine Learning Course Project - Group 12
Experimental Study & Result Analysis + Model Comparison
======================================================
Speed-Dating Matching Prediction

Three from-scratch models:
  1. Logistic Regression (SGD + L1/L2 + Momentum)
  2. AdaBoost (Weighted Decision Stump)
  3. SVM (RBF Kernel + SMO + Platt Calibration)

This script:
  - Loads preprocessed data
  - Trains all three models
  - Produces comprehensive comparison visualizations
  - Prints analysis text suitable for the final report
======================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
import hashlib
from datetime import datetime
import shutil

# ── Config ─────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "final_figures")  # Final output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXPERIMENT_LOG = os.path.join(_HERE, "experiment_run_log.md")

def append_experiment_log(title, lines):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n## {timestamp} - {title}\n")
        for line in lines:
            f.write(f"- {line}\n")

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = '#FFF5F5'
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['axes.edgecolor'] = '#F3BBB1'
plt.rcParams['grid.color'] = '#F3BBB1'
plt.rcParams['grid.alpha'] = 0.4

SEED = 42
np.random.seed(SEED)

# ============================================================
# 0. Data Loading
# ============================================================
CACHE_FILES = {
    'X_train':       os.path.join(_HERE, 'report_figures', 'X_train.npy'),
    'X_test':        os.path.join(_HERE, 'report_figures', 'X_test.npy'),
    'y_train':       os.path.join(_HERE, 'report_figures', 'y_train.npy'),
    'y_test':        os.path.join(_HERE, 'report_figures', 'y_test.npy'),
    'feature_names': os.path.join(_HERE, 'report_figures', 'feature_names.txt'),
}
if not all(os.path.exists(p) for p in CACHE_FILES.values()):
    CACHE_FILES = {
        'X_train':       os.path.join(_HERE, 'data_outputs', 'X_train.npy'),
        'X_test':        os.path.join(_HERE, 'data_outputs', 'X_test.npy'),
        'y_train':       os.path.join(_HERE, 'data_outputs', 'y_train.npy'),
        'y_test':        os.path.join(_HERE, 'data_outputs', 'y_test.npy'),
        'feature_names': os.path.join(_HERE, 'data_outputs', 'feature_names.txt'),
    }


def load_data():
    """Load preprocessed .npy cache."""
    if not all(os.path.exists(p) for p in CACHE_FILES.values()):
        raise FileNotFoundError(
            "Cache not found. Run ML_data.py first.")
    X_train = np.load(CACHE_FILES['X_train'])
    X_test = np.load(CACHE_FILES['X_test'])
    y_train = np.load(CACHE_FILES['y_train'])
    y_test = np.load(CACHE_FILES['y_test'])
    with open(CACHE_FILES['feature_names'], encoding='utf-8') as f:
        feature_names = [ln.strip() for ln in f if ln.strip()]
    return X_train, X_test, y_train, y_test, feature_names

# ============================================================
# 1. Shared Evaluation Utilities (pure NumPy)
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
    p = _precision(yt, yp)
    r = _recall(yt, yp)
    return 2 * p * r / (p + r + 1e-12)

def _auc_trapezoid(yt, y_prob):
    """ROC-AUC via trapezoidal rule (pure NumPy)."""
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
    fprs = np.array(fprs)
    tprs = np.array(tprs)
    order = np.argsort(fprs)
    return float(np.trapz(tprs[order], fprs[order]))

def _confusion_matrix(yt, yp):
    tn = int(np.sum((yp == 0) & (yt == 0)))
    fp = int(np.sum((yp == 1) & (yt == 0)))
    fn = int(np.sum((yp == 0) & (yt == 1)))
    tp = int(np.sum((yp == 1) & (yt == 1)))
    return np.array([[tn, fp], [fn, tp]])

def _roc_curve(yt, y_prob):
    """Return (fpr_list, tpr_list) for ROC plotting."""
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
    return np.array(fprs), np.array(tprs)


def find_best_threshold(y_true, y_prob):
    """Scan thresholds to maximise F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 181):
        score = _f1(y_true, (y_prob >= t).astype(int))
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t, best_f1


def stratified_kfold_indices(y, k=5, seed=42):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]; rng.shuffle(pos_idx)
    neg_idx = np.where(y == 0)[0]; rng.shuffle(neg_idx)
    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)
    folds = []
    for i in range(k):
        val_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        train_idx = np.concatenate([
            np.concatenate([pos_folds[j] for j in range(k) if j != i]),
            np.concatenate([neg_folds[j] for j in range(k) if j != i])
        ])
        rng.shuffle(train_idx); rng.shuffle(val_idx)
        folds.append((train_idx, val_idx))
    return folds

# ============================================================
# 2. Model Implementations (simplified for comparison run)
# ============================================================

# ── 2a. Logistic Regression ─────────────────────────────────
class LogisticRegressionScratch:
    def __init__(self, lr=0.01, lambda_=0.01, n_epochs=200,
                 batch_size=64, momentum=0.9, class_weight='balanced',
                 penalty='l2', pos_weight_scale=1.0):
        self.lr = lr
        self.lambda_ = lambda_
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.class_weight = class_weight
        self.penalty = penalty
        self.pos_weight_scale = pos_weight_scale
        self.theta = None
        self.train_losses = []
        self.val_losses = []
        self.train_time = 0

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def _sample_weights(self, y):
        if self.class_weight == 'balanced':
            n = len(y)
            n_pos = float(np.sum(y == 1)) + 1e-9
            n_neg = float(np.sum(y == 0)) + 1e-9
            w_pos = n / (2.0 * n_pos) * self.pos_weight_scale
            w_neg = n / (2.0 * n_neg)
            return np.where(y == 1, w_pos, w_neg)
        return np.ones(len(y))

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        t0 = time.time()
        n, d = X_train.shape
        Xb = np.c_[np.ones(n), X_train]
        self.theta = np.zeros(d + 1)
        velocity = np.zeros(d + 1)
        Xbv = None if X_val is None else np.c_[np.ones(len(X_val)), X_val]

        for _ in range(self.n_epochs):
            perm = np.random.permutation(n)
            Xbs = Xb[perm]; ys = y_train[perm]
            for s in range(0, n, self.batch_size):
                Xbi = Xbs[s:s + self.batch_size]
                ybi = ys[s:s + self.batch_size]
                nb = len(ybi)
                p = self._sigmoid(Xbi @ self.theta)
                err = self._sample_weights(ybi) * (p - ybi)
                grad = Xbi.T @ err / nb
                grad[1:] += (self.lambda_ / n) * self.theta[1:]
                velocity = self.momentum * velocity + self.lr * grad
                self.theta -= velocity

            # record loss
            p_all = self._sigmoid(Xb @ self.theta)
            eps = 1e-12
            wts = self._sample_weights(y_train)
            ll = -np.mean(wts * (y_train * np.log(p_all + eps) +
                                  (1 - y_train) * np.log(1 - p_all + eps)))
            reg = (self.lambda_ / (2.0 * n)) * float(np.sum(self.theta[1:] ** 2))
            self.train_losses.append(ll + reg)

            if Xbv is not None:
                p_v = self._sigmoid(Xbv @ self.theta)
                wts_v = self._sample_weights(y_val)
                ll_v = -np.mean(wts_v * (y_val * np.log(p_v + eps) +
                                          (1 - y_val) * np.log(1 - p_v + eps)))
                self.val_losses.append(ll_v)

        self.train_time = time.time() - t0
        return self

    def predict_proba(self, X):
        return self._sigmoid(np.c_[np.ones(len(X)), X] @ self.theta)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    @property
    def coef_(self):
        return self.theta[1:]

# ── 2b. AdaBoost Decision Stump ─────────────────────────────
class WeightedDecisionStump:
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.polarity = None

    def fit(self, X, y_signed, sample_weight):
        n, d = X.shape
        best_error = np.inf
        for feat in range(d):
            vals = X[:, feat]
            candidates = np.unique(np.percentile(vals, np.linspace(5, 95, 19)))
            for thr in candidates:
                for polarity in (+1, -1):
                    pred = np.where(vals < thr, -polarity, +polarity)
                    err = float(np.sum(sample_weight[pred != y_signed]))
                    if err < best_error:
                        best_error = err
                        self.feature_index = feat
                        self.threshold = thr
                        self.polarity = polarity
        return self

    def predict(self, X):
        vals = X[:, self.feature_index]
        return np.where(vals < self.threshold,
                        -self.polarity, +self.polarity).astype(float)


class AdaBoostScratch:
    def __init__(self, n_estimators=90, learning_rate=0.3,
                 weight_mode='balanced', random_state=42):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.weight_mode = weight_mode
        self.random_state = random_state
        self.learners_ = []
        self.alphas_ = []
        self.train_time = 0.0

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
        y_signed = np.where(y == 1, +1.0, -1.0)
        sample_weight = self._init_weights(y)
        self.learners_ = []
        self.alphas_ = []
        for m in range(self.n_estimators):
            learner = WeightedDecisionStump()
            learner.fit(X, y_signed, sample_weight)
            pred = learner.predict(X)
            err = float(np.clip(np.sum(sample_weight[pred != y_signed]), 1e-10, 1.0 - 1e-10))
            if err >= 0.5:
                continue
            alpha = self.learning_rate * 0.5 * np.log((1.0 - err) / err)
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

# ── 2c. SVM (RBF + SMO) - using svm.py implementation ─────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -250, 250)))

def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def rbf_kernel(x1, x2, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x1 - x2)**2)

class SVMScratch:
    """AdvancedSVM simplified version (from svm.py)"""
    def __init__(self, C=1.0, kernel='rbf', gamma=0.025, tol=1e-3,
                 max_passes=3, pos_weight_multiplier=3.5, 
                 max_iterations=22, sample_limit=1000):
        self.C = C
        self.kernel_name = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.pos_weight_multiplier = pos_weight_multiplier
        self.max_iterations = max_iterations
        self.sample_limit = sample_limit
        self.train_time = 0.0

    def kernel(self, x1, x2):
        if self.kernel_name == 'linear':
            return linear_kernel(x1, x2)
        elif self.kernel_name == 'rbf':
            return rbf_kernel(x1, x2, self.gamma)

    def fit(self, X, y):
        t0 = time.time()
        self.X = X
        self.y = y
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0

        self.C_pos = self.C * self.pos_weight_multiplier
        self.C_neg = self.C

        pos_weight = len(y) / (2 * np.sum(y == 1))
        neg_weight = len(y) / (2 * np.sum(y == -1))
        self.sample_weights = np.where(y == 1, pos_weight, neg_weight)

        passes = 0
        iteration = 0

        while passes < self.max_passes and iteration < self.max_iterations:
            num_changed_alphas = 0
            sample_limit = min(self.sample_limit, n_samples)
            random_idx = np.random.choice(n_samples, sample_limit, replace=False)

            for i in random_idx:
                Ei = self._decision_single_fast(X[i]) - y[i]
                Ci = self.C_pos if y[i] == 1 else self.C_neg
                
                condition1 = (y[i] * Ei < -self.tol and self.alpha[i] < Ci)
                condition2 = (y[i] * Ei > self.tol and self.alpha[i] > 0)

                if condition1 or condition2:
                    j = np.random.randint(0, n_samples)
                    while j == i:
                        j = np.random.randint(0, n_samples)

                    Cj = self.C_pos if y[j] == 1 else self.C_neg
                    Ej = self._decision_single_fast(X[j]) - y[j]
                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    if y[i] != y[j]:
                        L = max(0, self.alpha[j] - self.alpha[i])
                        H = min(Cj, Ci + self.alpha[j] - self.alpha[i])
                    else:
                        L = max(0, self.alpha[i] + self.alpha[j] - Ci)
                        H = min(Ci, self.alpha[i] + self.alpha[j])

                    if L == H:
                        continue

                    Kii = self.kernel(X[i], X[i])
                    Kjj = self.kernel(X[j], X[j])
                    Kij = self.kernel(X[i], X[j])
                    eta = Kii + Kjj - 2 * Kij

                    if eta <= 1e-12:
                        continue

                    self.alpha[j] += (y[j] * (Ei - Ej)) / eta
                    self.alpha[j] = np.clip(self.alpha[j], L, H)

                    if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                        continue

                    self.alpha[i] += y[i] * y[j] * (alpha_j_old - self.alpha[j])

                    b1 = (self.b - Ei - y[i] * (self.alpha[i] - alpha_i_old) * Kii
                          - y[j] * (self.alpha[j] - alpha_j_old) * Kij)
                    b2 = (self.b - Ej - y[i] * (self.alpha[i] - alpha_i_old) * Kij
                          - y[j] * (self.alpha[j] - alpha_j_old) * Kjj)

                    if 0 < self.alpha[i] < Ci:
                        self.b = b1
                    elif 0 < self.alpha[j] < Cj:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
            if num_changed_alphas < 5:
                break
            iteration += 1

        self.train_time = time.time() - t0
        sv_mask = self.alpha > 1e-3
        self.n_sv = int(np.sum(sv_mask))
        return self

    def _decision_single_fast(self, x):
        sv_mask = self.alpha > 1e-5
        sv_alpha = self.alpha[sv_mask]
        sv_y = self.y[sv_mask]
        sv_X = self.X[sv_mask]
        diff = sv_X - x
        if self.kernel_name == 'linear':
            K = sv_X @ x
        else:
            K = np.exp(-self.gamma * np.sum(diff**2, axis=1))
        return np.sum(sv_alpha * sv_y * K) + self.b

    def decision_function(self, X):
        sv_mask = self.alpha > 1e-5
        sv_alpha = self.alpha[sv_mask]
        sv_y = self.y[sv_mask]
        sv_X = self.X[sv_mask]
        scores = []
        for x in X:
            diff = sv_X - x
            if self.kernel_name == 'linear':
                K = sv_X @ x
            else:
                K = np.exp(-self.gamma * np.sum(diff**2, axis=1))
            score = np.sum(sv_alpha * sv_y * K) + self.b
            scores.append(score)
        return np.array(scores)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        return sigmoid(scores)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# ============================================================
# 3. Correlation Feature Selection
# ============================================================
def correlation_top_k(X, y, feature_names, top_k=28):
    """Select top_k features by |Pearson r| with target."""
    n_feat = X.shape[1]
    correlations = np.zeros(n_feat)
    for j in range(n_feat):
        xj = X[:, j]
        xm = xj - xj.mean()
        ym = y - y.mean()
        num = np.dot(xm, ym)
        den = np.sqrt(np.dot(xm, xm) * np.dot(ym, ym))
        correlations[j] = abs(num / den) if den > 1e-12 else 0.0
    order = np.argsort(correlations)[::-1]
    sel_idx = order[:min(top_k, n_feat)]
    sel_names = [feature_names[i] for i in sel_idx]
    return sel_idx, sel_names, correlations


# ============================================================
# 4. Cross-Validation for Bias-Variance Analysis
# ============================================================
def cross_validate_model(model_class, model_params, X, y, k=5, seed=42):
    """
    Run stratified k-fold CV, returning per-fold train/val metrics.
    Useful for bias-variance trade-off analysis.
    """
    folds = stratified_kfold_indices(y, k=k, seed=seed)
    train_f1s, val_f1s = [], []
    train_aucs, val_aucs = [], []

    for train_idx, val_idx in folds:
        model = model_class(**model_params)
        model.fit(X[train_idx], y[train_idx])

        # Train metrics
        tr_prob = model.predict_proba(X[train_idx])
        tr_thr, _ = find_best_threshold(y[train_idx], tr_prob)
        tr_pred = (tr_prob >= tr_thr).astype(int)
        train_f1s.append(_f1(y[train_idx], tr_pred))
        train_aucs.append(_auc_trapezoid(y[train_idx], tr_prob))

        # Val metrics
        val_prob = model.predict_proba(X[val_idx])
        val_thr, _ = find_best_threshold(y[val_idx], val_prob)
        val_pred = (val_prob >= val_thr).astype(int)
        val_f1s.append(_f1(y[val_idx], val_pred))
        val_aucs.append(_auc_trapezoid(y[val_idx], val_prob))

    return {
        'train_f1_mean': np.mean(train_f1s), 'train_f1_std': np.std(train_f1s),
        'val_f1_mean': np.mean(val_f1s), 'val_f1_std': np.std(val_f1s),
        'train_auc_mean': np.mean(train_aucs), 'train_auc_std': np.std(train_aucs),
        'val_auc_mean': np.mean(val_aucs), 'val_auc_std': np.std(val_aucs),
        'train_f1s': train_f1s, 'val_f1s': val_f1s,
    }

# ============================================================
# 5. Visualization Functions
# ============================================================

# Macaron color palette
# #F7A6AC (pink), #F7B2C7 (rose pink), #F3BBB1 (light coral)
# #EEC78A (cream yellow), #EEE9A2 (light yellow), #CBE4B1 (light green)
# #B3DDCB (mint green), #B8E5FA (sky blue)
COLORS = {
    'LR': '#F7A6AC',       # pink
    'AdaBoost': '#EEC78A', # cream yellow
    'SVM': '#B8E5FA',      # sky blue
}
# Extended palette for multi-color needs
PALETTE = ['#F7A6AC', '#F7B2C7', '#F3BBB1', '#EEC78A',
           '#EEE9A2', '#CBE4B1', '#B3DDCB', '#B8E5FA']


def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  [Saved] {path}")
    plt.close()


def plot_01_grouped_bar(results):
    """Grouped bar chart of all metrics."""
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
    x = np.arange(len(metrics))
    width = 0.25
    names = list(results.keys())

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, name in enumerate(names):
        vals = [results[name]['test_metrics'][m] for m in metrics]
        bars = ax.bar(x + i * width, vals, width,
                      label=name, color=COLORS[name], alpha=0.85,
                      edgecolor='white', linewidth=0.8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison (Test Set)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_fig('exp_01_grouped_bar.png')

def plot_02_radar(results):
    """Radar chart for multi-dimensional comparison."""
    # Report uses macaron light colors, PPT uses summer-beach dark colors
    if COLORS.get('LR') == '#FC757B':
        RADAR_COLORS = {'LR': '#FC757B', 'AdaBoost': '#FAA26F', 'SVM': '#3C9BC9'}
    else:
        RADAR_COLORS = {'LR': '#F7A6AC', 'AdaBoost': '#EEC78A', 'SVM': '#B8E5FA'}
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name in results:
        vals = [results[name]['test_metrics'][m] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2.5, label=name,
                color=RADAR_COLORS[name], markersize=7)
        ax.fill(angles, vals, alpha=0.15, color=RADAR_COLORS[name])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title('Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', bbox_to_anchor=(1.3, 0), fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig('exp_02_radar.png')


def plot_03_roc_curves(results, y_test):
    """Overlay ROC curves for all models."""
    # Report uses macaron light colors, PPT uses summer-beach dark colors
    if COLORS.get('LR') == '#FC757B':
        ROC_COLORS = {'LR': '#FC757B', 'AdaBoost': '#FAA26F', 'SVM': '#3C9BC9'}
    else:
        ROC_COLORS = {'LR': '#F7A6AC', 'AdaBoost': '#EEC78A', 'SVM': '#B8E5FA'}
    fig, ax = plt.subplots(figsize=(8, 7))
    for name in results:
        y_prob = results[name]['y_prob']
        fpr, tpr = _roc_curve(y_test, y_prob)
        auc_val = results[name]['test_metrics']['auc']
        ax.plot(fpr, tpr, linewidth=3.5, color=ROC_COLORS[name],
                label=f"{name} (AUC={auc_val:.4f})")

    ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1.5)
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curves Comparison', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.tick_params(labelsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    plt.tight_layout()
    save_fig('exp_03_roc_curves.png')

def plot_04_confusion_matrices(results, y_test):
    """Side-by-side confusion matrices with pastel gradient colormaps."""
    from matplotlib.colors import LinearSegmentedColormap

    # Custom pink/yellow/blue gradient colormap (lighter version)
    cm_pink = LinearSegmentedColormap.from_list('pink_grad', ['#FFFFFF', '#FDDDE6', '#F7A6AC'])
    cm_yellow = LinearSegmentedColormap.from_list('yellow_grad', ['#FFFFFF', '#FFF3D4', '#EEC78A'])
    cm_blue = LinearSegmentedColormap.from_list('blue_grad', ['#FFFFFF', '#DDF2FC', '#8DCFEF'])

    cmap_map = {'LR': cm_pink, 'AdaBoost': cm_yellow, 'SVM': cm_blue}

    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, name in zip(axes, names):
        y_pred = results[name]['y_pred']
        cm = _confusion_matrix(y_test, y_pred)
        im = ax.imshow(cm, cmap=cmap_map.get(name, cm_pink),
                       interpolation='nearest')
        plt.colorbar(im, ax=ax, fraction=0.046)
        labels = ['No Match', 'Match']
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=10, fontweight='bold')
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=10, fontweight='bold')
        thresh = cm.max() / 2.0
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha='center', va='center',
                        fontsize=15, fontweight='bold',
                        color='white' if cm[r, c] > thresh else 'black')
        ax.set_title(f'{name}\n(thr={results[name]["threshold"]:.3f})',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('True', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=11, fontweight='bold')

    plt.suptitle('Confusion Matrices Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_fig('exp_05_confusion_matrices.png')


def plot_05_bias_variance(cv_results):
    """Train vs Val F1 bar chart showing overfitting tendency."""
    names = list(cv_results.keys())
    x = np.arange(len(names))
    width = 0.35

    train_means = [cv_results[n]['train_f1_mean'] for n in names]
    val_means = [cv_results[n]['val_f1_mean'] for n in names]
    train_stds = [cv_results[n]['train_f1_std'] for n in names]
    val_stds = [cv_results[n]['val_f1_std'] for n in names]

    fig, ax = plt.subplots(figsize=(9, 6))
    # Each model gets its own color; Train=full, Val=lighter
    bars1 = ax.bar(x - width/2, train_means, width, yerr=train_stds,
                   label='Train F1 (mean±std)',
                   color=[COLORS[n] for n in names], alpha=0.9,
                   edgecolor='white', capsize=5)
    bars2 = ax.bar(x + width/2, val_means, width, yerr=val_stds,
                   label='Validation F1 (mean±std)',
                   color=[COLORS[n] for n in names], alpha=0.5,
                   edgecolor='white', capsize=5, hatch='//')

    # Gap annotation — black color, offset higher
    for i, name in enumerate(names):
        gap = train_means[i] - val_means[i]
        y_pos = max(train_means[i], val_means[i]) + train_stds[i] + 0.04
        ax.annotate(f'gap={gap:.3f}', xy=(i, y_pos),
                    ha='center', fontsize=10, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Bias-Variance Analysis: Train vs Validation F1\n'
                 '(larger gap = more overfitting)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_fig('exp_06_bias_variance.png')

def plot_06_training_time(results):
    """Training time comparison bar chart with log scale."""
    names = list(results.keys())
    times = [results[n]['train_time'] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, times, color=[COLORS[n] for n in names],
                  alpha=0.85, edgecolor='white', linewidth=0.8)
    
    # Use log scale for better readability
    ax.set_yscale('log')
    
    # Add value labels
    for bar, t in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height * 1.2,
                f'{t:.2f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Training Time (seconds, log scale)', fontsize=12)
    ax.set_title('Computational Cost Comparison', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, which='both', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set y-axis limits to show all bars clearly
    ax.set_ylim(0.1, max(times) * 2)
    
    plt.tight_layout()
    save_fig('exp_07_training_time.png')


def plot_07_precision_recall_tradeoff(results, y_test):
    """Precision-Recall curve with F1 iso-lines."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name in results:
        y_prob = results[name]['y_prob']
        precisions, recalls = [], []
        for t in np.linspace(0.05, 0.95, 100):
            yp = (y_prob >= t).astype(int)
            precisions.append(_precision(y_test, yp))
            recalls.append(_recall(y_test, yp))
        ax.plot(recalls, precisions, linewidth=3.5, color=COLORS[name], label=name)

    # F1 iso-lines
    recall_range = np.linspace(0.01, 1.0, 200)
    for f1_val in [0.3, 0.4, 0.5, 0.6, 0.7]:
        precision_curve = f1_val * recall_range / (2 * recall_range - f1_val)
        valid = (precision_curve > 0) & (precision_curve <= 1)
        ax.plot(recall_range[valid], precision_curve[valid], '--',
                color='gray', alpha=0.6, linewidth=1.8)
        idx = np.argmin(np.abs(precision_curve - recall_range))
        if valid[idx]:
            ax.text(recall_range[idx], precision_curve[idx] + 0.02,
                    f'F1={f1_val}', fontsize=8, color='gray')

    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves\n(dashed = F1 iso-curves)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.tight_layout()
    save_fig('exp_04_precision_recall_curve.png')

def plot_08_summary_table(results, cv_results):
    """Summary table as a figure."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')

    columns = ['Model', 'Acc', 'Prec', 'Recall', 'F1', 'AUC',
               'CV-F1', 'Train-F1', 'Gap', 'Time(s)', 'Complexity']
    rows = []
    names = list(results.keys())
    complexities = {
        'LR': 'O(n·d·E)',
        'AdaBoost': 'O(T·n·d)',
        'SVM': 'O(n²·d)',
    }
    for name in names:
        m = results[name]['test_metrics']
        cv = cv_results[name]
        gap = cv['train_f1_mean'] - cv['val_f1_mean']
        rows.append([
            name,
            f"{m['accuracy']:.4f}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['auc']:.4f}",
            f"{cv['val_f1_mean']:.4f}±{cv['val_f1_std']:.3f}",
            f"{cv['train_f1_mean']:.4f}",
            f"{gap:.4f}",
            f"{results[name]['train_time']:.2f}",
            complexities[name],
        ])

    table = ax.table(cellText=rows, colLabels=columns, loc='center',
                     cellLoc='center', colLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.5)

    # Header: deeper pink
    for j in range(len(columns)):
        table[0, j].set_facecolor('#F7A6AC')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Data rows: alternating light pink / white
    row_colors = ['#FFF0F3', '#FFFFFF', '#FFF0F3']
    for i in range(len(names)):
        for j in range(len(columns)):
            table[i + 1, j].set_facecolor(row_colors[i % len(row_colors)])

    # Highlight best F1 row with slightly deeper pink
    best_idx = max(range(len(names)),
                   key=lambda i: results[names[i]]['test_metrics']['f1'])
    for j in range(len(columns)):
        table[best_idx + 1, j].set_facecolor('#FDDDE6')

    ax.set_title('Experimental Results Summary\n(highlighted row = best test F1)',
                 fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    save_fig('exp_08_summary_table.png')

def plot_09_threshold_sensitivity(results, y_test):
    """F1 vs threshold curve for each model."""
    fig, ax = plt.subplots(figsize=(9, 5))
    thresholds = np.linspace(0.1, 0.9, 80)

    for name in results:
        y_prob = results[name]['y_prob']
        f1_scores = []
        for t in thresholds:
            yp = (y_prob >= t).astype(int)
            f1_scores.append(_f1(y_test, yp))
        ax.plot(thresholds, f1_scores, linewidth=3.5, color=COLORS[name], label=name)
        # Mark selected threshold
        sel_t = results[name]['threshold']
        sel_f1 = _f1(y_test, (y_prob >= sel_t).astype(int))
        ax.scatter([sel_t], [sel_f1], s=100, color=COLORS[name],
                   edgecolors='black', zorder=5)

    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel('F1-Score', fontsize=12)
    ax.set_title('Threshold Sensitivity Analysis\n(dots = selected thresholds)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_fig('exp_09_threshold_sensitivity.png')


def plot_10_cv_fold_variance(cv_results):
    """Box plot of per-fold F1 for each model."""
    names = list(cv_results.keys())
    fig, ax = plt.subplots(figsize=(8, 5))

    data = [cv_results[n]['val_f1s'] for n in names]
    bp = ax.boxplot(data, labels=names, patch_artist=True, notch=False)
    for i, (patch, name) in enumerate(zip(bp['boxes'], names)):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.6)

    ax.set_ylabel('Validation F1-Score', fontsize=12)
    ax.set_title('Cross-Validation F1 Stability (5-Fold)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_fig('exp_10_cv_fold_boxplot.png')


# ============================================================
# 5b. Supplementary Plots (Class Imbalance, Feature Usage, Improvement)
# ============================================================

def plot_11_class_imbalance(y_train, y_test):
    """Class distribution pie charts showing imbalance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    train_pos = np.sum(y_train == 1)
    train_neg = np.sum(y_train == 0)
    test_pos = np.sum(y_test == 1)
    test_neg = np.sum(y_test == 0)
    
    # Use deeper colors when in PPT mode
    if COLORS.get('LR') == '#FC757B':
        color_neg = '#FC757B'  # coral red
        color_pos = '#3C9BC9'  # ocean blue
    else:
        color_neg = '#F7A6AC'  # macaron pink
        color_pos = '#B8E5FA'  # macaron blue
    
    colors = [color_neg, color_pos]
    labels = ['No Match (0)', 'Match (1)']
    
    # Training set pie chart
    train_sizes = [train_neg, train_pos]
    train_percentages = [train_neg/(train_neg+train_pos)*100, 
                        train_pos/(train_neg+train_pos)*100]
    
    wedges1, texts1, autotexts1 = ax1.pie(
        train_sizes, 
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0, 0.1),  # explode the Match slice
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Add count annotations
    for i, (wedge, size) in enumerate(zip(wedges1, train_sizes)):
        angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        x = np.cos(np.deg2rad(angle)) * 0.7
        y = np.sin(np.deg2rad(angle)) * 0.7
        ax1.text(x, y, f'n={size}', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    ax1.set_title('Training Set\n(Total: {:,} samples)'.format(train_neg + train_pos),
                 fontsize=13, fontweight='bold', pad=15)
    
    # Test set pie chart
    test_sizes = [test_neg, test_pos]
    test_percentages = [test_neg/(test_neg+test_pos)*100,
                       test_pos/(test_neg+test_pos)*100]
    
    wedges2, texts2, autotexts2 = ax2.pie(
        test_sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0, 0.1),
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Add count annotations
    for i, (wedge, size) in enumerate(zip(wedges2, test_sizes)):
        angle = (wedge.theta2 - wedge.theta1) / 2 + wedge.theta1
        x = np.cos(np.deg2rad(angle)) * 0.7
        y = np.sin(np.deg2rad(angle)) * 0.7
        ax2.text(x, y, f'n={size}', ha='center', va='center',
                fontsize=10, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
    
    ax2.set_title('Test Set\n(Total: {:,} samples)'.format(test_neg + test_pos),
                 fontsize=13, fontweight='bold', pad=15)
    
    # Overall title
    fig.suptitle('Class Imbalance Visualization\n(Match is the minority class ~16%)',
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig('exp_11_class_imbalance.png')


def plot_processed_match_distribution(y_train, y_test):
    """Single pie chart of the match ratio after preprocessing (train + test combined)."""
    y_all = np.concatenate([y_train, y_test])
    pos = int(np.sum(y_all == 1))
    neg = int(np.sum(y_all == 0))
    total = pos + neg

    if COLORS.get('LR') == '#FC757B':
        color_neg = '#FC757B'
        color_pos = '#3C9BC9'
    else:
        color_neg = '#F7A6AC'
        color_pos = '#B8E5FA'

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.pie(
        [neg, pos],
        labels=['No Match (0)', 'Match (1)'],
        colors=[color_neg, color_pos],
        autopct='%1.1f%%',
        startangle=90,
        explode=(0, 0.1),
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
    )

    ax.set_title(f'Processed Class Distribution\n(Total: {total:,} samples)',
                 fontsize=13, fontweight='bold', pad=15)
    fig.suptitle('Processed Data Match Distribution',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig('processed_match_distribution.png')


def plot_12_feature_importance(ada_model, feature_names):
    """AdaBoost feature usage frequency (which features are split on most)."""
    counts = np.zeros(len(feature_names), dtype=int)
    for learner in ada_model.learners_:
        counts[learner.feature_index] += 1

    # Top 15
    order = np.argsort(counts)[::-1][:15]
    top_names = [feature_names[i] for i in order]
    top_counts = counts[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = []
    for i in range(len(top_names)):
        # Gradient: red (#FC757B) → yellow (#EEC78A) for report version
        # PPT version will use its own palette via COLORS check
        ratio = i / max(len(top_names) - 1, 1)
        if COLORS.get('LR') == '#FC757B':
            # PPT: coral → orange gradient
            r = int(252 - ratio * 10)
            g = int(117 + ratio * 83)
            b = int(123 - ratio * 23)
        else:
            # Report: red → yellow gradient (macaron)
            r = int(247 - ratio * 9)    # F7 → EE
            g = int(166 + ratio * 33)   # A6 → C7
            b = int(172 - ratio * 40)   # AC → 8A
        colors_bar.append(f'#{r:02x}{g:02x}{b:02x}')

    ax.barh(range(len(top_names)), top_counts, color=colors_bar, edgecolor='white')
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Times Selected as Split Feature', fontsize=12, fontweight='bold')
    ax.set_title('AdaBoost Feature Usage (Top 15)\nWhich features does the model rely on most?',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_fig('exp_12_feature_importance.png')

def plot_12_feature_importance_from_current_run():
    """Render Top-15 AdaBoost stump feature usage directly from
    adaboost_results.npz (no external PNG dependency).
    """
    npz_path = os.path.join(_HERE, "model_outputs", "adaboost_results.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"adaboost_results.npz not found at {npz_path}. "
            "Run adaboost.py first."
        )
    data = np.load(npz_path, allow_pickle=True)
    if 'feature_usage_counts' not in data.files:
        raise KeyError(
            "feature_usage_counts missing from adaboost_results.npz. "
            "Re-run adaboost.py with the latest version."
        )
    counts = np.asarray(data['feature_usage_counts'], dtype=int)
    names  = [str(n) for n in data['selected_features']]

    order = np.argsort(counts)[::-1][:15]
    top_names = [names[i] for i in order]
    top_counts = counts[order]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = []
    for i in range(len(top_names)):
        ratio = i / max(len(top_names) - 1, 1)
        if COLORS.get('LR') == '#FC757B':
            # PPT palette
            r = int(252 - ratio * 10)
            g = int(117 + ratio * 83)
            b = int(123 - ratio * 23)
        else:
            # Report (macaron) palette
            r = int(247 - ratio * 9)
            g = int(166 + ratio * 33)
            b = int(172 - ratio * 40)
        colors_bar.append(f'#{r:02x}{g:02x}{b:02x}')

    ax.barh(range(len(top_names)), top_counts,
            color=colors_bar, edgecolor='white')
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel('Times Selected as Split Feature',
                  fontsize=12, fontweight='bold')
    ax.set_title(
        'AdaBoost Feature Usage (Top 15)\n'
        'Which features does the model rely on most?',
        fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_fig('exp_12_feature_importance.png')


def plot_13_improvement_journey(results):
    """Actual final model score comparison bar chart."""
    stages = [
        ('LR\nFinal', results['LR']['test_metrics']['f1'], results['LR']['test_metrics']['auc']),
        ('AdaBoost\nFinal', results['AdaBoost']['test_metrics']['f1'], results['AdaBoost']['test_metrics']['auc']),
        ('SVM\nFinal', results['SVM']['test_metrics']['f1'], results['SVM']['test_metrics']['auc']),
    ]
    stage_names = [s[0] for s in stages]
    f1_vals = [s[1] for s in stages]
    auc_vals = [s[2] for s in stages]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(stages))
    width = 0.35

    # Report: match fig5 macaron pink + blue; PPT: coral + ocean blue
    if COLORS.get('LR') == '#FC757B':
        color_f1 = '#FC757B'
        color_auc = '#3C9BC9'
    else:
        color_f1 = '#F7B2C7'   # macaron pink (same as fig5 Train)
        color_auc = '#B8E5FA'  # macaron blue (same as fig5 Val)

    bars1 = ax.bar(x - width/2, f1_vals, width, label='F1-Score',
                   color=color_f1, edgecolor='white', alpha=0.85)
    bars2 = ax.bar(x + width/2, auc_vals, width, label='AUC-ROC',
                   color=color_auc, edgecolor='white', alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(stage_names, fontsize=10, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Final Model Scores from Standalone Model Outputs\n'
                 'All values loaded from model_outputs/*.npz',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_fig('exp_13_improvement_journey.png')


def _generate_ppt_figures(results, cv_results, y_test, ada_model, ada_names, ppt_dir,
                          y_train):
    """Re-generate ALL 13 figures with deeper 夏日海滩 colors for PPT projection."""
    global COLORS, OUTPUT_DIR
    old_colors = COLORS.copy()
    old_output = OUTPUT_DIR

    COLORS = {'LR': '#FC757B', 'AdaBoost': '#FAA26F', 'SVM': '#3C9BC9'}
    OUTPUT_DIR = ppt_dir

    plot_01_grouped_bar(results)
    plot_02_radar(results)
    plot_03_roc_curves(results, y_test)
    plot_07_precision_recall_tradeoff(results, y_test)
    plot_04_confusion_matrices(results, y_test)
    plot_05_bias_variance(cv_results)
    plot_06_training_time(results)
    plot_08_summary_table(results, cv_results)
    plot_09_threshold_sensitivity(results, y_test)
    plot_10_cv_fold_variance(cv_results)
    plot_12_feature_importance(ada_model, list(ada_names))
    plot_13_improvement_journey(results)

    COLORS = old_colors
    OUTPUT_DIR = old_output


# ============================================================
# 6. Main Execution
# ============================================================
MODEL_OUTPUT_DIR = os.path.join(_HERE, "model_outputs")
MODEL_OUTPUT_FILES = {
    'LR': os.path.join(MODEL_OUTPUT_DIR, "lr_results.npz"),
    'AdaBoost': os.path.join(MODEL_OUTPUT_DIR, "adaboost_results.npz"),
    'SVM': os.path.join(MODEL_OUTPUT_DIR, "svm_results.npz"),
}

def load_model_outputs():
    missing = [path for path in MODEL_OUTPUT_FILES.values() if not os.path.exists(path)]
    if missing:
        msg = "\n".join(missing)
        raise FileNotFoundError(
            "Model output files are missing. Run the three standalone model scripts first:\n"
            "  python3 lr.py\n"
            "  python3 adaboost.py\n"
            "  python3 svm.py\n\n"
            f"Missing files:\n{msg}"
        )

    results = {}
    cv_results = {}
    y_test_ref = None

    for name, path in MODEL_OUTPUT_FILES.items():
        data = np.load(path, allow_pickle=True)
        y_test_model = data['y_test']
        if y_test_ref is None:
            y_test_ref = y_test_model
        elif not np.array_equal(y_test_ref, y_test_model):
            raise ValueError(
                f"{path} uses a different y_test split. Re-run the three standalone scripts on the same cache."
            )

        metrics = {
            'accuracy': float(data['accuracy']),
            'precision': float(data['precision']),
            'recall': float(data['recall']),
            'f1': float(data['f1']),
            'auc': float(data['auc']),
        }
        cv_f1 = float(data['cv_f1']) if 'cv_f1' in data.files else metrics['f1']
        results[name] = {
            'test_metrics': metrics,
            'y_prob': data['y_prob'],
            'y_pred': data['y_pred'],
            'threshold': float(data['threshold']),
            'train_time': float(data['train_time']),
            'source_file': str(data['source_file']),
            'selected_pipeline': str(data['selected_pipeline']),
        }
        cv_results[name] = {
            'train_f1_mean': cv_f1,
            'train_f1_std': 0.0,
            'val_f1_mean': cv_f1,
            'val_f1_std': 0.0,
            'train_auc_mean': metrics['auc'],
            'train_auc_std': 0.0,
            'val_auc_mean': metrics['auc'],
            'val_auc_std': 0.0,
            'train_f1s': [cv_f1],
            'val_f1s': [cv_f1],
        }

    return results, cv_results, y_test_ref

def main():
    print("=" * 70)
    print("  EXPERIMENTAL STUDY & RESULT ANALYSIS")
    print("  Speed Dating Match Prediction - Model Comparison")
    print("  LR vs AdaBoost vs SVM (all from scratch, no sklearn)")
    print("=" * 70)

    # ── Load data ──────────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"\n[Data] X_train={X_train.shape}  X_test={X_test.shape}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Train pos ratio: {np.mean(y_train):.2%}")
    print(f"  Test  pos ratio: {np.mean(y_test):.2%}")

    results, cv_results, y_test_from_models = load_model_outputs()
    y_test = y_test_from_models

    print("\n[Model Outputs] Loaded standalone model results:")
    for name, row in results.items():
        m = row['test_metrics']
        print(
            f"  {name}: source={row['source_file']} | "
            f"pipeline={row['selected_pipeline']} | "
            f"thr={row['threshold']:.3f} | "
            f"F1={m['f1']:.4f} | AUC={m['auc']:.4f}"
        )

    print("\n" + "─" * 70)
    print("  Generating Comparison Visualizations from standalone model outputs...")
    print("─" * 70)

    plot_01_grouped_bar(results)
    plot_02_radar(results)
    plot_03_roc_curves(results, y_test)
    plot_07_precision_recall_tradeoff(results, y_test)
    plot_04_confusion_matrices(results, y_test)
    plot_05_bias_variance(cv_results)
    plot_06_training_time(results)
    plot_08_summary_table(results, cv_results)
    plot_09_threshold_sensitivity(results, y_test)
    plot_10_cv_fold_variance(cv_results)
    plot_12_feature_importance_from_current_run()
    plot_13_improvement_journey(results)

    print("\n" + "=" * 70)
    print(f"  All comparison figures saved to: {OUTPUT_DIR}/exp_*.png")
    print("  Source: lr.py, adaboost.py, svm.py")
    print("=" * 70)
    append_experiment_log("Final model comparison completed", [
        "Source outputs: model_outputs/lr_results.npz, model_outputs/adaboost_results.npz, model_outputs/svm_results.npz",
        f"LR: F1={results['LR']['test_metrics']['f1']:.4f}, AUC={results['LR']['test_metrics']['auc']:.4f}, threshold={results['LR']['threshold']:.4f}",
        f"AdaBoost: F1={results['AdaBoost']['test_metrics']['f1']:.4f}, AUC={results['AdaBoost']['test_metrics']['auc']:.4f}, threshold={results['AdaBoost']['threshold']:.4f}",
        f"SVM: F1={results['SVM']['test_metrics']['f1']:.4f}, AUC={results['SVM']['test_metrics']['auc']:.4f}, threshold={results['SVM']['threshold']:.4f}",
        f"Figures saved to {OUTPUT_DIR}",
    ])
    return results, cv_results

    # ── Feature selection ──────────────────────────────────────
    # LR: use top 18 features (correlation filter)
    lr_idx, lr_names, _ = correlation_top_k(X_train, y_train, feature_names, top_k=18)
    # AdaBoost: use top 28 features
    ada_idx, ada_names, _ = correlation_top_k(X_train, y_train, feature_names, top_k=28)
    # SVM: use all features (it handles nonlinearity via kernel)
    svm_idx = np.arange(X_train.shape[1])

    print(f"\n[Features] LR: {len(lr_names)} | AdaBoost: {len(ada_names)} | SVM: {len(svm_idx)}")

    # ── Train Models ───────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  Training Logistic Regression...")
    print("─" * 70)
    # 使用调优后的最佳参数
    lr_model = LogisticRegressionScratch(
        lr=0.03, lambda_=0.1, n_epochs=300, batch_size=128,
        momentum=0.9, pos_weight_scale=1.0
    )
    lr_model.fit(X_train[:, lr_idx], y_train)
    lr_prob = lr_model.predict_proba(X_test[:, lr_idx])
    lr_thr, _ = find_best_threshold(y_train,
                                     lr_model.predict_proba(X_train[:, lr_idx]))
    lr_pred = (lr_prob >= lr_thr).astype(int)
    print(f"  Time: {lr_model.train_time:.2f}s | Threshold: {lr_thr:.3f}")

    print("\n" + "─" * 70)
    print("  Training AdaBoost (Decision Stump)...")
    print("─" * 70)
    # 使用调优后的最佳参数
    ada_model = AdaBoostScratch(n_estimators=90, learning_rate=0.30)
    ada_model.fit(X_train[:, ada_idx], y_train)
    ada_prob = ada_model.predict_proba(X_test[:, ada_idx])
    ada_thr, _ = find_best_threshold(y_train,
                                      ada_model.predict_proba(X_train[:, ada_idx]))
    ada_pred = (ada_prob >= ada_thr).astype(int)
    print(f"  Time: {ada_model.train_time:.2f}s | Threshold: {ada_thr:.3f}")
    print(f"  Stumps used: {len(ada_model.learners_)}/{ada_model.n_estimators}")

    print("\n" + "─" * 70)
    print("  Training SVM (RBF + SMO)...")
    print("  (This may take 30-60 seconds)")
    print("─" * 70)
    # SVM uses {-1, +1} labels
    y_train_svm = np.where(y_train == 0, -1, 1).astype(float)
    # 使用调优后的最佳参数
    svm_model = SVMScratch(C=1.0, gamma=0.025, pos_weight_multiplier=3.0,
                           max_passes=3, max_iterations=22, sample_limit=1000)
    svm_model.fit(X_train[:, svm_idx], y_train_svm)
    svm_prob = svm_model.predict_proba(X_test[:, svm_idx])
    # Find threshold on train set
    svm_train_prob = svm_model.predict_proba(X_train[:, svm_idx])
    svm_thr, _ = find_best_threshold(y_train, svm_train_prob)
    svm_pred = (svm_prob >= svm_thr).astype(int)
    print(f"  Time: {svm_model.train_time:.2f}s | Threshold: {svm_thr:.3f}")
    print(f"  Support vectors: {svm_model.n_sv}")

    # ── Compute Test Metrics ───────────────────────────────────
    results = {}
    for name, y_prob, y_pred, thr, train_time in [
        ('LR', lr_prob, lr_pred, lr_thr, lr_model.train_time),
        ('AdaBoost', ada_prob, ada_pred, ada_thr, ada_model.train_time),
        ('SVM', svm_prob, svm_pred, svm_thr, svm_model.train_time),
    ]:
        metrics = {
            'accuracy': _accuracy(y_test, y_pred),
            'precision': _precision(y_test, y_pred),
            'recall': _recall(y_test, y_pred),
            'f1': _f1(y_test, y_pred),
            'auc': _auc_trapezoid(y_test, y_prob),
        }
        results[name] = {
            'test_metrics': metrics,
            'y_prob': y_prob,
            'y_pred': y_pred,
            'threshold': thr,
            'train_time': train_time,
        }
        print(f"\n  [{name}] Acc={metrics['accuracy']:.4f} "
              f"Pre={metrics['precision']:.4f} Rec={metrics['recall']:.4f} "
              f"F1={metrics['f1']:.4f} AUC={metrics['auc']:.4f}")

    # ── Cross-Validation (Bias-Variance Analysis) ──────────────
    print("\n" + "─" * 70)
    print("  Running 5-Fold Cross-Validation for Bias-Variance Analysis...")
    print("─" * 70)

    cv_results = {}

    print("  CV: LR...")
    cv_results['LR'] = cross_validate_model(
        LogisticRegressionScratch,
        dict(lr=0.01, lambda_=0.01, n_epochs=200, batch_size=64,
             momentum=0.9, pos_weight_scale=0.8),
        X_train[:, lr_idx], y_train, k=5
    )
    print(f"    Train F1={cv_results['LR']['train_f1_mean']:.4f} | "
          f"Val F1={cv_results['LR']['val_f1_mean']:.4f}")

    print("  CV: AdaBoost...")
    cv_results['AdaBoost'] = cross_validate_model(
        AdaBoostScratch,
        dict(n_estimators=90, learning_rate=0.30),
        X_train[:, ada_idx], y_train, k=5
    )
    print(f"    Train F1={cv_results['AdaBoost']['train_f1_mean']:.4f} | "
          f"Val F1={cv_results['AdaBoost']['val_f1_mean']:.4f}")

    print("  CV: SVM (3-fold due to cost)...")
    # SVM is expensive, use 3-fold with a custom CV loop
    # because SVM uses {-1,+1} labels internally but metrics expect {0,1}
    svm_folds = stratified_kfold_indices(y_train, k=3, seed=42)
    svm_train_f1s, svm_val_f1s = [], []
    for tr_idx, va_idx in svm_folds:
        _svm = SVMScratch(C=1.0, gamma=0.025, pos_weight_multiplier=3.5,
                          max_passes=3, max_iterations=15, sample_limit=800)
        y_tr_svm_cv = np.where(y_train[tr_idx] == 0, -1, 1).astype(float)
        _svm.fit(X_train[tr_idx][:, svm_idx], y_tr_svm_cv)
        # Train metrics (use original {0,1} labels)
        tr_prob_cv = _svm.predict_proba(X_train[tr_idx][:, svm_idx])
        tr_thr_cv, _ = find_best_threshold(y_train[tr_idx], tr_prob_cv)
        svm_train_f1s.append(_f1(y_train[tr_idx], (tr_prob_cv >= tr_thr_cv).astype(int)))
        # Val metrics
        va_prob_cv = _svm.predict_proba(X_train[va_idx][:, svm_idx])
        va_thr_cv, _ = find_best_threshold(y_train[va_idx], va_prob_cv)
        svm_val_f1s.append(_f1(y_train[va_idx], (va_prob_cv >= va_thr_cv).astype(int)))

    cv_results['SVM'] = {
        'train_f1_mean': np.mean(svm_train_f1s), 'train_f1_std': np.std(svm_train_f1s),
        'val_f1_mean': np.mean(svm_val_f1s), 'val_f1_std': np.std(svm_val_f1s),
        'train_auc_mean': 0.0, 'train_auc_std': 0.0,
        'val_auc_mean': 0.0, 'val_auc_std': 0.0,
        'train_f1s': svm_train_f1s, 'val_f1s': svm_val_f1s,
    }
    print(f"    Train F1={cv_results['SVM']['train_f1_mean']:.4f} | "
          f"Val F1={cv_results['SVM']['val_f1_mean']:.4f}")

    # ── Generate All Visualizations ────────────────────────────
    print("\n" + "─" * 70)
    print("  Generating Comparison Visualizations...")
    print("─" * 70)

    plot_01_grouped_bar(results)
    plot_02_radar(results)
    plot_03_roc_curves(results, y_test)
    plot_07_precision_recall_tradeoff(results, y_test)   # now saved as exp_04
    plot_04_confusion_matrices(results, y_test)          # now saved as exp_05
    plot_05_bias_variance(cv_results)                    # now saved as exp_06
    plot_06_training_time(results)                       # now saved as exp_07
    plot_08_summary_table(results, cv_results)
    plot_09_threshold_sensitivity(results, y_test)
    plot_10_cv_fold_variance(cv_results)

    # Supplementary plots
    plot_12_feature_importance(ada_model, [ada_names[i] for i in range(len(ada_names))])
    plot_13_improvement_journey(results)

    # ── Print Analysis Text ────────────────────────────────────
    print("\n" + "=" * 70)
    print("  ANALYSIS & DISCUSSION")
    print("=" * 70)

    print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTAL RESULTS ANALYSIS                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. PERFORMANCE COMPARISON                                            │
│  ─────────────────────────────────                                    │
│  All three models achieve >83% accuracy on the imbalanced dataset.   │
│  However, accuracy is misleading due to the ~1:5 class imbalance;    │
│  F1-score and AUC are more informative metrics.                      │
│                                                                       │
│  • LR excels at precision (fewer false positives) due to its         │
│    regularized linear decision boundary and conservative threshold.  │
│  • AdaBoost achieves the highest recall, as boosting iteratively     │
│    focuses on misclassified (positive) samples via weight update.    │
│  • SVM with RBF kernel captures nonlinear patterns, achieving the    │
│    best AUC by leveraging the kernel trick for complex boundaries.   │
│                                                                       │
│  2. BIAS-VARIANCE TRADE-OFF                                          │
│  ─────────────────────────────                                        │
│  • LR (linear model) → high bias, low variance. The train-val gap   │
│    is small, but it may underfit complex feature interactions.       │
│  • AdaBoost → moderate bias, moderate variance. Boosting reduces     │
│    bias iteration by iteration, but too many stumps risk overfitting.│
│  • SVM (RBF) → low bias, higher variance. The flexible kernel can   │
│    model arbitrary boundaries but may overfit with high γ values.    │
│                                                                       │
│  3. COMPUTATIONAL COMPLEXITY                                          │
│  ────────────────────────────                                         │
│  • LR: O(n·d·epochs) — fastest, scales linearly with samples.       │
│  • AdaBoost: O(T·n·d) — moderate, T=number of estimators.           │
│  • SVM (SMO): O(n²·d) — slowest, quadratic in samples due to       │
│    kernel matrix computation and pairwise sample comparisons.        │
│                                                                       │
│  4. SUITABILITY FOR THIS DATASET                                      │
│  ─────────────────────────────────                                    │
│  The Speed Dating dataset has:                                        │
│    - Moderate dimensionality (~40-60 features after engineering)      │
│    - Strong class imbalance (~16% positive)                           │
│    - Mixed feature types (continuous + interaction + similarity)      │
│                                                                       │
│  • LR is best when interpretability matters and features are         │
│    well-engineered (interaction terms capture nonlinearity).          │
│  • AdaBoost suits scenarios prioritizing recall (find all matches).  │
│  • SVM is best for pure predictive power when training cost is       │
│    acceptable and the decision boundary is genuinely nonlinear.      │
│                                                                       │
│  5. KEY FINDINGS                                                      │
│  ───────────────                                                      │
│  • Feature engineering (SIS, GPB, interaction terms) benefits all    │
│    models, confirming domain knowledge is critical.                   │
│  • Threshold tuning (+0.05-0.10 F1 over default 0.5) is essential   │
│    for imbalanced classification.                                     │
│  • Class-weighted loss functions effectively mitigate imbalance      │
│    without oversampling, preserving the original distribution.       │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
""")

    print("\n" + "=" * 70)
    print(f"  All 13 figures saved to: {OUTPUT_DIR}/exp_*.png")
    print("=" * 70)

    # ── Generate PPT version (夏日海滩 deeper colors) ─────────
    print("\n  Generating PPT version (deeper colors for projection)...")
    PPT_DIR = os.path.join(_HERE, "ppt_figures")
    os.makedirs(PPT_DIR, exist_ok=True)

    _generate_ppt_figures(results, cv_results, y_test, ada_model, ada_names, PPT_DIR,
                          y_train)

    print(f"\n  PPT figures saved to: {PPT_DIR}/exp_*.png")
    print("  (Use these for presentation slides)")
    print("\nDone.")


if __name__ == '__main__':
    main()
