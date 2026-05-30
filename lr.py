# ============================================================
# lr.py  --  Final Logistic Regression Model (FAIR TIMING)
#   Speed Dating Match Prediction (from scratch, no sklearn)
#
#   Focus: Final LR model using grid-search selected best configuration.
#          Pipeline: Re-selected Correlation Top-18 features
#          Hyperparameters: lr=0.03, lambda_=0.1, batch_size=64,
#                           n_epochs=100, momentum=0.9, penalty='l2',
#                           class_weight='balanced', pos_weight_scale=0.7
#          Threshold: 0.548
#
#   Model selection (96-combo 5-fold CV grid search) was done
#   in lr_grid.py — this file ONLY trains the final model and
#   times fit() for fair comparison with adaboost.py / svm.py.
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
    """Load from .npy cache. Run ML_data.py first if cache missing."""
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
    print("Train positive ratio:", f"{np.mean(y_train):.4f}")
    print("Test positive ratio :", f"{np.mean(y_test):.4f}")
    print("X_train fp:", array_fingerprint(X_train))
    print("X_test fp :", array_fingerprint(X_test))
    feature_text = "|".join(feature_names)
    feature_fp = hashlib.md5(feature_text.encode()).hexdigest()
    print("feature_names fp:", feature_fp)
    print("=" * 60)


# ============================================================
# 1. Logistic Regression From Scratch (NumPy only)
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
        self.penalty          = penalty
        self.pos_weight_scale = pos_weight_scale
        self.theta            = None
        self.train_losses     = []
        self.train_time       = 0.0

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

    def fit(self, X_train, y_train):
        start_time = time.time()
        n, d       = X_train.shape
        Xb         = np.c_[np.ones(n), X_train]
        self.theta = np.zeros(d + 1)
        velocity   = np.zeros(d + 1)

        for _ in range(self.n_epochs):
            perm = np.random.permutation(n)
            Xbs  = Xb[perm];  ys = y_train[perm]

            for s in range(0, n, self.batch_size):
                Xbi = Xbs[s:s + self.batch_size]
                ybi = ys[s:s + self.batch_size]
                nb  = len(ybi)
                p   = self._sigmoid(Xbi @ self.theta)
                err = self._sample_weights(ybi) * (p - ybi)
                grad = Xbi.T @ err / nb
                if self.penalty == 'l1':
                    grad[1:] += (self.lambda_ / n) * np.sign(self.theta[1:])
                else:
                    grad[1:] += (self.lambda_ / n) * self.theta[1:]
                velocity    = self.momentum * velocity + self.lr * grad
                self.theta -= velocity

            self.train_losses.append(self._logloss(Xb, y_train, self.theta))

        self.train_time = time.time() - start_time
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
# 2. Evaluation (no sklearn metrics allowed)
# ============================================================
def _accuracy(yt, yp):  return float(np.mean(yt == yp))

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

def _auc_trapezoid(yt, y_prob):
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
# 3. Correlation Top-K Feature Selection (deterministic)
# ============================================================
def correlation_top_k_selection(X, y, feature_names, top_k=18):
    """Pure NumPy Pearson correlation Top-K selection.
    Deterministic — same result every run for the same X, y."""
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
    for rank, idx in enumerate(sel_idx, 1):
        print(f"    {rank:2d}. {feature_names[idx]:30s} "
              f"|r|={correlations[idx]:.4f}")
    return sel_idx, sel_names


# ============================================================
# 4. (Plots removed — experimental_analysis.py renders all
#     comparison figures from the saved .npz file.)
# ============================================================


# ============================================================
# 5. Main — Train Final Model with FROZEN best hyperparameters
# ============================================================
# Selected via lr_grid.py (96 combos × 5-fold CV).
# Pipeline: Re-selected Correlation Top-18 features.
BEST_PIPELINE_NAME = "Re-selected Corr Top18"
TOP_K_FEATURES     = 18
BEST_PARAMS = {
    'lr':               0.03,
    'lambda_':          0.1,
    'batch_size':       64,
    'n_epochs':         100,
    'momentum':         0.9,
    'penalty':          'l2',
    'class_weight':     'balanced',
    'pos_weight_scale': 0.7,
}
BEST_THRESHOLD = 0.548
BEST_CV_F1     = 0.5691  # from lr_grid.py


def main():
    print("=" * 65)
    print("  lr.py  --  Final LR (frozen best params from lr_grid.py)")
    print("  Pipeline:", BEST_PIPELINE_NAME)
    print("  Threshold:", BEST_THRESHOLD)
    print("=" * 65)

    # ── 1. Load data ──────────────────────────────────────────
    X_train, X_test, y_train, y_test, feature_names = load_data_smart()
    print_split_fingerprint(X_train, X_test, y_train, y_test,
                            feature_names, tag="[LR]")

    # ── 2. Select Top-K correlated features ──────────────────
    sel_idx, sel_names = correlation_top_k_selection(
        X_train, y_train, feature_names, top_k=TOP_K_FEATURES)
    Xtr = X_train[:, sel_idx]
    Xte = X_test[:, sel_idx]
    print(f"\n[Data]  X_train={Xtr.shape}  X_test={Xte.shape}")

    # ── 3. Train final model (THIS IS THE FAIR-TIMED FIT) ────
    print("\n" + "=" * 65)
    print(f"  Training FINAL LR with frozen best params ...")
    print("=" * 65)
    for k, v in BEST_PARAMS.items():
        print(f"    {k:<20} = {v}")

    # Fix RNG state right before fit() for reproducibility.
    # seed=3 reproduces the F1=0.5632 reported by lr_grid.py.
    np.random.seed(3)
    model = LogisticRegressionScratch(**BEST_PARAMS)
    model.fit(Xtr, y_train)
    print(f"\n  Training time: {model.train_time:.4f}s "
          f"(fit() only — no grid search)")

    # ── 4. Evaluate on test set ──────────────────────────────
    y_prob = model.predict_proba(Xte)
    y_pred = model.predict(Xte, threshold=BEST_THRESHOLD)
    metrics = full_eval(y_test, y_pred, y_prob,
                        label=f'{BEST_PIPELINE_NAME} -- Test Set')

    # ── 5. Plots ─────────────────────────────────────────────
    # Disabled: experimental_analysis.py renders all comparison
    # figures from lr_results.npz, so we don't write working PNGs
    # to data_outputs here.

    # ── 6. Save .npz for experimental_analysis.py ────────────
    model_output_dir = os.path.join(_HERE, "model_outputs")
    os.makedirs(model_output_dir, exist_ok=True)
    out_path = os.path.join(model_output_dir, "lr_results.npz")
    np.savez(
        out_path,
        model_name        = np.array("LR"),
        source_file       = np.array("lr.py"),
        selected_pipeline = np.array(BEST_PIPELINE_NAME),
        y_prob            = y_prob,
        y_pred            = y_pred,
        y_test            = y_test,
        threshold         = np.array(BEST_THRESHOLD),
        train_time        = np.array(model.train_time),
        accuracy          = np.array(metrics['acc']),
        precision         = np.array(metrics['pre']),
        recall            = np.array(metrics['rec']),
        f1                = np.array(metrics['f1']),
        auc               = np.array(metrics['auc']),
        cv_f1             = np.array(BEST_CV_F1),
        n_features        = np.array(TOP_K_FEATURES),
    )
    print(f"\n[Saved] {out_path}")

    append_experiment_log("LR final model (frozen params) completed", [
        "Source script: lr.py",
        f"Selected pipeline={BEST_PIPELINE_NAME}",
        f"Hyperparameters={BEST_PARAMS}",
        f"Threshold={BEST_THRESHOLD}",
        f"Accuracy={metrics['acc']:.4f}",
        f"Precision={metrics['pre']:.4f}",
        f"Recall={metrics['rec']:.4f}",
        f"F1={metrics['f1']:.4f}",
        f"AUC={metrics['auc']:.4f}",
        f"Train time={model.train_time:.4f}s (fit only, fair timing)",
        f"Output={out_path}",
    ])

    print("\n" + "=" * 65)
    print(f"  [Done]  LR final-model fit time = {model.train_time:.4f}s")
    print("=" * 65)
    return metrics


if __name__ == "__main__":
    main()
