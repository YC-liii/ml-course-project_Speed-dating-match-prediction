"""
svm.py  --  Final SVM Ensemble Model (FAIR TIMING)
    Speed Dating Match Prediction (from scratch, no sklearn)

Focus: Final SVM Top-5 ensemble using configurations selected
       in svm_grid.py.
       - 5 RBF SVMs with different (gamma, pos_weight_multiplier)
       - Each model gets its own Platt scaling
       - Final score = mean of 5 calibrated probabilities
       - Threshold: 0.340 (selected on validation)

Model selection (15-config focused search + 64 ensemble variants
search) was done in svm_grid.py — this file ONLY trains the
final 5 SVMs + Platt + ensemble for fair comparison with
lr.py / adaboost.py.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data_outputs")
EXPERIMENT_LOG = os.path.join(_HERE, "experiment_run_log.md")


def append_experiment_log(title, lines):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n## {timestamp} - {title}\n")
        for line in lines:
            f.write(f"- {line}\n")


# ============================================================
# Final ensemble configuration (frozen from svm_grid.py)
# ============================================================
ENSEMBLE_CONFIGS = [
    {'name': 'focused_rbf_C_1.0_gamma_0.020_posw_3.0',
     'C': 1.0, 'gamma': 0.020, 'pos_weight_multiplier': 3.0},
    {'name': 'focused_rbf_C_1.0_gamma_0.018_posw_3.5',
     'C': 1.0, 'gamma': 0.018, 'pos_weight_multiplier': 3.5},
    {'name': 'focused_rbf_C_1.0_gamma_0.018_posw_4.0',
     'C': 1.0, 'gamma': 0.018, 'pos_weight_multiplier': 4.0},
    {'name': 'focused_rbf_C_1.0_gamma_0.025_posw_4.0',
     'C': 1.0, 'gamma': 0.025, 'pos_weight_multiplier': 4.0},
    {'name': 'focused_rbf_C_1.0_gamma_0.030_posw_4.0',
     'C': 1.0, 'gamma': 0.030, 'pos_weight_multiplier': 4.0},
]
ENSEMBLE_THRESHOLD = 0.340
ENSEMBLE_PIPELINE_NAME = "Top-5 val_f1 platt_mean validation ensemble"

# ============================================================
# Metrics (no sklearn)
# ============================================================
def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def precision(y_true, y_pred):
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    return tp / (tp + fp) if tp + fp > 0 else 0.0


def recall(y_true, y_pred):
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    return tp / (tp + fn) if tp + fn > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def compute_auc(y_true, scores):
    if np.max(scores) == np.min(scores):
        thresholds = np.array([np.inf, -np.inf])
    else:
        eps = 1e-12
        thresholds = np.linspace(
            np.max(scores) + eps,
            np.min(scores) - eps,
            200
        )
    tpr_list = []
    fpr_list = []
    P = float(np.sum(y_true == 1))
    N = float(np.sum(y_true == 0))
    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        tp = float(np.sum((y_pred == 1) & (y_true == 1)))
        fp = float(np.sum((y_pred == 1) & (y_true == 0)))
        tpr_list.append(tp / P if P > 0 else 0)
        fpr_list.append(fp / N if N > 0 else 0)
    auc = np.trapz(tpr_list, fpr_list)
    return float(auc), fpr_list, tpr_list


def compute_average_precision(y_true, scores):
    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]
    total_pos = float(np.sum(y_true == 1))
    if total_pos == 0:
        return 0.0
    tp = 0
    precision_sum = 0.0
    for rank, label in enumerate(y_sorted, start=1):
        if label == 1:
            tp += 1
            precision_sum += tp / rank
    return precision_sum / total_pos


def safe_divide(a, b):
    return a / (np.abs(b) + 1e-6)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# Feature Engineering (matches svm_grid.py exactly)
# ============================================================
def add_engineered_features(X_base, feature_names):
    idx = {name: i for i, name in enumerate(feature_names)}
    new_columns = []
    new_names = []

    def has(*names):
        return all(name in idx for name in names)

    def add(name, values):
        new_columns.append(values.reshape(-1, 1))
        new_names.append(name)

    if has('like', 'guess_prob_liked'):
        like = X_base[:, idx['like']]
        guess = X_base[:, idx['guess_prob_liked']]
        add('eng_like_x_guess_prob_liked', like * guess)
        add('eng_like_minus_guess_prob_liked', like - guess)
        add('eng_abs_like_minus_guess_prob_liked', np.abs(like - guess))
    if has('like', 'partner_avg_trait'):
        like = X_base[:, idx['like']]
        partner_avg = X_base[:, idx['partner_avg_trait']]
        add('eng_like_x_partner_avg_trait', like * partner_avg)
    if has('attractive_o', 'shared_interests_o'):
        a_o = X_base[:, idx['attractive_o']]
        s_o = X_base[:, idx['shared_interests_o']]
        add('eng_attractive_o_x_shared_interests_o', a_o * s_o)
    if has('age_diff'):
        ad = X_base[:, idx['age_diff']]
        add('eng_age_diff_squared', ad ** 2)
    if has('SIS', 'interest_cosine'):
        sis = X_base[:, idx['SIS']]
        ic = X_base[:, idx['interest_cosine']]
        add('eng_SIS_x_interest_cosine', sis * ic)
    if has('SIS', 'interest_euclidean_sim'):
        sis = X_base[:, idx['SIS']]
        ies = X_base[:, idx['interest_euclidean_sim']]
        add('eng_SIS_x_interest_euclidean_sim', sis * ies)
    if has('attractive', 'attractive_o'):
        a = X_base[:, idx['attractive']]
        ao = X_base[:, idx['attractive_o']]
        add('eng_attractive_pair_product', a * ao)
        add('eng_attractive_pair_abs_diff', np.abs(a - ao))
    if has('funny', 'funny_o'):
        f = X_base[:, idx['funny']]
        fo = X_base[:, idx['funny_o']]
        add('eng_funny_pair_product', f * fo)
    if has('like', 'age_diff'):
        like = X_base[:, idx['like']]
        ad = X_base[:, idx['age_diff']]
        add('eng_like_per_age_diff', safe_divide(like, ad))

    if not new_columns:
        return X_base, []
    return np.hstack([X_base] + new_columns), new_names


def augment_train_val_test_features(X_train_inner, X_val, X_test,
                                    feature_names):
    X_train_aug, eng_names = add_engineered_features(
        X_train_inner, feature_names)
    if not eng_names:
        return X_train_inner, X_val, X_test, feature_names

    X_val_aug, _ = add_engineered_features(X_val, feature_names)
    X_test_aug, _ = add_engineered_features(X_test, feature_names)

    n_new = len(eng_names)
    train_new = X_train_aug[:, -n_new:]
    new_mean = np.mean(train_new, axis=0)
    new_std = np.std(train_new, axis=0)
    new_std[new_std == 0] = 1

    X_train_aug[:, -n_new:] = (X_train_aug[:, -n_new:] - new_mean) / new_std
    X_val_aug[:, -n_new:]   = (X_val_aug[:, -n_new:]   - new_mean) / new_std
    X_test_aug[:, -n_new:]  = (X_test_aug[:, -n_new:]  - new_mean) / new_std

    return X_train_aug, X_val_aug, X_test_aug, feature_names + eng_names


# ============================================================
# Stratified splits (with deterministic seeds)
# ============================================================
def stratified_train_val_split(X, y, val_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_val = int(len(pos_idx) * val_size)
    neg_val = int(len(neg_idx) * val_size)
    val_idx = np.concatenate([pos_idx[:pos_val], neg_idx[:neg_val]])
    train_idx = np.concatenate([pos_idx[pos_val:], neg_idx[neg_val:]])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def stratified_index_split(y, val_size=0.5, seed=123):
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_n = int(len(pos_idx) * val_size)
    neg_n = int(len(neg_idx) * val_size)
    first  = np.concatenate([pos_idx[:pos_n], neg_idx[:neg_n]])
    second = np.concatenate([pos_idx[pos_n:], neg_idx[neg_n:]])
    rng.shuffle(first)
    rng.shuffle(second)
    return first, second


# ============================================================
# Kernels
# ============================================================
def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def rbf_kernel(x1, x2, gamma=0.1):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


# ============================================================
# AdvancedSVM (SMO optimizer, identical to svm_grid.py)
# ============================================================
class AdvancedSVM:

    def __init__(self, C=1.0, kernel='rbf', gamma=0.1, tol=1e-3,
                 max_passes=10, pos_weight_multiplier=5.0,
                 max_iterations=30, sample_limit=1200):
        self.C = C
        self.kernel_name = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.pos_weight_multiplier = pos_weight_multiplier
        self.max_iterations = max_iterations
        self.sample_limit = sample_limit

    def kernel(self, x1, x2):
        if self.kernel_name == 'linear':
            return linear_kernel(x1, x2)
        elif self.kernel_name == 'rbf':
            return rbf_kernel(x1, x2, self.gamma)

    def fit(self, X, y):
        self.X = X
        self.y = y
        n_samples, _ = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        self.C_pos = self.C * self.pos_weight_multiplier
        self.C_neg = self.C

        passes = 0
        iteration = 0
        print(f"\nTraining SVM | kernel={self.kernel_name}, C={self.C}, "
              f"gamma={self.gamma}, pos_w={self.pos_weight_multiplier}")

        while passes < self.max_passes and iteration < self.max_iterations:
            num_changed = 0
            sample_limit = min(self.sample_limit, n_samples)
            random_idx = np.random.choice(n_samples, sample_limit,
                                          replace=False)
            for i in random_idx:
                Ei = self.decision_single_fast(X[i]) - y[i]
                Ci = self.C_pos if y[i] == 1 else self.C_neg
                cond1 = (y[i] * Ei < -self.tol) and (self.alpha[i] < Ci)
                cond2 = (y[i] * Ei >  self.tol) and (self.alpha[i] > 0)
                if not (cond1 or cond2):
                    continue

                j = np.random.randint(0, n_samples)
                while j == i:
                    j = np.random.randint(0, n_samples)
                Cj = self.C_pos if y[j] == 1 else self.C_neg
                Ej = self.decision_single_fast(X[j]) - y[j]
                ai_old, aj_old = self.alpha[i], self.alpha[j]

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

                self.alpha[j] += y[j] * (Ei - Ej) / eta
                self.alpha[j] = np.clip(self.alpha[j], L, H)
                if abs(self.alpha[j] - aj_old) < 1e-5:
                    continue
                self.alpha[i] += y[i] * y[j] * (aj_old - self.alpha[j])

                b1 = (self.b - Ei
                      - y[i] * (self.alpha[i] - ai_old) * Kii
                      - y[j] * (self.alpha[j] - aj_old) * Kij)
                b2 = (self.b - Ej
                      - y[i] * (self.alpha[i] - ai_old) * Kij
                      - y[j] * (self.alpha[j] - aj_old) * Kjj)
                if 0 < self.alpha[i] < Ci:
                    self.b = b1
                elif 0 < self.alpha[j] < Cj:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2
                num_changed += 1

            passes = passes + 1 if num_changed == 0 else 0
            if num_changed < 5:
                print("  Early stopping triggered.")
                break
            iteration += 1
            print(f"  Iter={iteration} | Changed Alphas={num_changed}")

        sv = self.alpha > 1e-3
        self.support_vectors = self.X[sv]
        self.support_vector_labels = self.y[sv]
        self.support_vector_alphas = self.alpha[sv]
        print(f"  Training Finished. Support Vectors: "
              f"{len(self.support_vectors)}")

    def decision_single_fast(self, x):
        sv = self.alpha > 1e-5
        sv_alpha = self.alpha[sv]
        sv_y = self.y[sv]
        sv_X = self.X[sv]
        if self.kernel_name == 'linear':
            K = sv_X @ x
        else:
            diff = sv_X - x
            K = np.exp(-self.gamma * np.sum(diff ** 2, axis=1))
        return float(np.sum(sv_alpha * sv_y * K) + self.b)

    def decision_function(self, X):
        sv = self.alpha > 1e-5
        sv_alpha = self.alpha[sv]
        sv_y = self.y[sv]
        sv_X = self.X[sv]
        scores = []
        for x in X:
            if self.kernel_name == 'linear':
                K = sv_X @ x
            else:
                diff = sv_X - x
                K = np.exp(-self.gamma * np.sum(diff ** 2, axis=1))
            scores.append(np.sum(sv_alpha * sv_y * K) + self.b)
        return np.array(scores)


# ============================================================
# Platt Scaling (matches svm_grid.py)
# ============================================================
def fit_platt_scaling(scores, y, lr=0.01, epochs=800, l2=1e-3):
    scores = scores.reshape(-1)
    y = y.reshape(-1)
    score_mean = float(np.mean(scores))
    score_std  = float(np.std(scores))
    if score_std == 0:
        score_std = 1.0
    z = (scores - score_mean) / score_std
    a, b = 1.0, 0.0
    for _ in range(epochs):
        p = sigmoid(a * z + b)
        error = p - y
        grad_a = np.mean(error * z) + l2 * a
        grad_b = np.mean(error)
        a -= lr * grad_a
        b -= lr * grad_b
    return {'a': a, 'b': b, 'mean': score_mean, 'std': score_std}


def apply_platt_scaling(scores, params):
    z = (scores - params['mean']) / params['std']
    return sigmoid(params['a'] * z + params['b'])


# ============================================================
# Main — Train final 5-SVM ensemble + Platt + ensemble predict
# ============================================================
def main():
    print("=" * 65)
    print("  svm.py  --  Final SVM Top-5 Ensemble (frozen configs)")
    print(f"  Pipeline: {ENSEMBLE_PIPELINE_NAME}")
    print(f"  Threshold: {ENSEMBLE_THRESHOLD}")
    print("=" * 65)

    # ── Load data ────────────────────────────────────────────
    X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
    X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    with open(os.path.join(DATA_DIR, 'feature_names.txt'),
              encoding='utf-8') as f:
        feature_names = [ln.strip() for ln in f if ln.strip()]
    print(f"Train: {X_train.shape}   Test: {X_test.shape}")

    # ── Inner train / val split (for Platt + threshold) ──────
    X_train_inner, X_val, y_train_inner, y_val = stratified_train_val_split(
        X_train, y_train, val_size=0.2, seed=42)
    y_train_inner_svm = np.where(y_train_inner == 0, -1, 1)

    # ── Engineered features (12 extra cols) ──────────────────
    X_train_inner, X_val, X_test, feature_names = (
        augment_train_val_test_features(
            X_train_inner, X_val, X_test, feature_names))
    n_features = X_train_inner.shape[1]
    print(f"After feature engineering: {n_features} features")
    print(f"Train inner: {X_train_inner.shape}   Val: {X_val.shape}   "
          f"Test: {X_test.shape}")

    # ── Train 5 SVMs with FROZEN configs (THIS IS THE TIMED PART) ──
    print("\n" + "=" * 65)
    print(f"  Training {len(ENSEMBLE_CONFIGS)} SVMs + Platt scaling ...")
    print("=" * 65)

    start_time = time.time()
    member_models = []
    member_platts = []

    for cfg in ENSEMBLE_CONFIGS:
        # Match svm_grid.py's per-config seed for reproducibility.
        np.random.seed(
            int(cfg['gamma'] * 10000)
            + int(cfg['pos_weight_multiplier'] * 100)
            + int(cfg['C'] * 10)
        )
        model = AdvancedSVM(
            C=cfg['C'], kernel='rbf', gamma=cfg['gamma'],
            max_passes=3,
            pos_weight_multiplier=cfg['pos_weight_multiplier'],
            max_iterations=22, sample_limit=1000,
        )
        model.fit(X_train_inner, y_train_inner_svm)

        # Fit Platt scaling on the calibration half of the val set.
        calib_idx, _ = stratified_index_split(y_val, val_size=0.5, seed=123)
        raw_val_scores = model.decision_function(X_val)
        platt = fit_platt_scaling(raw_val_scores[calib_idx],
                                  y_val[calib_idx])

        member_models.append(model)
        member_platts.append(platt)

    train_time = time.time() - start_time
    print(f"\n  Total ensemble training time: {train_time:.4f}s "
          f"(5 SVMs + Platt fitting only — no search)")

    # ── Ensemble predict on TEST ─────────────────────────────
    test_calibrated_list = []
    for model, platt in zip(member_models, member_platts):
        raw = model.decision_function(X_test)
        test_calibrated_list.append(apply_platt_scaling(raw, platt))
    y_prob = np.mean(test_calibrated_list, axis=0)
    y_pred = (y_prob >= ENSEMBLE_THRESHOLD).astype(int)

    # ── Evaluate ─────────────────────────────────────────────
    acc = accuracy(y_test, y_pred)
    pre = precision(y_test, y_pred)
    rec = recall(y_test, y_pred)
    f1  = f1_score(y_test, y_pred)
    auc, _, _ = compute_auc(y_test, y_prob)
    avg_p = compute_average_precision(y_test, y_prob)

    print("\n" + "─" * 50)
    print(f"  Final Test Set Metrics ({ENSEMBLE_PIPELINE_NAME})")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {pre:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  AUC      : {auc:.4f}")
    print(f"  Avg Prec : {avg_p:.4f}")
    print("─" * 50)

    # ── Compute val F1 for log (optional but informative) ────
    val_calibrated_list = []
    for model, platt in zip(member_models, member_platts):
        rv = model.decision_function(X_val)
        val_calibrated_list.append(apply_platt_scaling(rv, platt))
    val_scores = np.mean(val_calibrated_list, axis=0)
    val_pred = (val_scores >= ENSEMBLE_THRESHOLD).astype(int)
    val_f1 = f1_score(y_val, val_pred)
    print(f"  Validation F1 (used for selection): {val_f1:.4f}")

    # ── Save .npz for experimental_analysis.py ───────────────
    model_output_dir = os.path.join(_HERE, "model_outputs")
    os.makedirs(model_output_dir, exist_ok=True)
    out_path = os.path.join(model_output_dir, "svm_results.npz")
    np.savez(
        out_path,
        model_name        = np.array("SVM"),
        source_file       = np.array("svm.py"),
        selected_pipeline = np.array(ENSEMBLE_PIPELINE_NAME),
        y_prob            = y_prob,
        y_pred            = y_pred,
        y_test            = y_test,
        threshold         = np.array(ENSEMBLE_THRESHOLD),
        train_time        = np.array(train_time),
        accuracy          = np.array(acc),
        precision         = np.array(pre),
        recall            = np.array(rec),
        f1                = np.array(f1),
        auc               = np.array(auc),
        avg_precision     = np.array(avg_p),
        cv_f1             = np.array(val_f1),
        n_features        = np.array(n_features),
    )
    print(f"\n[Saved] {out_path}")

    append_experiment_log("SVM final model (frozen ensemble) completed", [
        "Source script: svm.py",
        f"Selected pipeline={ENSEMBLE_PIPELINE_NAME}",
        f"Ensemble size={len(ENSEMBLE_CONFIGS)}",
        f"Configs={[c['name'] for c in ENSEMBLE_CONFIGS]}",
        f"Threshold={ENSEMBLE_THRESHOLD}",
        f"Test accuracy={acc:.4f}",
        f"Test precision={pre:.4f}",
        f"Test recall={rec:.4f}",
        f"Test F1={f1:.4f}",
        f"Test AUC={auc:.4f}",
        f"Avg precision={avg_p:.4f}",
        f"Train time={train_time:.4f}s (5 SVMs + Platt only, fair timing)",
        f"Output={out_path}",
    ])

    print("\n" + "=" * 65)
    print(f"  [Done]  SVM ensemble final-model fit time = {train_time:.4f}s")
    print("=" * 65)
    return {
        'accuracy': acc, 'precision': pre, 'recall': rec,
        'f1': f1, 'auc': auc, 'train_time': train_time,
    }


if __name__ == "__main__":
    main()
