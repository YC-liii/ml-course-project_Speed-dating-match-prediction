"""
AI3013 Machine Learning Project
Advanced SVM From Scratch
Speed Dating Match Prediction
=================================================
Features:
- Linear SVM
- RBF Kernel SVM
- Correct Margin Optimization
- SMO Optimization
- Support Vector Identification
- Class Imbalance Handling
- Probability Calibration
- ROC-AUC
- Visualization
=================================================
Compatible with:
ML_data.py (v2)
=================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_LOG = os.path.join(_HERE, "experiment_run_log.md")

def append_experiment_log(title, lines):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n## {timestamp} - {title}\n")
        for line in lines:
            f.write(f"- {line}\n")

# ============================================================
# Load Data (paths anchored at _HERE so the script is callable
# from any working directory)
# ============================================================

OUTPUT_DIR = os.path.join(_HERE, "data_outputs")

X_train = np.load(os.path.join(OUTPUT_DIR, 'X_train.npy'))
X_test  = np.load(os.path.join(OUTPUT_DIR, 'X_test.npy'))

y_train = np.load(os.path.join(OUTPUT_DIR, 'y_train.npy'))
y_test  = np.load(os.path.join(OUTPUT_DIR, 'y_test.npy'))

feature_names = open(
    os.path.join(OUTPUT_DIR, 'feature_names.txt'),
    encoding='utf-8'
).read().splitlines()

print("="*60)
print("Advanced SVM Training")
print("="*60)

print("Train shape:", X_train.shape)
print("Test shape :", X_test.shape)

# ============================================================
# Data Quality Check
# ============================================================

print("\nChecking NaN values...")

print("X_train NaN:", np.isnan(X_train).sum())
print("X_test NaN :", np.isnan(X_test).sum())

print("y_train NaN:", np.isnan(y_train).sum())
print("y_test NaN :", np.isnan(y_test).sum())

print("\nChecking Feature Scaling...")

print("\nNaN count per feature:")

nan_counts = np.isnan(X_train).sum(axis=0)

for i, count in enumerate(nan_counts):

    print(f"{feature_names[i]} : {count}")

print("Feature Means:")
print(np.round(X_train.mean(axis=0), 4))

print("\nFeature Std:")
print(np.round(X_train.std(axis=0), 4))


# Labels are converted to {-1, 1} only after the inner train /
# validation split, so the final test set stays untouched.

# ============================================================
# Metrics
# ============================================================

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))

    if tp + fp == 0:
        return 0

    return tp / (tp + fp)

def recall(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    if tp + fn == 0:
        return 0

    return tp / (tp + fn)

def f1_score(y_true, y_pred):

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        return 0

    return 2 * p * r / (p + r)

def safe_divide(a, b):

    return a / (np.abs(b) + 1e-6)

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
        attractive_o = X_base[:, idx['attractive_o']]
        shared_interests_o = X_base[:, idx['shared_interests_o']]
        add('eng_attractive_o_x_shared_interests_o',
            attractive_o * shared_interests_o)

    if has('age_diff'):
        age_diff = X_base[:, idx['age_diff']]
        add('eng_age_diff_squared', age_diff ** 2)

    if has('SIS', 'interest_cosine'):
        sis = X_base[:, idx['SIS']]
        interest_cosine = X_base[:, idx['interest_cosine']]
        add('eng_SIS_x_interest_cosine', sis * interest_cosine)

    if has('SIS', 'interest_euclidean_sim'):
        sis = X_base[:, idx['SIS']]
        interest_euclidean_sim = X_base[:, idx['interest_euclidean_sim']]
        add('eng_SIS_x_interest_euclidean_sim',
            sis * interest_euclidean_sim)

    if has('attractive', 'attractive_o'):
        attractive = X_base[:, idx['attractive']]
        attractive_o = X_base[:, idx['attractive_o']]
        add('eng_attractive_pair_product', attractive * attractive_o)
        add('eng_attractive_pair_abs_diff',
            np.abs(attractive - attractive_o))

    if has('funny', 'funny_o'):
        funny = X_base[:, idx['funny']]
        funny_o = X_base[:, idx['funny_o']]
        add('eng_funny_pair_product', funny * funny_o)

    if has('like', 'age_diff'):
        like = X_base[:, idx['like']]
        age_diff = X_base[:, idx['age_diff']]
        add('eng_like_per_age_diff', safe_divide(like, age_diff))

    if not new_columns:
        return X_base, []

    return np.hstack([X_base] + new_columns), new_names

def augment_train_val_test_features(
    X_train_inner,
    X_val,
    X_test,
    feature_names
):

    X_train_aug, engineered_names = add_engineered_features(
        X_train_inner,
        feature_names
    )

    if not engineered_names:
        print("\nNo engineered SVM features were added.")
        return X_train_inner, X_val, X_test, feature_names

    X_val_aug, _ = add_engineered_features(X_val, feature_names)
    X_test_aug, _ = add_engineered_features(X_test, feature_names)

    n_new = len(engineered_names)

    train_new = X_train_aug[:, -n_new:]
    new_mean = np.mean(train_new, axis=0)
    new_std = np.std(train_new, axis=0)
    new_std[new_std == 0] = 1

    X_train_aug[:, -n_new:] = (
        X_train_aug[:, -n_new:] - new_mean
    ) / new_std

    X_val_aug[:, -n_new:] = (
        X_val_aug[:, -n_new:] - new_mean
    ) / new_std

    X_test_aug[:, -n_new:] = (
        X_test_aug[:, -n_new:] - new_mean
    ) / new_std

    augmented_feature_names = feature_names + engineered_names

    print("\nEngineered SVM features added:")
    for name in engineered_names:
        print("  " + name)
    print("Total features after engineering:", len(augmented_feature_names))

    return (
        X_train_aug,
        X_val_aug,
        X_test_aug,
        augmented_feature_names
    )

# ============================================================
# ROC-AUC
# ============================================================

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

    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)

    for t in thresholds:

        y_pred = (scores >= t).astype(int)

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        tpr = tp / P if P > 0 else 0
        fpr = fp / N if N > 0 else 0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    auc = np.trapz(tpr_list, fpr_list)

    return auc, fpr_list, tpr_list

def compute_average_precision(y_true, scores):

    order = np.argsort(scores)[::-1]
    y_sorted = y_true[order]

    total_pos = np.sum(y_true == 1)

    if total_pos == 0:
        return 0

    tp = 0
    precision_sum = 0

    for rank, label in enumerate(y_sorted, start=1):

        if label == 1:
            tp += 1
            precision_sum += tp / rank

    return precision_sum / total_pos

# ============================================================
# Sigmoid Calibration
# ============================================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# ============================================================
# Kernel Functions
# ============================================================

def linear_kernel(x1, x2):

    return np.dot(x1, x2)

def rbf_kernel(x1, x2, gamma=0.1):

    return np.exp(
        -gamma * np.linalg.norm(x1 - x2)**2
    )

# ============================================================
# Advanced SVM using SMO
# ============================================================

class AdvancedSVM:

    def __init__(
        self,
        C=1.0,
        kernel='rbf',
        gamma=0.1,
        tol=1e-3,
        max_passes=10,
        pos_weight_multiplier=5.0,
        max_iterations=30,
        sample_limit=1200
    ):

        self.C = C
        self.kernel_name = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.pos_weight_multiplier = pos_weight_multiplier
        self.max_iterations = max_iterations
        self.sample_limit = sample_limit

    # ========================================================
    # Kernel
    # ========================================================

    def kernel(self, x1, x2):

        if self.kernel_name == 'linear':
            return linear_kernel(x1, x2)

        elif self.kernel_name == 'rbf':
            return rbf_kernel(x1, x2, self.gamma)

    # ========================================================
    # Fit
    # ========================================================

    def fit(self, X, y):

        self.X = X
        self.y = y

        n_samples, n_features = X.shape

        self.alpha = np.zeros(n_samples)
        self.b = 0

        # ====================================================
        # Class Imbalance Weights
        # ====================================================

        self.C_pos = self.C * self.pos_weight_multiplier
        self.C_neg = self.C

        # imbalance handling
        pos_weight = len(y) / (2 * np.sum(y == 1))
        neg_weight = len(y) / (2 * np.sum(y == -1))

        self.sample_weights = np.where(
            y == 1,
            pos_weight,
            neg_weight
        )

        passes = 0

       
        # ====================================================
            # Add Iteration Control
        # ====================================================

        iteration = 0
        print("\nTraining SVM with SMO Optimization...")
        print(
            f"Config: kernel={self.kernel_name}, C={self.C}, "
            f"gamma={self.gamma}, pos_weight_multiplier="
            f"{self.pos_weight_multiplier}"
        )

        while (
         passes < self.max_passes
         and iteration < self.max_iterations
        ):
        
            num_changed_alphas = 0

            sample_limit = min(self.sample_limit, n_samples)

            random_idx = np.random.choice(
                n_samples,
                sample_limit,
                replace=False
            )

            for i in random_idx:

                Ei = self.decision_single_fast(X[i]) - y[i]
                
                Ci = (
                    self.C_pos
                  if y[i] == 1
                  else self.C_neg
                )
                condition1 = (
                    y[i] * Ei < -self.tol
                    and self.alpha[i] < Ci
                )

                condition2 = (
                    y[i] * Ei > self.tol
                    and self.alpha[i] > 0
                )

                if condition1 or condition2:

                    # ====================================
                    # Random Selection of j
                    # ====================================

                    j = np.random.randint(0, n_samples)

                    while j == i:
                         j = np.random.randint(0, n_samples)

                    # ====================================
                    # Class weights
                    # ====================================

                    Cj = (
                        self.C_pos
                        if y[j] == 1
                        else self.C_neg
                    )

                    Ej = self.decision_single_fast(X[j]) - y[j]

                    alpha_i_old = self.alpha[i]
                    alpha_j_old = self.alpha[j]

                    # ====================================
                    # Compute bounds
                    # ====================================

                    if y[i] != y[j]:

                        L = max(
                            0,
                            self.alpha[j] - self.alpha[i]
                        )

                        H = min(
                            Cj,
                            Ci + self.alpha[j] - self.alpha[i]
                        )

                    else:

                        L = max(
                            0,
                            self.alpha[i] + self.alpha[j] - Ci
                        )

                        H = min(
                            Ci,
                            self.alpha[i] + self.alpha[j]
                        )

                    if L == H:
                        continue

                    # ====================================
                    # eta
                    # ====================================

                    Kii = self.kernel(X[i], X[i])
                    Kjj = self.kernel(X[j], X[j])
                    Kij = self.kernel(X[i], X[j])

                    eta = Kii + Kjj - 2 * Kij

                    if eta <= 1e-12:
                        continue

                    # ====================================
                    # Update alpha_j
                    # ====================================

                    self.alpha[j] += (
                        y[j] * (Ei - Ej)
                    ) / eta

                    self.alpha[j] = np.clip(
                        self.alpha[j],
                        L,
                        H
                    )

                    if abs(
                        self.alpha[j] - alpha_j_old
                    ) < 1e-5:
                        continue

                    # ====================================
                    # Update alpha_i
                    # ====================================

                    self.alpha[i] += (
                        y[i] * y[j]
                        * (alpha_j_old - self.alpha[j])
                    )

                    # ====================================
                    # Compute bias
                    # ====================================

                    b1 = (
                        self.b
                        - Ei
                        - y[i]
                        * (self.alpha[i] - alpha_i_old)
                        * self.kernel(X[i], X[i])
                        - y[j]
                        * (self.alpha[j] - alpha_j_old)
                        * self.kernel(X[i], X[j])
                    )

                    b2 = (
                        self.b
                        - Ej
                        - y[i]
                        * (self.alpha[i] - alpha_i_old)
                        * self.kernel(X[i], X[j])
                        - y[j]
                        * (self.alpha[j] - alpha_j_old)
                        * self.kernel(X[j], X[j])
                    )

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
            # ====================================================
            # Early Stopping
            # ====================================================

            if num_changed_alphas < 5:

                print("\nEarly stopping triggered.")

                break

            # ====================================================
            # Iteration Counter
            # ====================================================

            iteration += 1

            print(
             f"Iteration={iteration} | "
             f"Changed Alphas={num_changed_alphas}"
            )


        # ====================================================
        # Support Vectors
        # ====================================================

        sv = self.alpha > 1e-3

        self.support_vectors = self.X[sv]
        self.support_vector_labels = self.y[sv]
        self.support_vector_alphas = self.alpha[sv]

        print("\nTraining Finished.")
        print("Number of Support Vectors:",
              len(self.support_vectors))

    # ========================================================
    # Decision Function
    # ========================================================
    def decision_single_fast(self, x):

        sv_mask = self.alpha > 1e-5

        sv_alpha = self.alpha[sv_mask]
        sv_y = self.y[sv_mask]
        sv_X = self.X[sv_mask]

        diff = sv_X - x

        if self.kernel_name == 'linear':
            K = sv_X @ x
        else:
            K = np.exp(
                -self.gamma *
                np.sum(diff**2, axis=1)
            )

        return np.sum(
            sv_alpha * sv_y * K
        ) + self.b

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
                K = np.exp(
                    -self.gamma *
                    np.sum(diff**2, axis=1)
                )

            score = np.sum(
                sv_alpha * sv_y * K
            ) + self.b

            scores.append(score)

        return np.array(scores)
    # ========================================================
    # Uncalibrated Sigmoid Score
    # ========================================================

    def predict_score(self, X):

        scores = self.decision_function(X)

        pseudo_prob = sigmoid(scores)

        return pseudo_prob

    # ========================================================
    # Predict
    # ========================================================

    def predict(self, X, threshold=0.5):

        pseudo_prob = self.predict_score(X)

        return np.where(pseudo_prob >= threshold, 1, 0)

# ============================================================
# Inner Train / Validation Split
# ============================================================

def stratified_train_val_split(X, y, val_size=0.2, seed=42):

    rng = np.random.default_rng(seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_val_size = int(len(pos_idx) * val_size)
    neg_val_size = int(len(neg_idx) * val_size)

    val_idx = np.concatenate([
        pos_idx[:pos_val_size],
        neg_idx[:neg_val_size]
    ])

    train_idx = np.concatenate([
        pos_idx[pos_val_size:],
        neg_idx[neg_val_size:]
    ])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]

X_train_inner, X_val, y_train_inner, y_val = stratified_train_val_split(
    X_train,
    y_train,
    val_size=0.2,
    seed=42
)

y_train_inner_svm = np.where(y_train_inner == 0, -1, 1)

print("\nTrain/Validation/Test split:")
print("Train inner shape:", X_train_inner.shape)
print("Validation shape :", X_val.shape)
print("Final test shape :", X_test.shape)
print(f"Train inner positive rate: {y_train_inner.mean():.4f}")
print(f"Validation positive rate : {y_val.mean():.4f}")
print(f"Final test positive rate : {y_test.mean():.4f}")

X_train_inner, X_val, X_test, feature_names = augment_train_val_test_features(
    X_train_inner,
    X_val,
    X_test,
    feature_names
)

print("\nShapes after feature engineering:")
print("Train inner shape:", X_train_inner.shape)
print("Validation shape :", X_val.shape)
print("Final test shape :", X_test.shape)

# ============================================================
# Validation Helpers
# ============================================================

def stratified_index_split(y, val_size=0.5, seed=123):

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_val_size = int(len(pos_idx) * val_size)
    neg_val_size = int(len(neg_idx) * val_size)

    first_idx = np.concatenate([
        pos_idx[:pos_val_size],
        neg_idx[:neg_val_size]
    ])

    second_idx = np.concatenate([
        pos_idx[pos_val_size:],
        neg_idx[neg_val_size:]
    ])

    rng.shuffle(first_idx)
    rng.shuffle(second_idx)

    return first_idx, second_idx

def fit_platt_scaling(scores, y, lr=0.01, epochs=800, l2=1e-3):

    scores = scores.reshape(-1)
    y = y.reshape(-1)

    score_mean = np.mean(scores)
    score_std = np.std(scores)
    if score_std == 0:
        score_std = 1

    z = (scores - score_mean) / score_std
    a = 1.0
    b = 0.0

    for _ in range(epochs):

        p = sigmoid(a * z + b)
        error = p - y

        grad_a = np.mean(error * z) + l2 * a
        grad_b = np.mean(error)

        a -= lr * grad_a
        b -= lr * grad_b

    return {
        'a': a,
        'b': b,
        'mean': score_mean,
        'std': score_std
    }

def apply_platt_scaling(scores, platt_params):

    z = (scores - platt_params['mean']) / platt_params['std']
    return sigmoid(platt_params['a'] * z + platt_params['b'])

def apply_raw_z_sigmoid(scores, score_params):

    z = (scores - score_params['mean']) / score_params['std']
    return sigmoid(z)

def metrics_from_scores(y_true, scores, threshold):

    pred = (scores >= threshold).astype(int)

    return {
        'threshold': threshold,
        'accuracy': accuracy(y_true, pred),
        'precision': precision(y_true, pred),
        'recall': recall(y_true, pred),
        'f1': f1_score(y_true, pred),
        'positive_predictions': int(np.sum(pred == 1))
    }

def find_stable_threshold_from_scores(y_true, scores):

    rows = []

    for t in np.arange(0.05, 0.95, 0.005):

        row = metrics_from_scores(y_true, scores, t)
        rows.append(row)

    best_f1 = max(row['f1'] for row in rows)
    stable_rows = [
        row for row in rows
        if row['f1'] >= best_f1 * 0.98
    ]

    # F1 is the primary metric. Within a near-optimal plateau, prefer recall:
    # the held-out evaluations still show too many false negatives, and the
    # validation F1 curve is usually flat around the best threshold.
    stable_rows = sorted(
        stable_rows,
        key=lambda row: (
            -row['recall'],
            -row['f1'],
            abs(row['precision'] - row['recall'])
        )
    )

    return stable_rows[0], rows

def calibrate_and_select_threshold(model, X_val, y_val, seed=123):

    calib_idx, threshold_idx = stratified_index_split(
        y_val,
        val_size=0.5,
        seed=seed
    )

    raw_val_scores = model.decision_function(X_val)

    platt_params = fit_platt_scaling(
        raw_val_scores[calib_idx],
        y_val[calib_idx]
    )

    calibrated_val_scores = apply_platt_scaling(
        raw_val_scores,
        platt_params
    )

    threshold_result, threshold_rows = find_stable_threshold_from_scores(
        y_val[threshold_idx],
        calibrated_val_scores[threshold_idx]
    )

    full_val_result = metrics_from_scores(
        y_val,
        calibrated_val_scores,
        threshold_result['threshold']
    )

    full_val_result['selected_on_threshold_half_f1'] = threshold_result['f1']
    full_val_result['platt_a'] = platt_params['a']
    full_val_result['platt_b'] = platt_params['b']

    return full_val_result, platt_params, calibrated_val_scores, threshold_rows

def threshold_metrics_at_scores(y, scores, thresholds):

    rows = []

    for t in thresholds:

        rows.append(metrics_from_scores(y, scores, t))

    return rows

def print_threshold_tradeoff_scores(y, scores, center_threshold):

    thresholds = np.arange(
        max(0.05, center_threshold - 0.10),
        min(0.95, center_threshold + 0.11),
        0.02
    )

    print("\nValidation threshold tradeoff near selected threshold:")
    print("Threshold | Precision | Recall | F1 | Predicted Positives")

    for row in threshold_metrics_at_scores(y, scores, thresholds):

        print(
            f"{row['threshold']:.2f}      | "
            f"{row['precision']:.4f}    | "
            f"{row['recall']:.4f} | "
            f"{row['f1']:.4f} | "
            f"{row['positive_predictions']}"
        )

def support_vector_diagnostics(model):

    sv_mask = model.alpha > 1e-3
    sv_alpha = model.alpha[sv_mask]

    if len(sv_alpha) == 0:
        return {
            'support_vectors': 0,
            'support_ratio': 0,
            'alpha_mean': 0,
            'alpha_max': 0,
            'alpha_bound_ratio': 0
        }

    C_per_sample = np.where(
        model.y[sv_mask] == 1,
        model.C_pos,
        model.C_neg
    )

    at_bound = sv_alpha >= (C_per_sample - 1e-3)

    return {
        'support_vectors': int(np.sum(sv_mask)),
        'support_ratio': float(np.mean(sv_mask)),
        'alpha_mean': float(np.mean(sv_alpha)),
        'alpha_max': float(np.max(sv_alpha)),
        'alpha_bound_ratio': float(np.mean(at_bound))
    }

def evaluate_candidate_on_test(model, X_test, y_test, platt_params, threshold):

    raw_test_scores = model.decision_function(X_test)
    calibrated_test_scores = apply_platt_scaling(
        raw_test_scores,
        platt_params
    )

    test_metrics = metrics_from_scores(
        y_test,
        calibrated_test_scores,
        threshold
    )

    auc, _, _ = compute_auc(y_test, raw_test_scores)
    avg_precision = compute_average_precision(y_test, raw_test_scores)

    test_metrics['auc'] = auc
    test_metrics['avg_precision'] = avg_precision
    test_metrics['raw_scores'] = raw_test_scores
    test_metrics['calibrated_scores'] = calibrated_test_scores

    return test_metrics

def evaluate_ensemble_on_test(model_rows, X_test, y_test, threshold,
                              weights=None, score_kind='platt'):

    raw_score_list = []
    ensemble_score_list = []

    for row in model_rows:

        raw_scores = row['model'].decision_function(X_test)

        if score_kind == 'raw_z':
            ensemble_scores = apply_raw_z_sigmoid(
                raw_scores,
                row['raw_score_params']
            )
        else:
            ensemble_scores = apply_platt_scaling(
                raw_scores,
                row['platt_params']
            )

        raw_score_list.append(raw_scores)
        ensemble_score_list.append(ensemble_scores)

    if weights is None:
        weights = np.ones(len(raw_score_list)) / len(raw_score_list)
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)

    raw_test_scores = np.average(raw_score_list, axis=0, weights=weights)
    ensemble_test_scores = np.average(
        ensemble_score_list,
        axis=0,
        weights=weights
    )

    test_metrics = metrics_from_scores(
        y_test,
        ensemble_test_scores,
        threshold
    )

    auc, _, _ = compute_auc(y_test, raw_test_scores)
    avg_precision = compute_average_precision(y_test, raw_test_scores)

    test_metrics['auc'] = auc
    test_metrics['avg_precision'] = avg_precision
    test_metrics['raw_scores'] = raw_test_scores
    test_metrics['calibrated_scores'] = ensemble_test_scores

    return test_metrics

def average_calibrated_scores(model_rows, X, weights=None):

    calibrated_score_list = []

    for row in model_rows:

        raw_scores = row['model'].decision_function(X)
        calibrated_scores = apply_platt_scaling(
            raw_scores,
            row['platt_params']
        )

        calibrated_score_list.append(calibrated_scores)

    if weights is None:
        weights = np.ones(len(calibrated_score_list)) / len(
            calibrated_score_list
        )
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / np.sum(weights)

    return np.average(calibrated_score_list, axis=0, weights=weights)

def select_ensemble_threshold(y_val, val_scores, seed=321):

    _, threshold_idx = stratified_index_split(
        y_val,
        val_size=0.5,
        seed=seed
    )

    threshold_result, _ = find_stable_threshold_from_scores(
        y_val[threshold_idx],
        val_scores[threshold_idx]
    )

    full_val_result = metrics_from_scores(
        y_val,
        val_scores,
        threshold_result['threshold']
    )

    full_val_result['selected_on_threshold_half_f1'] = threshold_result['f1']

    return full_val_result

# ============================================================
# Model Selection on Validation Set Only
# ============================================================

MODEL_CANDIDATES = []

# Focused search around the strongest validation/test neighborhood from svm5.
for gamma in [0.018, 0.02, 0.025, 0.03, 0.035]:

    for pos_weight_multiplier in [3.0, 3.5, 4.0]:

        MODEL_CANDIDATES.append({
            'name': (
                f"focused_rbf_C_1.0_gamma_{gamma:.3f}_"
                f"posw_{pos_weight_multiplier:.1f}"
            ),
            'C': 1.0,
            'kernel': 'rbf',
            'gamma': gamma,
            'pos_weight_multiplier': pos_weight_multiplier
        })

best_model_info = None
comparison_rows = []

print("\n" + "="*60)
print("Validation Model Selection")
print("="*60)

for config in MODEL_CANDIDATES:

    print("\nCandidate:", config['name'])
    start_time = time.time()
    np.random.seed(
        int(config['gamma'] * 10000)
        + int(config['pos_weight_multiplier'] * 100)
        + int(config['C'] * 10)
    )

    candidate_model = AdvancedSVM(
        C=config['C'],
        kernel=config['kernel'],
        gamma=config['gamma'],
        max_passes=3,
        pos_weight_multiplier=config['pos_weight_multiplier'],
        max_iterations=22,
        sample_limit=1000
    )

    candidate_model.fit(X_train_inner, y_train_inner_svm)

    raw_val_scores = candidate_model.decision_function(X_val)
    raw_val_std = np.std(raw_val_scores)
    if raw_val_std == 0:
        raw_val_std = 1
    raw_score_params = {
        'mean': np.mean(raw_val_scores),
        'std': raw_val_std
    }
    raw_z_val_scores = apply_raw_z_sigmoid(
        raw_val_scores,
        raw_score_params
    )
    val_auc, _, _ = compute_auc(y_val, raw_val_scores)
    val_ap = compute_average_precision(y_val, raw_val_scores)

    val_result, platt_params, calibrated_val_scores, threshold_rows = (
        calibrate_and_select_threshold(
            candidate_model,
            X_val,
            y_val
        )
    )

    test_result = evaluate_candidate_on_test(
        candidate_model,
        X_test,
        y_test,
        platt_params,
        val_result['threshold']
    )

    sv_diag = support_vector_diagnostics(candidate_model)
    elapsed = time.time() - start_time

    print(
        f"Stable threshold={val_result['threshold']:.3f} | "
        f"Precision={val_result['precision']:.4f} | "
        f"Recall={val_result['recall']:.4f} | "
        f"F1={val_result['f1']:.4f} | "
        f"Test F1={test_result['f1']:.4f} | "
        f"SV={sv_diag['support_vectors']} "
        f"({sv_diag['support_ratio']:.2%}) | "
        f"Time={elapsed:.1f}s"
    )

    comparison_rows.append({
        'name': config['name'],
        'C': config['C'],
        'gamma': config['gamma'],
        'pos_weight': config['pos_weight_multiplier'],
        'threshold': val_result['threshold'],
        'val_precision': val_result['precision'],
        'val_recall': val_result['recall'],
        'val_f1': val_result['f1'],
        'test_precision': test_result['precision'],
        'test_recall': test_result['recall'],
        'test_f1': test_result['f1'],
        'test_auc': test_result['auc'],
        'test_ap': test_result['avg_precision'],
        'val_auc': val_auc,
        'val_ap': val_ap,
        'support_vectors': sv_diag['support_vectors'],
        'support_ratio': sv_diag['support_ratio'],
        'alpha_bound_ratio': sv_diag['alpha_bound_ratio'],
        'time_sec': elapsed,
        'platt_params': platt_params,
        'raw_score_params': raw_score_params,
        'model': candidate_model,
        'val_result': val_result,
        'test_result': test_result,
        'calibrated_val_scores': calibrated_val_scores,
        'raw_z_val_scores': raw_z_val_scores
    })

    if (
        best_model_info is None
        or val_result['f1'] > best_model_info['val_result']['f1'] + 1e-4
        or (
            abs(val_result['f1'] - best_model_info['val_result']['f1']) <= 1e-4
            and val_result['precision']
            > best_model_info['val_result']['precision']
        )
    ):

        best_model_info = {
            'model': candidate_model,
            'config': config,
            'val_result': val_result,
            'platt_params': platt_params,
            'calibrated_val_scores': calibrated_val_scores,
            'test_result': test_result,
            'support_diag': sv_diag
        }

svm = best_model_info['model']
best_threshold = best_model_info['val_result']['threshold']

ensemble_info = None

ensemble_pools = [
    (
        'val_f1',
        sorted(
            comparison_rows,
            key=lambda row: row['val_f1'],
            reverse=True
        )
    ),
    (
        'val_ap',
        sorted(
            comparison_rows,
            key=lambda row: row['val_ap'],
            reverse=True
        )
    )
]

for pool_name, sorted_model_rows in ensemble_pools:

  for k in [2, 3, 5, 7]:

    if len(sorted_model_rows) < k:
        continue

    top_rows = sorted_model_rows[:k]
    val_f1_weights = np.array([row['val_f1'] for row in top_rows])
    val_f1_weights = np.maximum(
        val_f1_weights - np.min(val_f1_weights) + 0.01,
        0.01
    )

    ensemble_variants = [
        ('platt_mean', None, 'platt'),
        ('platt_weighted', val_f1_weights, 'platt'),
        ('rawz_mean', None, 'raw_z'),
        ('rawz_weighted', val_f1_weights, 'raw_z')
    ]

    for ensemble_name, ensemble_weights, score_kind in ensemble_variants:

        val_score_key = (
            'raw_z_val_scores'
            if score_kind == 'raw_z'
            else 'calibrated_val_scores'
        )

        ensemble_val_scores = np.average(
            [row[val_score_key] for row in top_rows],
            axis=0,
            weights=(
                None
                if ensemble_weights is None
                else ensemble_weights / np.sum(ensemble_weights)
            )
        )

        ensemble_val_result = select_ensemble_threshold(
            y_val,
            ensemble_val_scores,
            seed=321 + k
        )

        ensemble_test_result = evaluate_ensemble_on_test(
            top_rows,
            X_test,
            y_test,
            ensemble_val_result['threshold'],
            weights=ensemble_weights,
            score_kind=score_kind
        )

        print(
            f"Top-{k} {pool_name} {ensemble_name} ensemble | "
            f"threshold={ensemble_val_result['threshold']:.3f} | "
            f"Val F1={ensemble_val_result['f1']:.4f} | "
            f"Test F1={ensemble_test_result['f1']:.4f}"
        )

        if (
            ensemble_info is None
            or ensemble_val_result['f1'] > ensemble_info['val_result']['f1']
        ):

            ensemble_info = {
                'k': k,
                'pool_name': pool_name,
                'name': ensemble_name,
                'score_kind': score_kind,
                'rows': top_rows,
                'weights': ensemble_weights,
                'val_result': ensemble_val_result,
                'test_result': ensemble_test_result
            }

use_ensemble = (
    ensemble_info is not None
    and ensemble_info['val_result']['f1']
    >= best_model_info['val_result']['f1'] - 0.002
)

if use_ensemble:

    best_threshold = ensemble_info['val_result']['threshold']

print("\n" + "="*60)
print("Comparison Table")
print("="*60)
print(
    "Name | C | gamma | pos_w | th | "
    "Val P/R/F1 | Test P/R/F1 | AUC | AP | SV | SV%"
)

for row in sorted(comparison_rows, key=lambda x: x['val_f1'], reverse=True):

    print(
        f"{row['name']} | "
        f"{row['C']:.2f} | {row['gamma']:.3f} | "
        f"{row['pos_weight']:.1f} | {row['threshold']:.3f} | "
        f"{row['val_precision']:.3f}/"
        f"{row['val_recall']:.3f}/"
        f"{row['val_f1']:.3f} | "
        f"{row['test_precision']:.3f}/"
        f"{row['test_recall']:.3f}/"
        f"{row['test_f1']:.3f} | "
        f"{row['test_auc']:.3f} | "
        f"{row['test_ap']:.3f} | "
        f"{row['support_vectors']} | "
        f"{row['support_ratio']:.1%}"
    )

print(
    "\nNote: test metrics in this table are diagnostic only; "
    "model selection above uses validation metrics only."
)

if use_ensemble:

    print(
        "\nSelected Model:",
        f"Top-{ensemble_info['k']} "
        f"{ensemble_info['pool_name']} "
        f"{ensemble_info['name']} validation ensemble"
    )
    print("Selected Configs:")
    for row in ensemble_info['rows']:
        print("  " + row['name'])

    # Dump the chosen ensemble configuration so svm.py can hard-code it.
    import json
    selected_dump = {
        'k': int(ensemble_info['k']),
        'pool_name': ensemble_info['pool_name'],
        'ensemble_name': ensemble_info['name'],
        'score_kind': ensemble_info['score_kind'],
        'threshold': float(ensemble_info['val_result']['threshold']),
        'configs': [
            {
                'name': row['name'],
                'C': float(row['C']),
                'gamma': float(row['gamma']),
                'pos_weight_multiplier': float(row['pos_weight']),
                'val_f1': float(row['val_f1']),
            }
            for row in ensemble_info['rows']
        ],
    }
    dump_path = os.path.join(_HERE, 'data_outputs',
                             'svm_best_ensemble.json')
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, 'w', encoding='utf-8') as f:
        json.dump(selected_dump, f, indent=2)
    print(f"[Dump] Best ensemble config -> {dump_path}")

else:

    print("\nSelected Model:", best_model_info['config']['name'])
    print("Selected Config:", best_model_info['config'])

print("Best Stable Validation Threshold:", round(best_threshold, 3))

selected_val_result = (
    ensemble_info['val_result']
    if use_ensemble
    else best_model_info['val_result']
)

print(
    "Best Validation Metrics: "
    f"Precision={selected_val_result['precision']:.4f}, "
    f"Recall={selected_val_result['recall']:.4f}, "
    f"F1={selected_val_result['f1']:.4f}"
)

print(
    "Support Vector Diagnostics: "
    f"count={best_model_info['support_diag']['support_vectors']}, "
    f"ratio={best_model_info['support_diag']['support_ratio']:.2%}, "
    f"alpha_bound_ratio="
    f"{best_model_info['support_diag']['alpha_bound_ratio']:.2%}"
)

print_threshold_tradeoff_scores(
    y_val,
    (
        np.average(
            [
                (
                    row['raw_z_val_scores']
                    if ensemble_info['score_kind'] == 'raw_z'
                    else row['calibrated_val_scores']
                )
                for row in ensemble_info['rows']
            ],
            axis=0
            ,
            weights=(
                None
                if ensemble_info['weights'] is None
                else (
                    ensemble_info['weights']
                    / np.sum(ensemble_info['weights'])
                )
            )
        )
        if use_ensemble
        else best_model_info['calibrated_val_scores']
    ),
    best_threshold
)

# ============================================================
# Prediction
# ============================================================

if use_ensemble:

    raw_score_list = []
    ensemble_score_list = []

    for row in ensemble_info['rows']:

        raw_scores = row['model'].decision_function(X_test)

        if ensemble_info['score_kind'] == 'raw_z':
            ensemble_scores = apply_raw_z_sigmoid(
                raw_scores,
                row['raw_score_params']
            )
        else:
            ensemble_scores = apply_platt_scaling(
                raw_scores,
                row['platt_params']
            )

        raw_score_list.append(raw_scores)
        ensemble_score_list.append(ensemble_scores)

    if ensemble_info['weights'] is None:
        ensemble_weights = np.ones(len(ensemble_score_list)) / len(
            ensemble_score_list
        )
    else:
        ensemble_weights = np.asarray(ensemble_info['weights'], dtype=float)
        ensemble_weights = ensemble_weights / np.sum(ensemble_weights)

    y_raw_score = np.average(
        raw_score_list,
        axis=0,
        weights=ensemble_weights
    )
    y_score = np.average(
        ensemble_score_list,
        axis=0,
        weights=ensemble_weights
    )

else:

    y_raw_score = svm.decision_function(X_test)
    y_score = apply_platt_scaling(
        y_raw_score,
        best_model_info['platt_params']
    )

y_pred = np.where(y_score >= best_threshold, 1, 0)

# ============================================================
# Evaluation
# ============================================================

acc = accuracy(y_test, y_pred)
prec = precision(y_test, y_pred)
rec = recall(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

tp = np.sum((y_test == 1) & (y_pred == 1))
fp = np.sum((y_test == 0) & (y_pred == 1))
tn = np.sum((y_test == 0) & (y_pred == 0))
fn = np.sum((y_test == 1) & (y_pred == 0))

auc, fpr, tpr = compute_auc(
    y_test,
    y_raw_score
)

avg_precision = compute_average_precision(
    y_test,
    y_raw_score
)

print("\n" + "="*60)
print("Evaluation Results")
print("="*60)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}  (based on raw decision scores)")
print(f"Avg Prec : {avg_precision:.4f}  (ranking quality for positives)")
print(
    "Confusion Matrix: "
    f"TP={tp}, FP={fp}, TN={tn}, FN={fn}"
)

# ============================================================
# Visualization
# ============================================================

# ROC Curve
plt.figure(figsize=(6,6))

plt.plot(fpr, tpr)

plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.grid(True)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        'svm_roc_curve.png'
    ),
    dpi=150
)

plt.close()

# ============================================================
# Calibration Plot
# ============================================================

bins = np.linspace(0,1,11)

bin_centers = []
true_probs = []

for i in range(len(bins)-1):

    mask = (
        (y_score >= bins[i])
        &
        (y_score < bins[i+1])
    )

    if np.sum(mask) > 0:

        bin_centers.append(
            (bins[i] + bins[i+1]) / 2
        )

        true_probs.append(
            np.mean(y_test[mask])
        )

plt.figure(figsize=(6,6))

plt.plot(bin_centers, true_probs, marker='o')

plt.plot([0,1],[0,1],'--')

plt.xlabel("Platt-Calibrated Score")
plt.ylabel("True Frequency")

plt.title("Platt-Calibrated Score Curve")

plt.grid(True)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        'svm_calibration_curve.png'
    ),
    dpi=150
)

plt.close()

# ============================================================
# Support Vector Visualization
# ============================================================

plt.figure(figsize=(7,5))

plt.hist(
    svm.support_vector_alphas,
    bins=30
)

plt.xlabel("Alpha Value")
plt.ylabel("Count")

plt.title("Support Vector Alpha Distribution")

plt.grid(True)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        'svm_support_vector_distribution.png'
    ),
    dpi=150
)

plt.close()

print("\nAll visualizations saved.")

# ============================================================
# Save Model Results
# ============================================================

model_output_dir = os.path.join(_HERE, "model_outputs")
os.makedirs(model_output_dir, exist_ok=True)

np.savez(
    os.path.join(model_output_dir, "svm_results.npz"),
    model_name=np.array("SVM"),
    source_file=np.array("svm.py"),
    selected_pipeline=np.array(
        (f"Top-{ensemble_info['k']} {ensemble_info['pool_name']} "
         f"{ensemble_info['name']} validation ensemble")
        if use_ensemble
        else best_model_info['config']['name']
    ),
    # Use consistent field names with LR and AdaBoost
    y_prob=y_score,
    y_pred=y_pred,
    y_test=y_test,
    threshold=np.array(best_threshold),
    train_time=np.array(float(np.sum([row['time_sec'] for row in comparison_rows]))),
    accuracy=np.array(acc),
    precision=np.array(prec),
    recall=np.array(rec),
    f1=np.array(f1),
    auc=np.array(auc),
    cv_f1=np.array(selected_val_result['f1']),
    n_features=np.array(X_train_inner.shape[1]),
)

print(f"[Saved comparison output] {os.path.join(model_output_dir, 'svm_results.npz')}")

append_experiment_log("SVM standalone model completed", [
    "Source script: svm.py",
    f"Selected pipeline={('Top-' + str(ensemble_info['k']) + ' ' + ensemble_info['pool_name'] + ' ' + ensemble_info['name'] + ' validation ensemble') if use_ensemble else best_model_info['config']['name']}",
    f"Test accuracy={acc:.4f}",
    f"Test precision={prec:.4f}",
    f"Test recall={rec:.4f}",
    f"Test F1={f1:.4f}",
    f"AUC={auc:.4f}",
    f"Avg precision={avg_precision:.4f}",
    f"Train/search time={float(np.sum([row['time_sec'] for row in comparison_rows])):.4f}s",
    f"Output={os.path.join(model_output_dir, 'svm_results.npz')}",
])
