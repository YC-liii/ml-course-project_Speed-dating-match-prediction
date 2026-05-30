
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil


_HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_HERE, "final_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_DIR = os.path.join(_HERE, "data_outputs")

# Font settings
import matplotlib
try:
    matplotlib.rcParams['font.sans-serif'] = ['PingFang SC', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans']
except:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = '#FFF5F5'
plt.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams['axes.edgecolor'] = '#F3BBB1'
plt.rcParams['grid.color'] = '#F3BBB1'
plt.rcParams['grid.alpha'] = 0.4

# macaron配色方案
COLORS = {
    'primary': '#F7A6AC',      # pink
    'secondary': '#EEC78A',    # cream yellow
    'tertiary': '#B8E5FA',     # sky blue
    'accent1': '#F7B2C7',      # rose pink
    'accent2': '#F3BBB1',      # light coral
    'accent3': '#CBE4B1',      # light green
    'accent4': '#B3DDCB',      # mint green
}

PALETTE = ['#F7A6AC', '#EEC78A', '#B8E5FA', '#F7B2C7', 
           '#F3BBB1', '#CBE4B1', '#B3DDCB']

ENGINEERED_KEYWORDS = [
    '_pair', '_gap', '_centered', 'SIS', 'interest_cosine',
    'interest_euclidean_sim', 'gender_', 'male_', '_x_',
    'female_samerace'
]

def load_latest_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    with open(os.path.join(DATA_DIR, "feature_names.txt"), encoding="utf-8") as f:
        feature_names = [line.strip() for line in f if line.strip()]
    return X_train, y_train, feature_names

def pearson_importance(X, y):
    y_centered = y - y.mean()
    scores = []
    for j in range(X.shape[1]):
        x = X[:, j]
        x_centered = x - x.mean()
        denom = np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
        scores.append(abs(np.dot(x_centered, y_centered) / denom) if denom > 1e-12 else 0.0)
    return np.array(scores)

def is_engineered_feature(name):
    return any(key in name for key in ENGINEERED_KEYWORDS)

# ============================================================
# Figure 1: GPB Before vs After Normalization (Bar Chart)
# ============================================================

def plot_figure_1_gpb_comparison():
    """GPB归一化前后对比 - 条形图版本（始终用macaron英文版本，避免与 ML_data.py 的中文深蓝版冲突）"""
    interests = ['Music', 'Sports']
    person_a_before = [9, 8]
    person_b_before = [6, 5]
    person_a_after = [1.2, 0.8]
    person_b_after = [1.1, 0.7]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('GPB Before vs After Normalization', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    x = np.arange(len(interests))
    width = 0.35
    
    # Left: Before Normalization
    bars1 = ax1.bar(x - width/2, person_a_before, width, 
                    label='Person A', color=COLORS['primary'], 
                    alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, person_b_before, width, 
                    label='Person B', color=COLORS['tertiary'], 
                    alpha=0.85, edgecolor='white', linewidth=1.5)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_xlabel('Interest', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Rating Score', fontsize=12, fontweight='bold')
    ax1.set_title('Before Normalization\n(Large Difference)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(interests, fontsize=11)
    ax1.legend(fontsize=11, framealpha=0.9, loc='upper right')
    ax1.set_ylim(0, 10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Right: After GPB
    bars3 = ax2.bar(x - width/2, person_a_after, width, 
                    label='Person A', color=COLORS['primary'], 
                    alpha=0.85, edgecolor='white', linewidth=1.5)
    bars4 = ax2.bar(x + width/2, person_b_after, width, 
                    label='Person B', color=COLORS['tertiary'], 
                    alpha=0.85, edgecolor='white', linewidth=1.5)
    
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'+{height:.1f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color='#2E7D32')
    
    ax2.set_xlabel('Interest', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Normalized Increment', fontsize=12, fontweight='bold')
    ax2.set_title('After GPB Normalization\n(Reduced Gap, More Fair)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(interests, fontsize=11)
    ax2.legend(fontsize=11, framealpha=0.9, loc='upper right')
    ax2.set_ylim(0, 1.5)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(OUTPUT_DIR, 'fig1_gpb_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='#FFF5F5', edgecolor='none')
    print(f"✓ Figure 1 saved: {output_path}")
    plt.close()

# ============================================================
# Figure 2: Raw Data Feature Importance
# ============================================================

def plot_figure_2_raw_features():
    """Raw feature importance"""
    X_train, y_train, feature_names = load_latest_data()
    scores = pearson_importance(X_train, y_train)
    raw_idx = [i for i, name in enumerate(feature_names) if not is_engineered_feature(name)]
    raw_scores = scores[raw_idx]
    top = np.argsort(raw_scores)[::-1][:15]
    features = [feature_names[raw_idx[i]] for i in top]
    importance_scores = raw_scores[top]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sorted_idx = np.argsort(importance_scores)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_scores = importance_scores[sorted_idx]
    
    colors = []
    for score in sorted_scores:
        if score > 0.7:
            colors.append(COLORS['primary'])
        elif score > 0.5:
            colors.append(COLORS['secondary'])
        elif score > 0.3:
            colors.append(COLORS['tertiary'])
        else:
            colors.append(COLORS['accent4'])
    
    y_pos = np.arange(len(sorted_features))
    bars = ax.barh(y_pos, sorted_scores, 
                   color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.2)
    
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}',
                ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Raw Data Feature Importance', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, max(sorted_scores) * 1.25)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['primary'], alpha=0.85, label='High (>0.7)'),
        Patch(facecolor=COLORS['secondary'], alpha=0.85, label='Medium (0.5-0.7)'),
        Patch(facecolor=COLORS['tertiary'], alpha=0.85, label='Low (0.3-0.5)'),
        Patch(facecolor=COLORS['accent4'], alpha=0.85, label='Very Low (<0.3)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', 
              fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'fig2_raw_feature_importance.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='#FFF5F5', edgecolor='none')
    print(f"✓ Figure 2 saved: {output_path}")
    plt.close()

# ============================================================
# Figure 3: Top-10 Features (Engineered vs Raw)
# ============================================================

def plot_figure_3_top10_features():
    """Top-10 feature importance (engineered vs raw features)"""
    X_train, y_train, feature_names = load_latest_data()
    scores = pearson_importance(X_train, y_train)
    top_idx = np.argsort(scores)[::-1][:10]
    features = [feature_names[i] for i in top_idx]
    importance_scores = scores[top_idx]
    is_engineered = [is_engineered_feature(name) for name in features]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    sorted_idx = np.argsort(importance_scores)
    sorted_features = [features[i] for i in sorted_idx]
    sorted_scores = importance_scores[sorted_idx]
    sorted_types = [is_engineered[i] for i in sorted_idx]
    
    colors = [COLORS['primary'] if eng else COLORS['tertiary'] for eng in sorted_types]
    
    y_pos = np.arange(len(sorted_features))
    bars = ax.barh(y_pos, sorted_scores, 
                   color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5)
    
    for i, (bar, score) in enumerate(zip(bars, sorted_scores)):
        width = bar.get_width()
        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=11)
    ax.set_xlabel('Feature Importance Score', fontsize=13, fontweight='bold')
    ax.set_title('Top-10 Features by Importance\n(Engineered vs Raw)', 
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xlim(0, 1.05)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['primary'], alpha=0.85, 
              edgecolor='white', linewidth=1.5,
              label='Engineered Features'),
        Patch(facecolor=COLORS['tertiary'], alpha=0.85, 
              edgecolor='white', linewidth=1.5,
              label='Raw Features')
    ]
    ax.legend(handles=legend_elements, loc='lower right', 
              fontsize=11, framealpha=0.95, 
              edgecolor='#F3BBB1', fancybox=True)
    
    n_engineered = sum(sorted_types)
    n_raw = len(sorted_types) - n_engineered
    stats_text = f'Engineered: {n_engineered} | Raw: {n_raw}'
    ax.text(0.02, 0.02, stats_text,
            transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', 
                     alpha=0.8, edgecolor='#F3BBB1'))
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, 'fig3_top10_features.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='#FFF5F5', edgecolor='none')
    print(f"✓ Figure 3 saved: {output_path}")
    plt.close()

# ============================================================
# Figure 4: Feature Engineering Before vs After Comparison
# ============================================================

def plot_figure_4_feature_engineering_comparison():
    """Feature engineering before/after - side-by-side comparison"""
    X_train, y_train, feature_names = load_latest_data()
    scores = pearson_importance(X_train, y_train)
    raw_idx = [i for i, name in enumerate(feature_names) if not is_engineered_feature(name)]
    eng_idx = [i for i, name in enumerate(feature_names) if is_engineered_feature(name)]
    raw_top = np.argsort(scores[raw_idx])[::-1][:10]
    eng_top = np.argsort(scores[eng_idx])[::-1][:10]
    raw_features = [feature_names[raw_idx[i]] for i in raw_top]
    eng_features = [feature_names[eng_idx[i]] for i in eng_top]
    importance_before = scores[[raw_idx[i] for i in raw_top]]
    importance_after = scores[[eng_idx[i] for i in eng_top]]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    fig.suptitle('Feature Importance: Before vs After Feature Engineering', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    sorted_before_idx = np.argsort(importance_before)
    sorted_after_idx = np.argsort(importance_after)
    sorted_raw_features = [raw_features[i] for i in sorted_before_idx]
    sorted_eng_features = [eng_features[i] for i in sorted_after_idx]
    sorted_before = importance_before[sorted_before_idx]
    sorted_after = importance_after[sorted_after_idx]
    
    y_pos = np.arange(len(sorted_before))
    
    # Left: Before
    bars1 = ax1.barh(y_pos, sorted_before, 
                     color=COLORS['tertiary'], alpha=0.85,
                     edgecolor='white', linewidth=1.5)
    
    for bar, score in zip(bars1, sorted_before):
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(sorted_raw_features, fontsize=10)
    ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax1.set_title('Before Feature Engineering\n(Raw Features Only)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlim(0, 1.0)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    ax1.invert_xaxis()
    
    # Right: After
    bars2 = ax2.barh(y_pos, sorted_after, 
                     color=COLORS['primary'], alpha=0.85,
                     edgecolor='white', linewidth=1.5)
    
    for bar, score in zip(bars2, sorted_after):
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{score:.2f}',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(sorted_eng_features, fontsize=10)
    ax2.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    ax2.set_title('After Feature Engineering\n(With Engineered Features)', 
                  fontsize=13, fontweight='bold', pad=15)
    ax2.set_xlim(0, 1.0)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    avg_raw = float(np.mean(sorted_before))
    avg_eng = float(np.mean(sorted_after))
    improvement = (avg_eng - avg_raw) / max(avg_raw, 1e-12) * 100
    fig.text(0.5, 0.90, f'Avg |r|\n{avg_raw:.3f} → {avg_eng:.3f}\n{improvement:+.1f}%',
             ha='center', va='center', fontsize=10, fontweight='bold',
             color='#2E7D32',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                       edgecolor='#2E7D32', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = os.path.join(OUTPUT_DIR, 'fig4_feature_engineering_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='#FFF5F5', edgecolor='none')
    print(f"✓ Figure 4 saved: {output_path}")
    plt.close()

# ============================================================
# Main function
# ============================================================

def generate_all_figures():
    """生成所有报告图表"""
    
    print("=" * 70)
    print("ML-LR Term Project - Figure Generation")
    print("Unified Macaron Color Scheme (Report Version)")
    print("=" * 70)
    
    print("\nGenerating figures...")
    
    plot_figure_1_gpb_comparison()
    plot_figure_2_raw_features()
    plot_figure_3_top10_features()
    plot_figure_4_feature_engineering_comparison()
    sync_model_comparison_table()
    generate_top7_features_table()
    
    print("\n" + "=" * 70)
    print("All figures generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)
    
    print("\nGenerated files:")
    print("  1. fig1_gpb_comparison.png")
    print("  2. fig2_raw_feature_importance.png")
    print("  3. fig3_top10_features.png")
    print("  4. fig4_feature_engineering_comparison.png")
    print("  5. model_comparison_table.png")
    print("  6. top7_features_table.png")


# ============================================================
# Table figures (merged from sync_final_figure_tables.py)
# ============================================================

def sync_model_comparison_table():
    source = os.path.join(OUTPUT_DIR, "exp_08_summary_table.png")
    target = os.path.join(OUTPUT_DIR, "model_comparison_table.png")
    if not os.path.exists(source):
        print(f"  [Skip] {source} not found; run experimental_analysis.py first.")
        return
    shutil.copy2(source, target)
    print(f"✓ model_comparison_table.png synced from exp_08_summary_table.png")


def generate_top7_features_table():
    X_train, y_train, feature_names = load_latest_data()
    scores = pearson_importance(X_train, y_train)
    order = np.argsort(scores)[::-1][:7]
    rows = [[str(rank), feature_names[idx], f"{scores[idx]:.4f}"]
            for rank, idx in enumerate(order, start=1)]

    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor("#FFF5F5")
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Rank", "Feature", "|Pearson r|"],
        cellLoc="center",
        colColours=[COLORS['primary'], COLORS['secondary'], COLORS['tertiary']],
        colWidths=[0.12, 0.62, 0.22],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.55)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        cell.set_linewidth(1.2)
        if r == 0:
            cell.set_text_props(weight="bold", color="black")
        else:
            cell.set_facecolor("#FFFFFF" if r % 2 else "#FFF5F5")
            if c == 1:
                cell.set_text_props(ha="left")

    ax.set_title(
        "Top-7 Features by Absolute Pearson Correlation (Latest data_outputs)",
        fontsize=14, fontweight="bold", pad=16,
    )
    plt.tight_layout()

    target = os.path.join(OUTPUT_DIR, "top7_features_table.png")
    plt.savefig(target, dpi=300, bbox_inches="tight", facecolor="#FFF5F5")
    plt.close()
    print(f"✓ top7_features_table.png saved from real data_outputs")


if __name__ == '__main__':
    generate_all_figures()
