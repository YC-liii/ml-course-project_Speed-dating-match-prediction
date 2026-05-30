# Speed Dating Match Prediction — 项目说明

> **课程**：AI3013 机器学习  
> **任务**：基于 Speed Dating 数据集，使用**三种 from-scratch 分类器**（LR / AdaBoost / SVM）预测两人是否会 `match`  
> **核心思路**：数据清洗 + 个人内标准化 (GPB) + 兴趣相似度特征工程 + wave-level split 防泄漏 + 三模型统一对比

---

## 1. 快速上手

```bash
cd ml_lr_term

# Step 0 (一次性，已完成):  调参,生成最优超参 (开发期跑过,演示无需再跑)
python3 lr_grid.py        # LR  网格搜索 (96 combos x 5-fold CV x 3 pipelines)
python3 adaboost_grid.py  # AdaBoost 重复 5-fold CV TopK 搜索
python3 svm_grid.py       # SVM  15 候选 + 64 ensemble 变体搜索

# Step 1: 数据预处理(首次运行,生成 data_outputs/)
python3 ML_data.py

# Step 2: 训练三个最终模型 (公平计时!仅 fit() 时间)
python3 lr.py        # ~0.3s   LR 最终训练 → lr_results.npz
python3 adaboost.py  # ~3s     AdaBoost 最终训练 → adaboost_results.npz
python3 svm.py       # ~30s    SVM 5-ensemble + Platt → svm_results.npz

# Step 3: 统一对比 + 生成 13 张实验图 (final_figures/exp_*.png)
python3 experimental_analysis.py

# Step 4: 生成报告专用图 (final_figures/fig*.png + 表格)
python3 generate_all_figures.py
```

**演示推荐流程**: 跳过 Step 0(grid 脚本调参时一次性跑过), 直接 Step 1→4。
三个最终模型脚本都 **只用最优参数训练 final 模型**, 训练时间公平可比。

**或者一键跑流水线 (Step 1-4)**:

```bash
python3 ML_data.py && python3 lr.py && python3 adaboost.py && python3 svm.py && python3 experimental_analysis.py && python3 generate_all_figures.py
```

---

## 2. 项目目录

```
ml_lr_term/
├── Speed Dating Data.csv           # 原始数据 (8378 行 × 195 列)
│
├── ML_data.py                      # 数据预处理 + 特征工程
│
├── lr_grid.py                      # LR 调参脚本 (网格搜索, 一次性跑)
├── lr.py                           # LR 最终模型 (用最优参数训练)
│
├── adaboost_grid.py                # AdaBoost 调参脚本 (一次性跑)
├── adaboost.py                     # AdaBoost 最终模型 (用最优参数训练)
│
├── svm_grid.py                     # SVM 调参脚本 (一次性跑)
├── svm.py                          # SVM 最终模型 (5-ensemble + Platt)
│
├── experimental_analysis.py        # 加载 3 个 .npz, 13 张实验图
├── generate_all_figures.py         # 报告专用图 (fig1-4 + 表格)
│
├── data_outputs/                   # 预处理产物 + 最优 ensemble 配置
├── model_outputs/                  # 模型结果 (lr/adaboost/svm_results.npz)
├── final_figures/                  # 最终报告图 (exp_*.png, fig*.png)
│
├── experiment_run_log.md           # 运行日志
└── README.md                       # 本文件
```

**调参 vs 最终训练分离**: 每个模型都拆成 `_grid.py` (调参, 慢) + `.py` (最终训练, 快)。
演示时只跑 `lr.py / adaboost.py / svm.py`, 训练时间公平可比。

---

## 3. 数据 & 任务

| 项 | 值 |
|---|---|
| 数据集 | Columbia Speed Dating Experiment (Fisman 2006) |
| 样本量 | 8378 行 (每行 = 一对 4 分钟约会) |
| 原始列数 | 195 (含简写列名 `attr`/`sinc`/`intel`/`fun`/`amb` ...) |
| 目标变量 | `match` (0/1)，双方都打勾即 match |
| 类别不平衡 | **1 : 4.9** (正样本仅 16.9%) |

**清洗后**：`8156 行 × 54 个特征`，wave-level split → train 5707, test 2449。

---

## 4. 整体管线 (上游 → 下游)

```
Speed Dating Data.csv
        │
        ▼
ML_data.py 全管道
  │
  ├─ STEP 1 EDA: 缺失率 / 目标分布 / 关键特征分布 / 兴趣相关性
  ├─ STEP 2 清洗: 删 match=NaN → 删核心特征>30%缺失行 → 3σ 截断 → 中位数/众数填充
  ├─ STEP 3 GPB 个人内标准化: μ_i, σ_i 消除打分基准偏差 → 17 维兴趣 z-score
  │           └─ 衍生: SIS / interest_cosine / interest_euclidean_sim
  ├─ STEP 4 交互特征: gender × {attractive, sincere, ...}, age_diff, trait_pair, ...
  ├─ STEP 5 相关性分析: Top-20 与 match 相关性条形图 + 共线性热图
  ├─ STEP 6 Wave-level split: 按实验 wave 分组，训练/测试参与者无重叠
  ├─ STEP 7 特征矩阵构建: 训练集 μ/σ 标准化 (防泄露)
  └─ STEP 8 保存: X_train.npy / X_test.npy / ... → data_outputs/
        │
        ▼
lr.py / adaboost.py / svm.py (并行训练)
  │
  ├─ lr.py:
  │     • 特征选择: L1+L2 Fusion / Correlation Filter
  │     • 5-Fold CV 网格搜索
  │     • 手写 LR (mini-batch SGD + Momentum + class_weight)
  │     • 保存 model_outputs/lr_results.npz
  │
  ├─ adaboost.py:
  │     • 手写 WeightedDecisionStump + AdaBoost.SAMME
  │     • TopK 特征筛选 + Stump TopK 微调
  │     • 保存 model_outputs/adaboost_results.npz
  │
  └─ svm.py:
        • 手写 SMO 优化器 + RBF Kernel
        • Platt scaling 概率校准
        • 候选超参数搜索 + Top-5 ensemble
        • 保存 model_outputs/svm_results.npz
        │
        ▼
experimental_analysis.py
  │
  ├─ 加载三个 model_outputs/*.npz
  ├─ 统一对比 + 13 张实验图 → final_figures/exp_*.png
  └─ 追加日志 → experiment_run_log.md
        │
        ▼
generate_all_figures.py
  │
  └─ 生成报告专用图 → final_figures/fig*.png + 表格
```

---

## 5. 上游：`ML_data.py` 关键技术

### 5.1 GPB (Global Persona Baseline) 个人内标准化 ⭐核心创新⭐

**问题**：同一项兴趣 `sports=8`，对"打分手松"的人是中等水平，对"打分手紧"的人却是极高水平。直接用原始分会引入**评分基准偏差**。

**解法**：对每个参与者 $i$，先统计 ta 在所有 39 个评分列上的均值 $\mu_i$ 和标准差 $\sigma_i$：

$$ z_{i,k} = \frac{\text{RawScore}_{i,k} - \mu_i}{\sigma_i} $$

得到 17 维"消除偏差的相对兴趣强度"，存为 `z_sports`, `z_tvsports`, ...

### 5.2 兴趣相似度三件套

基于 z-score 向量 $\mathbf{v}_A, \mathbf{v}_B$ 计算三种相似度：

| 特征名 | 公式 | 物理含义 |
|---|---|---|
| `SIS` (Standardized Interest Similarity) | $1 - \frac{1}{17}\sum \|z_{A,k} - z_{B,k}\|$ | 平均绝对差异 (越大越像) |
| `interest_cosine` | $\frac{\mathbf{v}_A \cdot \mathbf{v}_B}{\|\mathbf{v}_A\|\|\mathbf{v}_B\|}$ | **方向相似** (品味结构对齐) |
| `interest_euclidean_sim` | $1 - \frac{\sqrt{\sum w_k (z_{A,k}-z_{B,k})^2}}{\max}$ | **强度相似** (兴趣浓度相近) |

其中 `interest_euclidean` 用方差归一化权重 $w_k$ — 区分度高的兴趣维度权重大。

### 5.3 交互特征 (捕捉性别偏好差异)

```
gender × {attractive, sincere, intelligence, funny, ambition, income}  → 6 个
_pair 特征: attractive_pair, funny_pair, trait_pair, ...              → 多个
_gap 特征: age_diff, ...                                              → 多个
_centered 特征: like_iid_centered, ...                                → 多个
```

### 5.4 Wave-level Split 防数据泄露 ⭐严格切分⭐

**问题**：同一个参与者会在多个 wave 中出现，如果随机切分会导致同一个人的数据同时出现在训练和测试集。

**解法**：按实验 wave 分组，训练集和测试集使用完全不同的 wave，确保参与者无重叠。

```
Train waves: [4, 7, 8, 9, 10, 11, 13, 14, 17, 18, 19, 21]
Test  waves: [1, 2, 3, 5, 6, 12, 15, 16, 20]
Participant overlap: 0
```

### 5.5 防数据泄露

- **Wave-level split 优先**：先按 wave 划分，再用**训练集** $\mu, \sigma$ 标准化训练 + 测试
- 缺失填充用全集中位数 (合理，因为是 imputation 而非模型参数)

---

## 6. 下游：三个 from-scratch 模型

### 6.1 `lr.py` — Logistic Regression

> ✅ **全部 NumPy 实现**，零 sklearn 依赖。

| 组件 | 实现 |
|---|---|
| 优化器 | Mini-batch SGD + Momentum (默认 m=0.9) |
| 正则 | L1 (子梯度 sign(θ)) 或 L2，bias 不参与惩罚 |
| 类别权重 | `balanced` (正样本权重 = n / (2·n_pos)) |
| Loss | 加权 log-loss + L1/L2 regularization |
| 数值稳定 | `np.clip(z, -250, 250)` 防 sigmoid overflow |

**特征选择**：

- **Pipeline A: L1+L2 Fusion** (嵌入式)
  - L1 提供稀疏选择，L2 提供重要性排序
  - fusion_score = (L1 系数非零) × |L2 系数|

- **Pipeline B: Correlation Filter** (过滤式)
  - Pearson 相关系数排序
  - 自适应阈值 = 中位数

**最终选定**：**Re-selected Corr Top18** 管线

### 6.2 `adaboost.py` — AdaBoost

> ✅ **全部 NumPy 实现**，零 sklearn 依赖。

| 组件 | 实现 |
|---|---|
| 基学习器 | WeightedDecisionStump (决策桩) |
| 权重更新 | AdaBoost.SAMME 类型 |
| 特征选择 | TopK 特征筛选 + Stump TopK 微调 |
| 类别权重 | 初始化时对正样本加权 |

**最终选定**：**top28** 管线

### 6.3 `svm.py` — RBF Kernel SVM

> ✅ **全部 NumPy 实现**，零 sklearn 依赖。

| 组件 | 实现 |
|---|---|
| 优化器 | SMO (Sequential Minimal Optimization) |
| Kernel | RBF (Radial Basis Function) |
| 类别权重 | pos_weight_multiplier 对正样本加权 |
| 概率校准 | Platt scaling + raw-z sigmoid 双策略 |
| 超参数搜索 | gamma × pos_weight_multiplier 网格 (15 个候选) |
| Ensemble Pool | val_f1 + val_ap 双 pool 策略 |
| Ensemble 变体 | 16 种组合 (2 pools × 2 score types × 2 weight types × 4 k值) |
| 集成 | Top-5 val_f1 platt_mean validation ensemble |

**最终选定**：**Top-5 val_f1 platt_mean validation ensemble** 管线  
**搜索空间**：15 个单模型 + 64 个 ensemble 候选 = **79 个配置**

---

## 7. 最新结果

来自 `experiment_run_log.md` 最新记录 (2026-05-28, **公平计时版**):

| Model | Accuracy | Precision | Recall | F1 | AUC | Threshold | Train Time |
|---|---|---|---|---|---|---|---|
| **LR** | 0.8024 | 0.4496 | 0.7536 | **0.5632** | **0.8474** | 0.548 | **0.27s** |
| **AdaBoost** | 0.8085 | 0.4568 | 0.7029 | 0.5538 | 0.8410 | 0.660 | **2.9s** |
| **SVM** | 0.8195 | 0.4741 | 0.6184 | 0.5367 | 0.8229 | 0.340 | **32.5s** |

**计时说明 (公平对比)**:
- LR: 1 个模型的 `fit()` 时间, 不含 grid search
- AdaBoost: 110 stumps 训练时间, 不含 TopK 搜索
- SVM: 5 个 SMO + Platt 拟合时间, 不含 79 候选 ensemble 搜索

**观察**:
- **LR 在 F1 / AUC 上最强**, 训练最快 (0.27s, 单模型 + 18 维特征)
- AdaBoost 次之, 110 stumps 串行训练 ~3s
- SVM ensemble 慢是模型本身代价 (5 个 RBF SMO + 5 套 Platt)
- 三模型 F1 都在 0.53 ~ 0.56, AUC 0.82 ~ 0.85 (1:5 不平衡下相当扎实)
- `class_weight='balanced'` 刻意提高 Recall → Precision 被压低, 这是抓 Match 不漏报的取舍

---

## 8. 真实 Top-7 特征

来自 `generate_all_figures.py` 最新生成 (2026-05-24 22:48:22)：

```
1. attractive_pair             0.3973
2. funny_pair                  0.3891
3. like_x_shared_interests_o   0.3876
4. trait_pair                  0.3731
5. like_x_guess_prob_liked     0.3247
6. like_iid_centered           0.3164
7. like                        0.3047
```

**观察**：
- **最强信号都是工程特征**（`_pair`、`_x_`、`_centered`）
- 特征工程确实关键，原始 `like` 排第 7，但工程后的 `like_x_shared_interests_o` 排第 3
- `attractive_pair` / `funny_pair` / `trait_pair` 等"双方匹配度"特征最重要

---

## 9. 依赖

```bash
# 唯一依赖 — 零 sklearn / TensorFlow / PyTorch
numpy >= 1.20
pandas >= 1.3
matplotlib >= 3.3
seaborn >= 0.11
```

Python 3.8+。Mac/Linux/Windows 通用 (无 GPU 需求)。
项目全部代码 (模型 + 特征选择 + 评估) 都是 from-scratch 实现，满足课程要求。

---

## 10. 常见操作

### 强制重跑预处理 (修改了上游代码后)

```bash
rm -rf data_outputs/
python3 ML_data.py
```

### 只重跑某个模型

```bash
python3 lr.py          # 只重跑 LR
python3 adaboost.py    # 只重跑 AdaBoost
python3 svm.py         # 只重跑 SVM
```

### 重新生成对比图

```bash
python3 experimental_analysis.py    # 生成 exp_*.png
python3 generate_all_figures.py     # 生成 fig*.png + 表格
```

### 禁止弹图 (在服务器/CI 上跑)

```bash
MPLBACKEND=Agg python3 ML_data.py
MPLBACKEND=Agg python3 lr.py
# ... 其他脚本同理
```

---

## 11. 文件依赖图

```
                  Speed Dating Data.csv
                          │
                          ▼
              ┌─────────────────────────┐
              │  ML_data.py             │
              │  STEP 1 → 8 全管道       │
              └────────────┬────────────┘
                           │
                           ▼
              data_outputs/*.npy (缓存)
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
         ┌────────┐  ┌──────────┐  ┌──────┐
         │ lr.py  │  │adaboost  │  │svm.py│
         │        │  │.py       │  │      │
         └───┬────┘  └────┬─────┘  └───┬──┘
             │            │            │
             └────────────┼────────────┘
                          │
                          ▼
              model_outputs/*.npz
                          │
                          ▼
         ┌────────────────────────────────┐
         │  experimental_analysis.py      │
         │  • 加载三个 .npz               │
         │  • 统一对比 + 13 张实验图      │
         └────────────┬───────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │  generate_all_figures.py       │
         │  • 报告专用图 (fig1-4)         │
         │  • 表格 (model_comparison +    │
         │           top7_features)       │
         └────────────┬───────────────────┘
                      │
                      ▼
              final_figures/*.png
```

---

## 12. 给伙伴的协作建议

1. **改预处理逻辑** → 改 `ML_data.py` → 删 `data_outputs/` → 重跑 `ML_data.py`
2. **改 LR / AdaBoost / SVM** → 只改对应 `.py` → 直接 `python3 xxx.py`
3. **数据集换了** → 检查 `ML_data.py` 的列名映射，必要时新增；删 `data_outputs/`
4. **加新模型** → 写个新 `.py`，保存 `model_outputs/xxx_results.npz`，在 `experimental_analysis.py` 里加载
5. **加新指标** → 在各模型的评估函数里加，并改 `experimental_analysis.py` 的 metric list

---

## 13. 项目亮点

1. **严格的 wave-level split**：参与者无重叠，防止数据泄漏
2. **GPB 个人内标准化**：消除评分基准偏差，提升特征质量
3. **丰富的特征工程**：兴趣相似度三件套 + 交互特征 + _pair/_gap/_centered 系列
4. **三种 from-scratch 模型**：LR / AdaBoost / SVM 全部 NumPy 实现，零 sklearn
5. **统一对比框架**：`experimental_analysis.py` 读取三个 `.npz`，生成 13 张实验图
6. **真实数据驱动**：所有图表来自真实训练结果，无硬编码假数据
7. **完整日志追踪**：`experiment_run_log.md` 记录每次运行的详细信息

---

## 14. 致谢

- 数据集：Columbia Speed Dating Experiment (Fisman et al., 2006)
- 课程：AI3013 机器学习
- 团队成员：[在此填写团队成员]

---

**最后更新**：2026-05-25  
**版本**：v2.1 (升级 SVM ensemble 策略 + 完整流程验证)
