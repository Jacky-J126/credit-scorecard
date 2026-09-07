# 金融信用评分卡（A 卡）：基于监督学习的实现

> 吉林大学计算机科学与技术学院辅修学士学位毕业论文《基于监督学习的金融信用评分系统的实现》的完整代码、数据与结果。
> 指导教师：张宗升 教授 ｜ 完成时间：2026 年 5 月

[![CI](https://github.com/Jacky-J126/credit-scorecard/actions/workflows/ci.yml/badge.svg)](https://github.com/Jacky-J126/credit-scorecard/actions/workflows/ci.yml)

基于 Kaggle 公开数据集 [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit)（约 15 万条个人消费贷款记录），构建端到端的申请评分卡（A 卡）模型：数据预处理 → 缺失值填补 → 最优分箱与 WOE/IV 特征筛选 → 逻辑回归建模 → 模型评估 → 标准化评分卡 → 批量自动评分系统。

## 方法论总览

| 阶段 | 方法 | 关键参数 |
|---|---|---|
| 缺失值处理 | 随机森林回归填补 | `n_estimators=200, max_depth=3, random_state=0`；填补 MonthlyIncome 19.8% 缺失 |
| 异常值处理 | 业务规则过滤 | 剔除 `age=0`、`NumberOfTime30-59DaysPastDueNotWorse≥90` 样本 |
| 数据集划分 | 分层随机划分 | 训练:测试 = 7:3，`random_state=0` |
| 变量分箱 | 最优分箱 + Spearman 单调性检验 | 自动分箱（`mono_bin`）与手动分箱结合 |
| 特征筛选 | WOE 转换 + IV 阈值 | 10 个候选变量保留 5 个（IV<0.02 剔除） |
| 建模 | 逻辑回归（statsmodels `Logit`） | 全部变量 p<0.05，系数方向与经济意义一致 |
| 评估 | 预留 30% 测试集 | AUC = 0.85（ROC 曲线下面积） |
| 评分卡 | 对数几率线性映射 | 基准分 600、PDO 20、基准好坏比 20:1 |
| 落地 | Python 批量评分函数 | `compute_score()` → `ScoreData.csv` |

### 进入模型的 5 个变量

1. `RevolvingUtilizationOfUnsecuredLines` — 循环贷款利用率（自动分箱）
2. `age` — 借款人年龄（自动分箱）
3. `NumberOfTime30-59DaysPastDueNotWorse` — 30–59 天逾期次数（手动分箱）
4. `NumberOfTimes90DaysLate` — 90 天及以上逾期次数（手动分箱）
5. `NumberOfTime60-89DaysPastDueNotWorse` — 60–89 天逾期次数（手动分箱）

被剔除的变量（IV<0.02，预测力不足）：`DebtRatio`、`MonthlyIncome`、`NumberOfOpenCreditLinesAndLoans`、`NumberRealEstateLoansOrLines`、`NumberOfDependents`。

### 评分卡换算逻辑

```
p = 20 / ln(2)                      # 评分刻度因子（PDO=20）
q = 600 - 20 * ln(20) / ln(2)       # 基准偏移（基准分 600，好坏比 20:1）
baseScore = round(q + p * β₀)       # 基础得分常数（约 795，随截距系数微调）
各分箱得分 = round(βⱼ × WOEⱼᵢ × p)
```

## 目录结构

```
.
├── credit_scoring.py        # 完整建模流水线（单文件，可一键复现）
├── requirements.txt
├── conftest.py              # pytest 根配置（保证 tests/ 可导入 experiments/）
├── LICENSE                  # MIT
├── experiments/             # 三项后续改进实验（与主流水线同口径）
│   ├── common.py            # 公共预处理/建模函数（与主流水线口径一致）
│   ├── tree_models.py       # 实验 1：XGBoost/LightGBM 与逻辑回归对比
│   ├── reject_inference.py  # 实验 2：模拟拒绝推断
│   ├── imbalance.py         # 实验 3：类别不平衡处理对比
│   ├── results/             # 各实验 JSON 结果
│   └── figures/             # 各实验图形
├── tests/
│   └── test_baseline.py     # 基线指标回归测试（锁定两种口径的核心数字）
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI（主流水线 + 测试 + 三实验）
├── data/
│   ├── cs-training.csv      # 原始数据（Kaggle Give Me Some Credit）
│   ├── DataDescribe.csv     # 原始数据描述性统计
│   ├── MissingData.csv      # 随机森林填补 + 去重后数据
│   ├── TrainData.csv        # 训练集（70%）
│   ├── TestData.csv         # 测试集（30%）
│   ├── WoeData.csv          # WOE 转换后数据
│   └── ScoreData.csv        # 评分卡批量计算结果
├── figures/                 # 运行生成的图形（ROC 曲线、评分分布等）
└── sql/                     # SQL 风控特征工程练习（SQLite，零第三方依赖）
    ├── run_queries.py       # 建库 + 执行全部查询
    ├── queries.sql          # 10 个风控面试高频查询（缺失率/分箱/WOE-IV/规则模拟…）
    ├── README.md            # 查询清单与结果解读
    └── results/             # 全部查询结果 CSV（已提交，可直接查看）
```

## 快速开始

```bash
pip install -r requirements.txt
python credit_scoring.py
```

脚本按顺序完成：数据加载 → 缺失填补 → 异常值处理 → 训练/测试集划分 → 分箱与 WOE/IV 计算 → 逻辑回归拟合 → 测试集 ROC/AUC、KS、Gini 评估 → 评分卡生成 → 批量评分。所有中间结果与最终评分表写入 `data/`，图形输出到 `figures/`。

> 完整运行约需数分钟（随机森林填补 200 棵树）。

```bash
# 回归测试（锁定主流水线与实验两种口径的核心指标）
python -m pytest tests/ -v

# 三项后续改进实验（每个约数分钟）
python experiments/tree_models.py
python experiments/reject_inference.py
python experiments/imbalance.py
```

### SQL 特征工程练习

```bash
python sql/run_queries.py
```

将数据导入 SQLite 并执行 10 个风控面试高频查询（缺失率统计、等频分箱、WOE/IV 纯 SQL 计算、策略规则模拟等），结果见 `sql/results/`，解读见 `sql/README.md`。

## 结果

以下结果可在任意机器上按上文步骤完整复现（随机种子固定，随机森林与数据集划分均确定性可重现）。

### 测试集模型指标（n = 43,607）

| 指标 | 数值 |
|---|---|
| AUC（ROC 曲线下面积） | **0.8478** |
| KS 统计量 | **0.5428** |
| Gini 系数 | 0.6956 |
| 测试集构成 | 好客户 40,714 / 坏客户 2,893 |

### 逻辑回归系数（全部 p<0.001）

| 变量 | 系数 β | z 值 |
|---|---|---|
| const（截距） | 9.7249 | 102.978 |
| RevolvingUtilizationOfUnsecuredLines | 0.6556 | 50.004 |
| age | 0.4866 | 18.827 |
| NumberOfTime30-59DaysPastDueNotWorse | 1.0266 | 41.513 |
| NumberOfTimes90DaysLate | 1.7705 | 50.539 |
| NumberOfTime60-89DaysPastDueNotWorse | 1.1462 | 29.561 |

### 评分卡示例（测试集前 3 条记录）

| BaseScore | x1（利用率） | x2（年龄） | x3（30-59 逾期） | x7（90 天逾期） | x9（60-89 逾期） | Score |
|---|---|---|---|---|---|---|
| 794 | −21 | −5 | −70 | −140 | −61 | 497 |
| 794 | 23 | 4 | −27 | −100 | −61 | 633 |
| 794 | 6 | −2 | −27 | −100 | −61 | 610 |

> 注：基础得分常数由 `baseScore = round(q + p·β₀)` 计算，本次运行结果为 794（论文正文记为 795，系四舍五入表述差异）。
>
> 完整评分卡（各变量分箱与对应得分）见论文表 3-2；批量评分结果见 `data/ScoreData.csv`。

### 输出图形（figures/）

- `roc_curve.png` — 测试集 ROC 曲线（AUC = 0.85）
- `score_distribution.png` — 测试集信用评分分布
- `iv_bar.png` — 各候选变量 IV 值对比（筛选依据）
- `corr_heatmap.png` — 变量相关性热力图

## 后续改进实验（experiments/）

针对原「已知局限与后续改进方向」中可本地完成的三项，新增 `experiments/` 三个独立实验脚本；公共函数抽在 `experiments/common.py`，与主流水线的清洗、切分、分箱口径完全一致。

> **口径说明**：主流水线沿用论文做法，逻辑回归在「全量清洗数据」上拟合（上表 KS 0.5428 为该口径）；`experiments/` 均为标准 train-only 拟合（无测试集泄漏），故实验中的 LR 基线 KS 为 0.5444。两种口径均由 `tests/test_baseline.py` 锁定回归。

### 实验 1：XGBoost / LightGBM 与逻辑回归对比（`tree_models.py`）

原始特征训练两种梯度提升树（`n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=0`），与主流水线同款 WOE 逻辑回归、标准化原始特征逻辑回归对比：

| 模型 | AUC | KS | Gini |
|---|---|---|---|
| LR + WOE（主流水线同款 5 变量） | 0.8478 | 0.5444 | 0.6957 |
| LR + 标准化原始特征 | 0.8089 | 0.4861 | 0.6178 |
| XGBoost | 0.8611 | 0.5686 | 0.7222 |
| LightGBM | 0.8613 | 0.5698 | 0.7226 |

结论：树模型相对 LR+WOE 有约 +0.013 AUC 的稳定增益；LR+WOE 的优势在于可解释性——系数可直接换算为评分卡，A 卡场景仍为首选。图形：`experiments/figures/tree_roc_comparison.png`、`tree_feature_importance.png`。

### 实验 2：模拟拒绝推断（`reject_inference.py`）

真实业务中只有获批样本的结果标签可观测，只对获批样本建模会产生选择偏差。本实验按训练集分数 60% 分位数模拟审批通过线（通过线 0.9640，获批 61,082 / 被拒 40,665），对比四种建模方式在测试集的表现（统一使用 L2 正则化逻辑回归 C=1.0——获批样本近似完全分离，无正则项会黑塞矩阵奇异）：

| 方案 | AUC | KS | Gini |
|---|---|---|---|
| baseline（全量训练样本，参考上界） | 0.8478 | 0.5444 | 0.6955 |
| approved-only（拒绝推断前） | 0.7050 | 0.3302 | 0.4100 |
| parcelling（被拒样本按分数分段推断标签） | 0.8203 | 0.5018 | 0.6406 |
| hard-cutoff（被拒样本全部判坏） | 0.8262 | 0.5190 | 0.6523 |

结论：仅用获批样本建模 AUC 损失约 0.14；两种拒绝推断方法均可恢复大部分损失，hard-cutoff 在本数据上略优。注意：本实验为教学模拟（Kaggle 数据无真实审批记录），被拒样本标签不可观测，真实业务评估口径以获批样本为准。图形：`experiments/figures/reject_inference_roc.png`。

### 实验 3：类别不平衡处理对比（`imbalance.py`）

训练集坏客户占比约 6.7%。对比类别权重、坏样本过采样至 1:1、XGBoost `scale_pos_weight`：

| 模型 | AUC | KS | Gini |
|---|---|---|---|
| LR + WOE（不平衡基线） | 0.8478 | 0.5444 | 0.6957 |
| LR + WOE（class_weight=balanced） | 0.8485 | 0.5462 | 0.6971 |
| LR + WOE（坏样本过采样至 1:1） | 0.8485 | 0.5461 | 0.6970 |
| XGBoost（scale_pos_weight） | 0.8613 | 0.5669 | 0.7226 |

结论：排序指标（AUC/KS）对先验概率不敏感，不平衡处理提升有限（KS +0.0018），其真正作用在概率校准与阈值选择；树模型增益来自模型容量，与实验 1 结论一致。图形：`experiments/figures/imbalance_roc.png`。

## 数据来源与许可

- 数据：Kaggle 竞赛公开数据集 [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit)（2011 年发布，约 15 万条个人消费贷款记录），仅用于教学与研究。
- 代码：MIT License，见 [LICENSE](LICENSE)。

## 已知局限与后续改进方向

已完成（见上文「后续改进实验」）：

1. 树模型对比实验（XGBoost / LightGBM 与逻辑回归）；
2. 模拟拒绝推断（parcelling / hard-cutoff）；
3. 类别不平衡处理对比（类别权重 / 过采样）。

待完成（需外部数据，未在本仓库内进行）：

- PSI（群体稳定性指数）与跨期验证需要同一人群跨时间窗口的监控样本，Kaggle 公开数据为单期截面数据，无法在本仓库内完成；
- 本系统为教学与研究原型，未接入生产环境。

## 致谢

感谢指导教师张宗升教授在选题、方法论与研究细节上的指导。
