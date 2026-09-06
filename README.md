# 金融信用评分卡（A 卡）：基于监督学习的实现

> 吉林大学计算机科学与技术学院辅修学士学位毕业论文《基于监督学习的金融信用评分系统的实现》的完整代码、数据与结果。
> 指导教师：张宗升 教授 ｜ 完成时间：2026 年 5 月

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
├── LICENSE                  # MIT
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

## 数据来源与许可

- 数据：Kaggle 竞赛公开数据集 [Give Me Some Credit](https://www.kaggle.com/competitions/GiveMeSomeCredit)（2011 年发布，约 15 万条个人消费贷款记录），仅用于教学与研究。
- 代码：MIT License，见 [LICENSE](LICENSE)。

## 已知局限与后续改进方向

- 逻辑回归为强可解释性基线模型，可与 XGBoost/LightGBM 做对比实验；
- 可补充 KS、PSI（群体稳定性）、跨期验证与拒绝推断（Reject Inference）；
- 类别不平衡问题可通过欠采样/代价敏感学习进一步处理；
- 本系统为教学与研究原型，未接入生产环境。

## 致谢

感谢指导教师张宗升教授在选题、方法论与研究细节上的指导。
