# -*- coding: utf-8 -*-
"""
experiments/common.py

从 credit_scoring.py 主流水线提取的公共预处理与建模函数，与主流水线保持同口径
（相同清洗规则、相同 70/30 切分 random_state=0、相同 WOE 分箱与逻辑回归）。

experiments/ 下的实验脚本统一从这里导入，避免复制粘贴导致口径漂移。
主流水线 credit_scoring.py 是论文数字的唯一复现入口，本文件不修改其输出。

口径说明：主流水线沿用论文做法，逻辑回归在「全量清洗数据」上拟合（含测试集，
model_metrics.json 中 AUC 0.8478 / KS 0.5428 / Gini 0.6956 为该口径）。
本文件的 fit_logit 为标准 train-only 拟合（实验口径，无测试集泄漏），
因此实验里的 LR 基线 KS 为 0.5444，与 model_metrics.json 略有差异，README 已说明。

用法：
    from experiments import common          # 仓库根目录执行
    或
    import common                           # experiments/ 目录内脚本互导
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

TARGET = 'SeriousDlqin2yrs'

# 主流水线保留的 5 个入模特征（WOE 逻辑回归）
KEPT_FEATURES = [
    'RevolvingUtilizationOfUnsecuredLines',
    'age',
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse',
]

# 主流水线因 IV<0.02 / 业务原因剔除的特征
DROPPED_FEATURES = [
    'DebtRatio',
    'MonthlyIncome',
    'NumberOfOpenCreditLinesAndLoans',
    'NumberRealEstateLoansOrLines',
    'NumberOfDependents',
]

ALL_FEATURES = KEPT_FEATURES + DROPPED_FEATURES

# 手动分箱特征的分箱切点（与 credit_scoring.py 第 6 节完全一致）
MANUAL_CUTS = {
    'NumberOfTime30-59DaysPastDueNotWorse': [float('-inf'), 0, 1, 3, 5, float('inf')],
    'NumberOfOpenCreditLinesAndLoans': [float('-inf'), 1, 2, 3, 5, float('inf')],
    'NumberOfTimes90DaysLate': [float('-inf'), 0, 1, 3, 5, float('inf')],
    'NumberRealEstateLoansOrLines': [float('-inf'), 0, 1, 2, 3, float('inf')],
    'NumberOfTime60-89DaysPastDueNotWorse': [float('-inf'), 0, 1, 3, float('inf')],
    'NumberOfDependents': [float('-inf'), 0, 1, 2, 3, 5, float('inf')],
}


def set_missing(df):
    """随机森林填补 MonthlyIncome 缺失值（与主流水线第 2 节一致）。"""
    process_df = df.iloc[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
    known = process_df[process_df.MonthlyIncome.notnull()].to_numpy()
    unknown = process_df[process_df.MonthlyIncome.isnull()].to_numpy()
    X = known[:, 1:]
    y = known[:, 0]
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X, y)
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
    return df


def load_clean_data(path='data/cs-training.csv'):
    """数据加载 + 清洗 + 标签反转（与主流水线第 1-2 节一致）。"""
    data = pd.read_csv(path)
    data = set_missing(data)
    data = data.dropna()
    data = data.drop_duplicates()
    data = data[data['age'] > 0]
    data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]
    # 反转 SeriousDlqin2yrs：1=好客户，0=坏客户
    data[TARGET] = 1 - data[TARGET]
    return data.reset_index(drop=True)


def split_train_test(data, test_size=0.3, random_state=0):
    """70/30 切分（与主流水线第 3 节一致）。"""
    Y = data[TARGET]
    X = data.iloc[:, 1:]
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=random_state)
    train = pd.concat([Y_train, X_train], axis=1).reset_index(drop=True)
    test = pd.concat([Y_test, X_test], axis=1).reset_index(drop=True)
    return train, test


def mono_bin(Y, X, n=20):
    """自动单调分箱并计算 WOE/IV（与主流水线第 5 节一致，去掉打印）。"""
    r = 0
    good = Y.sum()
    bad = Y.count() - good
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n, duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index=True, observed=False)
        r, p = stats.spearmanr(d2.mean().X, d2.mean().Y)
        n = n - 1
        if n <= 1:
            break
    d3 = pd.DataFrame(d2.X.min(), columns=['min'])
    d3['min'] = d2.min().X
    d3['max'] = d2.max().X
    d3['sum'] = d2.sum().Y
    d3['total'] = d2.count().Y
    d3['rate'] = d2.mean().Y
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    # 防护：样本高度偏移时可能出现纯好/纯坏分箱（WOE=±inf），裁剪并归零。
    # 全量训练集上无纯分箱，基线数值不受影响。
    d3['woe'] = d3['woe'].clip(-10, 10).fillna(0.0)
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = d3.sort_values(by='min').reset_index(drop=True)
    actual_bins = len(d4)
    cut = [float('-inf')]
    for i in range(1, actual_bins):
        qua = X.quantile(i / actual_bins)
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4, iv, cut, woe


def calc_woe_manual(Y, X, cut):
    """手动分箱特征计算 WOE/IV（与主流水线第 6 节一致，去掉打印）。"""
    good = Y.sum()
    bad = Y.count() - good
    d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.cut(X, cut)})
    d2 = d1.groupby('Bucket', as_index=True, observed=False)
    d3 = pd.DataFrame({
        'min': d2.min().X,
        'max': d2.max().X,
        'sum': d2.sum().Y,
        'total': d2.count().Y,
        'rate': d2.mean().Y
    })
    d3['woe'] = np.log((d3['rate'] / (1 - d3['rate'])) / (good / bad))
    # 防护：同 mono_bin——纯好/纯坏分箱的 WOE=±inf，裁剪并归零。
    d3['woe'] = d3['woe'].clip(-10, 10).fillna(0.0)
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d3 = d3.reset_index(drop=True)
    woe = list(d3['woe'].round(3))
    return d3, iv, woe


def build_woe_pipeline(train):
    """基于训练集构建全部 10 个特征的 WOE 分箱切点与 WOE 值表。

    返回 (cuts, woes) 两个字典，key 为特征名。
    """
    Y = train[TARGET]
    cuts, woes = {}, {}
    for feat in ALL_FEATURES:
        if feat in MANUAL_CUTS:
            cut = MANUAL_CUTS[feat]
            _, _, woe = calc_woe_manual(Y, train[feat], cut)
        else:
            _, _, cut, woe = mono_bin(Y, train[feat])
        cuts[feat] = cut
        woes[feat] = woe
    return cuts, woes


def replace_woe(series, cut, woe):
    """按切点把原始值替换为 WOE 值（与主流水线第 7 节一致）。"""
    result = []
    i = 0
    while i < len(series):
        value = series.iloc[i] if hasattr(series, 'iloc') else series[i]
        j = len(cut) - 2
        m = len(cut) - 2
        while j >= 0:
            if value >= cut[j]:
                j = -1
            else:
                j -= 1
                m -= 1
        result.append(woe[m])
        i += 1
    return result


def apply_woe(frame, cuts, woes):
    """对 DataFrame 的全部特征做 WOE 替换，返回副本。"""
    out = frame.copy()
    for feat in ALL_FEATURES:
        out[feat] = replace_woe(out[feat], cuts[feat], woes[feat])
    return out


def fit_logit(train_woe):
    """在 WOE 特征上拟合逻辑回归（保留 5 个入模特征，与主流水线第 8 节一致）。"""
    Y = train_woe[TARGET]
    X = train_woe[KEPT_FEATURES]
    X1 = sm.add_constant(X)
    return sm.Logit(Y, X1).fit()


def predict_logit(fit, frame_woe):
    """用拟合好的逻辑回归预测（好客户概率，方向与主流水线一致）。"""
    X = frame_woe[KEPT_FEATURES]
    return fit.predict(sm.add_constant(X))


def eval_metrics(y_true, y_prob):
    """AUC / KS / Gini（与主流水线第 11 节同一口径）。"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    a = auc(fpr, tpr)
    return {
        'auc': round(float(a), 4),
        'ks': round(float(max(tpr - fpr)), 4),
        'gini': round(float(2 * a - 1), 4),
    }
