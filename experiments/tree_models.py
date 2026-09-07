# -*- coding: utf-8 -*-
"""
实验 1：XGBoost / LightGBM 与逻辑回归对比

回答 README「后续改进方向」第一条：逻辑回归为强可解释性基线模型，可与
XGBoost/LightGBM 做对比实验。

口径：
- 与主流水线相同的清洗与 70/30 切分（random_state=0）；
- 树模型直接使用原始特征（树模型自带非线性与自动交互），LR 提供两个版本：
  WOE 特征版（主流水线同款 5 变量，train-only 拟合）与标准化原始特征版；
- 评估指标：AUC / KS / Gini（与主流水线同一计算口径）。

运行：python experiments/tree_models.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import common

OUT_DIR = 'experiments/results'
FIG_DIR = 'experiments/figures'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def main():
    print('=' * 70)
    print('实验 1：XGBoost / LightGBM 与逻辑回归对比')
    print('=' * 70)

    data = common.load_clean_data()
    train, test = common.split_train_test(data)
    y_te = test[common.TARGET]
    X_tr_raw = train[common.ALL_FEATURES]
    X_te_raw = test[common.ALL_FEATURES]

    models = {}

    # 基线 1：LR + WOE（主流水线同款 5 变量，train-only 拟合）
    cuts, woes = common.build_woe_pipeline(train)
    fit_woe = common.fit_logit(common.apply_woe(train, cuts, woes))
    prob = common.predict_logit(fit_woe, common.apply_woe(test, cuts, woes))
    models['logistic_woe'] = {'metrics': common.eval_metrics(y_te, prob),
                              'prob': prob,
                              'note': 'WOE 特征，主流水线同款 5 变量（train-only）'}

    # 基线 2：LR + 标准化原始特征（公平基线：同一组原始特征）
    scaler = StandardScaler().fit(X_tr_raw)
    lr_raw = LogisticRegression(penalty=None, max_iter=2000, solver='lbfgs')
    lr_raw.fit(scaler.transform(X_tr_raw), train[common.TARGET])
    prob = lr_raw.predict_proba(scaler.transform(X_te_raw))[:, 1]
    models['logistic_raw'] = {'metrics': common.eval_metrics(y_te, prob),
                              'prob': prob,
                              'note': '标准化原始特征'}

    # XGBoost
    print('训练 XGBoost ...')
    xgbm = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='auc', random_state=0, n_jobs=-1)
    xgbm.fit(X_tr_raw, train[common.TARGET])
    prob = xgbm.predict_proba(X_te_raw)[:, 1]
    models['xgboost'] = {'metrics': common.eval_metrics(y_te, prob),
                         'prob': prob,
                         'note': '原始特征',
                         'importance': xgbm.feature_importances_}

    # LightGBM
    print('训练 LightGBM ...')
    lg = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=0, n_jobs=-1, verbose=-1)
    lg.fit(X_tr_raw, train[common.TARGET])
    prob = lg.predict_proba(X_te_raw)[:, 1]
    models['lightgbm'] = {'metrics': common.eval_metrics(y_te, prob),
                          'prob': prob,
                          'note': '原始特征',
                          'importance': lg.feature_importances_}

    # 结果表
    rows = []
    print('\n模型对比（测试集，70/30 切分 random_state=0，train-only 拟合）：')
    print('-' * 70)
    print('%-16s %8s %8s %8s   %s' % ('model', 'AUC', 'KS', 'Gini', 'note'))
    for name in ('logistic_woe', 'logistic_raw', 'xgboost', 'lightgbm'):
        m = models[name]['metrics']
        rows.append({'model': name, **m, 'note': models[name]['note']})
        print('%-16s %8.4f %8.4f %8.4f   %s' % (
            name, m['auc'], m['ks'], m['gini'], models[name]['note']))
    print('-' * 70)

    # 保存结果
    result = {
        'split': '70/30 random_state=0',
        'fitting': 'train-only（实验口径；主流水线论文口径为全量拟合）',
        'n_test': int(len(test)),
        'models': rows,
    }
    with open(f'{OUT_DIR}/tree_models.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'结果已保存至 {OUT_DIR}/tree_models.json')

    # 图 1：ROC 对比
    from sklearn.metrics import roc_curve, auc as auc_fn
    plt.figure(figsize=(7, 6))
    for name, color in (('logistic_woe', '#1f77b4'), ('logistic_raw', '#ff7f0e'),
                        ('xgboost', '#2ca02c'), ('lightgbm', '#d62728')):
        m = models[name]
        fpr, tpr, _ = roc_curve(y_te, m['prob'])
        plt.plot(fpr, tpr, color=color, lw=1.6,
                 label=f"{name} (AUC={m['metrics']['auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'r--', lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Comparison: Logistic Regression vs Gradient Boosting')
    plt.legend(loc='lower right')
    plt.savefig(f'{FIG_DIR}/tree_roc_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 图 2：树模型特征重要性
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, color) in zip(axes, (('xgboost', '#2ca02c'), ('lightgbm', '#d62728'))):
        imp = pd.Series(models[name]['importance'], index=common.ALL_FEATURES)
        imp = imp.sort_values()
        imp.plot(kind='barh', ax=ax, color=color)
        ax.set_title(f'{name} feature importance')
        ax.set_xlabel('importance')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/tree_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'图形已保存至 {FIG_DIR}/')

    print('实验 1 完成。')


if __name__ == '__main__':
    main()
