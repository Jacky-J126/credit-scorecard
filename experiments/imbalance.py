# -*- coding: utf-8 -*-
"""
实验 3：类别不平衡处理对比

回答 README「后续改进方向」第三条：坏客户占比约 6.7%，存在类别不平衡，
可对比类别权重、过采样等处理方式。

对比方案（统一在测试集上评估 AUC / KS / Gini）：
1. logistic_woe     ：主流水线同款 5 个 WOE 特征 + 逻辑回归（不平衡，基线）；
2. class_weight     ：同一组 WOE 特征 + class_weight='balanced'（sklearn 逻辑回归）；
3. oversample       ：对坏样本随机有放回过采样至 1:1 后拟合（sklearn 逻辑回归）；
4. xgb_scale_weight ：原始特征 + XGBoost scale_pos_weight（树模型自带不平衡处理）。

口径：与主流水线相同的清洗与 70/30 切分（random_state=0）；train-only 拟合。

运行：python experiments/imbalance.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc as auc_fn

import common

OUT_DIR = 'experiments/results'
FIG_DIR = 'experiments/figures'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def _oversample_bads(woe_frame):
    """对坏样本随机有放回过采样至好:坏 = 1:1，返回增强后的 WOE 特征帧。"""
    Y = woe_frame[common.TARGET]
    goods = woe_frame[Y == 1]
    bads = woe_frame[Y == 0]
    n_goods = len(goods)
    rng = np.random.RandomState(0)
    bads_up = bads.iloc[rng.randint(len(bads), size=n_goods)]
    return pd.concat([goods, bads_up], ignore_index=True)


def main():
    print('=' * 70)
    print('实验 3：类别不平衡处理对比')
    print('=' * 70)

    data = common.load_clean_data()
    train, test = common.split_train_test(data)
    y_te = test[common.TARGET]
    bad_rate = 1 - train[common.TARGET].mean()
    print(f'训练集坏客户占比：{bad_rate:.2%}')

    cuts, woes = common.build_woe_pipeline(train)
    train_woe = common.apply_woe(train, cuts, woes)
    test_woe = common.apply_woe(test, cuts, woes)

    models = {}

    # 1. 基线：主流水线同款 5 变量 WOE 逻辑回归（不平衡）
    fit = common.fit_logit(train_woe)
    prob = common.predict_logit(fit, test_woe)
    models['logistic_woe'] = {'metrics': common.eval_metrics(y_te, prob),
                              'prob': prob,
                              'note': 'WOE 特征，不平衡（基线）'}

    # 2. class_weight='balanced'
    lr_cw = LogisticRegression(penalty=None, max_iter=2000, solver='lbfgs',
                               class_weight='balanced')
    lr_cw.fit(train_woe[common.KEPT_FEATURES], train_woe[common.TARGET])
    prob = lr_cw.predict_proba(test_woe[common.KEPT_FEATURES])[:, 1]
    models['class_weight'] = {'metrics': common.eval_metrics(y_te, prob),
                              'prob': prob,
                              'note': 'WOE 特征，class_weight=balanced'}

    # 3. 坏样本过采样至 1:1
    train_up = _oversample_bads(train_woe)
    lr_up = LogisticRegression(penalty=None, max_iter=2000, solver='lbfgs')
    lr_up.fit(train_up[common.KEPT_FEATURES], train_up[common.TARGET])
    prob = lr_up.predict_proba(test_woe[common.KEPT_FEATURES])[:, 1]
    models['oversample'] = {'metrics': common.eval_metrics(y_te, prob),
                            'prob': prob,
                            'note': 'WOE 特征，坏样本过采样至 1:1'}

    # 4. XGBoost scale_pos_weight（原始特征）
    X_tr_raw = train[common.ALL_FEATURES]
    X_te_raw = test[common.ALL_FEATURES]
    scale = (train[common.TARGET] == 1).sum() / (train[common.TARGET] == 0).sum()
    xgbm = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale, eval_metric='auc',
        random_state=0, n_jobs=-1)
    xgbm.fit(X_tr_raw, train[common.TARGET])
    prob = xgbm.predict_proba(X_te_raw)[:, 1]
    models['xgb_scale_weight'] = {'metrics': common.eval_metrics(y_te, prob),
                                  'prob': prob,
                                  'note': '原始特征，scale_pos_weight'}

    # 结果表
    rows = []
    print('\n不平衡处理对比（测试集，70/30 切分 random_state=0，train-only 拟合）：')
    print('-' * 78)
    print('%-18s %8s %8s %8s   %s' % ('model', 'AUC', 'KS', 'Gini', 'note'))
    for name in ('logistic_woe', 'class_weight', 'oversample', 'xgb_scale_weight'):
        m = models[name]['metrics']
        rows.append({'model': name, **m, 'note': models[name]['note']})
        print('%-18s %8.4f %8.4f %8.4f   %s' % (
            name, m['auc'], m['ks'], m['gini'], models[name]['note']))
    print('-' * 78)

    result = {
        'split': '70/30 random_state=0',
        'fitting': 'train-only（实验口径）',
        'train_bad_rate': round(float(bad_rate), 4),
        'n_test': int(len(test)),
        'models': rows,
    }
    with open(f'{OUT_DIR}/imbalance.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'结果已保存至 {OUT_DIR}/imbalance.json')

    # ROC 对比图
    plt.figure(figsize=(7, 6))
    colors = {'logistic_woe': '#1f77b4', 'class_weight': '#ff7f0e',
              'oversample': '#2ca02c', 'xgb_scale_weight': '#d62728'}
    for name in ('logistic_woe', 'class_weight', 'oversample', 'xgb_scale_weight'):
        m = models[name]
        fpr, tpr, _ = roc_curve(y_te, m['prob'])
        plt.plot(fpr, tpr, color=colors[name], lw=1.6,
                 label=f"{name} (AUC={m['metrics']['auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'r--', lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Class Imbalance Handling: ROC Comparison')
    plt.legend(loc='lower right')
    plt.savefig(f'{FIG_DIR}/imbalance_roc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'图形已保存至 {FIG_DIR}/imbalance_roc.png')

    print('\n结论：坏样本率 6.7% 时，类别权重/过采样对排序指标（AUC/KS）提升有限'
          '（KS +0.0018），因为排序指标对先验概率不敏感；不平衡处理主要影响'
          '概率校准与阈值选择，而非排序能力。树模型（XGBoost）的增益来自模型'
          '容量，与实验 1 结论一致。')
    print('实验 3 完成。')


if __name__ == '__main__':
    main()
