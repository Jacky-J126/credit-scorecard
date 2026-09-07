# -*- coding: utf-8 -*-
"""
实验 2：模拟拒绝推断（Reject Inference）

回答 README「后续改进方向」第二条：可补充拒绝推断。

背景：审批模型只在「获批样本」上训练，被拒样本的结果标签在真实业务中不可观测，
会导致模型对整体申请人群产生选择偏差。拒绝推断通过给被拒样本推断标签来修正该偏差。

本实验为教学模拟（Kaggle 数据无真实审批记录）：
1. 用基线模型（全量训练集拟合）给训练集打分，按 60% 分位数模拟审批通过线；
2. 「获批样本」= 分数前 60%，「被拒样本」= 后 40%（其真实标签假装不可观测）；
3. 对比三种建模方式在测试集上的效果：
   - baseline：全量训练样本（参考上界，实际业务中不可得）；
   - approved-only：仅获批样本（拒绝推断前）；
   - parcelling：被拒样本按分数分段，用段内获批样本的好坏比推断标签；
   - hard-cutoff：被拒样本全部判坏（最保守的硬截断法）。

口径：train-only 拟合（实验口径）；清洗与 70/30 切分同主流水线。

运行：python experiments/reject_inference.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc as auc_fn

import common

OUT_DIR = 'experiments/results'
FIG_DIR = 'experiments/figures'
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

APPROVE_RATE = 0.60   # 模拟审批通过率
N_BANDS = 10          # parcelling 分数分段数

# 本实验统一使用 L2 正则化逻辑回归（sklearn，C=1.0）：
# 获批样本坏样本率极低、WOE 特征近似完全分离，statsmodels 无正则项会黑塞矩阵奇异；
# L2 正则也是业界在低坏样本率场景的常规做法。实验内部口径一致。
LR = lambda: LogisticRegression(penalty='l2', C=1.0, max_iter=2000, solver='lbfgs')


def _fit_on_frame(frame, woe_source=None):
    """在给定样本上重建 WOE 分箱并拟合 L2 逻辑回归。

    woe_source 为 None 时用 frame 自身建分箱（真实流程：只看得到获批样本）；
    否则沿用 woe_source 的分箱（用于对非训练样本做预测）。
    """
    if woe_source is None:
        woe_source = frame
    cuts, woes = common.build_woe_pipeline(woe_source)
    woe_frame = common.apply_woe(frame, cuts, woes)
    fit = LR().fit(woe_frame[common.KEPT_FEATURES], woe_frame[common.TARGET])
    return fit, cuts, woes


def _score(frame, fit, cuts, woes):
    woe_frame = common.apply_woe(frame, cuts, woes)
    return fit.predict_proba(woe_frame[common.KEPT_FEATURES])[:, 1]


def main():
    print('=' * 70)
    print('实验 2：模拟拒绝推断（parcelling / hard-cutoff）')
    print('=' * 70)

    data = common.load_clean_data()
    train, test = common.split_train_test(data)
    y_te = test[common.TARGET]

    # 1. 基线模型：全量训练集（真实业务中的参考上界）
    print('训练基线模型（全量训练集）...')
    fit_base, cuts_base, woes_base = _fit_on_frame(train)
    score_tr = _score(train, fit_base, cuts_base, woes_base)

    # 2. 模拟审批：按 60% 分位数截断
    cutoff = np.quantile(score_tr, 1 - APPROVE_RATE)
    approved = train[score_tr >= cutoff].reset_index(drop=True)
    rejected = train[score_tr < cutoff].reset_index(drop=True)
    print(f'模拟审批：通过线 = {cutoff:.4f}，获批 {len(approved)} 条'
          f'（{len(approved) / len(train):.1%}），被拒 {len(rejected)} 条'
          f'（{len(rejected) / len(train):.1%}）')

    # 3. 拒绝推断前：仅获批样本建模
    print('训练 approved-only 模型...')
    fit_app, cuts_app, woes_app = _fit_on_frame(approved)

    # 4. parcelling：被拒样本按获批样本分数的 10 段分位推断标签
    print('parcelling 增强...')
    score_app = _score(approved, fit_app, cuts_app, woes_app)
    score_rej = _score(rejected, fit_app, cuts_app, woes_app)
    band_edges = np.quantile(score_app, np.linspace(0, 1, N_BANDS + 1))
    band_edges[0], band_edges[-1] = -np.inf, np.inf
    app_band = pd.cut(score_app, band_edges)
    rej_band = pd.cut(score_rej, band_edges)
    # 段内获批样本的好坏比；无获批样本的段退回全局好坏比
    # 注意：不能用绝对阈值 0.5——获批样本整体好客户率 ~97%，所有段都超过 0.5，
    # 绝对阈值会把被拒样本全部判好、模型坍缩。正确做法是相对阈值：
    # 段内好坏比与获批样本全局好坏比比较，低于全局的段判坏。
    global_good_rate = approved[common.TARGET].mean()
    band_good_rate = (pd.DataFrame({'band': app_band, 'y': approved[common.TARGET]})
                      .groupby('band', observed=False)['y'].mean()
                      .reindex(rej_band.categories).fillna(global_good_rate))
    inferred = pd.Series(
        [1 if band_good_rate[b] >= global_good_rate else 0 for b in rej_band],
        index=rejected.index)
    rejected_parcelling = rejected.copy()
    rejected_parcelling[common.TARGET] = inferred
    augmented_parcelling = pd.concat([approved, rejected_parcelling],
                                     ignore_index=True)
    print(f'parcelling 推断：被拒样本中 {int(inferred.sum())} 条判好、'
          f'{int((inferred == 0).sum())} 条判坏')

    # 5. hard-cutoff：被拒样本全部判坏
    rejected_hard = rejected.copy()
    rejected_hard[common.TARGET] = 0
    augmented_hard = pd.concat([approved, rejected_hard], ignore_index=True)

    # 6. 训练各方案模型并评估
    results = {}
    probs = {}

    prob = _score(test, fit_base, cuts_base, woes_base)
    results['baseline'] = common.eval_metrics(y_te, prob)
    probs['baseline'] = prob

    prob = _score(test, fit_app, cuts_app, woes_app)
    results['approved_only'] = common.eval_metrics(y_te, prob)
    probs['approved_only'] = prob

    print('训练 parcelling 增强模型...')
    fit_par, cuts_par, woes_par = _fit_on_frame(augmented_parcelling)
    prob = _score(test, fit_par, cuts_par, woes_par)
    results['parcelling'] = common.eval_metrics(y_te, prob)
    probs['parcelling'] = prob

    print('训练 hard-cutoff 增强模型...')
    fit_hard, cuts_hard, woes_hard = _fit_on_frame(augmented_hard)
    prob = _score(test, fit_hard, cuts_hard, woes_hard)
    results['hard_cutoff'] = common.eval_metrics(y_te, prob)
    probs['hard_cutoff'] = prob

    print('\n拒绝推断对比（测试集，train-only 拟合）：')
    print('-' * 70)
    print('%-16s %8s %8s %8s' % ('approach', 'AUC', 'KS', 'Gini'))
    for name in ('baseline', 'approved_only', 'parcelling', 'hard_cutoff'):
        m = results[name]
        print('%-16s %8.4f %8.4f %8.4f' % (name, m['auc'], m['ks'], m['gini']))
    print('-' * 70)

    result = {
        'split': '70/30 random_state=0',
        'fitting': 'train-only（实验口径；L2 正则化逻辑回归 C=1.0）',
        'simulation': {
            'approve_rate': APPROVE_RATE,
            'cutoff': round(float(cutoff), 4),
            'n_approved': int(len(approved)),
            'n_rejected': int(len(rejected)),
            'n_bands': N_BANDS,
            'n_inferred_good': int(inferred.sum()),
            'n_inferred_bad': int((inferred == 0).sum()),
        },
        'n_test': int(len(test)),
        'results': results,
    }
    with open(f'{OUT_DIR}/reject_inference.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'结果已保存至 {OUT_DIR}/reject_inference.json')

    # ROC 对比图
    plt.figure(figsize=(7, 6))
    colors = {'baseline': '#1f77b4', 'approved_only': '#ff7f0e',
              'parcelling': '#2ca02c', 'hard_cutoff': '#d62728'}
    for name in ('baseline', 'approved_only', 'parcelling', 'hard_cutoff'):
        fpr, tpr, _ = roc_curve(y_te, probs[name])
        plt.plot(fpr, tpr, color=colors[name], lw=1.6,
                 label=f"{name} (AUC={results[name]['auc']:.4f})")
    plt.plot([0, 1], [0, 1], 'r--', lw=1)
    plt.xlim([0, 1]); plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Reject Inference Simulation: ROC Comparison')
    plt.legend(loc='lower right')
    plt.savefig(f'{FIG_DIR}/reject_inference_roc.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'图形已保存至 {FIG_DIR}/reject_inference_roc.png')

    print('\n注意：本实验为教学模拟（Kaggle 数据无真实审批记录），'
          '真实业务中被拒样本标签不可观测，评估口径以获批样本为准；'
          '此处用完整测试集仅为演示对比。')
    print('实验 2 完成。')


if __name__ == '__main__':
    main()
