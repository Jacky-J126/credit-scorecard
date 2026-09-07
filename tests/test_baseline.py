# -*- coding: utf-8 -*-
"""基线回归测试：锁定主流水线两种口径的核心指标，防止后续改动破坏可复现性。

主流水线（论文口径，在全量清洗数据上拟合 LR，再对测试集评估）：
    AUC 0.8478 / KS 0.5428 / Gini 0.6956，n_test = 43607
实验口径（仅训练集拟合，无测试集泄漏）：
    AUC 0.8478 / KS 0.5444 / Gini 0.6957

两种口径的差异原因见 README「口径说明」与 experiments/common.py 顶部注释。
"""
import pytest

from experiments import common


@pytest.fixture(scope='module')
def data():
    return common.load_clean_data()


def test_clean_bad_rate(data):
    """清洗后坏客户占比约 6.7%（标签已反转为 1=好客户）。"""
    assert 0.93 <= data[common.TARGET].mean() <= 0.94


def test_paper_caliber_full_fit(data):
    """主流水线第 8 节口径：全量清洗数据上拟合 LR，测试集评估。"""
    _, test = common.split_train_test(data)
    assert len(test) == 43607
    cuts, woes = common.build_woe_pipeline(data)
    fit = common.fit_logit(common.apply_woe(data, cuts, woes))
    prob = common.predict_logit(fit, common.apply_woe(test, cuts, woes))
    m = common.eval_metrics(test[common.TARGET], prob)
    assert m['auc'] == pytest.approx(0.8478, abs=1e-3)
    assert m['ks'] == pytest.approx(0.5428, abs=1e-3)
    assert m['gini'] == pytest.approx(0.6956, abs=1e-3)


def test_train_only_fit(data):
    """实验口径：仅训练集拟合 LR，测试集评估。"""
    train, test = common.split_train_test(data)
    cuts, woes = common.build_woe_pipeline(train)
    fit = common.fit_logit(common.apply_woe(train, cuts, woes))
    prob = common.predict_logit(fit, common.apply_woe(test, cuts, woes))
    m = common.eval_metrics(test[common.TARGET], prob)
    assert m['auc'] == pytest.approx(0.8478, abs=1e-3)
    assert m['ks'] == pytest.approx(0.5444, abs=1e-3)
    assert m['gini'] == pytest.approx(0.6957, abs=1e-3)
