# ============================================================
# 金融信用评分卡模型

import pandas as pd
import numpy as np
import warnings
import matplotlib
matplotlib.use('Agg')  # 无界面环境（服务器/CI）下静默生成图形，不弹窗
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

warnings.filterwarnings('ignore')

# 设置中文字体（如果系统支持）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. 数据加载
# ============================================================
data = pd.read_csv('data/cs-training.csv')
data.head()

# 数据集缺失和分布探索
data.describe().to_csv('data/DataDescribe.csv')
data.describe()

# ============================================================
# 2. 使用随机森林填补缺失值
# ============================================================
def set_missing(df):
    # 把已有的数值特征取出
    process_df = df.iloc[:, [5, 0, 1, 2, 3, 4, 6, 7, 8, 9]]
    # 分成已知特征和未知特征两部分
    known = process_df[process_df.MonthlyIncome.notnull()].to_numpy()
    unknown = process_df[process_df.MonthlyIncome.isnull()].to_numpy()
    # X为特征属性值
    X = known[:, 1:]
    # y为结果标签值
    y = known[:, 0]
    # fit到RandomForestRegressor中
    rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
    rfr.fit(X, y)
    # 用得到的模型进行未知特征值预测
    predicted = rfr.predict(unknown[:, 1:]).round(0)
    print(predicted)
    # 用得到的预测结果填补原缺失数据
    df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
    return df  # 修复：正确缩进到函数内部


data = set_missing(data)  # 用随机森林填补比较多的缺失值
data = data.dropna()  # 删除比较少的缺失值
data = data.drop_duplicates()  # 删除重复项
data.to_csv('data/MissingData.csv', index=False)

# 异常值处理
data = data[data['age'] > 0]  # 去掉年龄为0的异常值
data = data[data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]  # 剔除异常值
# 反转SeriousDlqin2yrs（好客户为1，坏客户为0）
data['SeriousDlqin2yrs'] = 1 - data['SeriousDlqin2yrs']
# 修复：dropna/drop_duplicates/布尔过滤后index不连续，reset避免后续pd.Series对齐出错
data = data.reset_index(drop=True)

# ============================================================
# 3. 划分训练集和测试集
# ============================================================
Y = data['SeriousDlqin2yrs']
X = data.iloc[:, 1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
train = pd.concat([Y_train, X_train], axis=1)
test = pd.concat([Y_test, X_test], axis=1)
# 修复：train_test_split 后 index 也不连续，需重置
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
clasTest = test.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
train.to_csv('data/TrainData.csv', index=False)
test.to_csv('data/TestData.csv', index=False)

# ============================================================
# 4. 相关性热力图
# ============================================================
corr = data.corr()
xticks = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
yticks = list(corr.index)
fig = plt.figure()
ax1 = fig.add_subplot(1, 1, 1)
sns.heatmap(corr, annot=True, cmap='rainbow', ax=ax1,
            annot_kws={'size': 9, 'weight': 'bold', 'color': 'blue'})
ax1.set_xticklabels(xticks, rotation=0, fontsize=10)
ax1.set_yticklabels(yticks, rotation=0, fontsize=10)
plt.savefig('figures/corr_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 5. 定义自动分箱函数（合并原两次定义，保留含IV计算的版本）
# ============================================================
def mono_bin(Y, X, n=20):
    r = 0
    good = Y.sum()
    bad = Y.count() - good
    # 保存原始n，用于后续生成cut点
    n_orig = n
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"X": X, "Y": Y, "Bucket": pd.qcut(X, n, duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index=True)
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
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d4 = d3.sort_values(by='min').reset_index(drop=True)
    print("=" * 60)
    print(d4)
    # 修复：使用实际分箱数量构建cut点，保证与woe长度匹配
    actual_bins = len(d4)
    cut = [float('-inf')]
    for i in range(1, actual_bins):
        qua = X.quantile(i / actual_bins)
        cut.append(round(qua, 4))
    cut.append(float('inf'))
    woe = list(d4['woe'].round(3))
    return d4, iv, cut, woe


# ============================================================
# 6. 计算每个特征的WOE、IV和分箱切点
# ============================================================
Y_train_series = train['SeriousDlqin2yrs']

# x1: RevolvingUtilizationOfUnsecuredLines（自动分箱）
df1, ivx1, cutx1, woex1 = mono_bin(Y_train_series, train['RevolvingUtilizationOfUnsecuredLines'])
# x2: age（自动分箱）
df2, ivx2, cutx2, woex2 = mono_bin(Y_train_series, train['age'])
# x3: NumberOfTime30-59DaysPastDueNotWorse（手动分箱，修复：'ninf'/'pinf' → float）
cutx3 = [float('-inf'), 0, 1, 3, 5, float('inf')]
# x4: DebtRatio（自动分箱）
df4, ivx4, cutx4, woex4 = mono_bin(Y_train_series, train['DebtRatio'])
# x5: MonthlyIncome（自动分箱）
df5, ivx5, cutx5, woex5 = mono_bin(Y_train_series, train['MonthlyIncome'])
# x6: NumberOfOpenCreditLinesAndLoans（手动分箱）
cutx6 = [float('-inf'), 1, 2, 3, 5, float('inf')]
# x7: NumberOfTimes90DaysLate（手动分箱）
cutx7 = [float('-inf'), 0, 1, 3, 5, float('inf')]
# x8: NumberRealEstateLoansOrLines（手动分箱）
cutx8 = [float('-inf'), 0, 1, 2, 3, float('inf')]
# x9: NumberOfTime60-89DaysPastDueNotWorse（手动分箱）
cutx9 = [float('-inf'), 0, 1, 3, float('inf')]
# x10: NumberOfDependents（手动分箱）
cutx10 = [float('-inf'), 0, 1, 2, 3, 5, float('inf')]


# 为手动分箱的特征计算WOE和IV
def calc_woe_manual(Y, X, cut):
    """为手动分箱的特征计算WOE和IV"""
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
    d3['goodattribute'] = d3['sum'] / good
    d3['badattribute'] = (d3['total'] - d3['sum']) / bad
    iv = ((d3['goodattribute'] - d3['badattribute']) * d3['woe']).sum()
    d3 = d3.reset_index(drop=True)
    print("=" * 60)
    print(d3)
    woe = list(d3['woe'].round(3))
    return d3, iv, woe


df3, ivx3, woex3 = calc_woe_manual(Y_train_series, train['NumberOfTime30-59DaysPastDueNotWorse'], cutx3)
df6, ivx6, woex6 = calc_woe_manual(Y_train_series, train['NumberOfOpenCreditLinesAndLoans'], cutx6)
df7, ivx7, woex7 = calc_woe_manual(Y_train_series, train['NumberOfTimes90DaysLate'], cutx7)
df8, ivx8, woex8 = calc_woe_manual(Y_train_series, train['NumberRealEstateLoansOrLines'], cutx8)
df9, ivx9, woex9 = calc_woe_manual(Y_train_series, train['NumberOfTime60-89DaysPastDueNotWorse'], cutx9)
df10, ivx10, woex10 = calc_woe_manual(Y_train_series, train['NumberOfDependents'], cutx10)

# IV柱状图
ivlist = [ivx1, ivx2, ivx3, ivx4, ivx5, ivx6, ivx7, ivx8, ivx9, ivx10]
index = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10']
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
x = np.arange(len(index)) + 1
ax1.bar(x, ivlist, width=0.4)
ax1.set_xticks(x)
ax1.set_xticklabels(index, rotation=0, fontsize=12)
ax1.set_ylabel('IV(Information Value)', fontsize=14)
for a, b in zip(x, ivlist):
    plt.text(a, b + 0.01, '%.4f' % b, ha='center', va='bottom', fontsize=10)
plt.savefig('figures/iv_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 7. WOE替换函数
# ============================================================
def replace_woe(series, cut, woe):
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


# 对全部数据（data）进行WOE替换
data['RevolvingUtilizationOfUnsecuredLines'] = pd.Series(
    replace_woe(data['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
data['age'] = pd.Series(replace_woe(data['age'], cutx2, woex2))
data['NumberOfTime30-59DaysPastDueNotWorse'] = pd.Series(
    replace_woe(data['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
data['DebtRatio'] = pd.Series(replace_woe(data['DebtRatio'], cutx4, woex4))
data['MonthlyIncome'] = pd.Series(replace_woe(data['MonthlyIncome'], cutx5, woex5))
data['NumberOfOpenCreditLinesAndLoans'] = pd.Series(
    replace_woe(data['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
data['NumberOfTimes90DaysLate'] = pd.Series(
    replace_woe(data['NumberOfTimes90DaysLate'], cutx7, woex7))
data['NumberRealEstateLoansOrLines'] = pd.Series(
    replace_woe(data['NumberRealEstateLoansOrLines'], cutx8, woex8))
data['NumberOfTime60-89DaysPastDueNotWorse'] = pd.Series(
    replace_woe(data['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
data['NumberOfDependents'] = pd.Series(
    replace_woe(data['NumberOfDependents'], cutx10, woex10))
data.to_csv('data/WoeData.csv', index=False)

# ============================================================
# 8. 逻辑回归建模
# ============================================================
data = pd.read_csv('data/WoeData.csv')
# 因变量
Y = data['SeriousDlqin2yrs']
# 自变量，剔除对模型不明显的变量
X = data.drop(['SeriousDlqin2yrs', 'DebtRatio', 'MonthlyIncome',
               'NumberOfOpenCreditLinesAndLoans', 'NumberRealEstateLoansOrLines',
               'NumberOfDependents'], axis=1)
X1 = sm.add_constant(X)
logit = sm.Logit(Y, X1)
result = logit.fit()
print(result.summary())

coe = result.params
print("模型系数 coe:", coe.values)

# ============================================================
# 9. 模型评估 - ROC曲线
# ============================================================
# 此处对test数据同样做WOE替换后再评估
test_woe = test.copy()
test_woe['RevolvingUtilizationOfUnsecuredLines'] = pd.Series(
    replace_woe(test_woe['RevolvingUtilizationOfUnsecuredLines'], cutx1, woex1))
test_woe['age'] = pd.Series(replace_woe(test_woe['age'], cutx2, woex2))
test_woe['NumberOfTime30-59DaysPastDueNotWorse'] = pd.Series(
    replace_woe(test_woe['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, woex3))
test_woe['DebtRatio'] = pd.Series(replace_woe(test_woe['DebtRatio'], cutx4, woex4))
test_woe['MonthlyIncome'] = pd.Series(replace_woe(test_woe['MonthlyIncome'], cutx5, woex5))
test_woe['NumberOfOpenCreditLinesAndLoans'] = pd.Series(
    replace_woe(test_woe['NumberOfOpenCreditLinesAndLoans'], cutx6, woex6))
test_woe['NumberOfTimes90DaysLate'] = pd.Series(
    replace_woe(test_woe['NumberOfTimes90DaysLate'], cutx7, woex7))
test_woe['NumberRealEstateLoansOrLines'] = pd.Series(
    replace_woe(test_woe['NumberRealEstateLoansOrLines'], cutx8, woex8))
test_woe['NumberOfTime60-89DaysPastDueNotWorse'] = pd.Series(
    replace_woe(test_woe['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, woex9))
test_woe['NumberOfDependents'] = pd.Series(
    replace_woe(test_woe['NumberOfDependents'], cutx10, woex10))

Y_test = test_woe['SeriousDlqin2yrs']
X_test_data = test_woe.drop(['SeriousDlqin2yrs', 'DebtRatio', 'MonthlyIncome',
                              'NumberOfOpenCreditLinesAndLoans',
                              'NumberRealEstateLoansOrLines',
                              'NumberOfDependents'], axis=1)
X3 = sm.add_constant(X_test_data)
resu = result.predict(X3)
fpr, tpr, threshold = roc_curve(Y_test, resu)
rocauc = auc(fpr, tpr)
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % rocauc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('真正率')
plt.xlabel('假正率')
plt.savefig('figures/roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================
# 10. 评分卡生成
# ============================================================
# 假设取600分为基础分值，PDO为20（每20分好坏比翻一番），好客户取20分
p = 20 / math.log(2)
q = 600 - 20 * math.log(20) / math.log(2)
# 修复：coe 现在已从 result.params 提取
baseScore = round(q + p * coe[0], 0)


def get_score(coe_val, woe, factor):
    scores = []
    for w in woe:
        score = round(coe_val * w * factor, 0)
        scores.append(score)
    return scores


# 保留变量：const, RevolvingUtilization, age, 30-59Days, 90DaysLate, 60-89Days
# 对应 coe[0]~coe[5]
x1_score = get_score(coe.iloc[1], woex1, p)   # RevolvingUtilizationOfUnsecuredLines
x2_score = get_score(coe.iloc[2], woex2, p)   # age
x3_score = get_score(coe.iloc[3], woex3, p)   # NumberOfTime30-59DaysPastDueNotWorse
x7_score = get_score(coe.iloc[4], woex7, p)   # NumberOfTimes90DaysLate
x9_score = get_score(coe.iloc[5], woex9, p)   # NumberOfTime60-89DaysPastDueNotWorse


# 根据变量计算分数的函数
def compute_score(series, cut, score):
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
        result.append(score[m])
        i += 1
    return result


test1 = pd.read_csv('data/TestData.csv')
test1['BaseScore'] = pd.Series(np.zeros(len(test1))) + baseScore
test1['x1'] = pd.Series(compute_score(test1['RevolvingUtilizationOfUnsecuredLines'], cutx1, x1_score))
test1['x2'] = pd.Series(compute_score(test1['age'], cutx2, x2_score))
test1['x3'] = pd.Series(compute_score(test1['NumberOfTime30-59DaysPastDueNotWorse'], cutx3, x3_score))
test1['x7'] = pd.Series(compute_score(test1['NumberOfTimes90DaysLate'], cutx7, x7_score))
test1['x9'] = pd.Series(compute_score(test1['NumberOfTime60-89DaysPastDueNotWorse'], cutx9, x9_score))
# 修复：基础的 baseScore 需单独加（原代码最后用 baseScore 而非 BaseScore）
test1['Score'] = test1['x1'] + test1['x2'] + test1['x3'] + test1['x7'] + test1['x9'] + test1['BaseScore']
test1.to_csv('data/ScoreData.csv', index=False)
print("评分卡生成完毕！")
print(test1[['BaseScore', 'x1', 'x2', 'x3', 'x7', 'x9', 'Score']].head())

# ============================================================
# 11. 补充模型指标（开源整理时新增：KS / Gini / 评分分布图）
# ============================================================
import json

ks = float(max(tpr - fpr))
gini = float(2 * rocauc - 1)
print("=" * 60)
print("测试集指标：AUC = %.4f | KS = %.4f | Gini = %.4f" % (rocauc, ks, gini))
print("测试集样本量：%d（好客户 %d / 坏客户 %d）" % (
    len(Y_test), int(Y_test.sum()), int((Y_test == 0).sum())))

metrics = {
    "n_test": int(len(Y_test)),
    "n_good": int(Y_test.sum()),
    "n_bad": int((Y_test == 0).sum()),
    "auc": round(rocauc, 4),
    "ks": round(ks, 4),
    "gini": round(gini, 4),
}
with open('data/model_metrics.json', 'w', encoding='utf-8') as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("指标已保存至 data/model_metrics.json")

# 评分分布图
plt.figure(figsize=(8, 5))
plt.hist(test1['Score'], bins=50, color='steelblue', edgecolor='white')
plt.xlabel('信用评分')
plt.ylabel('样本数')
plt.title('测试集信用评分分布（基准分 600，PDO 20）')
plt.savefig('figures/score_distribution.png', dpi=150, bbox_inches='tight')
print("评分分布图已保存至 figures/score_distribution.png")
