-- ============================================================
-- 风控特征工程 SQL 练习（SQLite）
-- 数据：Kaggle「Give Me Some Credit」（15 万条个人消费贷款记录）
-- 目标变量 target = SeriousDlqin2yrs：1 = 未来 90 天以上严重逾期（坏客户），0 = 正常
-- 分箱口径与论文（credit_scoring.py）保持一致，结果可对照 data/ 与论文核验
-- 定位：求职准备的自建练习（SQL 技能的证据支撑），非实习/工作经历
-- ============================================================

-- 查询 1：缺失率统计（数据质量检查）
-- 考点：COUNT(列) 会跳过 NULL；AVG(CASE WHEN ...) 是求占比的常用写法
SELECT COUNT(*) AS n,
       ROUND(100.0 * AVG(CASE WHEN monthly_income IS NULL THEN 1.0 ELSE 0 END), 2) AS monthly_income_missing_pct,
       ROUND(100.0 * AVG(CASE WHEN dependents IS NULL THEN 1.0 ELSE 0 END), 2) AS dependents_missing_pct
FROM loan;

-- 查询 2：整体坏账率（样本结构与基准线）
SELECT COUNT(*) AS n,
       ROUND(100.0 * AVG(target), 2) AS bad_rate_pct
FROM loan;

-- 查询 3：年龄分层坏账率（CASE WHEN 分箱 + GROUP BY）
-- 考点：连续变量手工分箱做业务分层；ORDER BY 分箱下界保持顺序
SELECT CASE
         WHEN age < 25 THEN '<25'
         WHEN age < 35 THEN '25-34'
         WHEN age < 45 THEN '35-44'
         WHEN age < 55 THEN '45-54'
         WHEN age < 65 THEN '55-64'
         ELSE '>=65'
       END AS age_band,
       COUNT(*) AS n,
       ROUND(100.0 * AVG(target), 2) AS bad_rate_pct
FROM loan
GROUP BY age_band
ORDER BY MIN(age);

-- 查询 4：循环贷款利用率十分位分箱（NTILE 窗口函数等频分箱）
-- 考点：窗口函数做等频分箱；坏账率随利用率单调上升（风险单调性，与论文 WOE 结论一致）
WITH deciles AS (
  SELECT target,
         NTILE(10) OVER (ORDER BY revolving_util) AS util_decile
  FROM loan
)
SELECT util_decile,
       COUNT(*) AS n,
       ROUND(100.0 * AVG(target), 2) AS bad_rate_pct
FROM deciles
GROUP BY util_decile
ORDER BY util_decile;

-- 查询 5：WOE/IV 计算（90 天逾期次数，分箱边界与论文 3.4 节一致）
-- 考点：CTE + CROSS JOIN + LN；风控特征工程核心公式（WOE/IV）的纯 SQL 实现
WITH total AS (
  SELECT SUM(1 - target) AS good_all, SUM(target) AS bad_all
  FROM loan
),
binned AS (
  SELECT CASE
           WHEN overdue_90 <= 0 THEN '0'
           WHEN overdue_90 <= 1 THEN '1'
           WHEN overdue_90 <= 3 THEN '2-3'
           WHEN overdue_90 <= 5 THEN '4-5'
           ELSE '>5'
         END AS bin,
         SUM(1 - target) AS good_cnt,
         SUM(target) AS bad_cnt
  FROM loan
  GROUP BY bin
)
SELECT b.bin,
       b.good_cnt,
       b.bad_cnt,
       ROUND(100.0 * b.bad_cnt * 1.0 / (b.bad_cnt + b.good_cnt), 2) AS bad_rate_pct,
       ROUND(LN((b.good_cnt * 1.0 / t.good_all) / (b.bad_cnt * 1.0 / t.bad_all)), 3) AS woe,
       ROUND(((b.good_cnt * 1.0 / t.good_all) - (b.bad_cnt * 1.0 / t.bad_all))
             * LN((b.good_cnt * 1.0 / t.good_all) / (b.bad_cnt * 1.0 / t.bad_all)), 4) AS iv_contribution
FROM binned b
CROSS JOIN total t
ORDER BY b.bin;

-- 查询 6：逾期行为交叉组合（多维交叉坏账率透视）
-- 考点：多个 CASE WHEN 做组合分组；交叉表的业务解读
SELECT CASE WHEN overdue_90 >= 1 THEN '曾90天逾期' ELSE '无90天逾期' END AS ever_90,
       CASE WHEN overdue_60_89 >= 1 THEN '曾60-89逾期' ELSE '无60-89逾期' END AS ever_60_89,
       COUNT(*) AS n,
       ROUND(100.0 * AVG(target), 2) AS bad_rate_pct
FROM loan
GROUP BY ever_90, ever_60_89
ORDER BY bad_rate_pct DESC;

-- 查询 7：逾期总次数特征构造（子查询内联派生字段 + HAVING 过滤稀疏桶）
-- 考点：派生字段构造特征；HAVING 对聚合结果过滤（WHERE 做不到）
SELECT total_delq,
       COUNT(*) AS n,
       ROUND(100.0 * AVG(target), 2) AS bad_rate_pct
FROM (
  SELECT target,
         overdue_30_59 + overdue_60_89 + overdue_90 AS total_delq
  FROM loan
) t
GROUP BY total_delq
HAVING COUNT(*) >= 100
ORDER BY total_delq;

-- 查询 8：收入五分位坏账率（收入为什么不是强风险因子？）
-- 考点：窗口函数 NTILE 分位分层；结论印证论文中 MonthlyIncome 因 IV<0.02 被剔除
SELECT inc_quintile,
       COUNT(*) AS n,
       ROUND(AVG(monthly_income), 0) AS avg_income,
       ROUND(100.0 * AVG(target), 2) AS bad_rate_pct
FROM (
  SELECT target,
         monthly_income,
         NTILE(5) OVER (ORDER BY monthly_income) AS inc_quintile
  FROM loan
  WHERE monthly_income IS NOT NULL
) t
GROUP BY inc_quintile
ORDER BY inc_quintile;

-- 查询 9：客户收入 vs 同龄人平均水平（JOIN 预聚合表）
-- 考点：先聚合再 JOIN 的特征工程手法（比大表自连接快得多）；CTE 可读性
WITH age_avg AS (
  SELECT age, AVG(monthly_income) AS avg_inc
  FROM loan
  WHERE monthly_income IS NOT NULL
  GROUP BY age
)
SELECT l.id,
       l.age,
       l.monthly_income,
       ROUND(a.avg_inc, 0) AS age_avg_income
FROM loan l
JOIN age_avg a ON l.age = a.age
WHERE l.monthly_income IS NOT NULL
ORDER BY l.id
LIMIT 5;

-- 查询 10：策略规则模拟（高利用率 + 多次 30-59 逾期）
-- 考点：把风控策略规则翻译成 WHERE 条件；命中率与坏账率决定规则可用性
SELECT COUNT(*) AS hit_n,
       ROUND(100.0 * COUNT(*) * 1.0 / (SELECT COUNT(*) FROM loan), 2) AS hit_rate_pct,
       ROUND(100.0 * AVG(target), 2) AS bad_rate_pct
FROM loan
WHERE revolving_util >= 0.9
  AND overdue_30_59 >= 2;
