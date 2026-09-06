"""SQL 风控特征工程练习运行器

用法：python sql/run_queries.py
流程：建库（credit.db）→ 导入 data/cs-training.csv → 顺序执行 queries.sql 全部查询
      → 控制台打印摘要 → 结果写入 sql/results/query_*.csv

仅依赖 Python 标准库（sqlite3 + csv），无需安装任何第三方包。
"""

import csv
import os
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CSV_PATH = os.path.join(REPO, "data", "cs-training.csv")
DB_PATH = os.path.join(HERE, "credit.db")
QUERIES_PATH = os.path.join(HERE, "queries.sql")
RESULTS_DIR = os.path.join(HERE, "results")


def _to_float(s):
    return float(s) if s.strip() else None


def build_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE loan (
            id               INTEGER PRIMARY KEY,  -- 原始文件行序（从 0 起），便于与论文产出对照
            target           INTEGER,              -- SeriousDlqin2yrs：1=坏客户（90天+严重逾期），0=正常
            revolving_util   REAL,                 -- 循环贷款利用率
            age              INTEGER,              -- 年龄
            overdue_30_59    INTEGER,              -- 30-59 天逾期次数
            debt_ratio       REAL,                 -- 负债比率
            monthly_income   REAL,                 -- 月收入（约 19.8% 缺失）
            open_credit_lines INTEGER,             -- 信贷账户数量
            overdue_90       INTEGER,              -- 90 天及以上逾期次数
            real_estate_loans INTEGER,             -- 不动产贷款数量
            overdue_60_89    INTEGER,              -- 60-89 天逾期次数
            dependents       REAL                  -- 抚养人数（约 2.6% 缺失）
        )
        """
    )
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            rows.append(
                (
                    i,
                    int(float(row["SeriousDlqin2yrs"])),
                    _to_float(row["RevolvingUtilizationOfUnsecuredLines"]),
                    int(float(row["age"])),
                    int(float(row["NumberOfTime30-59DaysPastDueNotWorse"])),
                    _to_float(row["DebtRatio"]),
                    _to_float(row["MonthlyIncome"]),
                    int(float(row["NumberOfOpenCreditLinesAndLoans"])),
                    int(float(row["NumberOfTimes90DaysLate"])),
                    int(float(row["NumberRealEstateLoansOrLines"])),
                    int(float(row["NumberOfTime60-89DaysPastDueNotWorse"])),
                    _to_float(row["NumberOfDependents"]),
                )
            )
        cur.executemany(
            "INSERT INTO loan VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows
        )
    conn.commit()
    n = cur.execute("SELECT COUNT(*) FROM loan").fetchone()[0]
    conn.close()
    print(f"[init] 建库完成：{os.path.relpath(DB_PATH, REPO)}，导入 {n:,} 行")


def run_queries():
    with open(QUERIES_PATH, encoding="utf-8") as f:
        cleaned = "\n".join(
            line for line in f.read().splitlines() if not line.strip().startswith("--")
        )
    statements = [s.strip() for s in cleaned.split(";") if s.strip()]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    for idx, stmt in enumerate(statements, 1):
        cur = conn.execute(stmt)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        print(f"\n===== 查询 {idx}（{len(rows)} 行结果）=====")
        for r in rows[:8]:
            print("  " + " | ".join(str(r[c]) for c in cols))
        if len(rows) > 8:
            print(f"  ...（共 {len(rows)} 行）")

        out = os.path.join(RESULTS_DIR, f"query_{idx:02d}.csv")
        with open(out, "w", newline="", encoding="utf-8") as fw:
            w = csv.writer(fw)
            w.writerow(cols)
            w.writerows([tuple(r[c] for c in cols) for r in rows])
        print(f"  → 已保存 {os.path.relpath(out, REPO)}")
    conn.close()
    print("\n[完成] 全部查询执行完毕，结果见 sql/results/，解释见 sql/README.md")


if __name__ == "__main__":
    build_db()
    run_queries()
