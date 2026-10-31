"""Part A, task 6 -- run the seven queries of sql/queries.sql with pd.read_sql_query and check the reference results."""
import re
import sqlite3
from contextlib import closing
import pandas as pd
from common import DB_PATH, SQL_FILE, Report

rep = Report("a4_sql")

text = SQL_FILE.read_text(encoding="utf-8")
parts = re.split(r"^-- name:\s*(\S+)\s*$", text, flags=re.M)
queries = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}
assert len(queries) == 7, f"expected 7 queries, found {len(queries)}"

with closing(sqlite3.connect(DB_PATH)) as con:
    res = {name: pd.read_sql_query(sql, con) for name, sql in queries.items()}
    avg_price = pd.read_sql_query("SELECT AVG(price_inr) AS a FROM products", con)["a"][0]

pd.set_option("display.width", 160)
Q1, Q2, Q3, Q4, Q5, Q6, Q7 = res.values()
checks = []

rep("Q1 WHERE then HAVING -- customers whose completed orders above INR 500 total more than INR 8,000")
rep(f"   rows = {len(Q1)} (reference 30)"); rep(Q1.head(5).to_string(index=False)); checks.append(len(Q1) == 30)

rep("\nQ2 three-table join -- revenue by city and category")
rep(f"   rows = {len(Q2)} (reference 30); total = INR {Q2['revenue_inr'].sum():,.2f} (reference 2,998,668.80)")
top = Q2.iloc[0]
rep(f"   largest cell: {top['city']} / {top['category']} = INR {top['revenue_inr']:,.2f} (reference Bengaluru / Electronics 233,909.45)")
checks += [len(Q2) == 30, round(Q2["revenue_inr"].sum(), 2) == 2998668.80,
           (top["city"], top["category"], round(top["revenue_inr"], 2)) == ("Bengaluru", "Electronics", 233909.45)]

rep("\nQ3 scalar subquery -- products priced above the catalogue average")
rep(f"   catalogue average price = INR {avg_price:,.2f}; products above it = {len(Q3)} (reference 11)")
rep(Q3.to_string(index=False)); checks.append(len(Q3) == 11)

rep("\nQ4 LEFT JOIN -- customers with no completed order")
rep(f"   rows = {len(Q4)} (reference 25)"); checks.append(len(Q4) == 25)

rep("\nQ5 CTE -- customers whose total completed spend is above the average customer total")
rep(f"   rows = {len(Q5)} (reference 550); average = INR {Q5['avg_total_spend_inr'].iloc[0]:,.2f} (reference 2,552.06)")
checks += [len(Q5) == 550, Q5["avg_total_spend_inr"].iloc[0] == 2552.06]

rep("\nQ6 ranking window functions -- top customer per city")
tops = Q6[Q6["spend_row_number"] == 1][["city", "customer_id", "total_spend"]]
rep(tops.to_string(index=False))
expected_tops = {"Bengaluru": ("C0082", 12856.20), "Chennai": ("C0930", 10151.40), "Delhi": ("C0291", 13742.60),
                 "Hyderabad": ("C0743", 12730.05), "Mumbai": ("C0693", 14766.55), "Pune": ("C0426", 10377.10)}
got_tops = {r.city: (r.customer_id, round(r.total_spend, 2)) for r in tops.itertuples()}
n_diff = int((Q6["spend_rank"] != Q6["spend_row_number"]).sum())
rep(f"   rows where RANK() differs from ROW_NUMBER(): {n_diff} (reference 8)")
checks += [got_tops == expected_tops, n_diff == 8]
ties = Q6[Q6.duplicated(["city", "total_spend"], keep=False)].sort_values(["city", "spend_rank"])
rep("   tied customers (same city, same spend):"); rep(ties.to_string(index=False))

rep("\nQ7 running total -- daily completed revenue")
rep(f"   daily rows = {len(Q7)} (reference 183); last cumulative = INR {Q7['cumulative_revenue_inr'].iloc[-1]:,.2f} (reference 2,998,668.80)")
rep(Q7.tail(3).to_string(index=False))
checks += [len(Q7) == 183, round(Q7["cumulative_revenue_inr"].iloc[-1], 2) == 2998668.80]

rep()
rep(f"SQL reference checks passed: {sum(checks)} of {len(checks)}")
rep.save()
assert all(checks), "one or more SQL results differ from the brief's references"
