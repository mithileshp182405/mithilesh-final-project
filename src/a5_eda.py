"""Part A, task 7 -- pandas cross-check of Q2 and exploratory statistics of completed order_amount."""
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd
from common import DB_PATH, CLEAN, Report

rep = Report("a5_eda")
pd.set_option("display.width", 160)

orders = pd.read_csv(CLEAN / "orders.csv")
customers = pd.read_csv(CLEAN / "customers.csv")
products = pd.read_csv(CLEAN / "products.csv")
done = (orders[orders["status"] == "completed"]
        .merge(customers[["customer_id", "city"]], on="customer_id")
        .merge(products[["product_id", "category"]], on="product_id"))

# ------------------------------------------------------------------ (a) cross-check of Q2 and per-city aggregation
rep("(a) PIVOT TABLE with margins=True (rebuilds SQL Q2)")
pivot = pd.pivot_table(done, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True, margins_name="Total")
rep(pivot.round(2).to_string())
with closing(sqlite3.connect(DB_PATH)) as con:
    sql_total = pd.read_sql_query("SELECT ROUND(SUM(order_amount), 2) AS t FROM orders WHERE status='completed'", con)["t"][0]
grand = round(pivot.loc["Total", "Total"], 2)
rep(f"pivot grand total = INR {grand:,.2f} | SQL total = INR {sql_total:,.2f} | equal: {grand == sql_total}")
assert grand == sql_total == 2998668.80
cell = pivot.drop(index="Total", columns="Total").stack().sort_values(ascending=False)
rep(f"largest cell: {cell.index[0][0]} / {cell.index[0][1]} = INR {cell.iloc[0]:,.2f}")

rep("\nper-city named aggregation of completed order_amount")
per_city = done.groupby("city").agg(total_inr=("order_amount", "sum"), mean_inr=("order_amount", "mean"),
                                    n_orders=("order_amount", "count")).round(2)
rep(per_city.to_string())
assert per_city.loc["Bengaluru"].tolist() == [523643.40, 276.47, 1894]
assert per_city.loc["Mumbai"].tolist() == [479058.50, 297.00, 1613]

# ------------------------------------------------------------------ (b) distribution of completed order_amount
amt = done["order_amount"]
q1, med, q3 = amt.quantile([.25, .5, .75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_hi, n_lo = int((amt > hi).sum()), int((amt < lo).sum())
rep("\n(b) DISTRIBUTION of completed order_amount (INR)")
rep(f"n = {len(amt):,} | mean {amt.mean():.2f} | median {med:.2f} | std {amt.std():.2f} | min {amt.min():.2f} | max {amt.max():.2f}")
rep(f"skewness (pandas, sample) = {amt.skew():.2f}")
rep(f"Q1 {q1:.2f} | Q3 {q3:.2f} | IQR {iqr:.2f} | lower fence {lo:.2f} | upper fence {hi:.2f}")
rep(f"outliers above upper fence: {n_hi} | below lower fence: {n_lo}")
assert (round(amt.mean(), 2), med, round(amt.skew(), 2), q1, q3, round(hi, 2), n_hi, n_lo) == (269.91, 135.0, 4.61, 64.8, 297.0, 645.3, 897, 0)

corr_cols = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]
corr = done[corr_cols].corr()
rep("\ncorrelation matrix (completed orders)")
rep(corr.round(3).to_string())
pairs = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
strongest = pairs.abs().sort_values(ascending=False)
rep("pairs by absolute correlation: " + "; ".join(f"{a} & {b} = {pairs[(a, b)]:+.3f}" for (a, b) in strongest.index[:3]))
assert strongest.index[0] == ("order_amount", "discount_pct") and round(pairs[strongest.index[0]], 3) == -0.095

share_hi = n_hi / len(amt) * 100
elec_share = done.loc[done["order_amount"] > hi, "category"].value_counts(normalize=True).mul(100).round(1).to_dict()
rep(f"\noutlier share of orders: {share_hi:.1f}% | category mix of outliers (%): {elec_share}")

ANSWER = (
    f"What is it about? Quick-commerce orders from six Indian cities, April-September 2025: {len(amt):,} completed orders worth "
    f"INR {amt.sum():,.2f}. "
    f"Distribution: order_amount is strongly right-skewed (mean {amt.mean():.2f} against median {med:.0f}, skewness {amt.skew():.2f}); "
    f"half of all orders sit between INR {q1:.0f} and INR {q3:.0f}. "
    f"Outliers: {n_hi} orders ({share_hi:.1f}%) exceed the upper fence of INR {hi:.0f} and none fall below the lower one; "
    f"they are mostly electronics purchases, genuine high-value baskets rather than errors, so I keep them. "
    f"Relationships: all correlations are weak; the largest in absolute value is order_amount with discount_pct "
    f"({pairs[strongest.index[0]]:+.3f}), just ahead of order_amount with quantity ({pairs[('order_amount', 'quantity')]:+.3f}). "
    f"Correlation of this size does not establish causation."
)
words = len(ANSWER.split())
rep(f"\nANSWER ({words} words, limit 200):\n{ANSWER}")
assert words <= 200
rep.save()
