"""Part A, tasks 3-5 -- clean customers and orders with explicit rules, then store CSV + SQLite.

Every rule logs how many rows/values it changed; the log is saved to reports/a2_clean.txt and quoted in the README.
"""
import sqlite3
from contextlib import closing
import numpy as np
import pandas as pd
from common import RAW, CLEAN, DB_PATH, Report

rep = Report("a2_clean")

# ------------------------------------------------------------------ load
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")
orders = pd.read_csv(RAW / "orders.csv")

# =================================================================== CUSTOMERS
rep("CLEAN CUSTOMERS")
raw_city = customers["city"].copy()
customers["city"] = customers["city"].str.strip().str.title()
rep(f"C1 city: strip + title case -> {int((raw_city != customers['city']).sum())} of {len(customers)} values changed; "
    f"{customers['city'].nunique()} distinct cities remain (was {raw_city.nunique()})")

GENDER_MAP = {"m": "male", "male": "male", "f": "female", "female": "female"}
raw_gender = customers["gender"].copy()
customers["gender"] = customers["gender"].str.strip().str.lower().map(GENDER_MAP).fillna("unknown")
rep(f"C2 gender: mapped to male/female/unknown -> {int((raw_gender.fillna('') != customers['gender']).sum())} values changed; "
    f"blank -> unknown: {int(raw_gender.isna().sum())}")

bad_age = (customers["age"] < 10) | (customers["age"] > 100)
customers.loc[bad_age, "age"] = np.nan
n_missing_before_fill = int(customers["age"].isna().sum())
customers["age"] = customers["age"].fillna(customers.groupby("city")["age"].transform("median"))
rep(f"C3 age: {int(bad_age.sum())} impossible values (outside 10-100) set to missing; "
    f"{n_missing_before_fill} missing ages filled with the city median (groupby + transform); "
    f"{int(customers['age'].isna().sum())} missing remain")

customers["signup_date"] = pd.to_datetime(customers["signup_date"], format="%Y-%m-%d")
rep(f"C4 signup_date: converted to datetime64 for {len(customers)} rows "
    f"({customers['signup_date'].min().date()} .. {customers['signup_date'].max().date()})")
rep()

# =================================================================== ORDERS
rep("CLEAN ORDERS")
rep(f"O0 raw rows: {len(orders):,}")

orders["order_ts"] = pd.to_datetime(orders["order_ts"], format="%Y-%m-%d %H:%M:%S")
rep(f"O1 order_ts: converted to datetime64 for {len(orders):,} rows (range {orders['order_ts'].min()} .. {orders['order_ts'].max()})")

before = len(orders)
orders = orders.drop_duplicates().copy()
rep(f"O2 exact duplicates: {before - len(orders)} rows removed -> {len(orders):,} rows")

before = len(orders)
n_pending_before = int((orders["status"] == "pending").sum())
orders = (orders.sort_values(["order_id", "order_ts"], kind="mergesort")
                .drop_duplicates(subset="order_id", keep="last")
                .sort_values("order_id", kind="mergesort")
                .reset_index(drop=True))
rep(f"O3 latest record per order_id: {before - len(orders)} superseded rows removed "
    f"({n_pending_before} pending rows existed before, {int((orders['status'] == 'pending').sum())} remain) -> {len(orders):,} rows, "
    f"{orders['order_id'].nunique():,} distinct order_id")

orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
n_amount_missing = int(orders["order_amount"].isna().sum())
price = orders["product_id"].map(products.set_index("product_id")["price_inr"])
recomputed = (price * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(recomputed)
rep(f"O4 order_amount: unparseable ('N/A', 'error') turned to missing = {n_amount_missing} values; "
    f"filled from price_inr x quantity x (1 - discount_pct/100), rounded to 2 decimals; "
    f"{int(orders['order_amount'].isna().sum())} missing remain")

orders["category"] = orders["product_id"].map(products.set_index("product_id")["category"])
n_negative = int((orders["delivery_minutes"] < 0).sum())
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
completed = orders["status"] == "completed"
cat_median = orders["delivery_minutes"].where(completed).groupby(orders["category"]).transform("median")
n_fill = int((completed & orders["delivery_minutes"].isna()).sum())
orders["delivery_minutes"] = orders["delivery_minutes"].fillna(cat_median.where(completed))
rep(f"O5 delivery_minutes: {n_negative} negative values set to missing; "
    f"{n_fill} missing values of completed orders filled with the median of their product category among completed orders; "
    f"cancelled orders left missing on purpose")
rep("   category medians used: " + ", ".join(f"{k}={v:g}" for k, v in
    orders[completed].groupby("category")["delivery_minutes"].median().items()))
orders = orders.drop(columns="category")

# ------------------------------------------------------------------ checkpoints printed for the log
rep()
completed = orders["status"] == "completed"
rep(f"rows {len(orders):,} | completed {int(completed.sum()):,} | cancelled {int((orders['status'] == 'cancelled').sum()):,} | "
    f"pending {int((orders['status'] == 'pending').sum())}")
rep(f"missing order_amount {int(orders['order_amount'].isna().sum())} | missing delivery (completed) "
    f"{int(orders.loc[completed, 'delivery_minutes'].isna().sum())} | missing delivery (cancelled) "
    f"{int(orders.loc[~completed, 'delivery_minutes'].isna().sum()):,}")
rep(f"completed revenue INR {orders.loc[completed, 'order_amount'].sum():,.2f}")
rep(f"customers by city: " + ", ".join(f"{k} {v}" for k, v in customers['city'].value_counts().items()))
rep(f"customers by gender: " + ", ".join(f"{k} {v}" for k, v in customers['gender'].value_counts().items()))
rep(f"age: missing {int(customers['age'].isna().sum())}, min {customers['age'].min():g}, max {customers['age'].max():g}, "
    f"non-integer ages {int((customers['age'] % 1 != 0).sum())}")

# =================================================================== STORE
customers_out = customers.copy()
customers_out["signup_date"] = customers_out["signup_date"].dt.strftime("%Y-%m-%d")
orders_out = orders.copy()
orders_out["order_ts"] = orders_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")   # text, as the brief requires

customers_out.to_csv(CLEAN / "customers.csv", index=False)
products.to_csv(CLEAN / "products.csv", index=False)
orders_out.to_csv(CLEAN / "orders.csv", index=False)

DDL = """
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY,
    city        TEXT NOT NULL,
    signup_date TEXT NOT NULL,
    age         REAL,
    gender      TEXT NOT NULL
);
CREATE TABLE products (
    product_id   TEXT PRIMARY KEY,
    product_name TEXT NOT NULL,
    category     TEXT NOT NULL,
    price_inr    INTEGER NOT NULL
);
CREATE TABLE orders (
    order_id         TEXT PRIMARY KEY,
    customer_id      TEXT NOT NULL REFERENCES customers(customer_id),
    product_id       TEXT NOT NULL REFERENCES products(product_id),
    quantity         INTEGER NOT NULL,
    order_ts         TEXT NOT NULL,
    status           TEXT NOT NULL,
    discount_pct     INTEGER NOT NULL,
    order_amount     REAL NOT NULL,
    delivery_minutes REAL
);
CREATE INDEX idx_orders_customer ON orders(customer_id);
CREATE INDEX idx_orders_status_ts ON orders(status, order_ts);
"""
if DB_PATH.exists():
    DB_PATH.unlink()
with closing(sqlite3.connect(DB_PATH)) as con:
    con.executescript(DDL)
    customers_out.to_sql("customers", con, if_exists="append", index=False)
    products.to_sql("products", con, if_exists="append", index=False)
    orders_out.to_sql("orders", con, if_exists="append", index=False)
    con.commit()
    counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("customers", "products", "orders")}
rep()
rep(f"STORED data/clean/*.csv and data/instamart.db -> table row counts {counts}")
rep.save()
