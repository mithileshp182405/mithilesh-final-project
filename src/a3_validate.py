"""Part A, task 5 -- validate the clean tables against the brief's reference checkpoints.

Prints exactly `ALL CHECKPOINTS PASSED` when every check holds (use --verbose to list the checks).
Reads both the clean CSV files and the SQLite database, so a mismatch between the two is also caught.
"""
import sqlite3
import sys
from contextlib import closing
import pandas as pd
from common import CLEAN, DB_PATH

verbose = "--verbose" in sys.argv

customers = pd.read_csv(CLEAN / "customers.csv")
orders = pd.read_csv(CLEAN / "orders.csv")
products = pd.read_csv(CLEAN / "products.csv")
with closing(sqlite3.connect(DB_PATH)) as con:
    db_orders = pd.read_sql_query("SELECT * FROM orders", con)
    db_tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    db_counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("customers", "products", "orders")}
    db_revenue = con.execute("SELECT ROUND(SUM(order_amount), 2) FROM orders WHERE status='completed'").fetchone()[0]

done = orders["status"] == "completed"
canc = orders["status"] == "cancelled"
city = customers["city"].value_counts().to_dict()
gender = customers["gender"].value_counts().to_dict()

checks = [
    ("orders: 12,309 rows", len(orders) == 12309),
    ("orders: one row per order_id", orders["order_id"].is_unique and orders["order_id"].nunique() == 12309),
    ("orders: 11,110 completed", int(done.sum()) == 11110),
    ("orders: 1,199 cancelled", int(canc.sum()) == 1199),
    ("orders: no pending rows", int((orders["status"] == "pending").sum()) == 0),
    ("orders: 0 missing order_amount", int(orders["order_amount"].isna().sum()) == 0),
    ("orders: 0 missing delivery_minutes among completed", int(orders.loc[done, "delivery_minutes"].isna().sum()) == 0),
    ("orders: 1,199 missing delivery_minutes among cancelled", int(orders.loc[canc, "delivery_minutes"].isna().sum()) == 1199),
    ("orders: no negative delivery_minutes", int((orders["delivery_minutes"] < 0).sum()) == 0),
    ("orders: completed revenue = INR 2,998,668.80", round(orders.loc[done, "order_amount"].sum(), 2) == 2998668.80),
    ("customers: 1,200 rows, exactly 6 cities", len(customers) == 1200 and len(city) == 6),
    ("customers: city counts Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184",
     city == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184}),
    ("customers: female 610, male 560, unknown 30", gender == {"female": 610, "male": 560, "unknown": 30}),
    ("customers: 0 missing ages, all between 18 and 60",
     int(customers["age"].isna().sum()) == 0 and customers["age"].between(18, 60).all()),
    ("database: tables customers, products, orders exist", {"customers", "products", "orders"} <= db_tables),
    ("database: row counts 1200 / 40 / 12309", db_counts == {"customers": 1200, "products": 40, "orders": 12309}),
    ("database: completed revenue = INR 2,998,668.80", db_revenue == 2998668.80),
    ("database: order_ts stored as 'YYYY-MM-DD HH:MM:SS' text",
     db_orders["order_ts"].str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all()),
    ("database and clean CSV hold the same order ids", set(db_orders["order_id"]) == set(orders["order_id"])),
]

failed = [name for name, ok in checks if not ok]
if verbose:
    for name, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + name)
if failed:
    print("CHECKPOINT FAILURES: " + "; ".join(failed))
    sys.exit(1)
print("ALL CHECKPOINTS PASSED")
