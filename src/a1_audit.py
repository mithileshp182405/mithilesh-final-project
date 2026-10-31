"""Part A, task 2 -- audit the three raw tables (shape, dtypes, missing values, planted defects)."""
import pandas as pd
from common import RAW, Report

rep = Report("a1_audit")

orders = pd.read_csv(RAW / "orders.csv")
customers = pd.read_csv(RAW / "customers.csv")
products = pd.read_csv(RAW / "products.csv")

# pandas silently turns the literal string "N/A" into NaN, which would hide half of the unparseable amounts.
# Re-read the amount column as plain text (no NA conversion) to count what really cannot be parsed.
amount_text = pd.read_csv(RAW / "orders.csv", usecols=["order_amount"], dtype=str, keep_default_na=False)["order_amount"]
amount_numeric = pd.to_numeric(amount_text, errors="coerce")

for name, df in [("orders.csv", orders), ("customers.csv", customers), ("products.csv", products)]:
    rep(f"== {name}: shape {df.shape}")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_pct": (df.isna().mean() * 100).round(2),
    })
    rep(summary.to_string())
    rep()

rep("NOTE: order_amount shows 53 missing above only because pandas turned the text 'N/A' into NaN;")
rep("      the real number of unparseable amounts is counted from the raw text below.")
rep()

audit = {
    "orders.csv raw rows": len(orders),
    "orders.csv exact duplicate rows": int(orders.duplicated().sum()),
    "orders.csv rows with status pending": int((orders["status"] == "pending").sum()),
    "orders.csv distinct order_id": int(orders["order_id"].nunique()),
    "orders.csv non-numeric order_amount": int(amount_numeric.isna().sum()),
    "orders.csv negative delivery_minutes": int((orders["delivery_minutes"] < 0).sum()),
    "orders.csv blank delivery_minutes": int(orders["delivery_minutes"].isna().sum()),
    "customers.csv blank age": int(customers["age"].isna().sum()),
    "customers.csv age outside 10-100": int(((customers["age"] < 10) | (customers["age"] > 100)).sum()),
    "customers.csv distinct raw city spellings": int(customers["city"].nunique()),
    "customers.csv distinct non-blank gender spellings": int(customers["gender"].nunique()),
    "customers.csv blank gender": int(customers["gender"].isna().sum()),
}
reference = [13084, 119, 656, 12309, 120, 106, 2541, 61, 6, 24, 10, 30]

rep(f"{'audit figure':<52}{'measured':>10}{'reference':>11}  match")
for (label, value), ref in zip(audit.items(), reference):
    rep(f"{label:<52}{value:>10,}{ref:>11,}  {'OK' if value == ref else 'DIFF'}")
assert list(audit.values()) == reference, "audit figures differ from the brief's reference values"

rep()
rep("Raw city spellings: " + ", ".join(repr(x) for x in sorted(customers["city"].unique())))
rep("Raw gender spellings: " + ", ".join(repr(x) for x in sorted(customers["gender"].dropna().unique())))
rep("Raw amount strings that are not numbers: " + ", ".join(sorted(amount_text[amount_numeric.isna()].unique())))
rep.save()
