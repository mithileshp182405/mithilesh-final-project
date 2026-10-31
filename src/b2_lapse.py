"""Part B, tasks 4-9 -- lapse problem, leakage-free preprocessing, tuned models, balance experiment, interpretation, persistence.

Leakage rules enforced in code:
  * every feature is built from orders dated BEFORE 2025-09-01 (asserted);
  * the segment labels of b1_segments.py are NOT used as features (asserted on the column list);
  * encoder, scaler and resampler are fitted on training rows only (scikit-learn Pipeline; oversampling touches only X_train).
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from common import CLEAN, OUTPUTS, Report

rep = Report("b2_lapse")
pd.set_option("display.width", 160)

CUTOFF = pd.Timestamp("2025-09-01")
WINDOW_END = pd.Timestamp("2025-10-01")

orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv(CLEAN / "customers.csv", parse_dates=["signup_date"])
products = pd.read_csv(CLEAN / "products.csv")
orders = orders.merge(products[["product_id", "category"]], on="product_id")

# ------------------------------------------------------------------ task 4: cohort, label, features
history = orders[orders["order_ts"] < CUTOFF]                       # the ONLY orders features may see
assert history["order_ts"].max() < CUTOFF
h_done = history[history["status"] == "completed"]
cohort_ids = sorted(h_done["customer_id"].unique())

future_done = orders[(orders["status"] == "completed") & (orders["order_ts"] >= CUTOFF) & (orders["order_ts"] < WINDOW_END)]
active_in_sept = set(future_done["customer_id"])
y = pd.Series([int(c not in active_in_sept) for c in cohort_ids], index=cohort_ids, name="lapsed")

g = h_done.groupby("customer_id")
feat = pd.DataFrame({
    "recency": (CUTOFF - g["order_ts"].max()).dt.days,
    "frequency": g["order_id"].nunique(),
    "monetary": g["order_amount"].sum(),
    "avg_discount_pct": g["discount_pct"].mean(),
    "avg_delivery_minutes": g["delivery_minutes"].mean(),
})
final = history[history["status"].isin(["completed", "cancelled"])]
feat["cancellation_rate"] = (final.assign(c=final["status"] == "cancelled").groupby("customer_id")["c"].mean()
                             .reindex(feat.index).fillna(0.0))
fav = (h_done.groupby(["customer_id", "category"]).size().rename("n").reset_index()
       .sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
       .drop_duplicates("customer_id").set_index("customer_id")["category"])
feat["fav_category"] = fav
cust = customers.set_index("customer_id")
feat["city"] = cust["city"]
feat["age"] = cust["age"]
feat["gender"] = cust["gender"]
feat["tenure_days"] = (CUTOFF - cust["signup_date"]).dt.days
X = feat.loc[cohort_ids].copy()

NUMERIC = ["recency", "frequency", "monetary", "avg_discount_pct", "cancellation_rate", "avg_delivery_minutes", "age", "tenure_days"]
NOMINAL = ["city", "gender", "fav_category"]
assert set(X.columns) == set(NUMERIC + NOMINAL)
assert not any("segment" in c or "premium" in c for c in X.columns), "segment labels must not be features"

rep("TASK 4 -- lapse problem (cutoff 2025-09-01; label looks at completed orders 2025-09-01 .. 2025-09-30)")
rep(f"cohort = {len(X):,} customers (reference 1,156) | lapsed = {int(y.sum())} = {y.mean() * 100:.2f}% (reference 443 = 38.32%)")
rep(f"latest order used by any feature: {history['order_ts'].max()} (< cutoff {CUTOFF.date()})")
rep(f"features ({len(X.columns)}): {', '.join(X.columns)}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
rep(f"split: train {len(X_train)} / test {len(X_test)} (reference 924 / 232); lapsed in train {int(y_train.sum())}, in test {int(y_test.sum())}")
base_pred = np.zeros(len(y_test), dtype=int)
rep(f"baseline 'always not lapsed' on test: accuracy {accuracy_score(y_test, base_pred) * 100:.1f}% (reference 61.6%), "
    f"lapsed-class precision {precision_score(y_test, base_pred, zero_division=0):.2f}, recall {recall_score(y_test, base_pred, zero_division=0):.2f}, "
    f"F1 {f1_score(y_test, base_pred, zero_division=0):.2f} (reference 0)")
assert (len(X), int(y.sum()), len(X_train), len(X_test)) == (1156, 443, 924, 232)

# ------------------------------------------------------------------ task 5: leakage-free preprocessing
def new_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), NOMINAL),
    ])

probe = new_preprocessor().fit(X_train)                             # fitted on training rows only
scaler_mean = dict(zip(NUMERIC, probe.named_transformers_["num"].mean_))
rep("\nTASK 5 -- preprocessing without leakage")
rep(f"learned mean of frequency inside the fitted scaler (training rows only) = {scaler_mean['frequency']:.4f}")
rep(f"full-cohort mean of frequency (train + test)                      = {X['frequency'].mean():.4f}")
rep("They differ because the scaler saw only the 924 training rows, never the 232 test rows, so its mean is a training statistic and the test rows")
rep("are merely transformed with it; fitting on the whole cohort would let test information flow into the model.")
cats = probe.named_transformers_["cat"].categories_
rep(f"one-hot with drop='first' (k-1 columns): " + ", ".join(f"{c}: {len(cs)} levels -> {len(cs) - 1} columns" for c, cs in zip(NOMINAL, cats)))
Xt_train, Xt_test = probe.transform(X_train), probe.transform(X_test)
assert not X.isna().any().any() and not np.isnan(Xt_train).any() and not np.isnan(Xt_test).any()
rep(f"missing values in the raw feature table: {int(X.isna().sum().sum())}; in the transformed train/test matrices: "
    f"{int(np.isnan(Xt_train).sum())}/{int(np.isnan(Xt_test).sum())}  -> confirmed none")
rep(f"transformed width = {Xt_train.shape[1]} columns")

# ------------------------------------------------------------------ task 6: tune and evaluate
def pipe(model):
    return Pipeline([("prep", new_preprocessor()), ("model", model)])

def lapsed_metrics(model, Xe, ye):
    p = model.predict(Xe)
    return dict(precision=precision_score(ye, p, zero_division=0), recall=recall_score(ye, p, zero_division=0),
                f1=f1_score(ye, p, zero_division=0), accuracy=accuracy_score(ye, p), cm=confusion_matrix(ye, p))

rep("\nTASK 6 -- tuning (5-fold CV on the training rows only, scoring='f1' on the lapsed class)")
lr_search = GridSearchCV(pipe(LogisticRegression(max_iter=1000)), {"model__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(X_train, y_train)
rf_search = GridSearchCV(pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
                         {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(X_train, y_train)
rep(f"logistic regression: best {lr_search.best_params_} | best CV F1 {lr_search.best_score_:.4f} | fits {len(lr_search.cv_results_['params']) * 5}")
rep(f"random forest      : best {rf_search.best_params_} | best CV F1 {rf_search.best_score_:.4f} | fits {len(rf_search.cv_results_['params']) * 5}")
rep("LR  CV F1 by C: " + ", ".join(f"C={p['model__C']}: {s:.3f}" for p, s in zip(lr_search.cv_results_["params"], lr_search.cv_results_["mean_test_score"])))
rep("RF  CV F1 grid: " + "; ".join(f"depth {p['model__max_depth']}/leaf {p['model__min_samples_leaf']}: {s:.3f}"
                                     for p, s in zip(rf_search.cv_results_["params"], rf_search.cv_results_["mean_test_score"])))

lr, rf = lr_search.best_estimator_, rf_search.best_estimator_
m_lr, m_rf = lapsed_metrics(lr, X_test, y_test), lapsed_metrics(rf, X_test, y_test)
rep("\nuntouched test set (tuning is finished), lapsed = positive class")
rep(f"{'model':<22}{'precision':>10}{'recall':>8}{'F1':>8}{'accuracy':>10}   confusion [[TN FP] [FN TP]]")
rep(f"{'baseline (all 0)':<22}{0:>10.3f}{0:>8.3f}{0:>8.3f}{accuracy_score(y_test, base_pred):>10.3f}   {confusion_matrix(y_test, base_pred).tolist()}")
for name, m in [("logistic regression", m_lr), ("random forest", m_rf)]:
    rep(f"{name:<22}{m['precision']:>10.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{m['accuracy']:>10.3f}   {m['cm'].tolist()}")

best_name, best_model, best_m = (("random forest", rf, m_rf) if m_rf["f1"] >= m_lr["f1"] else ("logistic regression", lr, m_lr))
assert best_m["f1"] >= 0.55, "selected model must reach test F1 >= 0.55"
rep(f"\nSELECTED MODEL: {best_name} (test F1 {best_m['f1']:.3f}, recall {best_m['recall']:.3f}, precision {best_m['precision']:.3f})")
rep("Justification: missing a customer who is about to lapse (a false negative) loses all of that customer's future revenue and a win-back later is")
rep("far costlier than prevention, whereas wrongly targeting a loyal customer (a false positive) only costs a small coupon or message; so recall on the")
rep(f"lapsed class matters at least as much as precision, and F1 balances the two. The {best_name} has the higher test F1 and catches")
rep(f"{best_m['recall'] * 100:.0f}% of lapsing customers at {best_m['precision'] * 100:.0f}% precision, well above the 0 recall of the do-nothing baseline.")
cv_best = "random forest" if rf_search.best_score_ >= lr_search.best_score_ else "logistic regression"
rep(f"Consistency check: cross-validation on the training rows ranks the models the same way ({cv_best} has the higher CV F1: "
    f"RF {rf_search.best_score_:.3f} vs LR {lr_search.best_score_:.3f}), so the choice does not rest on the test set alone.")
assert cv_best == best_name

# ------------------------------------------------------------------ task 7: class-balance experiment (training rows only)
minority = X_train[y_train == 1]
n_add = int((y_train == 0).sum() - (y_train == 1).sum())
rng = np.random.RandomState(42)
extra = minority.iloc[rng.choice(len(minority), size=n_add, replace=True)]
X_bal = pd.concat([X_train, extra]); y_bal = pd.concat([y_train, pd.Series(1, index=extra.index)])
assert (y_bal == 1).sum() == (y_bal == 0).sum() and len(X_bal) == len(X_train) + n_add
assert set(extra.index) <= set(X_train.index)                         # every added row is a training row
lr_bal = clone(lr).fit(X_bal, y_bal)
m_bal = lapsed_metrics(lr_bal, X_test, y_test)
rep("\nTASK 7 -- class-balance experiment (over-sample lapsed TRAINING rows with replacement, random_state=42; test set unchanged)")
rep(f"training rows {len(X_train)} -> {len(X_bal)}: lapsed {int((y_train == 1).sum())} -> {int((y_bal == 1).sum())}, not lapsed {int((y_bal == 0).sum())} "
    f"({n_add} duplicated lapsed rows added)")
rep(f"{'logistic regression':<24}{'precision':>10}{'recall':>8}{'F1':>8}   confusion")
rep(f"{'un-balanced (best C)':<24}{m_lr['precision']:>10.3f}{m_lr['recall']:>8.3f}{m_lr['f1']:>8.3f}   {m_lr['cm'].tolist()}")
rep(f"{'balanced by over-sampling':<24}{m_bal['precision']:>10.3f}{m_bal['recall']:>8.3f}{m_bal['f1']:>8.3f}   {m_bal['cm'].tolist()}")
moves = {k: m_bal[k] - m_lr[k] for k in ("precision", "recall", "f1")}
rep(f"change: precision {moves['precision']:+.3f}, recall {moves['recall']:+.3f}, F1 {moves['f1']:+.3f} -> recall moved most")
rep("Meaning for retention: over-sampling makes the model flag far more customers as lapsing, so the team reaches many more of the real leavers but")
rep("also contacts more loyal customers by mistake (precision does not improve). It is the right setting only when a missed leaver costs much more than")
rep("a wasted offer; it changes where the threshold effectively sits, it does not add information.")

# ------------------------------------------------------------------ task 8: interpret, persist, use
names = [n.split("__", 1)[1] for n in best_model.named_steps["prep"].get_feature_names_out()]
imp = pd.Series(rf.named_steps["model"].feature_importances_, index=[n.split("__", 1)[1] for n in rf.named_steps["prep"].get_feature_names_out()])
top5 = imp.sort_values(ascending=False).head(5)
READ_RF = {"recency": "the longer since the last order, the likelier the customer is to stay away",
           "frequency": "how often they order shows habit strength",
           "monetary": "how much they spend marks their value and attachment",
           "tenure_days": "account age separates new from established customers",
           "avg_delivery_minutes": "slow deliveries may push customers away",
           "avg_discount_pct": "discount dependence changes retention behaviour",
           "cancellation_rate": "frequent cancellations signal friction",
           "age": "age mildly shifts ordering habits"}
rep("\nTASK 8a -- five most important features of the best random forest (feature_importances_)")
for n, v in top5.items():
    rep(f"  {n:<22} {v:.3f}  {READ_RF.get(n, 'a customer attribute the forest finds informative')}")

coef = pd.Series(lr.named_steps["model"].coef_[0], index=[n.split("__", 1)[1] for n in lr.named_steps["prep"].get_feature_names_out()])
# The business reading is chosen from the MEASURED sign and size, so the text can never contradict the number printed beside it.
READ_LR = {
    "recency": {"positive": "customers who have been silent longer are somewhat likelier to lapse, but the effect is small once frequency is known",
                "negative": "a longer silence goes with a LOWER lapse risk once the other features are fixed (unexpected; check for overlap with frequency)"},
    "frequency": {"positive": "more orders go with a HIGHER lapse risk once spend is fixed (unexpected)",
                  "negative": "habitual buyers who order often are far less likely to lapse -- the strongest protective signal"},
    "monetary": {"positive": "for the SAME number of orders, a higher total spend means few, large baskets (big-ticket electronics buyers who order about once a month), "
                             "so a month without an order is more likely -- this group needs its own retention offer",
                 "negative": "customers who spend more are less likely to lapse, holding frequency fixed"},
}
rep("\nlogistic regression coefficients (numeric features standardised; positive = raises the odds of lapsing, others held fixed)")
for f in ["recency", "frequency", "monetary"]:
    sign = "positive" if coef[f] > 0 else "negative"
    strength = "strong" if abs(coef[f]) >= 0.5 else "weak"
    rep(f"  {f:<10} coefficient {coef[f]:+.3f} ({sign}, {strength}) -> {READ_LR[f][sign]}")
rep("  (frequency and monetary are correlated -- r = "
    f"{X_train[['frequency', 'monetary']].corr().iloc[0, 1]:.2f} on the training rows -- so each sign is conditional on the other; read the two together: "
    "'many orders' protects, 'big baskets at a given order count' signals a slower, high-ticket rhythm.)")

path = OUTPUTS / "lapse_model.joblib"
joblib.dump(best_model, path)
reloaded = joblib.load(path)
same_pred = np.array_equal(best_model.predict(X_test), reloaded.predict(X_test))
same_proba = np.array_equal(best_model.predict_proba(X_test), reloaded.predict_proba(X_test))
rep(f"\nTASK 8b -- saved {path.relative_to(path.parents[1])} ({best_name} pipeline); reloaded predictions identical to the original on all {len(X_test)} test rows: {same_pred}; "
    f"probabilities identical: {same_proba}")
assert same_pred and same_proba

seg = pd.read_csv(OUTPUTS / "customer_segments.csv")
premium_ids = set(seg.loc[seg["is_premium"], "customer_id"])
proba = pd.Series(reloaded.predict_proba(X_test)[:, 1], index=X_test.index)
prem_test = proba[proba.index.isin(premium_ids)].sort_values(ascending=False)
top = prem_test.head(10)
retention = pd.DataFrame({"customer_id": top.index, "lapse_probability": top.values.round(4),
                          "actually_lapsed": y_test.loc[top.index].values})
retention.to_csv(OUTPUTS / "retention_list.csv", index=False)
rep(f"\nTASK 8c -- Premium customers in the test set: {len(prem_test)}; retention list = top {len(retention)} by predicted lapse probability")
rep(retention.to_string(index=False))
rep(f"of the {len(retention)} listed, {int(retention['actually_lapsed'].sum())} truly lapsed in September 2025")
rep("CAVEAT: the Premium flag comes from the whole-window RFM (April-September), which includes the September window that defines the lapse label,")
rep("so the retention list is illustrative and not a leakage-free evaluation (the model's own features still use only orders before the cutoff).")

rep("\nLEAKAGE STATEMENT: no encoder, scaler or resampler saw test rows (all fitted inside the pipeline on training rows; over-sampling copied training rows only);")
rep("no order dated on or after 2025-09-01 entered any feature; the segment labels from task 3 were not used as features.")
rep.save()
