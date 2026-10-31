"""Part B, tasks 1-3 -- RFM table, K-means (K = 2..8), chosen segments with business names, PCA, customer_segments.csv."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from common import CLEAN, OUTPUTS, Report

rep = Report("b1_segments")
pd.set_option("display.width", 160)

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
orders = pd.read_csv(CLEAN / "orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]

# ------------------------------------------------------------------ task 1: RFM
last = done.groupby("customer_id")["order_ts"].max()
rfm = pd.DataFrame({
    "recency": (SNAPSHOT - last).dt.days,                              # whole days since the last completed order
    "frequency": done.groupby("customer_id")["order_id"].nunique(),    # distinct completed orders
    "monetary": done.groupby("customer_id")["order_amount"].sum().round(2),
}).reset_index()

rep("TASK 1 -- RFM (snapshot 2025-10-01 00:00:00, completed orders only)")
rep(f"customers = {len(rfm):,} (reference 1,175)")
rep(f"recency   median {rfm.recency.median():g} | min {rfm.recency.min()} | max {rfm.recency.max()}   (reference 19 / 0 / 180)")
rep(f"frequency mean {rfm.frequency.mean():.2f} | max {rfm.frequency.max()}   (reference 9.46 / 33)")
rep(f"monetary  mean {rfm.monetary.mean():.2f} | max {rfm.monetary.max():.2f}   (reference 2,552.06 / 14,766.55)")
assert (len(rfm), rfm.recency.median(), rfm.recency.min(), rfm.recency.max()) == (1175, 19, 0, 180)
assert (round(rfm.frequency.mean(), 2), rfm.frequency.max(), round(rfm.monetary.mean(), 2), rfm.monetary.max()) == (9.46, 33, 2552.06, 14766.55)

# ------------------------------------------------------------------ task 2: K-means sweep
scaler = StandardScaler()
X = scaler.fit_transform(rfm[["recency", "frequency", "monetary"]])
sweep = []
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    sweep.append((k, km.inertia_, silhouette_score(X, km.labels_)))
sweep = pd.DataFrame(sweep, columns=["K", "inertia", "silhouette"])
reference = {2: (1820.8, .465), 3: (1268.2, .445), 4: (743.1, .526), 5: (617.4, .481), 6: (503.2, .482), 7: (408.5, .416), 8: (366.5, .412)}
sweep["ref_inertia"] = sweep["K"].map(lambda k: reference[k][0])
sweep["ref_silhouette"] = sweep["K"].map(lambda k: reference[k][1])
sweep["inertia_drop_pct"] = (1 - sweep["inertia"] / sweep["inertia"].shift()).mul(100)
rep("\nTASK 2 -- K-means sweep (StandardScaler on R, F, M; n_init=10, random_state=42)")
rep(sweep.round({"inertia": 1, "silhouette": 3, "inertia_drop_pct": 1}).to_string(index=False))
for r in sweep.itertuples():
    assert abs(r.inertia / r.ref_inertia - 1) <= .05, f"inertia K={r.K} outside +-5%"
    assert abs(r.silhouette - r.ref_silhouette) <= .05, f"silhouette K={r.K} outside +-0.05"
rep("all seven (inertia, silhouette) pairs are within the stated tolerances of the reference.")

candidates = sweep[sweep["K"].isin([3, 4, 5])].set_index("K")
K = 4
rep(f"\nElbow: inertia falls by {sweep.set_index('K').loc[3, 'inertia_drop_pct']:.0f}% from K=2 to 3 and by "
    f"{sweep.set_index('K').loc[4, 'inertia_drop_pct']:.0f}% from K=3 to 4, but only {sweep.set_index('K').loc[5, 'inertia_drop_pct']:.0f}% "
    f"from K=4 to 5 and {sweep.set_index('K').loc[6, 'inertia_drop_pct']:.0f}% from K=5 to 6 -> the bend (elbow) is at K=4.")
rep(f"Silhouette among the allowed K: K=3 {candidates.loc[3, 'silhouette']:.3f}, K=4 {candidates.loc[4, 'silhouette']:.3f}, "
    f"K=5 {candidates.loc[5, 'silhouette']:.3f} -> K=4 is clearly the most separated.")
assert candidates["silhouette"].idxmax() == K
rep("CHOSEN K = 4: best silhouette, elbow at 4, and four segments is a number the campaign team can staff and act on")
rep("(Premium big-ticket / loyal regulars / lapsing / occasional) whereas five or more would split the loyal group into near-duplicates.")
rep("Why K=8 always has lower inertia than K=2: inertia is the within-cluster sum of squared distances, and every extra centroid can")
rep("only move points closer to a centre, so inertia falls monotonically down to 0 at K=n. A lower value therefore says nothing about")
rep("quality by itself -- it must be weighed against the elbow, the silhouette (cluster separation) and whether the groups are actionable.")

# ------------------------------------------------------------------ task 3: profile, names, PCA, output
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_
prof = (rfm.groupby("segment_id").agg(size=("customer_id", "count"), mean_recency=("recency", "mean"),
                                      mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean")).round(2))
rep("\nTASK 3a -- segment profile (cluster numbers are arbitrary labels with no ordering)")
rep(prof.to_string())

# Names are assigned from the profile itself, never from the raw cluster number, so they survive a different label order.
remaining = set(prof.index)
premium_id = prof["mean_monetary"].idxmax();            remaining.discard(premium_id)
loyal_id = prof.loc[list(remaining), "mean_frequency"].idxmax(); remaining.discard(loyal_id)
lapsing_id = prof.loc[list(remaining), "mean_recency"].idxmax(); remaining.discard(lapsing_id)
(occasional_id,) = remaining
SEGMENTS = {
    premium_id:    ("Premium Big-Ticket", "Early access to new electronics, free-delivery perks and a 1-hour premium support desk; protect them from churn first."),
    loyal_id:      ("Loyal Regulars", "Reward consistency with a points/streak scheme and basket-builder offers on grocery and dairy to raise basket size."),
    lapsing_id:    ("Lapsing Low-Spenders", "Win-back push within days of going quiet: a time-limited coupon and a reminder of their usual items."),
    occasional_id: ("Occasional Shoppers", "Light-touch nudges (weekly digest, first-order-of-the-week coupon) to move them toward the regular rhythm."),
}
rfm["segment_name"] = rfm["segment_id"].map(lambda s: SEGMENTS[s][0])
rfm["is_premium"] = rfm["segment_id"] == premium_id
prof["segment_name"] = [SEGMENTS[s][0] for s in prof.index]
prof["is_premium"] = [s == premium_id for s in prof.index]
rep("\nTASK 3b -- names, Premium flag and recommended action")
for s in prof.index:
    p = prof.loc[s]
    rep(f"  segment {s}: {SEGMENTS[s][0]:<22} n={int(p['size']):>3} | R {p.mean_recency:5.1f} d | F {p.mean_frequency:5.1f} | M INR {p.mean_monetary:8.2f}"
        f"{'  <- PREMIUM (highest mean monetary)' if s == premium_id else ''}")
    rep(f"      action: {SEGMENTS[s][1]}")

# reference sanity for K=4 (sizes about 436/138/419/182; premium 138 with M~7130, F~6.8; loyal 419 with F~19.4, R~8.8)
sizes = sorted(prof["size"].astype(int).tolist())
assert all(abs(a - b) <= 25 for a, b in zip(sizes, sorted([436, 138, 419, 182]))), sizes
assert abs(prof.loc[premium_id, "mean_monetary"] / 7130 - 1) < .05 and abs(prof.loc[premium_id, "mean_frequency"] - 6.8) < .5
assert abs(prof.loc[loyal_id, "mean_frequency"] - 19.4) < 1 and abs(prof.loc[loyal_id, "mean_recency"] - 8.8) < 1.5
assert rfm.groupby("segment_id")["is_premium"].first().sum() == 1

pca = PCA(n_components=2).fit(X)
ev = pca.explained_variance_ratio_
rep(f"\nTASK 3c -- PCA on the scaled RFM matrix: explained_variance_ratio_ = {ev[0]:.3f}, {ev[1]:.3f}; total {ev.sum():.3f} "
    f"(reference 0.634, 0.218, 0.852 +-0.02)")
assert abs(ev[0] - .634) <= .02 and abs(ev[1] - .218) <= .02 and abs(ev.sum() - .852) <= .02
rep("Two components keep about 85% of the variance of the three standardised RFM columns, so a 2-D picture of the segments is faithful.")

out = rfm[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]]
out.to_csv(OUTPUTS / "customer_segments.csv", index=False)
rep(f"\nwrote outputs/customer_segments.csv ({len(out):,} rows); premium customers = {int(out['is_premium'].sum())}")
rep.save()
