# Instamart Customer Intelligence & Support Copilot

## Overview

A single repository that turns messy Instamart-style order exports into a trusted database (Part A), segments customers and predicts who will stop ordering (Part B), triages support tickets safely (Part C), and answers policy questions and decides refund requests with a tool-using agent (Part D). All data is synthetic and fictional, produced by the supplied `generate_data.py`; all language-model calls go through the supplied offline `mock_llm.py` (no key, no network, no cost). All money is in INR.

**How the Parts connect.** Part A writes `data/clean/*.csv` and `data/instamart.db`. Part B reads the clean CSVs and writes `outputs/customer_segments.csv`, which Part C reads to route tickets (Premium customers). Part D's refund agent reads `data/instamart.db` from Part A and reuses the policy retrieval built in the same Part.

**Submission facts.** One public repository, branch `main`, text and code only (no images, PDFs, decks, audio or video; no charts committed). Libraries: Python standard library, pandas, numpy, scikit-learn, joblib. `generate_data.py` and `mock_llm.py` are unmodified copies of the supplied files.

### Repository layout

| Path | Purpose |
|---|---|
| `generate_data.py`, `mock_llm.py` | supplied files, unmodified |
| `requirements.txt`, `.gitignore` | pinned libraries (`pip freeze`); ignores `.env`, `.venv/`, `__pycache__/` |
| `src/common.py` | repository paths and a small report writer shared by all scripts |
| `src/a1_audit.py` ... `src/a5_eda.py` | Part A: audit, cleaning + storage, validation, SQL runner, pandas analysis |
| `sql/queries.sql` | the seven Part A queries |
| `src/b1_segments.py`, `src/b2_lapse.py` | Part B: RFM + K-means + PCA; lapse model |
| `src/triage.py`, `src/c1_triage.py` | Part C: pipeline library; runner for every Part C task |
| `src/rag.py`, `src/d1_rag_eval.py` | Part D: chunking, TF-IDF retrieval, grounded answers; evaluation |
| `src/agent_tools.py`, `src/agent.py`, `src/d2_agent_run.py` | Part D: five tools, ReAct loop with limits and resume; runner |
| `tests/test_pipeline.py` | 27 unit tests of the safety-critical logic (standard-library `unittest`) |
| `data/raw`, `data/policies`, `data/clean`, `data/instamart.db` | generated data (deterministic) |
| `outputs/`, `state/`, `reports/` | model, results, ledger, agent checkpoints; `reports/` holds the full console log of every script; `outputs/agent_demos.jsonl` records the stop-safely and crash-and-resume demonstration runs |

### Headline results

| Part | Result |
|---|---|
| A | every cleaning and SQL checkpoint matches; validation prints `ALL CHECKPOINTS PASSED`; completed revenue INR 2,998,668.80 |
| B | K = 4 segments (Premium = 138 customers); lapse model: random forest, test F1 0.655 on the lapsed class (baseline 0) |
| C | 147 of 156 eligible tickets classified (135 JSON + 12 keyword), 9 sent to humans; accuracy 85.7% zero-shot, 100% few-shot |
| D | policy assistant 12/12 retrieval, 12/12 grounded, 2/2 fallback; refund agent solves all seven scenarios, stops safely, resumes after a crash |

## Reproduce

Python 3.11 or newer. The versions in `requirements.txt` (pandas 3.1.0, numpy 2.4.6, scikit-learn 1.9.1, scipy 1.17.1, joblib 1.6.0) were resolved on Python 3.11, the lowest version the brief allows, and the whole pipeline was run end to end on Python 3.11, 3.12 and 3.13 with identical results (every report, CSV, model and JSONL output is byte-identical; the only differences are the timestamps in the refund ledger and the binary header of `data/instamart.db`, which records the version of the SQLite library that wrote it; the tables and rows inside are identical). Run from the repository root; every command is deterministic (fixed seeds), and each script asserts the brief's reference checkpoints, so a non-zero exit code means a number drifted.

1. `python -m venv .venv`
2. `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py` (last line must be `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python src/a1_audit.py` (Part A: audit of the raw data)
6. `python src/a2_clean.py` (Part A: cleaning, `data/clean/*.csv`, `data/instamart.db`)
7. `python src/a3_validate.py` (Part A: must print `ALL CHECKPOINTS PASSED`)
8. `python src/a4_sql.py` (Part A: runs `sql/queries.sql`)
9. `python src/a5_eda.py` (Part A: pandas cross-check and statistics)
10. `python src/b1_segments.py` (Part B: RFM, K-means, PCA, `outputs/customer_segments.csv`)
11. `python src/b2_lapse.py` (Part B: lapse model, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`)
12. `python src/c1_triage.py` (Part C: `outputs/triage_results.csv`, `outputs/triage_budget_demo.csv`)
13. `python src/d1_rag_eval.py` (Part D: chunking, retrieval, evaluation)
14. `python src/d2_agent_run.py` (Part D: `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/`)
15. `python -m unittest discover -s tests -v` (27 unit tests, must end with `OK`)

Each script also saves its console output to `reports/<script>.txt`. Re-running the pipeline regenerates every file; the only content that changes between runs is the timestamp inside `outputs/refund_ledger.jsonl`.

---

## Part A

### A1. Setup

Repository on branch `main`; virtual environment created with `python -m venv`; libraries installed with pip; `requirements.txt` written with `pip freeze`; `.gitignore` lists `.env`, `.venv/`, `venv/`, `env/`, `__pycache__/`; the two supplied files copied unmodified; `generate_data.py` run (last line as required); Reproduce section above.

### A2. Audit of the raw data (`src/a1_audit.py`)

Shapes: `orders.csv` (13,084, 9), `customers.csv` (1,200, 5), `products.csv` (40, 4). Data types are as pandas reads them (`str` = text, called `object` in older pandas).

| Table | Column | dtype | Missing | Missing % |
|---|---|---|---|---|
| orders | order_id, customer_id, product_id, order_ts, status | str | 0 | 0.00 |
| orders | quantity, discount_pct | int64 | 0 | 0.00 |
| orders | order_amount | str | 53 | 0.41 |
| orders | delivery_minutes | float64 | 2,541 | 19.42 |
| customers | customer_id, city, signup_date | str | 0 | 0.00 |
| customers | age | float64 | 61 | 5.08 |
| customers | gender | str | 30 | 2.50 |
| products | product_id, product_name, category | str | 0 | 0.00 |
| products | price_inr | int64 | 0 | 0.00 |

Note on `order_amount`: pandas silently converts the text `N/A` to a missing value, so the 53 above hides half of the problem. Re-reading the column as plain text shows **120** entries that are not numbers (the strings `N/A` and `error`), which is the figure used below.

| Audit figure | Measured | Reference |
|---|---|---|
| raw orders.csv rows | 13,084 | 13,084 |
| exact duplicate rows | 119 | 119 |
| rows with status pending | 656 | 656 |
| distinct order_id | 12,309 | 12,309 |
| non-numeric order_amount | 120 | 120 |
| negative delivery_minutes | 106 | 106 |
| blank delivery_minutes | 2,541 | 2,541 |
| customers: blank age | 61 | 61 |
| customers: age outside 10-100 | 6 | 6 |
| customers: distinct raw city spellings | 24 | 24 |
| customers: distinct non-blank gender spellings | 10 | 10 |
| customers: blank gender | 30 | 30 |

### A3. Cleaning rules and how many values each changed (`src/a2_clean.py`)

**Customers**

| Rule | Values changed | Result |
|---|---|---|
| C1 city: strip spaces, title case | 476 of 1,200 | 24 spellings become 6 cities |
| C2 gender: map to male / female / unknown (blank becomes unknown) | 950 | 30 blanks become `unknown` |
| C3 age: values outside 10-100 set to missing, then all missing filled with the city median (`groupby` + `transform`) | 6 impossible values, 67 filled (61 blank + 6) | 0 missing; ages 18 to 60 |
| C4 signup_date converted to datetime | 1,200 | 2024-01-01 to 2025-03-31 |

Reference after cleaning (all matched): Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184; female 610, male 560, unknown 30; 0 missing ages, every age between 18 and 60. (21 filled ages end in .5 because a city median of an even count is a half; they are kept as medians, not rounded.)

**Orders**, applied in this order

| Rule | Values changed | Rows after |
|---|---|---|
| O1 order_ts converted to datetime | 13,084 | 13,084 |
| O2 exact duplicate rows removed | 119 rows removed | 12,965 |
| O3 per `order_id` keep only the record with the latest `order_ts` (drops the earlier pending record) | 656 superseded rows removed; 0 pending remain | 12,309 (one per order_id) |
| O4 order_amount to numeric (unparseable become missing), then filled with `round(price_inr x quantity x (1 - discount_pct/100), 2)` | 112 amounts filled (the 120 in the raw file include rows removed by O2 and O3) | 0 missing |
| O5 delivery_minutes: negatives set to missing; missing values of completed orders filled with the median of the product category among completed orders (`groupby` + `transform`); cancelled orders left missing | 105 negatives set to missing; 774 completed orders filled; 1,199 cancelled orders deliberately left missing | 0 missing among completed |

Category medians used in O5: Dairy 25, Electronics 25, Grocery 25, Personal Care 25.5, Snacks 25. Result: 12,309 rows (11,110 completed, 1,199 cancelled, 0 pending), 0 missing `order_amount`, completed revenue INR 2,998,668.80.

### A4. Store and validate (`src/a2_clean.py`, `src/a3_validate.py`)

The clean tables are written to `data/clean/{customers,products,orders}.csv` and to `data/instamart.db` (tables `customers`, `products`, `orders`, with primary keys, foreign keys and two indexes; `order_ts` stored as text `YYYY-MM-DD HH:MM:SS`). Row counts in the database: 1,200 / 40 / 12,309.

`src/a3_validate.py` asserts 19 checkpoints (orders 12,309 rows with one per order_id; 11,110 completed; 1,199 cancelled; no pending; 0 missing amounts; 0 missing delivery among completed; 1,199 among cancelled; no negative delivery; revenue INR 2,998,668.80; 6 cities with the reference counts; genders 610 / 560 / 30; ages complete and between 18 and 60; the three tables exist with the right row counts; the database revenue equals the CSV revenue; timestamp text format; database and CSV hold the same order ids). It prints exactly `ALL CHECKPOINTS PASSED` (run with `--verbose` to list the checks).

### A5. SQL analysis (`sql/queries.sql`, run by `src/a4_sql.py` with `pd.read_sql_query`)

| Query | Technique | Result | Reference |
|---|---|---|---|
| Q1 | `WHERE` (orders above INR 500) then `HAVING` (customer total above INR 8,000) | 30 customers (top: C0693 with INR 13,776.55 over 6 orders) | 30 |
| Q2 | three-table join, revenue by city and category | 30 rows; total INR 2,998,668.80; largest cell Bengaluru / Electronics INR 233,909.45 | 30; 2,998,668.80; 233,909.45 |
| Q3 | scalar subquery | 11 products above the average catalogue price of INR 289.32 | 11 |
| Q4 | `LEFT JOIN` (status test in the `ON` clause) | 25 customers without a completed order | 25 |
| Q5 | CTE (customer totals, then their average) | 550 customers above the average of INR 2,552.06 | 550; 2,552.06 |
| Q6 | `RANK`, `DENSE_RANK`, `ROW_NUMBER` over `PARTITION BY city ORDER BY total_spend DESC` | top per city below; 8 rows where RANK differs from ROW_NUMBER | 8 |
| Q7 | running total with `SUM(revenue) OVER (ORDER BY day)` | 183 daily rows; last cumulative INR 2,998,668.80 | 183; 2,998,668.80 |

Q6 top customer per city (all match the reference): Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10).

*Why ties make RANK differ from ROW_NUMBER:* RANK gives tied customers the same rank and then skips the numbers they occupy, while ROW_NUMBER breaks the tie arbitrarily and keeps counting, so only the second and later members of each tie group get different values (8 rows: 7 tie groups, one of them with 3 customers) and the two numberings line up again right after the group.

### A6. pandas cross-check and exploratory analysis (`src/a5_eda.py`)

**(a)** `pd.pivot_table(..., margins=True)` of completed `order_amount` by city and category gives a grand total of INR 2,998,668.80, equal to the SQL total of Q2, and the same largest cell (Bengaluru / Electronics, INR 233,909.45). Per city with named aggregation:

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

**(b)** Completed `order_amount` (n = 11,110): mean 269.91, median 135.00, skewness 4.61 (right-skewed), Q1 64.80, Q3 297.00, IQR 232.20, fences -283.50 and 645.30; 897 outliers above the upper fence, none below.

Correlation matrix (completed orders):

| | order_amount | discount_pct | quantity | delivery_minutes |
|---|---|---|---|---|
| order_amount | 1.000 | -0.095 | 0.087 | -0.003 |
| discount_pct | -0.095 | 1.000 | -0.000 | -0.010 |
| quantity | 0.087 | -0.000 | 1.000 | -0.013 |
| delivery_minutes | -0.003 | -0.010 | -0.013 | 1.000 |

The pair with the largest absolute correlation is `order_amount` with `discount_pct` (-0.095).

**Answer to the four questions (104 words):** What is it about? Quick-commerce orders from six Indian cities, April-September 2025: 11,110 completed orders worth INR 2,998,668.80. Distribution: order_amount is strongly right-skewed (mean 269.91 against median 135, skewness 4.61); half of all orders sit between INR 65 and INR 297. Outliers: 897 orders (8.1%) exceed the upper fence of INR 645 and none fall below the lower one; they are mostly electronics purchases, genuine high-value baskets rather than errors, so I keep them. Relationships: all correlations are weak; the largest in absolute value is order_amount with discount_pct (-0.095), just ahead of order_amount with quantity (+0.087). Correlation of this size does not establish causation.

---

## Part B

Only completed orders are used for customer behaviour, as the brief says.

### B1. RFM table (`src/b1_segments.py`)

Snapshot 2025-10-01 00:00:00; recency = `(snapshot - last completed order).dt.days`; frequency = distinct completed orders; monetary = sum of completed `order_amount`.

| Quantity | Measured | Reference |
|---|---|---|
| customers with at least one completed order | 1,175 | 1,175 |
| recency median / min / max (days) | 19 / 0 / 180 | 19 / 0 / 180 |
| frequency mean / max | 9.46 / 33 | 9.46 / 33 |
| monetary mean / max (INR) | 2,552.06 / 14,766.55 | 2,552.06 / 14,766.55 |

### B2. K-means (StandardScaler, `KMeans(n_clusters=K, n_init=10, random_state=42)`)

| K | Inertia | Reference | Silhouette | Reference | Inertia drop vs K-1 |
|---|---|---|---|---|---|
| 2 | 1820.8 | 1820.8 | 0.465 | 0.465 | - |
| 3 | 1268.2 | 1268.2 | 0.445 | 0.445 | 30.3% |
| 4 | 743.1 | 743.1 | 0.526 | 0.526 | 41.4% |
| 5 | 617.4 | 617.4 | 0.480 | 0.481 | 16.9% |
| 6 | 503.2 | 503.2 | 0.482 | 0.482 | 18.5% |
| 7 | 408.5 | 408.5 | 0.416 | 0.416 | 18.8% |
| 8 | 366.5 | 366.5 | 0.412 | 0.412 | 10.3% |

**Choice: K = 4.** The elbow is at 4: inertia falls 30% from K=2 to 3 and 41% from 3 to 4, but only 17% from 4 to 5 and 18% from 5 to 6. Among the allowed values {3, 4, 5} K=4 also has the best silhouette (0.526 against 0.445 and 0.480). Four segments is a number the retention team can staff and act on (big-ticket, loyal, lapsing, occasional), whereas five would split the loyal group into near-duplicates.

**Why the inertia of K=8 is below that of K=2 and still cannot pick K:** inertia is the within-cluster sum of squared distances, and every extra centroid can only move points closer to a centre, so it falls steadily down to 0 when K equals the number of customers. A lower value therefore says nothing about quality by itself; it must be weighed against the elbow, the silhouette and whether the groups are actionable.

### B3. Segment profile, names and PCA

Cluster numbers are only labels with no ordering; the names below are assigned from the profile (Premium = highest mean monetary value), never from the number.

| segment_id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | **Premium Big-Ticket** (`is_premium` = true) | 138 | 24.1 | 6.8 | 7,129.70 | early access to new electronics, free-delivery perks, 1-hour premium support desk; protect from churn first |
| 2 | Loyal Regulars | 419 | 8.8 | 19.4 | 3,410.02 | points / streak scheme and basket-builder offers on grocery and dairy |
| 0 | Occasional Shoppers | 436 | 28.9 | 3.8 | 1,057.62 | light-touch nudges (weekly digest, first-order-of-the-week coupon) toward a regular rhythm |
| 3 | Lapsing Low-Spenders | 182 | 109.0 | 2.2 | 685.99 | win-back push within days of going quiet: time-limited coupon and reminder of usual items |

The sizes (436 / 138 / 419 / 182) and the profile of the 138- and 419-customer segments match the reference. **PCA** on the scaled RFM matrix: `explained_variance_ratio_` = 0.634 and 0.218, total 0.852 (reference 0.634, 0.218, 0.852), so a 2-D view keeps about 85% of the variance. `outputs/customer_segments.csv` (1,175 rows) has the columns `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`; `is_premium` is true for exactly the highest-mean-monetary segment (138 customers).

### B4. The lapse problem (`src/b2_lapse.py`)

Cutoff 2025-09-01. Cohort: customers with at least one completed order before the cutoff. Label `lapsed` = 1 if the customer has no completed order from 2025-09-01 through 2025-09-30.

Features (11), all computed from orders dated before the cutoff plus the clean customer table: `recency` (days from the cutoff to the last completed order), `frequency`, `monetary`, `avg_discount_pct` (completed orders), `cancellation_rate` (cancelled orders divided by all orders with a final status), `avg_delivery_minutes` (completed orders), `fav_category` (most frequent product category, ties broken alphabetically), `city`, `age`, `gender`, `tenure_days` (cutoff minus signup date).

| Quantity | Measured | Reference |
|---|---|---|
| cohort | 1,156 | 1,156 |
| lapsed | 443 (38.32%) | 443 (38.32%) |
| train / test rows (`train_test_split(test_size=0.2, stratify=y, random_state=42)`) | 924 / 232 | 924 / 232 |
| lapsed in train / test | 354 / 89 | - |
| baseline "always not lapsed": test accuracy | 61.6% | 61.6% |
| baseline lapsed-class precision / recall / F1 (`zero_division=0`) | 0 / 0 / 0 | F1 = 0 |

**Leakage controls.** No order dated on or after 2025-09-01 entered any feature (the latest order used is 2025-08-31 22:35:45, asserted in code); the segment labels from B3 were not used as features (asserted on the column list).

### B5. Preprocessing without leakage

A scikit-learn `Pipeline` with a `ColumnTransformer`: `StandardScaler` on the 8 numeric features and `OneHotEncoder(drop="first")` on `city`, `gender`, `fav_category` (k-1 columns each: 5 + 2 + 4), 19 columns in total. Every encoder and scaler is fitted on the 924 training rows only and merely applied to the test rows; no encoder, scaler or resampler saw test rows.

| | Value |
|---|---|
| learned mean of `frequency` inside the fitted scaler (training rows) | 8.1147 |
| mean of `frequency` over the full cohort (train + test) | 8.0285 |

They differ because the scaler saw only the 924 training rows and never the 232 test rows, so its mean is a training statistic; fitting it on the whole cohort would let test information flow into the model. Missing values: 0 in the raw feature table and 0 in the transformed train and test matrices (asserted).

### B6. Tuning and evaluation (5-fold CV on training rows only, `scoring="f1"`)

| Model | Grid | Best parameters | Best CV F1 |
|---|---|---|---|
| LogisticRegression(max_iter=1000) | C in {0.01, 0.1, 1, 10, 100} (25 fits) | C = 100 | 0.651 |
| RandomForestClassifier(n_estimators=100, random_state=42) | max_depth in {3, 5, 8} x min_samples_leaf in {5, 10, 20} (45 fits) | max_depth = 8, min_samples_leaf = 10 | 0.681 |

LR cross-validated F1 by C: 0.562 (0.01), 0.646 (0.1), 0.645 (1), 0.649 (10), 0.651 (100); the curve is flat above C = 0.1, so the choice of C matters little (the reference reports C = 1, which is within 0.006 here). The random forest settings match the reference (depth 8, leaf 10).

Untouched test set, lapsed class as positive (confusion matrix as `[[TN, FP], [FN, TP]]`):

| Model | Precision | Recall | F1 | Accuracy | Confusion matrix |
|---|---|---|---|---|---|
| baseline (always not lapsed) | 0.000 | 0.000 | 0.000 | 0.616 | [[143, 0], [89, 0]] |
| logistic regression | 0.644 | 0.652 | 0.648 | 0.728 | [[111, 32], [31, 58]] |
| random forest | 0.659 | 0.652 | 0.655 | 0.737 | [[113, 30], [31, 58]] |

**Selected model: random forest (test F1 0.655, above the required 0.55).** Missing a customer who is about to lapse loses all of that customer's future orders and a win-back later costs far more than prevention, whereas wrongly targeting a loyal customer only costs a small coupon or message, so recall on the lapsed class matters at least as much as precision and F1 balances the two. Both models catch the same 58 of 89 lapsing customers, but the forest wastes two fewer offers (30 false positives against 32). Cross-validation on the training rows ranks them the same way (0.681 against 0.651), so the choice does not rest on the test set alone.

### B7. Class-balance experiment

The 354 lapsed training rows were over-sampled with replacement (`RandomState(42)`) by adding 216 duplicated lapsed training rows, giving 570 and 570 (1,140 rows); the best logistic regression was refitted (whole pipeline) and evaluated on the original, unchanged test set.

| Logistic regression | Precision | Recall | F1 | Confusion matrix |
|---|---|---|---|---|
| un-balanced | 0.644 | 0.652 | 0.648 | [[111, 32], [31, 58]] |
| balanced by over-sampling | 0.589 | 0.854 | 0.697 | [[90, 53], [13, 76]] |
| change | -0.055 | +0.202 | +0.049 | |

**Recall moved most** (0.652 to 0.854, the reference reports about 0.64 to 0.85), while precision did not rise (it fell to 0.589, reference "stays near 0.6"). For the retention team this means over-sampling makes the model flag far more customers: it reaches 76 of the 89 real leavers instead of 58, but also contacts 53 loyal customers by mistake instead of 32. It is the right setting only when a missed leaver costs clearly more than a wasted offer, and it behaves like moving the decision threshold; it adds no new information.

### B8. Interpretation, persistence and use

**Five most important features of the random forest** (`feature_importances_`):

| Feature | Importance | Business reading |
|---|---|---|
| frequency | 0.280 | how often they order shows how strong the habit is |
| monetary | 0.271 | how much they spend marks their value and attachment |
| recency | 0.115 | the longer since the last order, the likelier the customer stays away |
| avg_delivery_minutes | 0.061 | slow deliveries may push customers away |
| cancellation_rate | 0.052 | frequent cancellations signal friction |

**Logistic regression coefficients** (numeric features standardised; positive raises the odds of lapsing, the others held fixed):

| Feature | Coefficient | Sign | Business reading |
|---|---|---|---|
| recency | +0.040 | positive, weak | customers silent for longer are somewhat likelier to lapse, but the effect is small once frequency is known |
| frequency | -1.698 | negative, strong | habitual buyers who order often are far less likely to lapse; the strongest protective signal |
| monetary | +0.285 | positive, weak | for the same number of orders a higher spend means few large baskets (big-ticket electronics buyers who order about once a month), so a month without an order is more likely; this group needs its own retention offer |

Frequency and monetary are correlated (r = 0.44 on the training rows), so each sign is conditional on the other and should be read together.

**Persistence.** The selected pipeline is saved with `joblib.dump` to `outputs/lapse_model.joblib`; the reloaded model reproduces the original predictions on all 232 test rows exactly (and the probabilities too).

**Retention list.** The Premium segment holds 40 customers of the test set. The top 10 by predicted lapse probability are in `outputs/retention_list.csv` (`customer_id, lapse_probability, actually_lapsed`):

| customer_id | lapse_probability | actually_lapsed |
|---|---|---|
| C0862 | 0.4910 | 0 |
| C1112 | 0.4841 | 0 |
| C0060 | 0.4798 | 0 |
| C0802 | 0.4796 | 0 |
| C1008 | 0.4778 | 0 |
| C1000 | 0.4624 | 0 |
| C0525 | 0.4597 | 1 |
| C0133 | 0.4562 | 0 |
| C0267 | 0.4525 | 0 |
| C0167 | 0.4522 | 0 |

Of the 10 listed, **1** truly lapsed. The Premium flag comes from the whole-window RFM (April to September), which includes the September window that defines the lapse label, so it is a post-hoc whole-window label and **the retention list is illustrative and not a leakage-free evaluation** (the model's own features still use only orders before the cutoff).

**Leakage statement.** No encoder, scaler or resampler saw test rows; no data from 2025-09-01 onwards entered the features; the segment labels from task 3 were not used as features.

---

## Part C

All results come from `src/c1_triage.py` (library: `src/triage.py`). Tickets are processed in file order; `tickets.csv` is read with `keep_default_na=False` so empty texts stay empty strings.

### C1. Input guardrails (before any model call)

| Guardrail | Status | Tickets | Reference |
|---|---|---|---|
| empty or whitespace-only text | `skipped_empty` | 4 | 4 |
| `est_tokens(text)` > `MAX_INPUT_TOKENS` = 400 | `flagged_long` | 4 (each estimates to 734 tokens) | 4 (734) |
| eligible for the model | | 156 | 156 |

A unit test and a post-run assertion prove that no empty or long ticket triggered a model call.

### C2. Redaction

Applied in this order with exactly these patterns: email `[\w.+-]+@[\w-]+\.[\w.]+` to `[EMAIL]`, phone `\b[6-9]\d{9}\b` to `[PHONE]`, order id `ORD-\d{6}` to `[ORDER_ID]`. The user message sent to the model is exactly the redacted ticket text and nothing else.

| Type | Substitutions (all 164 tickets) | Reference |
|---|---|---|
| email | 55 | 55 |
| phone | 91 | 91 |
| order id | 132 | 132 |
| **total** | **278** | **278** |

Re-applying the three patterns to the redacted texts finds 0 matches. Every prompt actually sent is also scanned with an assertion after each run (no email, phone or order id reached the model). Example: `Extra money was deducted for order ORD-108532, please check the bill. Contact user61@example.com.` becomes `Extra money was deducted for order [ORDER_ID], please check the bill. Contact [EMAIL]` (the brief's email pattern also consumes the final full stop).

### C3. The two prompts

Both contain the word `CLASSIFY`, list the four categories (billing, delivery, technical, fraud) and require a JSON-only reply `{"category": "...", "reason": "..."}`.

- Zero-shot (24 words, 32 estimated tokens): `CLASSIFY the support ticket into exactly one category: billing, delivery, technical or fraud. Reply with JSON only, in the form {"category": "...", "reason": "..."}.` It does not contain `EXAMPLES:`.
- Few-shot (94 words, 123 estimated tokens): the same text followed by a line `EXAMPLES:` and four labelled examples, one per category and including a fraud example. The examples are written in fresh wording and are not copies of any ticket in the file.

### C4. Resilient calls

Every model call goes through `call_with_retry`: at most 3 retries per ticket; the wait before retry number n is `BASE_DELAY x 2**n x random.uniform(0.9, 1.15)` with `BASE_DELAY = 0.05`; one log line per retry; `mock_llm.reset()` is called immediately before each full run; the jitter generator is seeded (42) so the log is reproducible.

| Run | TransientLLMErrors | Retries logged | Permanent errors | Retry numbers seen | Waits |
|---|---|---|---|---|---|
| zero-shot | 19 (reference 19) | 19 | 0 | only 0 | 0.0453 to 0.0562 s |
| few-shot | 19 | 19 | 0 | only 0 | 0.0453 to 0.0562 s |

Every wait lies in the allowed range for retry 0, 0.045 to 0.0575 s (asserted in code). Sample log lines (full log in `reports/c1_retry_log.txt`):

```
retry ticket=T0002 retry_number=0 wait=0.0530s
retry ticket=T0116 retry_number=0 wait=0.0453s
retry ticket=T0153 retry_number=0 wait=0.0484s
```

### C5. Output validation

Parse the reply as JSON and accept it if `category` is one of the four values (`classified_json`). If it is not valid JSON, take `re.findall(r"[a-z]+", text.lower())` and accept it if exactly one allowed category word is among them (`classified_keyword`). Everything else, including valid JSON with a category outside the list, is `flagged_invalid_output` and goes to human review; the code never guesses.

| Status | Count (full run over the 156 eligible tickets) | Reference |
|---|---|---|
| `classified_json` | 135 | 135 |
| `classified_keyword` | 12 | 12 |
| `flagged_invalid_output` | 9 | 9 |

The 12 keyword replies are the mock's chatty answers `The customer appears to have a <category>-related problem.` Measured on the same 156 replies, a plain whitespace split gives 135 / 0 / 21 instead of 135 / 12 / 9 because it cannot split `billing-related`; this is why the regular-expression extraction is used. Both prompts give the same 135 / 12 / 9.

### C6. Zero-shot versus few-shot (no token budget)

| Prompt | Tickets classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|---|
| zero-shot | 147 | 126 | 85.7% | 9,400 | 63.9 |
| few-shot | 147 | 147 | 100.0% | 23,596 | 160.5 |

(Tokens per classified ticket = total tokens of all model calls divided by the 147 classified tickets.) Zero-shot confusion table (rows true, columns predicted):

| true \ predicted | billing | delivery | technical | fraud |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

All 21 zero-shot errors are fraud tickets predicted as billing (11) or technical tickets predicted as billing (10), matching the reference, and every one of them contains a billing keyword such as `charged` or `payment`. Few-shot makes 0 errors.

**When the extra few-shot tokens pay off.** The few-shot prompt costs 160.5 tokens per classified ticket against 63.9 for zero-shot (+96.6, about 151% more) and removes all 21 mistakes, lifting accuracy from 85.7% to 100%. The mistakes are systematic rather than random: charged-or-payment wording pulls fraud and technical tickets into billing, and a mis-filed fraud ticket misses the escalation policy's 2-hour security deadline, so for such high-stakes categories about 97 extra tokens per ticket is cheap insurance. When tickets are unambiguous and a wrong label is cheap to correct, zero-shot is enough. Caveat: the mock changes its keyword priority when the system prompt contains `EXAMPLES:`, so the 100% measures this mock's behaviour and not in-context learning; with a real model the gain must be re-measured on held-out tickets.

### C7. Budget, routing and outputs

**(a) Budget demo (`outputs/triage_budget_demo.csv`).** `TOKEN_BUDGET = 6000`, zero-shot prompt (32 estimated tokens): the budget is checked before each model call and the run stopped calling after 100 calls and 6,038 tokens, so **56 tickets** ended `budget_exhausted` and were routed to human review (reference: 51 at about 29 prompt tokens and 58 at about 33, so 56 at 32 is consistent). The last allowed call overshoots slightly (38 tokens here), because the check happens before the call.

**(b) Final output (`outputs/triage_results.csv`).** Few-shot prompt with `TOKEN_BUDGET = 100000` (23,596 tokens spent, budget never reached), 164 rows with the columns `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`. Routing: skipped / flagged / invalid / exhausted go to `human_review` (NORMAL); else fraud goes to `security_team` (HIGH); else a Premium customer goes to `premium_desk` (HIGH); else `support_queue` (NORMAL); a customer without a segment row counts as not Premium (none occurred).

| Route | Priority | Tickets |
|---|---|---|
| security_team | HIGH | 34 |
| premium_desk | HIGH | 11 |
| support_queue | NORMAL | 102 |
| human_review | NORMAL | 17 |

The routing was re-derived independently from the output file and matches for all 164 rows.

**Which outputs were rule-validated and which went to humans.** 147 tickets (135 `classified_json` + 12 `classified_keyword`) passed the output rules and were routed automatically. 17 went to human review: 4 empty, 4 too long, 9 with an invalid model answer, 0 for budget.

---

## Part D

Results come from `src/d1_rag_eval.py` (tasks 1 to 4) and `src/d2_agent_run.py` (tasks 5 to 9); the library code is `src/rag.py`, `src/agent_tools.py` and `src/agent.py`.

### D1. Chunking

Each of the five policy files is split into chunks of 40 words with an overlap of 10 (step 30; the first chunk covers words 0 to 39; stop after the first chunk whose end reaches the last word); the source file name is stored with every chunk.

| Document | Words | Chunks (overlap 10) | Reference | Chunks (overlap 0) |
|---|---|---|---|---|
| refund_policy.txt | 119 | 4 | 4 | 3 |
| delivery_policy.txt | 100 | 3 | 3 | 3 |
| escalation_policy.txt | 78 | 3 | 3 | 2 |
| payment_policy.txt | 86 | 3 | 3 | 3 |
| privacy_policy.txt | 68 | 2 | 2 | 2 |
| **total** | | **15** | **15** | 13 |

### D2. Vectorise and retrieve

`TfidfVectorizer()` is fitted on the 15 chunk texts (vocabulary 200 terms); a query is transformed with the same fitted vectorizer and chunks are ranked by `cosine_similarity`; `retrieve(query, k=3)` returns dictionaries with `source`, `score`, `text`. For the query `refund policy late delivery` the top three chunks all come from `refund_policy.txt` (scores 0.2719, 0.2514, 0.2454), as in the reference.

TF (term frequency) measures how often a word occurs inside one chunk, that is how much the chunk is about it; IDF (inverse document frequency) measures how rare the word is across all chunks (the log of chunks divided by chunks containing it), so common words count little and distinctive ones count a lot. Limitation: TF-IDF matches words, not meaning. The query `reimbursement tardy parcel` means a late-delivery refund but shares no word with the policy vocabulary (checked in code), so every score is 0.0 and the "top" chunks are just the first three in file order, all from `delivery_policy.txt`.

### D3. Augment and answer

`answer(question)` retrieves three chunks and sends two messages to `mock_llm.chat`: a system message containing `POLICY ASSISTANT` (answer only from the supplied context, otherwise say the information could not be found) and a user message of exactly the form `CONTEXT:`, newline, the three chunks each preceded by `[<source file>]` on its own line, newline, `QUESTION:`, newline, the question. It returns the model's text and the sources used. Example: "Who must approve a refund above INR 1000?" returns `A refund above INR 1000 must be approved by a human support lead before money is returned.` with sources `refund_policy.txt` x 3.

### D4. Retrieval and grounding evaluation (`data/raw/policy_eval.csv`: 12 answerable, 2 not covered)

| Setting | Chunks | Expected document in top-3 (of 12) | Answer contains expected fact (of 12) | Unanswerable questions given the fallback `I could not find...` (of 2) |
|---|---|---|---|---|
| chunk 40, overlap 10 | 15 | **12 / 12** | **12 / 12** | **2 / 2** |
| chunk 40, overlap 0 | 13 | 12 / 12 | 10 / 12 | 2 / 2 |

The overlap-10 results equal the reference (12/12, 12/12, 2/2). With overlap 0, **two answerable questions break**:

1. *"What delivery time makes a completed order eligible for a full refund?"* returns only `receives a full refund of the order amount.` instead of the sentence containing `more than 45 minutes`.
2. *"How soon is money returned when a payment fails but is deducted?"* returns `If a payment fails but money is deducted, the amount is returned automatically` (glued to an unrelated fragment) and loses `48 hours`.

**Why:** retrieval is not the problem (the expected document is still in the top-3, and the fact phrase is whole inside one chunk). The problem is that the sentence carrying the answer (words 27 to 47 of the refund policy, words 27 to 42 of the payment policy) is cut by the 40-word boundary into two fragments in two different chunks. The model sees two half-sentences and quotes whichever overlaps the question more, which is not always the half with the fact. With a 10-word overlap the boundary words are repeated in both chunks, so a complete sentence survives in at least one of them; that is the purpose of the overlap.

### D5. Tools and the tool map

`TOOLS_SPEC` lists five tools, each with a name, a one-sentence description and a JSON-style parameter specification; `TOOL_MAP` maps the name to the Python function. The model asks for a tool by name with an arguments dictionary, and the agent looks the name up in `TOOL_MAP`.

| Tool | Parameters | What it does |
|---|---|---|
| `get_order` | `order_id` | reads `data/instamart.db` (parameterised SQL) and returns `{order_id, status, order_amount, delivery_minutes}` or `{"error": "order not found"}` |
| `search_policy` | `query` | returns the retrieval output of D2 under the key `results` |
| `calculate_refund` | `order_amount`, `delivery_minutes` | delivered in more than 45 minutes: 100% of the amount; otherwise 0 |
| `issue_refund` | `order_id`, `refund_inr` | appends one JSON line to `outputs/refund_ledger.jsonl`; **refuses amounts above INR 1000 itself** |
| `escalate_to_human` | `order_id`, `refund_inr` | appends one JSON line to the ledger (a human lead must approve) |

Each ledger line has a timestamp, the tool name, the run id, the order id and the amount. A tool failure (bad arguments, unknown tool) becomes an error observation instead of crashing the loop.

### D6. The ReAct loop (`run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10)`)

The messages start with a system message containing the word `AGENT` and the user message; each model reply either is the final answer or requests a tool, which is executed and appended as an assistant message plus a tool message. The status is the word before the colon of the final answer. The result holds `status, final_answer, steps, llm_calls, tokens_used` (plus the run id and the refund amount). All 7 rows of `agent_scenarios.csv` were run with the default limits and appended to `outputs/agent_runs.jsonl` (one line per run with every step's tool, arguments and observation).

| Scenario | Status | Refund (INR) | Expected | Model calls | Tokens used | Tools executed |
|---|---|---|---|---|---|---|
| S1 | REFUND | 128.25 | 128.25 | 5 | 1,122 | get_order, search_policy, calculate_refund, issue_refund |
| S2 | REFUND | 135.00 | 135.00 | 5 | 1,122 | same four |
| S3 | ESCALATE | 1,999.00 | 1,999.00 | 5 | 1,109 | get_order, search_policy, calculate_refund, escalate_to_human |
| S4 | DENIED | 0.00 | 0.00 | 4 | 785 | get_order, search_policy, calculate_refund |
| S5 | DENIED | 0.00 | 0.00 | 3 | 472 | get_order, search_policy |
| S6 | NOT_FOUND | 0.00 | 0.00 | 2 | 162 | get_order |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 0.00 | 6 | 3,270 | search_policy x 5 (the 6th request was not executed) |

S1 takes 5 model calls as in the reference. S7 uses 3,270 estimated tokens for all six model calls, below the default 6,000, so it ends by iteration cap and not earlier. The ledger after these runs holds exactly the issued and escalated actions: S1 and S2 (issue), S3 (escalate).

### D7. Stop safely

Three limits and a graceful fallback; every fallback returns the steps completed so far and a short human-handoff summary in `final_answer`.

| Limit | Demonstration | Status | Detail |
|---|---|---|---|
| (a) `max_iterations` = 6 | S7 with default limits | `FALLBACK_MAX_ITERATIONS` | 6 model calls, 3,270 tokens; the 6th requested tool is not executed |
| (b) `token_limit` | S1 with `token_limit=200` | `FALLBACK_TOKEN_LIMIT` | stopped after 3 model calls and 489 tokens without executing the tool just requested (tools run: get_order, search_policy); no refund issued |
| (c) `time_limit_seconds` | S1 with `time_limit_seconds=0` | `FALLBACK_TIMEOUT` | stops before the first model call (0 calls, 0 steps) |

The two demonstration runs and the crash-and-resume run (D8) are recorded in `outputs/agent_demos.jsonl`, so `outputs/agent_runs.jsonl` keeps exactly one line per scenario (7 lines). The default `token_limit=6000` does not stop S7 early (see D6). When the iteration cap is reached with a tool still requested, that tool is deliberately not executed: a refund tool must not run when the loop can no longer observe its result (same rule as the token limit).

### D8. Resumable run

After every tool observation the agent writes `state/<run_id>.json` (atomic write) with the run id, scenario id, the full message list, tokens used, step number and the list of tools already executed (plus the limits and steps). `resume_agent(run_id)` loads that file and continues; a finished run is returned as stored.

Demonstration with S1 (`state/S1-crash-demo.json`): a simulated crash is raised right after the second tool execution (after its checkpoint is saved). The checkpoint showed `step = 2`, `tools_executed = [get_order, search_policy]`, 2 model calls, 192 tokens, 6 messages and no ledger entry yet. After `resume_agent`:

- **Tools executed across the crash and the resume, in order: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`** (four executions, none repeated, confirmed by an independent execution log and not only by the checkpoint list).
- Final status `REFUND` with INR 128.25, 5 model calls and 1,122 tokens, identical to the uninterrupted S1 run.
- Exactly one ledger line for that run. (The ledger therefore shows ORD-100048 twice, under run ids `S1-main` and `S1-crash-demo`: these are two separate simulated requests for the same scenario, not a double refund of one request; within one run a repeated tool call adds nothing.) Calling `resume_agent` again on the finished run executes nothing.

### D9. ReAct versus plan-and-execute (203 words)

Measured ReAct cost: S1 5 model calls / 1122 tokens, S4 4, S5 3, S6 2 / 162 tokens; every call re-sends the whole history, so tokens grow faster than calls. A plan-and-execute design would ask once for the full step list and run the tools in code: for S1 that is about 2 calls and roughly 360 tokens (my estimate from the same prompts and observations; the mock has no such mode), cheaper than 1122. The saving disappears when an observation is unexpected. For S6 (NOT_FOUND) and S5 (cancelled order) the fixed plan is already wrong after step 1 or 2, so it needs a replanning call: about 3 calls against ReAct's 2 and 3. A plan written before seeing the order also contains issue_refund before anyone knows the order is eligible, so every plan needs an extra guard that blocks side-effecting steps. I would choose ReAct for this refund workflow: the paths are short (at most 5 calls), the branches depend on what each lookup returns (not found, cancelled, on time, above INR 1000), money moves only after the evidence is seen, and the iteration, token and time limits bound the worst case (S7). Plan-and-execute pays off for long, predictable pipelines.

---

## Limitations and Next Steps

**Evaluation honesty**

- **The retention list is illustrative and not a leakage-free evaluation.** The Premium flag is a post-hoc whole-window label: it comes from the April-September RFM, which includes the September window that defines the lapse label. Only 1 of the 10 listed customers actually lapsed, and the top probabilities are only 0.45 to 0.49, so the list should not be used to judge the model.
- The lapse model is evaluated on one stratified split of 232 test customers (89 lapsed), so F1 0.655 carries a wide uncertainty; no probability calibration or cost-based decision threshold was tuned. Next step: repeated or time-based validation and a threshold chosen from the real cost of an offer versus a lost customer.
- The few-shot accuracy of 100% reflects how `mock_llm.py` behaves (it changes keyword priority when `EXAMPLES:` is present); it is not evidence of in-context learning. The policy evaluation has only 14 questions, so 12/12 is a smoke test, not a precise accuracy.
- All data are synthetic; the segments, ages and delivery times follow the generator's rules and say nothing about real customers.

**Behaviour that follows the brief but needs a decision before production**

- **Fraud tickets can wait.** 5 true fraud tickets (T0116, T0016, T0084, T0148, T0076) ended in `human_review` at NORMAL priority because their model answer failed validation; the escalation policy asks for fraud to reach the security team within 2 hours. Next step: a keyword safety net (for example `unauthorized`, `fraud`, `hacked`) that raises priority before human review.
- **Imputed delivery times hide late deliveries.** 774 completed orders had their delivery time filled with a category median of about 25 minutes, so the refund agent will never see them as late (more than 45 minutes). Next step: keep an `is_imputed` flag in the database and route such orders to a human for refund decisions.
- **No ownership or duplicate-refund checks.** `get_order` returns only the four fields the brief specifies, so the agent does not verify that the requester owns the order, does not check whether the order was already refunded, and does not enforce the 7-day window (the brief treats all requests as in-window). The ledger is idempotent only within one run. Next step: pass the customer id, store refund state per order in the database, and make `issue_refund` idempotent per order.
- The refund amount is computed in code (`calculate_refund`) and the model never decides money: this is deliberate. The token budget is checked before each call, so one call can overshoot it slightly.
- After a resume the time limit is measured from the start of the resumed process, not from the original start.

**Retrieval and modelling**

- TF-IDF matches words, not meaning (shown in D2), and fixed 40-word chunks can cut an answer sentence (shown in D4). Next step: sentence-aware chunks, hybrid lexical plus embedding retrieval, and a larger labelled question set.
- Customer age was imputed with a city median, which is crude; city, gender and category are one-hot encoded with `handle_unknown="ignore"`, which would silently ignore a new city.
- Next steps overall: put a real language model behind the same `chat` interface and re-measure Parts C and D on held-out tickets; add monitoring of status counts (invalid output rate, budget exhaustion); run the 27 unit tests and the reference checks in continuous integration.
