"""Part C runner -- guardrails, redaction, prompts, retries, validation, zero-shot vs few-shot, budget demo, routed output."""
import pandas as pd
from common import OUTPUTS, RAW, REPORTS, Report        # first: sets sys.path so mock_llm.py (repo root) is importable
from mock_llm import est_tokens
import triage as T

rep = Report("c1_triage")
pd.set_option("display.width", 160)

tickets = pd.read_csv(RAW / "tickets.csv", dtype=str, keep_default_na=False)     # empty texts stay empty strings
assert len(tickets) == 164

# ------------------------------------------------------------------ task 1: input guardrails
tok = tickets["text"].map(est_tokens)
empty = tickets["text"].str.strip() == ""
long_ = tok > T.MAX_INPUT_TOKENS
rep("TASK 1 -- input guardrails (checked before any model call)")
rep(f"empty / whitespace-only tickets -> skipped_empty: {int(empty.sum())} (reference 4)")
rep(f"tickets with est_tokens > {T.MAX_INPUT_TOKENS} -> flagged_long: {int(long_.sum())} (reference 4); their estimated tokens: {sorted(set(tok[long_]))} (reference 734)")
eligible = int((~empty & ~long_).sum())
rep(f"eligible for the model: {eligible} (reference 156)")
assert (int(empty.sum()), int(long_.sum()), eligible) == (4, 4, 156) and set(tok[long_]) == {734}

# ------------------------------------------------------------------ task 2: redaction
totals = {"email": 0, "phone": 0, "order_id": 0}
redacted_texts = []
for text in tickets["text"]:
    new, counts = T.redact(text)
    redacted_texts.append(new)
    for k, v in counts.items():
        totals[k] += v
rep("\nTASK 2 -- redaction (email -> [EMAIL], then phone -> [PHONE], then order id -> [ORDER_ID])")
rep(f"substitutions: emails {totals['email']} (reference 55) | phones {totals['phone']} (reference 91) | order ids {totals['order_id']} (reference 132) | "
    f"total {sum(totals.values())} (reference 278)")
left = sum(len(p.findall(t)) for t in redacted_texts for _, p, _ in T.REDACTIONS)
rep(f"re-applying the three patterns to the redacted texts finds {left} matches (reference 0)")
assert (totals["email"], totals["phone"], totals["order_id"], sum(totals.values()), left) == (55, 91, 132, 278, 0)
example = tickets.loc[tickets["text"].str.contains("@") & tickets["text"].str.contains("ORD-")].iloc[0]
rep(f"example {example['ticket_id']} before: {example['text']}")
rep(f"example {example['ticket_id']} after : {T.redact(example['text'])[0]}")

# ------------------------------------------------------------------ task 3: prompts
rep("\nTASK 3 -- two prompts")
for name, p in [("zero-shot", T.ZERO_SHOT_PROMPT), ("few-shot", T.FEW_SHOT_PROMPT)]:
    ok = "CLASSIFY" in p and all(c in p for c in T.CATEGORIES) and '{"category": "...", "reason": "..."}' in p and "JSON only" in p
    rep(f"{name:<10} {len(p.split()):>3} words = {est_tokens(p):>3} estimated tokens | CLASSIFY + 4 categories + JSON-only form present: {ok}")
    assert ok
assert "EXAMPLES:" not in T.ZERO_SHOT_PROMPT
assert "EXAMPLES:" in T.FEW_SHOT_PROMPT and T.FEW_SHOT_PROMPT.count("Ticket:") >= 2 and '"category": "fraud"' in T.FEW_SHOT_PROMPT
rep(f"few-shot prompt: line 'EXAMPLES:' + {T.FEW_SHOT_PROMPT.count('Ticket:')} labelled examples (one per category, including fraud); zero-shot has no 'EXAMPLES:'")

# ------------------------------------------------------------------ tasks 4-6: full runs
def summarise(run):
    df = pd.DataFrame(run.results)
    classified = df[df["status"].isin(T.RULE_VALIDATED)]
    correct = int((classified["category"] == classified["true_category"]).sum())
    return df, classified, correct

runs = {}
for name, prompt in [("zero-shot", T.ZERO_SHOT_PROMPT), ("few-shot", T.FEW_SHOT_PROMPT)]:
    runs[name] = T.run_pipeline(tickets, prompt, name)

rep("\nTASK 4 -- resilient calls (mock_llm.reset() before each run; <=3 retries per ticket; wait = 0.05 x 2^n x uniform(0.9, 1.15))")
for name, run in runs.items():
    df = pd.DataFrame(run.results)
    n_err = int((df["status"] == "llm_error").sum())
    retry_numbers = sorted({n for n, _ in run.waits})
    rep(f"{name:<10} TransientLLMErrors {run.transient_errors} (reference 19) | retries logged {len(run.retry_lines)} | permanent errors {n_err} | "
        f"retry numbers seen {retry_numbers} | waits {min(w for _, w in run.waits):.4f}..{max(w for _, w in run.waits):.4f} s "
        f"(allowed for retry 0: 0.0450..0.0575 s)")
    assert run.transient_errors == 19 and len(run.retry_lines) == 19 and n_err == 0
zs = runs["zero-shot"]
rep("first three retry log lines of the zero-shot run:")
for line in zs.retry_lines[:3]:
    rep("   " + line)
(REPORTS / "c1_retry_log.txt").write_text(
    "".join(f"== {n} run ==\n" + "\n".join(r.retry_lines) + "\n" for n, r in runs.items()), encoding="utf-8")
rep("full retry logs of both runs: reports/c1_retry_log.txt")

rep("\nTASK 5 -- validation of every answer (JSON first, then keyword extraction with re.findall(r'[a-z]+'), never guess)")
for name, run in runs.items():
    df, classified, correct = summarise(run)
    c = df["status"].value_counts().to_dict()
    rep(f"{name:<10} classified_json {c.get('classified_json', 0)} | classified_keyword {c.get('classified_keyword', 0)} | "
        f"flagged_invalid_output {c.get('flagged_invalid_output', 0)} | skipped_empty {c.get('skipped_empty', 0)} | flagged_long {c.get('flagged_long', 0)}"
        f"   (reference 135 / 12 / 9)")
    assert (c["classified_json"], c["classified_keyword"], c["flagged_invalid_output"]) == (135, 12, 9)
def naive_validate(content):
    """The wrong way: JSON first, then a plain whitespace split (so 'billing-related' is one unmatched word)."""
    import json
    try:
        parsed = json.loads(content)
        return "classified_json" if isinstance(parsed, dict) and parsed.get("category") in T.CATEGORIES else "flagged_invalid_output"
    except ValueError:
        hits = {w for w in content.lower().split() if w in T.CATEGORIES}
        return "classified_keyword" if len(hits) == 1 else "flagged_invalid_output"

replies = [r["reply"] for r in runs["zero-shot"].results if r["reply"]]
naive_counts = pd.Series([naive_validate(x) for x in replies]).value_counts().to_dict()
rep(f"Measured on the same {len(replies)} replies, a plain whitespace split gives classified_json {naive_counts.get('classified_json', 0)} / "
    f"classified_keyword {naive_counts.get('classified_keyword', 0)} / flagged_invalid_output {naive_counts.get('flagged_invalid_output', 0)} "
    "instead of 135 / 12 / 9, because it cannot split 'billing-related'.")
assert (naive_counts["classified_json"], naive_counts.get("classified_keyword", 0), naive_counts["flagged_invalid_output"]) == (135, 0, 21)

rep("\nTASK 6 -- zero-shot vs few-shot (no token budget)")
rows = []
for name, run in runs.items():
    df, classified, correct = summarise(run)
    rows.append({"prompt": name, "classified": len(classified), "correct": correct, "accuracy_pct": round(100 * correct / len(classified), 1),
                 "total_tokens": run.tokens, "tokens_per_classified_ticket": round(run.tokens / len(classified), 1),
                 "tokens_per_call": round(run.tokens / (len(run.called_ticket_ids)), 1)})
cmp_ = pd.DataFrame(rows)
rep(cmp_.to_string(index=False))
zs_acc, fs_acc = cmp_.loc[0, "accuracy_pct"], cmp_.loc[1, "accuracy_pct"]
assert abs(zs_acc - 85.7) <= 0.5 and abs(fs_acc - 100.0) <= 0.5 and cmp_["classified"].tolist() == [147, 147]

_, zs_cls, _ = summarise(runs["zero-shot"])
confusion = pd.crosstab(zs_cls["true_category"], zs_cls["category"]).reindex(index=T.CATEGORIES, columns=T.CATEGORIES, fill_value=0)
confusion.index.name, confusion.columns.name = "true \\ predicted", None
rep("\nzero-shot confusion table (rows = true category, columns = predicted; classified tickets only)")
rep(confusion.to_string())
errors = zs_cls[zs_cls["category"] != zs_cls["true_category"]].groupby(["true_category", "category"]).size()
rep("zero-shot errors: " + ", ".join(f"true {a} predicted {b}: {n}" for (a, b), n in errors.items()) + f" (total {int(errors.sum())})")
assert errors.to_dict() == {("fraud", "billing"): 11, ("technical", "billing"): 10}
wrong = zs_cls[zs_cls["category"] != zs_cls["true_category"]]
has_billing_word = wrong["text"].str.lower().str.contains(r"\b(?:charged|invoice|bill|refund|deducted|payment)\b")
rep(f"all {int(has_billing_word.sum())} of {len(wrong)} zero-shot errors contain a billing keyword (charged / payment ...) that the zero-shot mock checks first")
assert has_billing_word.all()
_, fs_cls, _ = summarise(runs["few-shot"])
rep(f"few-shot errors: {int((fs_cls['category'] != fs_cls['true_category']).sum())}")

extra = cmp_.loc[1, "tokens_per_classified_ticket"] - cmp_.loc[0, "tokens_per_classified_ticket"]
fixed = int(errors.sum())
rep("\nWHEN IS FEW-SHOT WORTH THE EXTRA TOKENS?")
rep(f"The few-shot prompt costs {cmp_.loc[1, 'tokens_per_classified_ticket']:.1f} tokens per classified ticket against {cmp_.loc[0, 'tokens_per_classified_ticket']:.1f} for zero-shot "
    f"(+{extra:.1f}, about {100 * extra / cmp_.loc[0, 'tokens_per_classified_ticket']:.0f}% more) and in return accuracy rises from {zs_acc}% to {fs_acc}%, i.e. all {fixed} of the")
rep(f"zero-shot mistakes disappear. The mistakes are not random: zero-shot sends every fraud and technical ticket that mentions a charge or payment to billing, and a")
rep(f"mis-filed fraud ticket misses the 2-hour security SLA, so for the high-stakes categories {extra:.0f} extra tokens per ticket is cheap insurance. When tickets are")
rep("short, the categories are unambiguous and a wrong label is cheap to fix, the cheaper zero-shot prompt is enough.")
rep("Honest caveat: the mock switches its keyword priority when the system prompt contains 'EXAMPLES:', so the 100% measures this mock's behaviour,")
rep("not in-context learning; with a real model the gain must be re-measured on held-out tickets.")

# ------------------------------------------------------------------ task 7: budget demo and routed output
segments_df = pd.read_csv(OUTPUTS / "customer_segments.csv")
segments = {r.customer_id: (r.segment_name, bool(r.is_premium)) for r in segments_df.itertuples()}
COLUMNS = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]

DEMO_BUDGET = 6000
demo = T.run_pipeline(tickets, T.ZERO_SHOT_PROMPT, "zero-shot, budget demo", token_budget=DEMO_BUDGET)
demo_out = pd.DataFrame([T.route_ticket(r, segments) for r in demo.results])[COLUMNS]
demo_out.to_csv(OUTPUTS / "triage_budget_demo.csv", index=False)
n_exh = int((demo_out["status"] == "budget_exhausted").sum())
first_exh = demo_out.index[demo_out["status"] == "budget_exhausted"][0]
rep("\nTASK 7a -- token budget demo (zero-shot prompt)")
rep(f"TOKEN_BUDGET = {DEMO_BUDGET}: tokens spent {demo.tokens} | tickets with status budget_exhausted = {n_exh} (at least one required; reference ~51 for a 29-token prompt) | "
    f"first exhausted row index {first_exh} | model calls made {len(demo.called_ticket_ids)}")
rep(f"every budget_exhausted ticket is routed to human_review: {bool((demo_out.loc[demo_out['status'] == 'budget_exhausted', 'route'] == 'human_review').all())}")
assert n_exh >= 1 and (demo_out.loc[demo_out["status"] == "budget_exhausted", "route"] == "human_review").all()
last_call_tokens = max(r["tokens_used"] for r in demo.results)
rep(f"the budget is checked BEFORE each call, so the last allowed call may overshoot it slightly: spent {demo.tokens} against a budget of {DEMO_BUDGET} "
    f"(overshoot {demo.tokens - DEMO_BUDGET}, at most one call = {last_call_tokens} tokens)")
assert DEMO_BUDGET <= demo.tokens < DEMO_BUDGET + last_call_tokens

FINAL_BUDGET = 100000
final = T.run_pipeline(tickets, T.FEW_SHOT_PROMPT, "few-shot, final", token_budget=FINAL_BUDGET)
final_out = pd.DataFrame([T.route_ticket(r, segments) for r in final.results])[COLUMNS]
final_out.to_csv(OUTPUTS / "triage_results.csv", index=False)
assert len(final_out) == 164 and final.tokens == runs["few-shot"].tokens        # same as the unbudgeted few-shot run
rep(f"\nTASK 7b -- outputs/triage_results.csv: few-shot, TOKEN_BUDGET = {FINAL_BUDGET}, {len(final_out)} rows, tokens spent {final.tokens} (budget not reached)")
rep("status counts : " + ", ".join(f"{k} {v}" for k, v in final_out["status"].value_counts().items()))
rep("route counts  : " + ", ".join(f"{k} {v}" for k, v in final_out["route"].value_counts().items()))
rep("priority      : " + ", ".join(f"{k} {v}" for k, v in final_out["priority"].value_counts().items()))
rep(pd.crosstab(final_out["route"], final_out["priority"]).to_string())

# consistency of routing with the stated rules, recomputed independently from the output file
chk = final_out.merge(segments_df[["customer_id", "is_premium"]], on="customer_id", how="left")
chk["is_premium"] = chk["is_premium"].fillna(False).astype(bool)
human = chk["status"].isin(T.HUMAN_STATUSES)
exp_route = pd.Series("support_queue", index=chk.index)
exp_route[chk["is_premium"]] = "premium_desk"
exp_route[chk["category"] == "fraud"] = "security_team"
exp_route[human] = "human_review"
assert (chk["route"] == exp_route).all()
assert (chk.loc[chk["route"].isin(["security_team", "premium_desk"]), "priority"] == "HIGH").all()
assert (chk.loc[~chk["route"].isin(["security_team", "premium_desk"]), "priority"] == "NORMAL").all()
no_seg = int((final_out["segment_name"] == "").sum() + final_out["segment_name"].isna().sum())
rep(f"routing re-derived independently from the file matches for all {len(chk)} rows; tickets whose customer has no segment row (treated as not premium): {no_seg}")

validated = int(final_out["status"].isin(T.RULE_VALIDATED).sum())
to_human = int((final_out["route"] == "human_review").sum())
rep(f"\nRULE-VALIDATED vs HUMAN: {validated} tickets passed output validation (classified_json {int((final_out['status'] == 'classified_json').sum())}, "
    f"classified_keyword {int((final_out['status'] == 'classified_keyword').sum())}) and were routed by rule; {to_human} went to human review "
    f"(skipped_empty {int((final_out['status'] == 'skipped_empty').sum())}, flagged_long {int((final_out['status'] == 'flagged_long').sum())}, "
    f"flagged_invalid_output {int((final_out['status'] == 'flagged_invalid_output').sum())}, budget_exhausted {int((final_out['status'] == 'budget_exhausted').sum())}).")
fraud_human = chk[(chk["route"] == "human_review")].merge(tickets[["ticket_id", "true_category"]], on="ticket_id")
fraud_human = fraud_human[fraud_human["true_category"] == "fraud"]
rep(f"RISK NOTE: {len(fraud_human)} true fraud ticket(s) ended in human_review at NORMAL priority because their model answer failed validation "
    f"({', '.join(fraud_human['ticket_id'])}); the brief's rules give them no HIGH priority -- see Limitations.")
rep.save()
