"""Part D, tasks 1-4 -- chunk counts, retrieval check, grounded answers, evaluation with overlap 10 and overlap 0."""
import re
import pandas as pd
from common import POLICIES, RAW, Report
import rag

rep = Report("d1_rag_eval")
pd.set_option("display.width", 170)

# ------------------------------------------------------------------ task 1: chunking
rep("TASK 1 -- chunking (40 words, overlap 10 => step 30)")
idx10 = rag.PolicyIndex(40, 10)
counts = pd.Series([c["source"] for c in idx10.chunks]).value_counts().sort_index()
words_per_doc = {p.name: len(p.read_text(encoding="utf-8").split()) for p in sorted(POLICIES.glob("*.txt"))}
rep(f"total chunks = {len(idx10.chunks)} (reference 15)")
for src, n in counts.items():
    spans = [(c["start_word"], c["end_word"]) for c in idx10.chunks if c["source"] == src]
    rep(f"  {src:<22} {words_per_doc[src]:>3} words -> {n} chunks {spans}")
assert len(idx10.chunks) == 15 and counts.to_dict() == {"delivery_policy.txt": 3, "escalation_policy.txt": 3, "payment_policy.txt": 3,
                                                       "privacy_policy.txt": 2, "refund_policy.txt": 4}
first = idx10.chunks[0]
assert first["start_word"] == 0 and first["end_word"] == 39
idx0 = rag.PolicyIndex(40, 0)
rep(f"for comparison, overlap 0 gives {len(idx0.chunks)} chunks: " + ", ".join(f"{k.split('_')[0]} {v}" for k, v in
    pd.Series([c['source'] for c in idx0.chunks]).value_counts().sort_index().items()))

# ------------------------------------------------------------------ task 2: retrieval
rep("\nTASK 2 -- TF-IDF retrieval (TfidfVectorizer() fitted on the 15 chunk texts, cosine similarity)")
q = "refund policy late delivery"
hits = rag.retrieve(q, k=3)
for h in hits:
    rep(f"  query '{q}' -> {h['source']:<20} score {h['score']:.4f}  {h['text'][:70]}...")
assert [h["source"] for h in hits] == ["refund_policy.txt"] * 3
rep(f"vocabulary size: {len(idx10.vectorizer.vocabulary_)} terms; matrix shape {idx10.matrix.shape}")
rep("TF (term frequency) measures how often a word occurs inside one chunk, i.e. how much the chunk is about it; IDF (inverse document frequency) measures how rare the word")
rep("is across all chunks (log of chunks / chunks containing it), so common words such as 'the' or 'order' count little and distinctive ones such as 'escalate' or 'rain' count a lot.")
paraphrase = "reimbursement tardy parcel"
analyzer = idx10.vectorizer.build_analyzer()
shared = sorted(set(analyzer(paraphrase)) & set(idx10.vectorizer.vocabulary_))
ph = rag.retrieve(paraphrase, k=3)
rep(f"Limitation: TF-IDF matches words, not meaning. '{paraphrase}' means a late-delivery refund, but its words shared with the policy vocabulary: {shared or 'none'}; "
    f"so every score is {[h['score'] for h in ph]} and the 'top' chunks are just the first three in file order ({[h['source'] for h in ph]}). "
    "Asking for 'money back' instead of 'refund' loses the strongest refund-policy signal in the same way.")
assert not shared and all(h["score"] == 0.0 for h in ph)

# ------------------------------------------------------------------ task 3: augment and answer
rep("\nTASK 3 -- augment and answer (two messages: system with POLICY ASSISTANT, user = CONTEXT / chunks / QUESTION)")
demo_q = "Who must approve a refund above INR 1000?"
demo_hits = rag.retrieve(demo_q)
msg = rag.build_user_message(demo_q, demo_hits)
rep("user message sent to the model (first lines):")
for line in msg.splitlines()[:4] + ["..."] + msg.splitlines()[-2:]:
    rep("   | " + (line[:110] + ("..." if len(line) > 110 else "")))
res = rag.answer(demo_q)
rep(f"answer: {res['answer']!r}")
rep(f"sources used: {res['sources']}")

# ------------------------------------------------------------------ task 4: evaluation
ev = pd.read_csv(RAW / "policy_eval.csv", dtype=str, keep_default_na=False)
assert len(ev) == 14


def evaluate(index, label):
    rows = []
    for r in ev.itertuples():
        hits = rag.retrieve(r.question, k=3, index=index)
        res = rag.answer(r.question, index=index)
        answerable = r.expected_doc != "NONE"
        words = {c["source"]: None for c in index.chunks}
        fact_whole_in_a_chunk = answerable and any(r.expected_fact in c["text"] for c in index.chunks if c["source"] == r.expected_doc)
        rows.append({"question": r.question, "answerable": answerable, "expected_doc": r.expected_doc,
                     "doc_in_top3": answerable and r.expected_doc in [h["source"] for h in hits],
                     "fact_in_answer": answerable and r.expected_fact in res["answer"],
                     "fact_whole_in_a_chunk": fact_whole_in_a_chunk,
                     "fallback": (not answerable) and res["answer"].startswith("I could not find"),
                     "answer": res["answer"], "expected_fact": r.expected_fact})
    df = pd.DataFrame(rows)
    a, u = df[df["answerable"]], df[~df["answerable"]]
    summary = {"label": label, "chunks": len(index.chunks), "doc_in_top3": int(a["doc_in_top3"].sum()), "n_answerable": len(a),
               "fact_in_answer": int(a["fact_in_answer"].sum()), "fallback": int(u["fallback"].sum()), "n_unanswerable": len(u)}
    return df, summary


rep("\nTASK 4 -- evaluation on data/raw/policy_eval.csv (12 answerable + 2 not covered)")
df10, s10 = evaluate(idx10, "chunk 40 / overlap 10")
df0, s0 = evaluate(idx0, "chunk 40 / overlap 0")
table = pd.DataFrame([{"setting": s["label"], "chunks": s["chunks"],
                       "expected doc in top-3": f"{s['doc_in_top3']}/{s['n_answerable']}",
                       "answer contains expected fact": f"{s['fact_in_answer']}/{s['n_answerable']}",
                       "unanswerable -> fallback": f"{s['fallback']}/{s['n_unanswerable']}"} for s in (s10, s0)])
rep(table.to_string(index=False))
assert (s10["doc_in_top3"], s10["fact_in_answer"], s10["fallback"]) == (12, 12, 2), s10
assert s0["fact_in_answer"] < 12 or s0["doc_in_top3"] < 12, "overlap 0 must break at least one answerable question"

rep("\nper-question detail, overlap 10:")
for r in df10.itertuples():
    ok = (r.doc_in_top3 and r.fact_in_answer) if r.answerable else r.fallback
    rep(f"  {'OK  ' if ok else 'FAIL'} {r.question[:70]:<70} -> {r.answer[:60]!r}")

broken = df0[df0["answerable"] & ~(df0["doc_in_top3"] & df0["fact_in_answer"])]
rep(f"\nwith overlap 0, {len(broken)} answerable question(s) break:")
for r in broken.itertuples():
    doc_text = " ".join((POLICIES / r.expected_doc).read_text(encoding="utf-8").split())   # same normalisation as the chunker
    char = doc_text.find(r.expected_fact)
    assert char >= 0
    first_word = len(doc_text[:char].split())                                   # word index where the fact starts
    last_word = len(doc_text[:char + len(r.expected_fact)].split()) - 1         # word index where it ends
    # The model (and the mock) reasons over whole SENTENCES, so find the sentence that carries the fact and its word span.
    start = 0
    for sentence in re.split(r"(?<=[.])\s+", doc_text):
        n_words = len(sentence.split())
        if r.expected_fact in sentence:
            s_first, s_last = start, start + n_words - 1
            break
        start += n_words
    c_first, c_last = s_first // 40, s_last // 40                               # chunk numbers when step = 40 (no overlap)
    sentence_split = c_first != c_last
    rep(f"  Q: {r.question}")
    rep(f"     fact {r.expected_fact!r} = words {first_word}-{last_word} of {r.expected_doc} -> whole inside one chunk: {r.fact_whole_in_a_chunk}; "
        f"expected doc in top-3: {r.doc_in_top3}")
    rep(f"     but the SENTENCE carrying it spans words {s_first}-{s_last}, i.e. chunk {c_first} and chunk {c_last} -> "
        f"{'cut by the chunk boundary at word ' + str(40 * c_last) if sentence_split else 'inside one chunk'}")
    assert sentence_split, "expected the failure to come from a sentence cut by the chunk boundary"
    rep(f"     answer given: {r.answer!r}")
assert len(broken) >= 1
rep("Explanation: retrieval is not the problem (the expected document is still in the top-3 and the fact phrase is whole inside one chunk). What breaks is the")
rep("ANSWER SENTENCE: a 40-word cut with no overlap slices the sentence that holds the answer into two fragments in two different chunks. The model then sees two")
rep("half-sentences and quotes whichever overlaps the question more -- the tail 'receives a full refund of the order amount.' for the first question (it loses the")
rep("'more than 45 minutes' head) and the head '... returned automatically' for the second (it loses the 'within 48 hours' tail, and is even glued to an unrelated chunk).")
rep("With a 10-word overlap the words around the boundary are repeated in both chunks, so a complete sentence survives in at least one of them -- that is its purpose.")
rep.save()
