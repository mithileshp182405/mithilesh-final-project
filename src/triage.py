"""Part C -- support-ticket triage library: guardrails, PII redaction, retries, output validation, budget, routing.

Everything here is deterministic and offline: the only model is the supplied mock_llm.py.
The runner (c1_triage.py) uses these functions; tests/test_pipeline.py checks them in isolation.
"""
import json
import logging
import random
import re
import time

import common  # noqa: F401  (side effect: puts the repository root on sys.path so mock_llm.py can be imported)
import mock_llm
from mock_llm import est_tokens

# ------------------------------------------------------------------ constants fixed by the brief
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
JITTER_LOW, JITTER_HIGH = 0.9, 1.15
JITTER_SEED = 42            # makes the jittered waits reproducible from run to run

# Applied in exactly this order, with exactly these placeholders.
REDACTIONS = [
    ("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]"),
]

ZERO_SHOT_PROMPT = (
    'CLASSIFY the support ticket into exactly one category: billing, delivery, technical or fraud. '
    'Reply with JSON only, in the form {"category": "...", "reason": "..."}.'
)
# The examples are written in fresh wording (they are not copies of any ticket in tickets.csv).
FEW_SHOT_PROMPT = ZERO_SHOT_PROMPT + """
EXAMPLES:
Ticket: A transaction I never authorised appeared on my statement. -> {"category": "fraud", "reason": "unauthorised transaction"}
Ticket: The app keeps freezing whenever I try to pay. -> {"category": "technical", "reason": "app fault"}
Ticket: My invoice shows a higher total than the price I saw. -> {"category": "billing", "reason": "wrong amount"}
Ticket: The rider has not reached my address and the order is late. -> {"category": "delivery", "reason": "late order"}"""

# Statuses that never reach a category and therefore always go to a human.
HUMAN_STATUSES = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "llm_error"}
RULE_VALIDATED = {"classified_json", "classified_keyword"}

retry_logger = logging.getLogger("triage.retry")


# ------------------------------------------------------------------ privacy guardrail
def redact(text):
    """Return (redacted_text, {type: number_of_substitutions}) using the three brief patterns in order."""
    counts = {}
    for name, pattern, placeholder in REDACTIONS:
        text, n = pattern.subn(placeholder, text)
        counts[name] = n
    return text, counts


def contains_pii(text):
    """True if any of the three patterns still matches (used by the assertions)."""
    return any(p.search(text) for _, p, _ in REDACTIONS)


# ------------------------------------------------------------------ output validation
def validate_reply(content):
    """Return (status, category). Never guesses: anything ambiguous is flagged for a human."""
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        words = set(re.findall(r"[a-z]+", (content or "").lower()))
        hits = [c for c in CATEGORIES if c in words]
        if len(hits) == 1:
            return "classified_keyword", hits[0]
        return "flagged_invalid_output", ""
    category = parsed.get("category") if isinstance(parsed, dict) else None
    if isinstance(category, str) and category in CATEGORIES:
        return "classified_json", category
    return "flagged_invalid_output", ""          # valid JSON but wrong/missing category: no keyword fallback


# ------------------------------------------------------------------ resilient model call
class BudgetExhausted(Exception):
    pass


class Run:
    """State of one full pass over the tickets."""

    def __init__(self, label, system_prompt, token_budget=None):
        self.label, self.system_prompt, self.token_budget = label, system_prompt, token_budget
        self.tokens = 0
        self.transient_errors = 0
        self.retry_lines = []                    # one line per retry
        self.waits = []                          # (retry_number, wait_seconds)
        self.prompts_sent = []                   # every message list that reached the model (for the PII assertion)
        self.called_ticket_ids = []
        self.results = []

    def budget_left(self):
        return self.token_budget is None or self.tokens < self.token_budget


def call_with_retry(messages, ticket_id, run):
    """chat() with up to MAX_RETRIES retries on TransientLLMError and exponential backoff with jitter.

    wait before retry n (n = 0, 1, 2) = BASE_DELAY * 2**n * random.uniform(0.9, 1.15). One log line per retry.
    The token budget is checked before EVERY model call, retries included.
    """
    for attempt in range(MAX_RETRIES + 1):
        if not run.budget_left():
            raise BudgetExhausted()
        run.prompts_sent.append(messages)
        try:
            return mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            run.transient_errors += 1
            if attempt == MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** attempt * random.uniform(JITTER_LOW, JITTER_HIGH)
            line = f"retry ticket={ticket_id} retry_number={attempt} wait={wait:.4f}s"
            run.retry_lines.append(line)
            run.waits.append((attempt, wait))
            retry_logger.info(line)
            time.sleep(wait)


# ------------------------------------------------------------------ one ticket
def triage_ticket(row, run):
    """Return the result dict for one ticket. Guardrails run before any model call."""
    base = {"ticket_id": row["ticket_id"], "customer_id": row["customer_id"], "status": "", "category": "",
            "tokens_used": 0, "true_category": row["true_category"], "text": text if (text := row["text"]) else "", "reply": ""}
    text = row["text"]
    if not text.strip():
        return {**base, "status": "skipped_empty"}
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return {**base, "status": "flagged_long"}

    redacted, _ = redact(text)
    messages = [{"role": "system", "content": run.system_prompt}, {"role": "user", "content": redacted}]
    assert messages[1]["content"] == redacted and not contains_pii(messages[1]["content"]), "PII reached a prompt"
    if not run.budget_left():
        return {**base, "status": "budget_exhausted"}

    try:
        run.called_ticket_ids.append(row["ticket_id"])
        response = call_with_retry(messages, row["ticket_id"], run)
    except BudgetExhausted:
        run.called_ticket_ids.pop()
        return {**base, "status": "budget_exhausted"}
    except mock_llm.TransientLLMError:
        return {**base, "status": "llm_error"}

    used = response.usage.total_tokens
    run.tokens += used
    reply = response.choices[0].message.content
    status, category = validate_reply(reply)
    return {**base, "status": status, "category": category, "tokens_used": used, "reply": reply or ""}


def run_pipeline(tickets, system_prompt, label, token_budget=None):
    """Full pass over the tickets in file order. Calls mock_llm.reset() immediately before the run."""
    run = Run(label, system_prompt, token_budget)
    random.seed(JITTER_SEED)
    mock_llm.reset()
    for _, row in tickets.iterrows():
        run.results.append(triage_ticket(row, run))
    # post-run proof of the guardrails
    for messages in run.prompts_sent:
        assert not any(contains_pii(m["content"]) for m in messages), "PII found in a prompt sent to the model"
    skipped = {r["ticket_id"] for r in run.results if r["status"] in ("skipped_empty", "flagged_long")}
    assert not skipped & set(run.called_ticket_ids), "an empty or long ticket triggered a model call"
    for n, wait in run.waits:
        assert BASE_DELAY * 2 ** n * JITTER_LOW <= wait <= BASE_DELAY * 2 ** n * JITTER_HIGH, "wait outside the jitter range"
    return run


# ------------------------------------------------------------------ routing
def route_ticket(result, segments):
    """Apply the brief's routing rules. `segments` maps customer_id -> (segment_name, is_premium)."""
    segment_name, is_premium = segments.get(result["customer_id"], ("", False))
    if result["status"] in HUMAN_STATUSES:
        route, priority = "human_review", "NORMAL"
    elif result["category"] == "fraud":
        route, priority = "security_team", "HIGH"
    elif is_premium:
        route, priority = "premium_desk", "HIGH"
    else:
        route, priority = "support_queue", "NORMAL"
    return {"ticket_id": result["ticket_id"], "customer_id": result["customer_id"], "status": result["status"],
            "category": result["category"], "segment_name": segment_name, "priority": priority, "route": route,
            "tokens_used": result["tokens_used"]}
