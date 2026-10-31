"""Part D, task 5 -- the five tools of the refund agent, their JSON-style specification and the TOOL_MAP.

The model asks for a tool by NAME plus an arguments dictionary; the agent looks the name up in TOOL_MAP and calls the function.
"""
import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import common
import rag

LATE_AFTER_MINUTES = 45          # delivered in MORE than 45 minutes -> full refund
AUTO_APPROVE_LIMIT_INR = 1000    # a refund ABOVE this goes to a human

LEDGER_PATH = common.OUTPUTS / "refund_ledger.jsonl"
CURRENT_RUN = {"run_id": None}   # set by the agent before it executes a tool, so ledger lines can be tied to a run
EXECUTION_LOG = []               # (run_id, tool) for every execution -- lets the demos prove "none repeated" independently of the checkpoints


# ------------------------------------------------------------------ the five tools
def get_order(order_id):
    """Read one order from data/instamart.db (parameterised query: the id comes from free text and is never pasted into SQL)."""
    with closing(sqlite3.connect(common.DB_PATH)) as con:          # closing(): `with connect()` alone would leave the connection open
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
                          (str(order_id),)).fetchone()
    if row is None:
        return {"error": "order not found"}
    oid, status, amount, minutes = row
    if minutes is not None and float(minutes).is_integer():
        minutes = int(minutes)
    return {"order_id": oid, "status": status, "order_amount": float(amount), "delivery_minutes": minutes}


def search_policy(query):
    """Reuses the retrieval built in tasks 1-3 (same chunks, same TF-IDF index)."""
    return {"results": rag.retrieve(str(query), k=3)}


def calculate_refund(order_amount, delivery_minutes):
    """Refund rules: delivered in more than 45 minutes -> 100% of the order amount; otherwise 0."""
    amount = float(order_amount)
    if delivery_minutes is None or float(delivery_minutes) <= LATE_AFTER_MINUTES:
        return {"refund_inr": 0.0}
    return {"refund_inr": round(amount, 2)}


def _ledger_write(tool, order_id, refund_inr):
    """Append one JSON line to the ledger. Idempotent per (run, tool, order): a replayed call in the same run adds nothing."""
    run_id = CURRENT_RUN["run_id"]
    if LEDGER_PATH.exists() and run_id is not None:
        for line in LEDGER_PATH.read_text(encoding="utf-8").splitlines():
            old = json.loads(line)
            if (old["run_id"], old["tool"], old["order_id"]) == (run_id, tool, order_id):
                return {"refund_inr": old["refund_inr"]}
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": tool, "run_id": run_id,
             "order_id": order_id, "refund_inr": round(float(refund_inr), 2)}
    with LEDGER_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return {"refund_inr": entry["refund_inr"]}


def issue_refund(order_id, refund_inr):
    """Issue an automatic refund. Defence in depth: the tool itself refuses amounts above INR 1000, whatever the model asks."""
    if float(refund_inr) > AUTO_APPROVE_LIMIT_INR:
        return {"error": f"refund above INR {AUTO_APPROVE_LIMIT_INR} must be escalated to a human"}
    if float(refund_inr) <= 0:
        return {"error": "refund amount must be positive"}
    return _ledger_write("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    """Record a refund request that a human support lead must approve (no money moves)."""
    return _ledger_write("escalate_to_human", order_id, refund_inr)


# ------------------------------------------------------------------ specification and dispatch
def _spec(name, description, properties, required):
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required}}


TOOLS_SPEC = [
    _spec("get_order", "Look up one order (status, amount, delivery minutes) in the Instamart database by its order id.",
          {"order_id": {"type": "string", "description": "Order id such as ORD-100123"}}, ["order_id"]),
    _spec("search_policy", "Search the support policy documents and return the three most relevant passages.",
          {"query": {"type": "string", "description": "Plain-language search text"}}, ["query"]),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes using the refund rules.",
          {"order_amount": {"type": "number", "description": "Order amount in INR"},
           "delivery_minutes": {"type": "number", "description": "Minutes the delivery took"}}, ["order_amount", "delivery_minutes"]),
    _spec("issue_refund", "Issue an automatic refund of at most INR 1000 and record it in the refund ledger.",
          {"order_id": {"type": "string", "description": "Order id"}, "refund_inr": {"type": "number", "description": "Refund in INR"}},
          ["order_id", "refund_inr"]),
    _spec("escalate_to_human", "Send a refund above INR 1000 to a human support lead and record it in the ledger.",
          {"order_id": {"type": "string", "description": "Order id"}, "refund_inr": {"type": "number", "description": "Requested refund in INR"}},
          ["order_id", "refund_inr"]),
]

TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}
assert [t["name"] for t in TOOLS_SPEC] == list(TOOL_MAP)


def execute_tool(name, arguments):
    """Look the tool up by name and call it with the model's arguments; failures become observations, never crashes."""
    EXECUTION_LOG.append((CURRENT_RUN["run_id"], name))
    fn = TOOL_MAP.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}"}
    try:
        return fn(**(arguments or {}))
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}
    except Exception as exc:                                    # keep the loop alive; the model sees the error as an observation
        return {"error": f"{type(exc).__name__}: {exc}"}
