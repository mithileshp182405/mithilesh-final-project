"""Part D, tasks 5-9 -- tools, the seven scenario runs, the stop-safely demonstrations, crash-and-resume, ReAct vs plan-and-execute."""
import json
import pandas as pd
from common import OUTPUTS, RAW, STATE, Report
from mock_llm import est_tokens
import agent
import agent_tools as tools

rep = Report("d2_agent_run")
pd.set_option("display.width", 170)

# ------------------------------------------------------------------ clean slate so the run is reproducible (no duplicated ledger lines)
RUNS_PATH, DEMOS_PATH = OUTPUTS / "agent_runs.jsonl", OUTPUTS / "agent_demos.jsonl"
for p in (RUNS_PATH, DEMOS_PATH, tools.LEDGER_PATH, *STATE.glob("*.json"), *STATE.glob("*.tmp")):
    p.unlink(missing_ok=True)
tools.EXECUTION_LOG.clear()


def ledger_lines(run_id=None):
    if not tools.LEDGER_PATH.exists():
        return []
    rows = [json.loads(x) for x in tools.LEDGER_PATH.read_text(encoding="utf-8").splitlines()]
    return [r for r in rows if run_id is None or r["run_id"] == run_id]


def jsonl_record(res, **extra):
    return {"scenario_id": res["scenario_id"], "run_id": res["run_id"], "status": res["status"], "final_answer": res["final_answer"],
            "llm_calls": res["llm_calls"], "tokens_used": res["tokens_used"], "refund_inr": res["refund_inr"],
            "steps": [{"tool": s["tool"], "arguments": s["arguments"], "observation": s["observation"]} for s in res["steps"]], **extra}


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ------------------------------------------------------------------ task 5: tools
rep("TASK 5 -- five tools (TOOLS_SPEC) and TOOL_MAP")
for spec in tools.TOOLS_SPEC:
    rep(f"  {spec['name']:<18} params {list(spec['parameters']['properties'])}  -- {spec['description']}")
assert set(tools.TOOL_MAP) == {s["name"] for s in tools.TOOLS_SPEC} and len(tools.TOOL_MAP) == 5
probe_ok = tools.execute_tool("get_order", {"order_id": "ORD-999999"})
rep(f"  get_order(unknown id)            -> {probe_ok}")
assert probe_ok == {"error": "order not found"}
cases = [(500, 46, 500.0), (500, 45, 0.0), (500, 12, 0.0), (1999, 60, 1999.0)]
for amount, minutes, expected in cases:
    out = tools.execute_tool("calculate_refund", {"order_amount": amount, "delivery_minutes": minutes})
    rep(f"  calculate_refund({amount}, {minutes:>2} min)    -> {out}   (rule: more than 45 minutes = full refund, else 0)")
    assert out == {"refund_inr": expected}
blocked = tools.execute_tool("issue_refund", {"order_id": "ORD-000000", "refund_inr": 1500})
rep(f"  issue_refund(1500) is refused by the tool itself -> {blocked}")
assert "error" in blocked and not ledger_lines()
assert "error" in tools.execute_tool("no_such_tool", {}) and "error" in tools.execute_tool("get_order", {"wrong": 1})
tools.EXECUTION_LOG.clear()

# ------------------------------------------------------------------ task 6: seven scenarios
scen = pd.read_csv(RAW / "agent_scenarios.csv", dtype=str)
assert len(scen) == 7
results = {}
for r in scen.itertuples():
    res = agent.run_agent(r.message, r.scenario_id, run_id=f"{r.scenario_id}-main")
    results[r.scenario_id] = res
    append_jsonl(RUNS_PATH, jsonl_record(res))

rep("\nTASK 6 -- the seven scenarios of data/raw/agent_scenarios.csv (default limits: 6 model calls, 6000 tokens, 10 s)")
rows = []
for r in scen.itertuples():
    res = results[r.scenario_id]
    expected_ok = res["status"].startswith("FALLBACK") if r.expected_outcome == "FALLBACK" else res["status"] == r.expected_outcome
    refund_ok = abs(res["refund_inr"] - float(r.expected_refund_inr)) <= 0.01
    rows.append({"scenario": r.scenario_id, "expected": r.expected_outcome, "status": res["status"], "refund_inr": f"{res['refund_inr']:.2f}",
                 "expected_refund": r.expected_refund_inr, "model_calls": res["llm_calls"], "tokens_used": res["tokens_used"],
                 "tools": ", ".join(s["tool"] for s in res["steps"]) or "-", "match": "OK" if expected_ok and refund_ok else "FAIL"})
table = pd.DataFrame(rows)
rep(table.to_string(index=False))
assert (table["match"] == "OK").all()
assert results["S1"]["llm_calls"] == 5 and results["S5"]["llm_calls"] == 3 and results["S6"]["llm_calls"] == 2
assert results["S7"]["status"] == "FALLBACK_MAX_ITERATIONS" and results["S7"]["llm_calls"] == 6 and results["S7"]["tokens_used"] <= 6000
rep(f"S7 used {results['S7']['tokens_used']} estimated tokens for all six model calls, below the default limit of 6000, so it ends with FALLBACK_MAX_ITERATIONS and not earlier.")
rep(f"S7 hand-off text: {results['S7']['final_answer']}")
rep(f"ledger after the seven scenarios: {[(x['run_id'], x['tool'], x['refund_inr']) for x in ledger_lines()]}")
assert [(x["run_id"], x["tool"]) for x in ledger_lines()] == [("S1-main", "issue_refund"), ("S2-main", "issue_refund"), ("S3-main", "escalate_to_human")]

# ------------------------------------------------------------------ task 7: stop safely
s1_msg = scen.loc[scen["scenario_id"] == "S1", "message"].iloc[0]
rep("\nTASK 7 -- stop safely")
n_ledger_before = len(ledger_lines())
tok = agent.run_agent(s1_msg, "S1", token_limit=200, run_id="S1-token-limit-demo")
rep(f"(b) S1 with token_limit=200 -> {tok['status']} after {tok['llm_calls']} model calls and {tok['tokens_used']} tokens; tools executed: {tok['tools_executed']}")
rep(f"    hand-off: {tok['final_answer']}")
assert tok["status"] == "FALLBACK_TOKEN_LIMIT" and tok["tokens_used"] > 200 and len(tok["tools_executed"]) < 4 and len(ledger_lines()) == n_ledger_before
tmo = agent.run_agent(s1_msg, "S1", time_limit_seconds=0, run_id="S1-timeout-demo")
rep(f"(c) S1 with time_limit_seconds=0 -> {tmo['status']} after {tmo['llm_calls']} model calls (it stops before the first call); steps so far: {len(tmo['steps'])}")
rep(f"    hand-off: {tmo['final_answer']}")
assert tmo["status"] == "FALLBACK_TIMEOUT" and tmo["llm_calls"] == 0 and len(ledger_lines()) == n_ledger_before
rep(f"(a) S7 default run -> {results['S7']['status']} (six model calls, see task 6); (b) and (c) executed no refund tool: ledger unchanged at {n_ledger_before} lines")
append_jsonl(DEMOS_PATH, jsonl_record(tok, demo="token_limit=200"))
append_jsonl(DEMOS_PATH, jsonl_record(tmo, demo="time_limit_seconds=0"))

# ------------------------------------------------------------------ task 8: crash and resume
rep("\nTASK 8 -- resumable run: S1 crashes right after its SECOND tool execution, then resume_agent(run_id)")
RUN = "S1-crash-demo"
try:
    agent.run_agent(s1_msg, "S1", run_id=RUN, crash_after_tool_executions=2)
    raise AssertionError("the simulated crash did not happen")
except agent.SimulatedCrash as exc:
    rep(f"crash: {exc}")
ck = json.loads(agent.checkpoint_path(RUN).read_text(encoding="utf-8"))
rep(f"checkpoint state/{RUN}.json: run_id={ck['run_id']}, scenario_id={ck['scenario_id']}, step={ck['step']}, tokens_used={ck['tokens_used']}, "
    f"llm_calls={ck['llm_calls']}, tools_executed={ck['tools_executed']}, messages={len(ck['messages'])}, finished={ck['finished']}")
assert ck["step"] == 2 and ck["tools_executed"] == ["get_order", "search_policy"] and not ck["finished"]
assert ledger_lines(RUN) == []                                     # nothing irreversible happened before the crash
before = list(tools.EXECUTION_LOG)
res = agent.resume_agent(RUN)
executed = [name for rid, name in tools.EXECUTION_LOG if rid == RUN]
rep(f"after resume: status {res['status']}, refund INR {res['refund_inr']:.2f}, model calls {res['llm_calls']}, tokens {res['tokens_used']}")
rep(f"tools executed across the crash and the resume, in order: {executed}")
assert executed == ["get_order", "search_policy", "calculate_refund", "issue_refund"], executed
assert res["status"] == "REFUND" and abs(res["refund_inr"] - 128.25) <= 0.01
assert res["tokens_used"] == results["S1"]["tokens_used"] and res["llm_calls"] == results["S1"]["llm_calls"] == 5
rep(f"independent execution log: {len(executed)} executions, {len(set(executed))} distinct -> none repeated; same total tokens ({res['tokens_used']}) and calls (5) as the uninterrupted S1 run")
rep(f"ledger lines for this run: {[(x['tool'], x['refund_inr']) for x in ledger_lines(RUN)]}")
assert len(ledger_lines(RUN)) == 1
again = agent.resume_agent(RUN)                                    # resuming a finished run must do nothing
assert again["status"] == "REFUND" and [n for r_, n in tools.EXECUTION_LOG if r_ == RUN] == executed
rep("calling resume_agent again on the finished run returns the stored result and executes nothing")
append_jsonl(DEMOS_PATH, jsonl_record(res, demo="crash after tool #2, then resume", tools_executed=executed))
rep(f"ledger now holds {len(ledger_lines())} lines (S1-main, S2-main, S3-main and the crash demo's own S1 request, each tied to its run id)")

# ------------------------------------------------------------------ task 9: ReAct vs plan-and-execute
s1 = results["S1"]
sys_user = est_tokens(agent.AGENT_SYSTEM_PROMPT) + est_tokens(s1_msg)
plan_text = '["get_order", "search_policy", "calculate_refund", "issue_refund"]'
obs_tokens = sum(est_tokens(json.dumps(s["observation"])) for s in s1["steps"])
pe_tokens = (sys_user + est_tokens(plan_text)) + (sys_user + est_tokens(plan_text) + obs_tokens + est_tokens(s1["final_answer"]))
react = {k: (v["llm_calls"], v["tokens_used"]) for k, v in results.items()}
essay = (
    f"Measured ReAct cost: S1 {react['S1'][0]} model calls / {react['S1'][1]} tokens, S4 {react['S4'][0]}, S5 {react['S5'][0]}, S6 {react['S6'][0]} / {react['S6'][1]} tokens; "
    f"every call re-sends the whole history, so tokens grow faster than calls. A plan-and-execute design would ask once for the full step list and run the tools "
    f"in code: for S1 that is about 2 calls and roughly {pe_tokens} tokens (my estimate from the same prompts and observations; the mock has no such mode), "
    f"cheaper than {react['S1'][1]}. "
    f"The saving disappears when an observation is unexpected. For S6 (NOT_FOUND) and S5 (cancelled order) the fixed plan is already wrong after step 1 or 2, so it "
    f"needs a replanning call: about 3 calls against ReAct's {react['S6'][0]} and {react['S5'][0]}. A plan written before seeing the order also contains issue_refund "
    f"before anyone knows the order is eligible, so every plan needs an extra guard that blocks side-effecting steps. "
    f"I would choose ReAct for this refund workflow: the paths are short (at most 5 calls), the branches depend on what each lookup returns (not found, cancelled, "
    f"on time, above INR 1000), money moves only after the evidence is seen, and the iteration, token and time limits bound the worst case (S7). "
    f"Plan-and-execute pays off for long, predictable pipelines."
)
words = len(essay.split())
rep(f"\nTASK 9 -- ReAct versus plan-and-execute ({words} words, limit 250)")
rep(essay)
assert words <= 250
rep.save()
