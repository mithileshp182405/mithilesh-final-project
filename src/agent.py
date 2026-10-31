"""Part D, tasks 6-8 -- the ReAct refund agent with three safety limits and crash-resumable checkpoints.

run_agent(...)     start a run          resume_agent(run_id)   continue a run from its last checkpoint
Limits: max_iterations model calls, token_limit estimated tokens, time_limit_seconds of wall-clock time (time.monotonic()).
"""
import json
import os
import re
import time
import uuid

import common
import mock_llm
import agent_tools as tools

AGENT_SYSTEM_PROMPT = (
    "You are a refund AGENT for Instamart support. Use the tools to look up the order, read the refund rules, "
    "calculate the refund, then issue it or escalate it to a human. Finish with one line starting with "
    "REFUND:, ESCALATE:, DENIED:, NOT_FOUND: or NEEDS_INFO:."
)


class SimulatedCrash(RuntimeError):
    """Raised on purpose by the crash-and-resume demonstration."""


# ------------------------------------------------------------------ checkpoints
def checkpoint_path(run_id):
    return common.STATE / f"{run_id}.json"


def save_checkpoint(state):
    """Write state/<run_id>.json atomically (temp file + rename), so a crash can never leave a half-written checkpoint."""
    path = checkpoint_path(state["run_id"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ result shaping
def _refund_amount(steps):
    for s in reversed(steps):
        if s["tool"] in ("issue_refund", "escalate_to_human") and "refund_inr" in s["observation"]:
            return float(s["observation"]["refund_inr"])
    return 0.0


def _result(state):
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": state["status"],
            "final_answer": state["final_answer"], "steps": state["steps"], "llm_calls": state["llm_calls"],
            "tokens_used": state["tokens_used"], "refund_inr": _refund_amount(state["steps"]),
            "tools_executed": list(state["tools_executed"])}


def _fallback(state, status, reason):
    """Stop safely: record the status and a short human-handoff summary of what was done."""
    done = ", ".join(state["tools_executed"]) or "none"
    last = json.dumps(state["steps"][-1]["observation"])[:120] if state["steps"] else "n/a"
    state.update(status=status, finished=True, final_answer=(
        f"HANDOFF TO HUMAN AGENT ({reason}). Scenario {state['scenario_id']}, request: {state['user_message']!r}. "
        f"Tools executed so far: {done}. Last observation: {last}. No further automatic action was taken."))
    save_checkpoint(state)
    return _result(state)


# ------------------------------------------------------------------ the loop (shared by run_agent and resume_agent)
def _loop(state, crash_after_tool_executions=None):
    lim = state["limits"]
    start = time.monotonic()                                        # elapsed time is measured per process (a resumed run gets a fresh clock)
    while True:
        # (c) time limit, checked at the top of every iteration
        if time.monotonic() - start >= lim["time_limit_seconds"]:
            return _fallback(state, "FALLBACK_TIMEOUT", f"time limit of {lim['time_limit_seconds']} s reached")

        response = mock_llm.chat(state["messages"], tools=tools.TOOLS_SPEC)
        state["llm_calls"] += 1
        state["tokens_used"] += response.usage.total_tokens

        # (b) token limit: stop at once, WITHOUT executing the tool that was just requested
        if state["tokens_used"] > lim["token_limit"]:
            return _fallback(state, "FALLBACK_TOKEN_LIMIT",
                             f"token limit {lim['token_limit']} exceeded: {state['tokens_used']} tokens after {state['llm_calls']} model calls")

        message = response.choices[0].message
        if not message.tool_calls:                                  # final answer: its first word before the colon is the status
            final = message.content or ""
            m = re.match(r"\s*([A-Z_]+)\s*:", final)
            state.update(status=m.group(1) if m else "UNPARSED", final_answer=final, finished=True)
            save_checkpoint(state)
            return _result(state)

        # (a) iteration cap: the last allowed call still wants a tool -> stop WITHOUT executing it (a refund tool must not run unobserved)
        if state["llm_calls"] >= lim["max_iterations"]:
            return _fallback(state, "FALLBACK_MAX_ITERATIONS",
                             f"no final answer after {lim['max_iterations']} model calls; the {state['llm_calls']}th call requested "
                             f"{message.tool_calls[0].name} which was not executed")

        calls = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in message.tool_calls]
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": calls})
        for call in calls:
            tools.CURRENT_RUN["run_id"] = state["run_id"]
            observation = tools.execute_tool(call["name"], call["arguments"])
            state["messages"].append({"role": "tool", "tool_call_id": call["id"], "name": call["name"],
                                      "content": json.dumps(observation)})
            state["tools_executed"].append(call["name"])
            state["step"] = len(state["tools_executed"])
            state["steps"].append({"step": state["step"], "tool": call["name"], "arguments": call["arguments"], "observation": observation})
            save_checkpoint(state)                                  # after EVERY tool observation
            if crash_after_tool_executions is not None and state["step"] == crash_after_tool_executions:
                raise SimulatedCrash(f"simulated crash right after tool execution #{state['step']} of run {state['run_id']}")


# ------------------------------------------------------------------ public API
def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              run_id=None, crash_after_tool_executions=None):
    run_id = run_id or f"{scenario_id}-{uuid.uuid4().hex[:8]}"
    state = {"run_id": run_id, "scenario_id": scenario_id, "user_message": user_message,
             "limits": {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds},
             "messages": [{"role": "system", "content": AGENT_SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": [],
             "status": None, "final_answer": None, "finished": False}
    return _loop(state, crash_after_tool_executions)


def resume_agent(run_id):
    """Load state/<run_id>.json and continue exactly where the run stopped; finished runs are returned as they are."""
    state = json.loads(checkpoint_path(run_id).read_text(encoding="utf-8"))
    if state["finished"]:
        return _result(state)
    return _loop(state)
