"""Unit tests for the safety-critical logic (standard-library unittest, no extra dependency).

Run after the pipeline has produced data/instamart.db:   python -m unittest discover -s tests -v
The agent tests write to temporary folders, so they never touch the committed outputs/ or state/.
"""
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import common      # noqa: E402  (also puts the repository root on sys.path)
import mock_llm    # noqa: E402
import triage as T  # noqa: E402
import rag         # noqa: E402
import agent       # noqa: E402
import agent_tools as tools  # noqa: E402


class TestRedaction(unittest.TestCase):
    def test_placeholders_and_counts(self):
        text, counts = T.redact("Mail a.b+c@shop.co.in, call 9876543210 about ORD-123456 please")
        self.assertEqual(text, "Mail [EMAIL], call [PHONE] about [ORDER_ID] please")   # the pattern stops before the comma
        self.assertEqual(counts, {"email": 1, "phone": 1, "order_id": 1})

    def test_phone_needs_exactly_ten_digits_starting_6_to_9(self):
        for not_a_phone in ("5876543210", "98765432101", "987654321"):
            self.assertEqual(T.redact(not_a_phone)[0], not_a_phone)

    def test_redaction_is_idempotent_and_leaves_no_pii(self):
        once = T.redact("x@y.com 9123456789 ORD-654321")[0]
        self.assertFalse(T.contains_pii(once))
        self.assertEqual(T.redact(once)[0], once)


class TestValidation(unittest.TestCase):
    def test_valid_json(self):
        self.assertEqual(T.validate_reply('{"category": "fraud", "reason": "x"}'), ("classified_json", "fraud"))

    def test_json_with_category_outside_the_list_is_not_guessed(self):
        self.assertEqual(T.validate_reply('{"category": "shipping", "reason": "x"}'), ("flagged_invalid_output", ""))

    def test_chatty_reply_uses_word_extraction(self):
        self.assertEqual(T.validate_reply("The customer appears to have a billing-related problem."), ("classified_keyword", "billing"))

    def test_ambiguous_or_empty_replies_go_to_a_human(self):
        for bad in ("could be billing or fraud", "no idea", "", None, "[1, 2]", '{"category": "Billing"}', '{"reason": "x"}'):
            self.assertEqual(T.validate_reply(bad)[0], "flagged_invalid_output", bad)


class TestRetryAndBudget(unittest.TestCase):
    def make_run(self, **kw):
        return T.Run("t", T.ZERO_SHOT_PROMPT, **kw)

    def test_backoff_waits_are_inside_the_jitter_range(self):
        run, calls = self.make_run(), []

        def flaky(messages, tools=None):
            calls.append(1)
            if len(calls) <= 2:
                raise mock_llm.TransientLLMError("busy")
            return "ok"
        with mock.patch.object(mock_llm, "chat", flaky), mock.patch.object(time, "sleep"):
            self.assertEqual(T.call_with_retry([{"role": "user", "content": "x"}], "T1", run), "ok")
        self.assertEqual([n for n, _ in run.waits], [0, 1])
        for n, wait in run.waits:
            self.assertTrue(0.05 * 2 ** n * 0.9 <= wait <= 0.05 * 2 ** n * 1.15)
        self.assertEqual(len(run.retry_lines), 2)
        self.assertIn("ticket=T1 retry_number=0", run.retry_lines[0])

    def test_gives_up_after_three_retries(self):
        run, calls = self.make_run(), []

        def always_fails(messages, tools=None):
            calls.append(1)
            raise mock_llm.TransientLLMError("busy")
        with mock.patch.object(mock_llm, "chat", always_fails), mock.patch.object(time, "sleep"):
            with self.assertRaises(mock_llm.TransientLLMError):
                T.call_with_retry([{"role": "user", "content": "x"}], "T1", run)
        self.assertEqual(len(calls), 4)          # 1 call + 3 retries
        self.assertEqual(len(run.retry_lines), 3)

    def test_exhausted_budget_makes_no_call(self):
        run = self.make_run(token_budget=100)
        run.tokens = 100                         # already AT the budget
        with mock.patch.object(mock_llm, "chat") as chat:
            with self.assertRaises(T.BudgetExhausted):
                T.call_with_retry([{"role": "user", "content": "x"}], "T1", run)
            chat.assert_not_called()


class TestGuardrails(unittest.TestCase):
    ROW = {"ticket_id": "T9", "customer_id": "C1", "order_id": "", "true_category": "billing"}

    def test_empty_and_long_tickets_never_reach_the_model(self):
        with mock.patch.object(mock_llm, "chat") as chat:
            for text, status in (("", "skipped_empty"), ("   \n\t ", "skipped_empty"), ("word " * 400, "flagged_long")):
                self.assertEqual(T.triage_ticket({**self.ROW, "text": text}, T.Run("t", T.ZERO_SHOT_PROMPT))["status"], status)
            chat.assert_not_called()

    def test_prompt_contains_exactly_the_redacted_text(self):
        seen = []

        def fake(messages, tools=None):
            seen.append(messages)
            return mock_llm._resp('{"category": "billing", "reason": "x"}', messages=messages)
        with mock.patch.object(mock_llm, "chat", fake):
            T.triage_ticket({**self.ROW, "text": "Charged twice for ORD-123456, mail me at a@b.com or 9876543210"}, T.Run("t", T.ZERO_SHOT_PROMPT))
        self.assertEqual(seen[0][1]["content"], "Charged twice for [ORDER_ID], mail me at [EMAIL] or [PHONE]")
        self.assertFalse(any(T.contains_pii(m["content"]) for m in seen[0]))

    def test_prompts_follow_the_brief(self):
        for p in (T.ZERO_SHOT_PROMPT, T.FEW_SHOT_PROMPT):
            self.assertIn("CLASSIFY", p)
            self.assertTrue(all(c in p for c in T.CATEGORIES))
        self.assertNotIn("EXAMPLES:", T.ZERO_SHOT_PROMPT)
        self.assertIn("EXAMPLES:", T.FEW_SHOT_PROMPT)


class TestRouting(unittest.TestCase):
    SEG = {"C1": ("Premium Big-Ticket", True), "C2": ("Loyal Regulars", False)}

    def route(self, customer, status, category=""):
        return T.route_ticket({"ticket_id": "T", "customer_id": customer, "status": status, "category": category, "tokens_used": 0}, self.SEG)

    def test_rules(self):
        self.assertEqual(self.route("C2", "classified_json", "fraud")["route"], "security_team")
        self.assertEqual(self.route("C1", "classified_json", "fraud")["route"], "security_team")      # fraud beats premium
        self.assertEqual(self.route("C1", "classified_json", "billing")["route"], "premium_desk")
        self.assertEqual(self.route("C2", "classified_keyword", "delivery")["route"], "support_queue")
        self.assertEqual(self.route("C404", "classified_json", "billing")["route"], "support_queue")  # unknown customer = not premium
        for status in T.HUMAN_STATUSES:
            r = self.route("C1", status)
            self.assertEqual((r["route"], r["priority"]), ("human_review", "NORMAL"))
        self.assertEqual(self.route("C1", "classified_json", "fraud")["priority"], "HIGH")
        self.assertEqual(self.route("C404", "classified_json", "billing")["segment_name"], "")


class TestChunking(unittest.TestCase):
    def test_window_arithmetic(self):
        words = [f"w{i}" for i in range(119)]
        self.assertEqual([s for s, _ in rag.chunk_words(words, 40, 10)], [0, 30, 60, 90])
        self.assertEqual([s for s, _ in rag.chunk_words(words, 40, 0)], [0, 40, 80])
        self.assertEqual(rag.chunk_words(words, 40, 10)[-1][1][-1], "w118")        # last chunk reaches the last word
        self.assertEqual(len(rag.chunk_words([f"w{i}" for i in range(40)], 40, 10)), 1)

    def test_real_corpus_has_15_chunks(self):
        self.assertEqual(len(rag.load_chunks()), 15)

    def test_user_message_format(self):
        hits = [{"source": "a.txt", "text": "AAA"}, {"source": "b.txt", "text": "BBB"}]
        self.assertEqual(rag.build_user_message("Q?", hits), "CONTEXT:\n[a.txt]\nAAA\n[b.txt]\nBBB\nQUESTION:\nQ?")


class TestRefundTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = tools.LEDGER_PATH
        tools.LEDGER_PATH = Path(self.tmp.name) / "ledger.jsonl"
        tools.CURRENT_RUN["run_id"] = "unit"

    def tearDown(self):
        tools.LEDGER_PATH = self.old
        self.tmp.cleanup()

    def test_refund_rule_boundary(self):
        self.assertEqual(tools.calculate_refund(500, 46), {"refund_inr": 500.0})
        self.assertEqual(tools.calculate_refund(500, 45), {"refund_inr": 0.0})
        self.assertEqual(tools.calculate_refund(500, None), {"refund_inr": 0.0})

    def test_auto_refund_is_capped_at_1000(self):
        self.assertIn("error", tools.issue_refund("ORD-100001", 1000.01))
        self.assertFalse(tools.LEDGER_PATH.exists())
        self.assertEqual(tools.issue_refund("ORD-100001", 1000), {"refund_inr": 1000.0})

    def test_ledger_is_idempotent_within_a_run(self):
        tools.issue_refund("ORD-100001", 50)
        tools.issue_refund("ORD-100001", 50)
        self.assertEqual(len(tools.LEDGER_PATH.read_text().splitlines()), 1)

    def test_get_order_not_found_and_sql_injection_is_inert(self):
        self.assertEqual(tools.get_order("ORD-999999"), {"error": "order not found"})
        self.assertEqual(tools.get_order("x' OR '1'='1"), {"error": "order not found"})

    def test_unknown_tool_and_bad_arguments_become_observations(self):
        self.assertIn("error", tools.execute_tool("drop_database", {}))
        self.assertIn("error", tools.execute_tool("get_order", {"nope": 1}))


class TestAgentLimits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pandas as pd
        sc = pd.read_csv(common.RAW / "agent_scenarios.csv", dtype=str).set_index("scenario_id")
        cls.msg = {k: sc.loc[k, "message"] for k in sc.index}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_state, self.old_ledger = common.STATE, tools.LEDGER_PATH
        common.STATE, tools.LEDGER_PATH = Path(self.tmp.name), Path(self.tmp.name) / "ledger.jsonl"
        tools.EXECUTION_LOG.clear()

    def tearDown(self):
        common.STATE, tools.LEDGER_PATH = self.old_state, self.old_ledger
        self.tmp.cleanup()

    def test_happy_path_and_branches(self):
        for sid, status, calls in (("S1", "REFUND", 5), ("S3", "ESCALATE", 5), ("S4", "DENIED", 4), ("S5", "DENIED", 3), ("S6", "NOT_FOUND", 2)):
            r = agent.run_agent(self.msg[sid], sid, run_id=f"t-{sid}")
            self.assertEqual((r["status"], r["llm_calls"]), (status, calls), sid)

    def test_iteration_cap_stops_without_running_the_pending_tool(self):
        r = agent.run_agent(self.msg["S7"], "S7", run_id="t-loop")
        self.assertEqual((r["status"], r["llm_calls"]), ("FALLBACK_MAX_ITERATIONS", 6))
        self.assertEqual(len(r["steps"]), 5)
        self.assertIn("HANDOFF", r["final_answer"])

    def test_token_limit_stops_before_executing_the_requested_tool(self):
        r = agent.run_agent(self.msg["S1"], "S1", token_limit=200, run_id="t-tok")
        self.assertEqual(r["status"], "FALLBACK_TOKEN_LIMIT")
        self.assertGreater(r["tokens_used"], 200)
        self.assertNotIn("issue_refund", r["tools_executed"])
        self.assertFalse(tools.LEDGER_PATH.exists())

    def test_zero_time_limit_stops_before_the_first_model_call(self):
        r = agent.run_agent(self.msg["S1"], "S1", time_limit_seconds=0, run_id="t-time")
        self.assertEqual((r["status"], r["llm_calls"], r["steps"]), ("FALLBACK_TIMEOUT", 0, []))

    def test_crash_and_resume_executes_each_tool_exactly_once(self):
        with self.assertRaises(agent.SimulatedCrash):
            agent.run_agent(self.msg["S1"], "S1", run_id="t-crash", crash_after_tool_executions=2)
        ck = json.loads(agent.checkpoint_path("t-crash").read_text())
        self.assertEqual((ck["step"], ck["tools_executed"]), (2, ["get_order", "search_policy"]))
        res = agent.resume_agent("t-crash")
        executed = [n for rid, n in tools.EXECUTION_LOG if rid == "t-crash"]
        self.assertEqual(executed, ["get_order", "search_policy", "calculate_refund", "issue_refund"])
        self.assertEqual(res["status"], "REFUND")
        self.assertAlmostEqual(res["refund_inr"], 128.25, places=2)
        agent.resume_agent("t-crash")                                          # finished run: nothing executes again
        self.assertEqual([n for rid, n in tools.EXECUTION_LOG if rid == "t-crash"], executed)
        self.assertEqual(len(tools.LEDGER_PATH.read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
