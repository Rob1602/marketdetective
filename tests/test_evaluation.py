import json

from marketdetective.evaluation import (
    contains_term,
    cause_found,
    load_ground_truth,
    load_runs,
    score_run,
    summarize,
    unsupported_numbers,
)

GT = """
events:
  - ticker: NVDA
    date: "2026-11-09"
    expected_driver: company-specific
    cause: Board investigation into the CEO
    key_terms:
      - [huang, ceo]
      - [investigation, probe, scandal]
"""


def run(mode="agent", causes=(("CEO Jensen Huang scandal", ["n1"]),), conclusion="Board probe into the CEO.",
        numbers=True, hypotheses=(), invalid=(), seconds=100.0, tin=1000, tout=200):
    return {
        "mode": mode, "model": "qwen3:8b", "ticker": "NVDA", "date": "2026-11-09",
        "numbers": {"day_return": "-5.32%", "explained_by_market": "+4.24%",
                    "explained_by_sector": "+0.80%", "company_specific_abnormal_return": "-10.34%",
                    "cumulative_abnormal_return_days_-1_to_+1": "-8.66%"} if numbers else None,
        "report": {"supported_causes": [{"cause": c, "evidence_ids": ids} for c, ids in causes],
                   "hypotheses": list(hypotheses), "conclusion": conclusion, "confidence": "high"},
        "citations_valid": ["n1"], "citations_invalid": list(invalid),
        "steps": [{"kind": "tool", "name": "analyze_event"}, {"kind": "llm", "name": "reasoning"}],
        "total_seconds": seconds, "prompt_tokens": tin, "output_tokens": tout, "error": "",
    }


def truths(tmp_path):
    p = tmp_path / "gt.yaml"
    p.write_text(GT, encoding="utf-8")
    return load_ground_truth(p)


def test_cause_found_requires_every_group(tmp_path):
    t = truths(tmp_path)[("NVDA", "2026-11-09")]
    assert cause_found(run(), t)[0]
    ok, missing = cause_found(run(causes=(("Market selloff", []),), conclusion="Tech fell."), t)
    assert not ok and len(missing) == 2


def test_cause_found_accepts_any_synonym(tmp_path):
    t = truths(tmp_path)[("NVDA", "2026-11-09")]
    assert cause_found(run(causes=(("Probe into the CEO", []),), conclusion=""), t)[0]


def test_terms_match_whole_words_only():
    # without word boundaries, 'no' would match inside 'not'/'announced' and 'ceo' inside 'ceos'
    assert contains_term("no specific cause was found", "no")
    assert not contains_term("nvidia announced it is not now listed", "no")
    assert contains_term("the ceo resigned", "ceo")
    assert not contains_term("tech ceos met", "ceo")
    assert contains_term("there is no clear driver", "no clear")


def test_unsupported_numbers_flags_imported_figures():
    r = run(conclusion="The stock fell over 5% and is 30% below its high.")
    assert unsupported_numbers(r) == ["30%"]          # 5% matches the computed -5.32%, 30% does not


def test_no_numbers_means_every_percentage_is_unverified():
    r = run(mode="rag", numbers=False, conclusion="It fell 5% pre-market.")
    assert unsupported_numbers(r) == ["5%"]


def test_score_and_summary(tmp_path):
    t = truths(tmp_path)
    scores = [score_run(run(), t),
              score_run(run(mode="rag", numbers=False, hypotheses=["a", "b"],
                            invalid=["x"], seconds=40.0, tin=900, tout=100), t)]
    s = summarize(scores)
    assert s["agent"]["with_numbers"] == 1 and s["rag"]["with_numbers"] == 0
    assert s["rag"]["citations_invalid"] == 1
    assert s["rag"]["avg_hypotheses"] == 2.0
    assert s["agent"]["avg_seconds"] == 100 and s["rag"]["avg_tokens"] == 1000
    assert scores[0].tokens == 1200


def test_load_runs_keeps_only_the_latest(tmp_path):
    for name, mode, secs in [("NVDA_2026-11-09_agent_20261113-120000.json", "agent", 10),
                             ("NVDA_2026-11-09_agent_20261113-130000.json", "agent", 20),
                             ("NVDA_2026-11-09_rag_20261113-131000.json", "rag", 5)]:
        (tmp_path / name).write_text(json.dumps(run(mode=mode, seconds=secs)), encoding="utf-8")
    latest = load_runs(tmp_path)
    assert len(latest) == 2
    assert {r["total_seconds"] for r in latest} == {20, 5}
    assert len(load_runs(tmp_path, latest_only=False)) == 3


def test_broken_json_is_skipped(tmp_path):
    (tmp_path / "a.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(run()), encoding="utf-8")
    assert len(load_runs(tmp_path)) == 1
