"""Evaluation: compares the reports in runs/ with the ground truth in data/ground_truth.yaml.

Four automatic measures per run:

1. cause_found    - the answer contains the ground truth key terms. Every group of synonyms must
                    have at least one term present.
2. citations      - how many citations are valid and how many fabricated (or published after the
                    close). The check happens during the run; the counts are reported here.
3. numeric_claims - percentages in the text matching no value produced by the event study. Shows
                    whether the model imports figures from the news instead of measuring them.
4. cost           - seconds, steps and tokens.

Judging the quality of an answer is left to a human reading the reports.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PCT = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*%")
# Event study figures the text may legitimately quote.
NUMERIC_FIELDS = ["day_return", "explained_by_market", "explained_by_sector",
                  "company_specific_abnormal_return", "cumulative_abnormal_return_days_-1_to_+1"]


@dataclass
class Truth:
    ticker: str
    date: str
    cause: str
    key_terms: list[list[str]]
    expected_driver: str = ""


def load_ground_truth(path: str | Path) -> dict[tuple[str, str], Truth]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    out = {}
    for e in raw.get("events", []):
        t = Truth(ticker=str(e["ticker"]).upper(), date=str(e["date"]), cause=str(e.get("cause", "")).strip(),
                  key_terms=[[str(x).lower() for x in g] for g in e.get("key_terms", [])],
                  expected_driver=str(e.get("expected_driver", "")))
        out[(t.ticker, t.date)] = t
    return out


def load_runs(runs_dir: str | Path, latest_only: bool = True) -> list[dict]:
    runs = []
    for p in sorted(Path(runs_dir).glob("*.json")):
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")) | {"file": p.name})
        except json.JSONDecodeError:
            continue
    if latest_only:   # filenames end with date and time, so the last one wins
        keep = {}
        for r in runs:
            keep[(r["ticker"], r["date"], r["mode"])] = r
        runs = list(keep.values())
    return sorted(runs, key=lambda r: (r["date"], r["mode"]))


def answer_text(run: dict) -> str:
    r = run.get("report") or {}
    parts = [c.get("cause", "") for c in r.get("supported_causes", [])]
    parts += r.get("hypotheses", []) + [r.get("conclusion", "")]
    return " ".join(parts).lower()


def contains_term(text: str, term: str) -> bool:
    """Match the term as a whole word.

    Without word boundaries, 'no' would be found inside 'not', 'now' or 'announced', and the
    check would report every cause as found.
    """
    return re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text) is not None


def cause_found(run: dict, truth: Truth) -> tuple[bool, list[str]]:
    """True if every key-term group is represented. Also returns the groups that are missing."""
    text = answer_text(run)
    missing = [" / ".join(g) for g in truth.key_terms
               if not any(contains_term(text, t) for t in g)]
    return (not missing and bool(truth.key_terms)), missing


def _values(numbers: dict | None) -> list[float]:
    vals = []
    for f in NUMERIC_FIELDS:
        v = (numbers or {}).get(f)
        if isinstance(v, str):
            m = PCT.search(v)
            if m:
                vals.append(abs(float(m.group(1).replace(",", "."))))
    return vals


def unsupported_numbers(run: dict, tolerance: float = 0.5) -> list[str]:
    """Percentages in the text that match no computed value.

    Tolerance in percentage points: '5%' passes if the computed value is -5.32%.
    A coarse check, to be read as a flag rather than a verdict.
    """
    known = _values(run.get("numbers"))
    bad = []
    for m in PCT.finditer(answer_text(run)):
        x = abs(float(m.group(1).replace(",", ".")))
        if not any(abs(x - k) <= tolerance for k in known):
            bad.append(m.group(0))
    return bad


@dataclass
class Score:
    ticker: str
    date: str
    mode: str
    model: str
    cause_found: bool
    missing_terms: list[str]
    citations_valid: int
    citations_invalid: int
    unsupported_numbers: list[str]
    has_numbers: bool
    n_hypotheses: int
    confidence: str
    seconds: float
    steps: int
    tool_calls: int
    prompt_tokens: int
    output_tokens: int
    error: str = ""
    notes: str = field(default="")

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


def score_run(run: dict, truths: dict[tuple[str, str], Truth]) -> Score:
    truth = truths.get((run["ticker"], run["date"]))
    found, missing = cause_found(run, truth) if truth else (False, ["(nessuna causa vera definita)"])
    report = run.get("report") or {}
    return Score(
        ticker=run["ticker"], date=run["date"], mode=run["mode"], model=run.get("model", ""),
        cause_found=found, missing_terms=missing,
        citations_valid=len(run.get("citations_valid", [])),
        citations_invalid=len(run.get("citations_invalid", [])),
        unsupported_numbers=unsupported_numbers(run),
        has_numbers=run.get("numbers") is not None,
        n_hypotheses=len(report.get("hypotheses", [])),
        confidence=report.get("confidence", ""),
        seconds=float(run.get("total_seconds", 0)),
        steps=len(run.get("steps", [])),
        tool_calls=sum(1 for s in run.get("steps", []) if s.get("kind") == "tool"),
        prompt_tokens=int(run.get("prompt_tokens", 0)),
        output_tokens=int(run.get("output_tokens", 0)),
        error=run.get("error", ""),
    )


def summarize(scores: list[Score]) -> dict[str, dict]:
    """Averages per mode."""
    out = {}
    for mode in sorted({s.mode for s in scores}):
        rows = [s for s in scores if s.mode == mode]
        n = len(rows)
        out[mode] = {
            "runs": n,
            "cause_found": sum(s.cause_found for s in rows),
            "citations_valid": sum(s.citations_valid for s in rows),
            "citations_invalid": sum(s.citations_invalid for s in rows),
            "unsupported_numbers": sum(len(s.unsupported_numbers) for s in rows),
            "avg_seconds": sum(s.seconds for s in rows) / n,
            "avg_tokens": sum(s.tokens for s in rows) / n,
            "avg_hypotheses": sum(s.n_hypotheses for s in rows) / n,
            "with_numbers": sum(s.has_numbers for s in rows),
            "errors": sum(bool(s.error) for s in rows),
        }
    return out
