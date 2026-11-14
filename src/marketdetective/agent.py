"""The agent: a local LLM that uses tools through function calling and writes a structured report.

Two modes, for the comparison:
- "agent": the model calls analyze_event and search_news as often as it needs, then writes the report.
- "rag":   baseline without tools. One search with the user's question, the retrieved news goes
           into the prompt and the model answers. No statistics at all.

Guarantees enforced in code rather than trusted to the model:
- the numbers in the report come from the event study, not from the model's text;
- every citation is checked: it must be a news item actually returned by search_news and published
  before the market close of the event day. Invalid citations are reported.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .tools import TOOL_SCHEMAS, Toolbox

# ------------------------------------------------------------- report schema


class Cause(BaseModel):
    cause: str = Field(description="One sentence describing the cause")
    evidence_ids: list[str] = Field(description="IDs of the news items that support it, e.g. ['fh-123']")


class LLMReport(BaseModel):
    """What the model must produce (structured output)."""
    supported_causes: list[Cause] = Field(description="Causes directly supported by the news found")
    hypotheses: list[str] = Field(description="Plausible explanations NOT directly confirmed by the news")
    conclusion: str = Field(description="2-3 sentences answering the question")
    confidence: Literal["low", "medium", "high"]


SYSTEM_AGENT = """You are a financial analyst investigating why a stock moved on a given day.
You have two tools:
- analyze_event: the numbers (return, market and sector contribution, abnormal return, significance).
- search_news: news published before the market close of that day.

Method:
1. Call analyze_event first.
2. If the move is mostly company-specific, search for company news. If it is mostly market- or
   sector-driven, say so and search for market or sector news.
3. Use 2 or 3 searches with SPECIFIC queries about candidate causes (earnings, guidance, analyst
   ratings, management, legal or regulatory issues, competition, products, China/export).
   Prefer news from the event day and the day before.
4. Stop when you have enough evidence.

Rules:
- Never invent numbers: use only the numbers returned by analyze_event.
- A cause is "supported" only if a news item you received states it. Otherwise it is a hypothesis.
- If the news does not explain the move, say so clearly."""

SYSTEM_RAG = """You are a financial analyst. Answer the question using ONLY the news items provided.
A cause is "supported" only if a news item states it; otherwise list it as a hypothesis.
If the news does not explain the move, say so clearly."""

FINAL_INSTRUCTION = """Now write the final report as JSON.
Cite news only by their exact "id" values from the search results.
Keep supported causes and hypotheses separate."""


# -------------------------------------------------------------------- trace


@dataclass
class Step:
    kind: str                  # "llm" | "tool"
    name: str = ""
    args: dict = field(default_factory=dict)
    seconds: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0
    summary: str = ""


@dataclass
class RunResult:
    mode: str
    model: str
    ticker: str
    date: str
    question: str
    numbers: dict | None
    report: dict
    citations_valid: list[str]
    citations_invalid: list[str]
    steps: list[Step]
    total_seconds: float
    error: str = ""

    @property
    def tokens(self) -> tuple[int, int]:
        return (sum(s.prompt_tokens for s in self.steps), sum(s.output_tokens for s in self.steps))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["prompt_tokens"], d["output_tokens"] = self.tokens
        return d

    def to_markdown(self) -> str:
        r, n = self.report, self.numbers
        lines = [f"# {self.ticker} · {self.date}", "", f"**Question:** {self.question}", "",
                 f"*Mode: {self.mode} · model: {self.model} · {self.total_seconds:.0f}s · "
                 f"{len(self.steps)} steps · tokens in/out {self.tokens[0]}/{self.tokens[1]}*", ""]
        if n:
            lines += ["## Numbers (from the event study)", "",
                      f"- Day return: **{n['day_return']}**",
                      f"- Explained by market ({n['benchmarks']['market']}): {n['explained_by_market']}",
                      f"- Explained by sector ({n['benchmarks']['sector']}): {n['explained_by_sector']}",
                      f"- Company-specific (abnormal): **{n['company_specific_abnormal_return']}** "
                      f"(t = {n['t_stat']}, p = {n['p_value']}) → "
                      f"{'anomalous' if n['is_anomalous'] else 'not anomalous'}",
                      f"- Abnormal return days −1 to +1: {n['cumulative_abnormal_return_days_-1_to_+1']}", ""]
        lines += ["## Supported causes", ""]
        for c in r.get("supported_causes", []) or [{"cause": "None found in the news.", "evidence_ids": []}]:
            ids = ", ".join(c.get("evidence_ids", [])) or "no source"
            lines.append(f"- {c['cause']} [{ids}]")
        lines += ["", "## Hypotheses (not confirmed by sources)", ""]
        lines += [f"- {h}" for h in r.get("hypotheses", [])] or ["- None"]
        lines += ["", "## Conclusion", "", r.get("conclusion", ""), "",
                  f"Confidence: **{r.get('confidence', '?')}**", "", "## Checks", ""]
        lines.append(f"- Valid citations: {len(self.citations_valid)}")
        if self.citations_invalid:
            lines.append(f"- **Invalid citations** (not retrieved or after the close): {', '.join(self.citations_invalid)}")
        if self.error:
            lines.append(f"- Error: {self.error}")
        return "\n".join(lines)


# -------------------------------------------------------------------- agent

ChatFn = Callable[..., object]


def _get(obj, key, default=None):
    """Read a field from either a dict or an object returned by the ollama library."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    try:
        val = obj[key]
        return default if val is None else val
    except (KeyError, TypeError, IndexError):
        return getattr(obj, key, default)


def _ollama_chat(**kwargs):
    import ollama
    return ollama.chat(**kwargs)


class MarketDetective:
    def __init__(self, toolbox: Toolbox, model: str = "qwen3:8b", temperature: float = 0.0,
                 max_tool_rounds: int = 6, chat: ChatFn | None = None, verbose: bool = True,
                 log: Callable[[str], None] | None = None):
        """`log` receives each step as it happens, so Streamlit can show progress live."""
        self.tb = toolbox
        self.model = model
        self.options = {"temperature": temperature}
        self.max_tool_rounds = max_tool_rounds
        self.chat = chat or _ollama_chat
        self.verbose = verbose
        self.log_fn = log

    def _log(self, msg: str) -> None:
        if self.log_fn is not None:
            self.log_fn(msg.strip())
        elif self.verbose:
            print(msg, flush=True)

    def _call(self, steps: list[Step], label: str, **kwargs):
        t0 = time.perf_counter()
        resp = self.chat(model=self.model, options=self.options, **kwargs)
        steps.append(Step(kind="llm", name=label, seconds=time.perf_counter() - t0,
                          prompt_tokens=int(_get(resp, "prompt_eval_count", 0) or 0),
                          output_tokens=int(_get(resp, "eval_count", 0) or 0)))
        return resp

    def _final_report(self, messages: list, steps: list[Step]) -> dict:
        resp = self._call(steps, "final_report", messages=messages + [{"role": "user", "content": FINAL_INSTRUCTION}],
                          format=LLMReport.model_json_schema())
        content = _get(_get(resp, "message"), "content", "") or ""
        # If the model wraps the JSON in extra text (a reasoning block, say), keep only the JSON.
        if "{" in content:
            content = content[content.index("{"): content.rindex("}") + 1]
        return LLMReport.model_validate_json(content).model_dump()

    def _check_citations(self, report: dict, ticker: str, date: str) -> tuple[list[str], list[str]]:
        valid, invalid = [], []
        for c in report.get("supported_causes", []):
            ok = [i for i in c.get("evidence_ids", []) if self.tb.is_valid_citation(i, ticker, date)]
            bad = [i for i in c.get("evidence_ids", []) if i not in ok]
            valid += ok
            invalid += bad
        return sorted(set(valid)), sorted(set(invalid))

    # ----------------------------------------------------------------- agent mode

    def investigate(self, ticker: str, date: str, question: str | None = None) -> RunResult:
        question = question or f"Why did {ticker} stock move on {date}? Was the move unusual, and what caused it?"
        self.tb.seen_news.clear()
        steps: list[Step] = []
        t_start = time.perf_counter()
        messages: list = [{"role": "system", "content": SYSTEM_AGENT},
                          {"role": "user", "content": f"{question}\nTicker: {ticker}. Date: {date}."}]
        numbers, error, reminded = None, "", False

        try:
            for _ in range(self.max_tool_rounds):
                resp = self._call(steps, "reasoning", messages=messages, tools=TOOL_SCHEMAS)
                msg = _get(resp, "message")
                calls = _get(msg, "tool_calls", []) or []
                messages.append(msg)
                if not calls:
                    if numbers is None and not reminded:
                        reminded = True
                        messages.append({"role": "user", "content": "Call analyze_event first, then search_news."})
                        continue
                    break
                for call in calls:
                    fn = _get(call, "function")
                    name, args = _get(fn, "name", ""), dict(_get(fn, "arguments", {}) or {})
                    t0 = time.perf_counter()
                    result = self.tb.run(name, args)
                    if name == "analyze_event" and "error" not in result:
                        numbers = result
                    n_res = len(result.get("results", [])) if name == "search_news" else None
                    summary = result.get("error") or (f"{n_res} news" if n_res is not None else result.get("main_driver", ""))
                    steps.append(Step(kind="tool", name=name, args=args,
                                      seconds=time.perf_counter() - t0, summary=str(summary)))
                    self._log(f"  → {name}({', '.join(f'{k}={v!r}' for k, v in args.items())})  [{summary}]")
                    messages.append({"role": "tool", "content": json.dumps(result), "tool_name": name})

            if numbers is None:   # the model never called analyze_event, so the code adds the numbers
                numbers = self.tb.analyze_event(ticker, date)
            self._log("  → writing the report...")
            report = self._final_report(messages, steps)
        except Exception as e:  # noqa: BLE001 - the error goes into the result, the trace is kept
            report, error = {}, f"{type(e).__name__}: {e}"

        valid, invalid = self._check_citations(report, ticker, date)
        return RunResult("agent", self.model, ticker.upper(), date, question, numbers, report,
                         valid, invalid, steps, time.perf_counter() - t_start, error)

    # ------------------------------------------------------------- RAG baseline

    def rag_baseline(self, ticker: str, date: str, question: str | None = None, k: int = 8) -> RunResult:
        question = question or f"Why did {ticker} stock move on {date}? Was the move unusual, and what caused it?"
        self.tb.seen_news.clear()
        steps: list[Step] = []
        t_start = time.perf_counter()
        report, error = {}, ""
        try:
            hits = self.tb.index.search_event(question, ticker, date, lookback_days=2, k=k)
            for h in hits:
                self.tb.seen_news[h.item.id] = h.item
            steps.append(Step(kind="tool", name="retrieve", args={"query": question}, summary=f"{len(hits)} news"))
            context = "\n".join(f"[{h.item.id}] {h.item.published[:16]} {h.item.source}: "
                                f"{h.item.headline}. {h.item.summary[:300]}" for h in hits) or "No news found."
            messages = [{"role": "system", "content": SYSTEM_RAG},
                        {"role": "user", "content": f"{question}\nTicker: {ticker}. Date: {date}.\n\nNews:\n{context}"}]
            self._log(f"  → retrieved {len(hits)} news, writing the report...")
            report = self._final_report(messages, steps)
        except Exception as e:  # noqa: BLE001
            error = f"{type(e).__name__}: {e}"
        valid, invalid = self._check_citations(report, ticker, date)
        return RunResult("rag", self.model, ticker.upper(), date, question, None, report,
                         valid, invalid, steps, time.perf_counter() - t_start, error)
