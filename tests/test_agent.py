import json

import pandas as pd

from conftest import EVENT_DATE, make_returns
from marketdetective.agent import MarketDetective
from marketdetective.config import EventStudyParams, load_config
from marketdetective.news import NewsArchive, NewsItem
from marketdetective.rag import NewsIndex
from marketdetective.tools import Toolbox
from test_rag import FakeEmbedder

DATE = "2024-04-19"


def news(i, when, headline):
    return NewsItem(id=f"n{i}", ticker="NVDA", published=pd.Timestamp(when, tz="UTC").isoformat(),
                    source="Test", headline=headline, summary="", url="")


class FakeStore:
    """Synthetic prices: the generator used by the statistics tests, turned into price levels."""

    def get_many(self, symbols, start, end=None):
        r = make_returns()
        prices = 100 * (r[["stock", "market", "sector"]]).cumsum().apply(lambda c: __import__("numpy").exp(c))
        return prices.rename(columns=dict(zip(["stock", "market", "sector"], [s.upper() for s in symbols])))


def toolbox(tmp_path):
    archive = NewsArchive(tmp_path)
    archive.add([
        news(1, "2024-04-19 14:00", "US weighs export ban on chips to China"),
        news(2, "2024-04-19 23:00", "China export ban confirmed after the close"),
    ])
    cfg = load_config()
    cfg = cfg.__class__(**{**cfg.__dict__, "event_study": EventStudyParams(estimation_window=200)})
    return Toolbox(cfg, NewsIndex(archive, FakeEmbedder()), store=FakeStore())


def call(name, **args):
    return {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]},
            "prompt_eval_count": 100, "eval_count": 10}


def text(content):
    return {"message": {"role": "assistant", "content": content}, "prompt_eval_count": 100, "eval_count": 20}


class ScriptedChat:
    """A fake LLM that replies with a fixed sequence and records what it receives."""

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


REPORT = json.dumps({
    "supported_causes": [{"cause": "Export ban fears", "evidence_ids": ["n1", "n2", "n999"]}],
    "hypotheses": ["Profit taking"],
    "conclusion": "The stock fell on export ban fears.",
    "confidence": "medium",
})


def test_toolbox_analyze_event_on_synthetic_prices(tmp_path):
    out = toolbox(tmp_path).analyze_event("nvda", EVENT_DATE)
    assert out["is_anomalous"] is True
    assert out["main_driver"] == "company-specific"
    assert out["day_return"].endswith("%")


def test_toolbox_errors_are_returned_to_the_model(tmp_path):
    tb = toolbox(tmp_path)
    assert "error" in tb.run("delete_everything", {})
    assert "error" in tb.run("analyze_event", {"ticker": "ZZZZ", "date": DATE})
    assert "error" in tb.run("analyze_event", {"ticker": "NVDA"})


def test_agent_loop_and_citation_checks(tmp_path):
    tb = toolbox(tmp_path)
    chat = ScriptedChat([
        call("analyze_event", ticker="NVDA", date=EVENT_DATE),
        call("search_news", ticker="NVDA", date=DATE, query="china export ban"),
        text("I have enough information."),
        text("Here is the report: " + REPORT),
    ])
    res = MarketDetective(tb, chat=chat, verbose=False).investigate("NVDA", DATE)

    assert res.error == ""
    assert [s.name for s in res.steps if s.kind == "tool"] == ["analyze_event", "search_news"]
    assert res.numbers["is_anomalous"] is True
    assert res.citations_valid == ["n1"]
    assert res.citations_invalid == ["n2", "n999"]        # n2 is after the close, n999 does not exist
    assert res.tokens == (400, 60)
    assert "tools" in chat.calls[0] and "format" in chat.calls[-1]
    md = res.to_markdown()
    assert "Invalid citations" in md and "Company-specific" in md


def test_agent_adds_numbers_if_model_never_calls_tools(tmp_path):
    chat = ScriptedChat([text("It fell."), text("Still no tools."), text(REPORT)])
    res = MarketDetective(toolbox(tmp_path), chat=chat, verbose=False).investigate("NVDA", EVENT_DATE)
    assert res.numbers is not None
    assert any("Call analyze_event" in str(m) for m in chat.calls[1]["messages"])


def test_rag_baseline_has_no_tools_and_no_numbers(tmp_path):
    chat = ScriptedChat([text(REPORT)])
    res = MarketDetective(toolbox(tmp_path), chat=chat, verbose=False).rag_baseline("NVDA", DATE)
    assert res.numbers is None
    assert "tools" not in chat.calls[0]
    assert "n2" not in chat.calls[0]["messages"][1]["content"]   # look-ahead excluded here too
    assert res.citations_valid == ["n1"]


def test_log_callback_receives_each_step(tmp_path):
    """Used by the Streamlit interface to show the steps as they happen."""
    lines = []
    chat = ScriptedChat([call("analyze_event", ticker="NVDA", date=EVENT_DATE),
                         call("search_news", ticker="NVDA", date=DATE, query="china"),
                         text("done"), text(REPORT)])
    MarketDetective(toolbox(tmp_path), chat=chat, verbose=False, log=lines.append).investigate("NVDA", DATE)
    assert any(line.startswith("→ analyze_event") for line in lines)
    assert any("search_news" in line for line in lines)
    assert lines[-1].startswith("→ writing the report")
    assert not any(line.startswith(" ") for line in lines)   # already stripped for the interface


def test_bad_json_is_reported_not_raised(tmp_path):
    chat = ScriptedChat([text("not json at all")])
    res = MarketDetective(toolbox(tmp_path), chat=chat, verbose=False).rag_baseline("NVDA", DATE)
    assert res.error and res.report == {}
