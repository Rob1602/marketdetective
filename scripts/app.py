"""Interfaccia web di MarketDetective.

Si avvia con:
    streamlit run scripts/app.py

Si apre nel browser. Tre schede: l'investigazione di un evento, i risultati dell'esperimento
e lo storico delle esecuzioni già fatte. I testi mostrati all'utente sono in inglese, come
il resto del repository.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from marketdetective.agent import MarketDetective  # noqa: E402
from marketdetective.config import load_config  # noqa: E402
from marketdetective.evaluation import load_ground_truth, load_runs, score_run, summarize  # noqa: E402
from marketdetective.news import NewsArchive  # noqa: E402
from marketdetective.rag import NewsIndex, OllamaEmbedder  # noqa: E402
from marketdetective.tools import Toolbox  # noqa: E402

RUNS = ROOT / "runs"

st.set_page_config(page_title="MarketDetective", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------- setup

@st.cache_resource(show_spinner="Loading the news archive and embeddings...")
def get_index(news_dir: str, embedding_model: str):
    index = NewsIndex(NewsArchive(news_dir), OllamaEmbedder(embedding_model))
    index.build()
    return index


def pct_to_float(value) -> float:
    """'-5.32%' -> -5.32. Restituisce 0.0 se il valore non è una percentuale."""
    try:
        return float(str(value).replace("%", "").replace("+", "").strip())
    except (TypeError, ValueError):
        return 0.0


cfg = load_config()


# ------------------------------------------------------------------- sidebar

with st.sidebar:
    st.title("🔎 MarketDetective")
    st.caption("Why did a stock move the way it did?")

    tickers = list(cfg.tickers) or ["NVDA"]
    ticker = st.selectbox("Stock", tickers,
                          format_func=lambda t: f"{t} · {cfg.tickers[t].name}" if t in cfg.tickers else t)
    event_date = st.date_input("Event date", value=date(2026, 11, 9),
                               min_value=date(2018, 1, 2), max_value=date.today())
    mode = st.radio("Mode", ["agent", "rag"],
                    format_func=lambda m: "Agent (with tools)" if m == "agent" else "RAG baseline (no tools)")
    st.caption("The agent runs the event study and chooses its own searches. The baseline does a "
               "single retrieval and never sees a number.")

    # Modello e domanda vengono dalla configurazione: l'esperimento richiede che siano
    # identici su tutti gli eventi. Per variarli c'è scripts/ask.py (--model, --question).
    model = cfg.llm.get("model", "qwen3:8b")

    run_clicked = st.button("Investigate", type="primary", use_container_width=True)
    st.caption(f"Running {model} locally. The agent takes about 5 minutes, the baseline about 2.")


# ---------------------------------------------------------------------- tabs

tab_run, tab_experiment = st.tabs(["Investigation", "Experiment"])


# ------------------------------------------------------------- report render

def show_numbers(n: dict) -> None:
    st.subheader("The numbers")
    c = st.columns(4)
    c[0].metric("Day return", n["day_return"])
    c[1].metric("Explained by market", n["explained_by_market"], help=f"Index: {n['benchmarks']['market']}")
    c[2].metric("Explained by sector", n["explained_by_sector"], help=f"ETF: {n['benchmarks']['sector']}")
    c[3].metric("Company-specific", n["company_specific_abnormal_return"],
                help=f"t = {n['t_stat']}, p = {n['p_value']}")

    verdict = "ANOMALOUS move" if n["is_anomalous"] else "Move within the normal range"
    (st.error if n["is_anomalous"] else st.success)(
        f"**{verdict}** · t = {n['t_stat']}, p = {n['p_value']} · "
        f"main driver: {n['main_driver']}")

    chart = pd.DataFrame(
        {"contribution (%)": [pct_to_float(n["explained_by_market"]),
                              pct_to_float(n["explained_by_sector"]),
                              pct_to_float(n["company_specific_abnormal_return"])]},
        index=["market", "sector", "company-specific"])
    st.bar_chart(chart, height=220)
    st.caption(f"Cumulative abnormal return, days −1 to +1: "
               f"{n['cumulative_abnormal_return_days_-1_to_+1']}")


def show_report(result, news_by_id: dict) -> None:
    r = result.report
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Evidence-backed causes")
        causes = r.get("supported_causes") or []
        if not causes:
            st.info("No cause supported by the available news.")
        for cause in causes:
            ids = cause.get("evidence_ids", [])
            bad = [i for i in ids if i in result.citations_invalid]
            st.markdown(f"**{cause['cause']}**")
            for i in ids:
                item = news_by_id.get(i)
                if item:
                    st.caption(f"✅ `{i}` · {item['source']}, {item['published'][:10]} · {item['headline']}")
                else:
                    st.caption(f"❌ `{i}` · invalid citation: never retrieved, or published after the close")
            if bad:
                st.warning(f"{len(bad)} of {len(ids)} citations are invalid.")

        st.subheader("Hypotheses (not confirmed)")
        for h in r.get("hypotheses") or ["None"]:
            st.markdown(f"- {h}")

        st.subheader("Conclusion")
        st.markdown(r.get("conclusion", ""))
        st.caption(f"Confidence stated by the model: **{r.get('confidence', '?')}**")

    with right:
        st.subheader("Automatic checks")
        st.metric("Valid citations", len(result.citations_valid))
        st.metric("Invalid citations", len(result.citations_invalid),
                  help="Articles never retrieved, or published after the close of the event day")
        if result.citations_invalid:
            st.error("The model cited sources it never received: " +
                     ", ".join(f"`{i}`" for i in result.citations_invalid))
        else:
            st.success("Every citation checks out.")

        st.subheader("Cost")
        tin, tout = result.tokens
        st.metric("Time", f"{result.total_seconds:.0f} s")
        st.metric("Tokens (in / out)", f"{tin} / {tout}")
        st.metric("Steps", len(result.steps))

        if result.error:
            st.error(f"Error: {result.error}")


# -------------------------------------------------------------- investigazione

with tab_run:
    if not run_clicked:
        st.markdown("""
        ### How it works

        1. **The numbers:** an *event study* measures how unusual the move was and splits it into
           market, sector and company-specific components.
        2. **The news:** semantic search over the archive, restricted to articles published
           **before** the market close of the event day.
        3. **The report:** evidence-backed causes and hypotheses stay separate, and every citation
           is verified in code.

        Pick a stock, a date and a mode in the sidebar, then press **Investigate**.
        """)
    else:
        date_str = event_date.strftime("%Y-%m-%d")
        st.header(f"{cfg.tickers[ticker].name} · {date_str}")

        try:
            index = get_index(str(cfg.news_dir), cfg.embedding_model)
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not prepare the archive: {e}\n\n"
                     "Check that Ollama is running and that the embedding model is installed: "
                     f"`ollama pull {cfg.embedding_model}`")
            st.stop()

        steps_box = st.status("Working...", expanded=True)
        lines: list[str] = []

        def log(msg: str) -> None:
            lines.append(msg)
            steps_box.write(msg)

        toolbox = Toolbox(cfg, index)
        agent = MarketDetective(toolbox, model=model,
                                temperature=float(cfg.llm.get("temperature", 0.0)),
                                verbose=False, log=log)
        runner = agent.investigate if mode == "agent" else agent.rag_baseline
        try:
            result = runner(ticker, date_str)
            steps_box.update(label=f"Done in {result.total_seconds:.0f} seconds", state="complete",
                             expanded=False)
        except Exception as e:  # noqa: BLE001
            steps_box.update(label="Run failed", state="error")
            st.error(f"{type(e).__name__}: {e}")
            st.stop()

        # Mappa id -> notizia, per mostrare le citazioni per esteso.
        result_news = {i: {"source": it.source, "published": it.published, "headline": it.headline}
                       for i, it in toolbox.seen_news.items()}

        if result.numbers:
            show_numbers(result.numbers)
        else:
            st.warning("Baseline mode: no measured figures. The model only sees the news, so every "
                       "percentage it quotes comes from an article rather than from a computation.")

        show_report(result, result_news)

        RUNS.mkdir(exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = RUNS / f"{result.ticker}_{result.date}_{result.mode}_{stamp}"
        base.with_suffix(".json").write_text(json.dumps(result.to_dict(), indent=2, default=str),
                                             encoding="utf-8")
        base.with_suffix(".md").write_text(result.to_markdown(), encoding="utf-8")
        st.download_button("Download the report as Markdown", result.to_markdown(),
                           file_name=f"{base.name}.md", mime="text/markdown")
        st.caption(f"Saved to `runs/{base.name}.md`")


# ----------------------------------------------------------------- esperimento

with tab_experiment:
    st.header("Does the agentic design earn its cost?")
    st.caption("Every event run in both modes, same model and same archive.")

    all_runs = load_runs(RUNS) if RUNS.exists() else []
    if not all_runs:
        st.info("No runs yet. Investigate a few events in both modes, then come back.")
    else:
        truths = load_ground_truth(ROOT / "data" / "ground_truth.yaml")
        scores = [score_run(r, truths) for r in all_runs]
        summary = summarize(scores)
        labels = {"agent": "Agent", "rag": "RAG baseline"}

        rows = {
            "Events analysed": lambda d: str(d["runs"]),
            "Correct cause identified": lambda d: f"{d['cause_found']}/{d['runs']}",
            "Valid citations": lambda d: str(d["citations_valid"]),
            "Fabricated citations": lambda d: str(d["citations_invalid"]),
            "Unverified numeric claims": lambda d: str(d["unsupported_numbers"]),
            "Reports measured figures": lambda d: f"{d['with_numbers']}/{d['runs']}",
            "Hypotheses per answer (avg)": lambda d: f"{d['avg_hypotheses']:.1f}",
            "Time per event (avg)": lambda d: f"{d['avg_seconds']:.0f} s",
            "Tokens per event (avg)": lambda d: f"{d['avg_tokens']:,.0f}",
            "Runs ending in an error": lambda d: str(d["errors"]),
        }
        st.dataframe(
            pd.DataFrame({"Metric": list(rows)} |
                         {labels.get(m, m): [fn(summary[m]) for fn in rows.values()] for m in summary}),
            use_container_width=True, hide_index=True)

        st.subheader("Per run")
        st.dataframe(pd.DataFrame([{
            "Event": s.date,
            "Mode": labels.get(s.mode, s.mode),
            "Cause found": "yes" if s.cause_found else "NO",
            "Valid cit.": s.citations_valid,
            "Fabricated cit.": s.citations_invalid,
            "Unverified numbers": len(s.unsupported_numbers),
            "Hypotheses": s.n_hypotheses,
            "Seconds": round(s.seconds),
            "Tokens": s.tokens,
        } for s in scores]), use_container_width=True, hide_index=True)
