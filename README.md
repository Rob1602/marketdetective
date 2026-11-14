# MarketDetective

**A local LLM agent that investigates unusual stock price moves.**

Ask *"Why did NVIDIA move on 2026-11-09?"* and the system measures how unusual the move was,
splits it into market, sector and company-specific components, searches a news archive for
candidate causes, and writes a report that keeps evidence-backed causes separate from hypotheses.

Everything runs locally and for free: Qwen3 8B via [Ollama](https://ollama.com), no paid API.

```
→ analyze_event(ticker='NVDA', date='2026-11-09')          [company-specific]
→ search_news(query='earnings results guidance', ...)      [5 news]
→ search_news(query='product launch regulatory issues')    [5 news]
→ writing the report...
```

---

## Key finding

The point of the project is not only to build the agent, but to test whether the agentic design is
worth its cost. It is compared against a RAG baseline: same model, same archive, same questions,
but no tools. The baseline does one retrieval, pastes the documents into the prompt and answers.

> On identifying causes, the agent gained very little: 7 of 9 events against the baseline's 6, at
> 4.8 times the tokens and 1.9 times the wall-clock time.
>
> What it gained is the ability to recognise when there is nothing to explain. On a control day
> with no anomalous move, the agent said so and showed the figures. The baseline listed three
> causes and described the day as unusual.

Three results are more informative than the totals.

**Borrowed figures.** The baseline has no way to measure anything, so it takes numbers from the
articles it reads. One report stated that the stock *"fell over 5% pre-market and reached its most
oversold level since 2019"*. That claim comes from an article about a different trading day. Across
nine events the baseline made 11 unverified numeric claims against the agent's 1. The agent cannot
do this: every figure in its report is inserted from the event study and never read from the
model's own text.

**The control day.** 2026-07-15 was included on purpose. NVDA rose 0.33% in line with the market,
abnormal return +0.01%, t = 0.00. The two systems answered very differently.

> **Agent:** *"The +0.33% return was entirely explained by broader market (SPY: +0.70%) and sector
> (SOXX: −0.24%) movements. No company-specific news or events were found to explain the move. The
> abnormal return (+0.01%) was statistically insignificant (p-value = 0.9973)."*
>
> **Baseline:** *"likely driven by analyst upgrades, confirmed business expansions (e.g., H200 chip
> shipments to China), and optimism about NVIDIA's AI leadership. The move appears unusual..."*

The baseline had no way to know that +0.33% was an ordinary day, so it looked for a reason and
found one. The agent had the statistics and used them to say that no explanation was needed. This
is the clearest argument for giving the model statistical tools, and it is visible only because the
control case was included. Accuracy on real events alone would not have shown it.

**Required fields get filled.** The agent was not faultless either. Its two causes on the control
day were correct (*"Market-driven movement"*, *"Sector-wide dynamics"*), but the output schema
requires an `evidence_ids` list on every supported cause and there was no news to cite. It produced
`MARKET_20260715_001`, `SPY_20260715_002` and `SOXX_20260715_003`, identifiers that look plausible
but do not exist. The citation checker rejected all three. They are the only fabricated citations in
the experiment. The analysis was correct and the evidence field was invented, because the schema
offered no way to state that there was no evidence.

---

## Results

9 events · 2 modes · 18 runs · 7,935 indexed articles · Qwen3 8B
[full tables](evaluation/results.md) · [raw runs](runs/)

| Metric | Agent | RAG baseline |
|---|---|---|
| Correct cause identified | 7/9 | 6/9 |
| Valid citations | 21 | 31 |
| **Fabricated citations** | **3** | **0** |
| **Unverified numeric claims** | **1** | **11** |
| **Reports measured figures** | **9/9** | **0/9** |
| Hypotheses per answer (avg) | 1.3 | 2.1 |
| Time per event (avg) | 315 s | 163 s |
| Tokens per event (avg) | 11,404 | 2,398 |
| Runs ending in an error | 0 | 0 |

Per event:

| Event | Move | Cause (ground truth) | Agent | Baseline |
|---|---|---|---|---|
| 2026-02-02 | −2.9% | $100bn OpenAI investment reported paused | no | no |
| 2026-04-27 | +3.9% | *No discrete catalyst*: AI build-out momentum, bullish analyst note | yes | yes |
| 2026-04-30 | −4.7% | Competition from hyperscalers' custom AI chips | yes | yes |
| 2026-06-01 | +6.1% | Computex: entry into the PC chip market (RTX Spark, Vera CPU) | yes | yes |
| **2026-07-15** | **+0.3%, t = 0.00** | **Control: nothing to explain** | yes *(but 3 fabricated citation IDs)* | no *(invented causes, called it "unusual")* |
| 2026-07-27 | −5.1% | Circular-financing fears over the $250bn OpenAI backstop | no | yes |
| 2026-08-27 | +8.4% | Q2 earnings beat and strong guidance | yes | yes |
| 2026-11-04 | −3.2% | *No clean attribution*: China chip shortfall dominates coverage, post-election selloff is the only explicit link | yes | no |
| 2026-11-09 | −5.3% *while the market rose 4.2%* | Board investigation into CEO Jensen Huang | yes | yes |

The agent missed 2026-02-02 and 2026-07-27. In both cases the cause was a financial or structural
story rather than a single corporate event. The agent's searches start from conventional categories
such as earnings, products and regulation, and do not reach that kind of story. This is a limit of
the prompt rather than of the retriever: the relevant articles were in the archive.

**Reading the tables.** An *unverified numeric claim* is a percentage in the answer that matches no
value produced by the event study, with a 0.5 pp tolerance. The baseline computes nothing, so every
percentage it states falls into that column, which is the point of the comparison. *Correct cause*
is a whole-word match against the ground truth's key terms. It is a screen rather than a judgment,
so every run was also read by hand.

---

## Method

### 1. Is the move unusual?

A market model is estimated over 120 trading days, separated from the event by a 10-day gap:

```
R_stock = α + β_m · R_market + β_s · (R_sector − R_market) + ε
```

The sector factor is expressed in excess of the market, so the two factors are less collinear and
the event-day return splits into parts that can be read separately:

**return = α + market + sector + company-specific (abnormal return)**

The move is flagged as anomalous when |t| of the abnormal return exceeds 2. For NVDA on 2026-11-09:
−5.32% = +4.24% (market) + 0.80% (sector) − 10.34% (company-specific), t = −5.75, p < 0.001.

### 2. No look-ahead, enforced by tests

- α and β are estimated only on data preceding the event window. A test multiplies all data from the
  event onward by 5 and checks that the estimates do not change.
- The retriever returns only news published before the 16:00 New York close of the event day. A test
  places an article 30 minutes after the close and checks that it is never retrieved.

Both rules matter because news published after a move explains it trivially. An agent allowed to
read it would score well without having learned anything.

### 3. Agent vs baseline

|  | RAG baseline | Agent |
|---|---|---|
| Who decides what to retrieve | the program (one query = the user's question) | the model |
| Retrievals per answer | 1, fixed | as many as it chooses |
| Access to statistics | none | `analyze_event` tool |
| Steps | 2 | 6–8 |

The agent is a plain tool-calling loop over Ollama's chat API, with no orchestration framework, so
every step can be inspected. Two tools are exposed: `analyze_event` (the event study) and
`search_news` (date-filtered retrieval).

### 4. Guardrails in code, not in the prompt

The model is not trusted to behave:

- **Numbers** in the final report are inserted from the event study, never read from model output.
- **Citations** are verified. Each cited ID must correspond to an article the model actually
  received and published before the close. Violations are reported as `Invalid citations`.
- **Output shape** is enforced with a Pydantic JSON schema, so supported causes and hypotheses are
  always kept apart.

---

## Limitations

- **9 events, 1 ticker.** This is a pilot, not an accuracy claim. 7/9 and 6/9 cannot be told apart
  at this sample size, and a single control day carries more weight than it should. The next step is
  30 or more events across several tickers and sectors, with several control days.
- **Cause matching is lexical and had to be corrected twice.** The automatic check looks for key
  terms from the ground truth. Version 1 matched substrings and scored the control case as passed
  for both systems, because `no` occurs inside `not` and `announced`. Version 2 matched whole words
  and scored it as failed for both. Reading the two reports showed that the agent had in fact
  answered correctly, in wording the key terms did not cover (*"no company-specific news"*,
  *"statistically insignificant"*), so the terms were extended. Both corrections are in the commit
  history. The lexical check is a measurement tool and needs checking like any other.
- **The ground truth is my own reading** of the retrieved articles, not an independent source. Two
  of the nine events (2026-04-27, 2026-11-04) have no clean attribution and are recorded as such:
  the press reports a dominant story without tying the price move to it.
- **News coverage is uneven.** Finnhub's free tier caps results per request, which is worked around
  by fetching one day at a time, and goes back about a year, so events have to be recent.
- **Headlines and summaries only**, not full article text, which limits how specific a cause can be.
- **One model.** Results may differ with a larger local model or a commercial API model.

---

## Setup (macOS, Apple Silicon)

```bash
conda create -n marketdetective python=3.11 -y
conda activate marketdetective
pip install -r requirements.txt

ollama pull qwen3:8b
ollama pull nomic-embed-text

# free key from finnhub.io
conda env config vars set FINNHUB_API_KEY=your_key
conda deactivate && conda activate marketdetective
```

## Usage

```bash
python -m pytest                                       # 47 tests

python scripts/find_events.py NVDA                     # find anomalous days
python scripts/find_events.py NVDA --quiet             # find quiet days (negative control)
python scripts/analyze_event.py NVDA 2026-11-09        # event study only
python scripts/download_news.py NVDA 2026-11-09        # fetch + archive news
python scripts/headlines.py NVDA 2026-11-09            # headlines used to set ground truth

python scripts/ask.py NVDA 2026-11-09                  # agent
python scripts/ask.py NVDA 2026-11-09 --mode rag       # baseline
python scripts/run_all.py NVDA DATE1 DATE2 DATE3       # whole experiment, unattended
python scripts/evaluate.py                             # comparison tables

streamlit run scripts/app.py                           # web interface
```

Tickers, benchmarks and all statistical parameters live in `config.yaml`.

## Project structure

```
config.yaml                 tickers, benchmarks, event-study parameters
src/marketdetective/
  stats.py                  market model, abnormal returns, decomposition
  data.py                   prices via yfinance, cached to CSV
  news.py                   Finnhub client, JSONL archive
  rag.py                    local embeddings, date-filtered retrieval
  tools.py                  the two tools exposed to the LLM
  agent.py                  tool-calling loop, RAG baseline, citation checks
  evaluation.py             scoring against ground truth
scripts/                    command-line entry points
  app.py                    Streamlit interface
data/news/                  news archive (committed, for reproducibility)
data/ground_truth.yaml      verified cause of each event
runs/                       every run, with steps, timings and token counts
evaluation/                 comparison tables
tests/                      47 tests on synthetic data and a scripted LLM
```

## Stack

Python · pandas · NumPy · SciPy · Pydantic · Streamlit · Ollama (Qwen3 8B, nomic-embed-text) · yfinance · Finnhub
