"""Compare the runs in runs/ with the ground truth in data/ground_truth.yaml.

Examples:
    python scripts/evaluate.py
    python scripts/evaluate.py --all     # include repeated runs, not only the latest one

Writes:
    evaluation/results.md           the comparison tables
    evaluation/results.csv          the same data as a spreadsheet
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketdetective.config import PROJECT_ROOT  # noqa: E402
from marketdetective.evaluation import load_ground_truth, load_runs, score_run, summarize  # noqa: E402

MODE_LABEL = {"agent": "Agent", "rag": "RAG baseline"}


def md_table(rows: list[list[str]], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score the saved runs against the ground truth")
    ap.add_argument("--all", action="store_true", help="include every run, not only the latest one per case")
    ap.add_argument("--runs", default=str(PROJECT_ROOT / "runs"))
    args = ap.parse_args()

    runs = load_runs(args.runs, latest_only=not args.all)
    if not runs:
        print(f"No runs in {args.runs}. Run scripts/ask.py first.")
        return
    truths = load_ground_truth(PROJECT_ROOT / "data" / "ground_truth.yaml")
    scores = [score_run(r, truths) for r in runs]

    out_dir = PROJECT_ROOT / "evaluation"
    out_dir.mkdir(exist_ok=True)

    # --- per-run table
    per_run_header = ["Event", "Mode", "Cause found", "Valid cit.", "Fabricated cit.",
                      "Unverified numbers", "Hypotheses", "Seconds", "Tokens"]
    per_run = [[s.date, MODE_LABEL.get(s.mode, s.mode), "yes" if s.cause_found else "NO",
                str(s.citations_valid), str(s.citations_invalid), str(len(s.unsupported_numbers)),
                str(s.n_hypotheses), f"{s.seconds:.0f}", str(s.tokens)] for s in scores]

    # --- summary table
    summary = summarize(scores)
    sum_header = ["Metric"] + [MODE_LABEL.get(m, m) for m in summary]
    def row(label, fn):
        return [label] + [fn(summary[m]) for m in summary]
    sum_rows = [
        row("Events analysed", lambda d: str(d["runs"])),
        row("Correct cause identified", lambda d: f"{d['cause_found']}/{d['runs']}"),
        row("Valid citations (total)", lambda d: str(d["citations_valid"])),
        row("Fabricated citations (total)", lambda d: str(d["citations_invalid"])),
        row("Unverified numeric claims", lambda d: str(d["unsupported_numbers"])),
        row("Reports measured figures", lambda d: f"{d['with_numbers']}/{d['runs']}"),
        row("Hypotheses per answer (avg)", lambda d: f"{d['avg_hypotheses']:.1f}"),
        row("Time per event (avg, s)", lambda d: f"{d['avg_seconds']:.0f}"),
        row("Tokens per event (avg)", lambda d: f"{d['avg_tokens']:.0f}"),
        row("Runs ending in an error", lambda d: str(d["errors"])),
    ]

    model = next((s.model for s in scores if s.model), "?")
    text = [f"# Evaluation · {len(scores)} runs · {model}", "",
            "## Comparison by mode", "", md_table(sum_rows, sum_header), "",
            "## Per run", "", md_table(per_run, per_run_header), "",
            "**Reading:** *cause found* is a whole-word match against the ground truth's key terms "
            "(`data/ground_truth.yaml`) — a screen, not a judge. *Fabricated citations* are articles "
            "never retrieved, or published after the close of the event day. *Unverified numbers* are "
            "percentages in the text matching no value produced by the event study (0.5 pp tolerance); "
            "for the RAG baseline, which computes nothing, every percentage falls here by "
            "construction.", ""]

    (out_dir / "results.md").write_text("\n".join(text), encoding="utf-8")

    # --- on-screen reminder of what to re-read by hand (kept out of the file)
    problems = [s for s in scores if s.missing_terms or s.citations_invalid or s.unsupported_numbers or s.error]
    screen = list(text)
    if problems:
        screen += ["## To re-read by hand (screen only)", ""]
        for s in problems:
            bits = []
            if s.missing_terms:
                bits.append("ground truth terms not mentioned: " + "; ".join(s.missing_terms))
            if s.citations_invalid:
                bits.append(f"{s.citations_invalid} invalid citations")
            if s.unsupported_numbers:
                bits.append("unverified percentages: " + ", ".join(s.unsupported_numbers))
            if s.error:
                bits.append(f"error: {s.error}")
            screen.append(f"- **{s.date} · {MODE_LABEL.get(s.mode, s.mode)}** — " + "; ".join(bits))
        screen.append("")

    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["event", "mode", "cause_found", "valid_citations", "fabricated_citations",
                    "unverified_numbers", "hypotheses", "confidence", "seconds", "steps",
                    "tool_calls", "tokens_in", "tokens_out", "error"])
        for s in scores:
            w.writerow([s.date, s.mode, int(s.cause_found), s.citations_valid, s.citations_invalid,
                        len(s.unsupported_numbers), s.n_hypotheses, s.confidence, round(s.seconds),
                        s.steps, s.tool_calls, s.prompt_tokens, s.output_tokens, s.error])

    print("\n".join(screen))
    print("Saved to evaluation/results.md and results.csv")


if __name__ == "__main__":
    main()
