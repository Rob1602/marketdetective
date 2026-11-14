# Evaluation · 18 runs · qwen3:8b

## Comparison by mode

| Metric | Agent | RAG baseline |
|---|---|---|
| Events analysed | 9 | 9 |
| Correct cause identified | 7/9 | 6/9 |
| Valid citations (total) | 23 | 31 |
| Fabricated citations (total) | 3 | 0 |
| Unverified numeric claims | 1 | 11 |
| Reports measured figures | 9/9 | 0/9 |
| Hypotheses per answer (avg) | 1.3 | 2.1 |
| Time per event (avg, s) | 339 | 163 |
| Tokens per event (avg) | 11416 | 2398 |
| Runs ending in an error | 0 | 0 |

## Per run

| Event | Mode | Cause found | Valid cit. | Fabricated cit. | Unverified numbers | Hypotheses | Seconds | Tokens |
|---|---|---|---|---|---|---|---|---|
| 2026-02-02 | Agent | NO | 1 | 0 | 0 | 3 | 456 | 12359 |
| 2026-02-02 | RAG baseline | NO | 4 | 0 | 0 | 2 | 193 | 2348 |
| 2026-04-27 | Agent | yes | 3 | 0 | 0 | 2 | 342 | 18090 |
| 2026-04-27 | RAG baseline | yes | 2 | 0 | 3 | 3 | 71 | 1834 |
| 2026-04-30 | Agent | yes | 1 | 0 | 0 | 2 | 349 | 11826 |
| 2026-04-30 | RAG baseline | yes | 2 | 0 | 4 | 3 | 234 | 3095 |
| 2026-06-01 | Agent | yes | 5 | 0 | 0 | 0 | 172 | 11714 |
| 2026-06-01 | RAG baseline | yes | 2 | 0 | 0 | 2 | 111 | 2607 |
| 2026-07-15 | Agent | yes | 0 | 3 | 0 | 0 | 223 | 4262 |
| 2026-07-15 | RAG baseline | NO | 3 | 0 | 0 | 2 | 289 | 2901 |
| 2026-07-27 | Agent | NO | 0 | 0 | 0 | 3 | 378 | 13081 |
| 2026-07-27 | RAG baseline | yes | 5 | 0 | 2 | 1 | 210 | 2877 |
| 2026-08-27 | Agent | yes | 4 | 0 | 0 | 2 | 275 | 7415 |
| 2026-08-27 | RAG baseline | yes | 4 | 0 | 1 | 2 | 110 | 1877 |
| 2026-11-04 | Agent | yes | 4 | 0 | 1 | 0 | 302 | 12006 |
| 2026-11-04 | RAG baseline | NO | 3 | 0 | 0 | 1 | 173 | 2060 |
| 2026-11-09 | Agent | yes | 5 | 0 | 0 | 0 | 558 | 11990 |
| 2026-11-09 | RAG baseline | yes | 6 | 0 | 1 | 3 | 76 | 1981 |

**Reading:** *cause found* is a whole-word match against the ground truth's key terms (`data/ground_truth.yaml`) — a screen, not a judge. *Fabricated citations* are articles never retrieved, or published after the close of the event day. *Unverified numbers* are percentages in the text matching no value produced by the event study (0.5 pp tolerance); for the RAG baseline, which computes nothing, every percentage falls here by construction.
