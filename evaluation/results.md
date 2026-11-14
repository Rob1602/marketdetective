# Valutazione · 18 esecuzioni · modello qwen3:8b

## Confronto fra modalità

| Misura | Agente | Baseline RAG |
|---|---|---|
| Eventi analizzati | 9 | 9 |
| Causa corretta trovata | 6/9 | 6/9 |
| Citazioni valide (totale) | 21 | 31 |
| Citazioni inventate (totale) | 3 | 0 |
| Percentuali non verificate | 1 | 11 |
| Fornisce numeri misurati | 9/9 | 0/9 |
| Ipotesi per risposta (media) | 1.3 | 2.1 |
| Tempo medio (s) | 315 | 163 |
| Token medi | 11404 | 2398 |
| Errori | 0 | 0 |

## Dettaglio per esecuzione

| Evento | Modalità | Causa trovata | Cit. valide | Cit. inventate | Numeri non verificati | Ipotesi | Secondi | Token |
|---|---|---|---|---|---|---|---|---|
| 2026-02-02 | Agente | NO | 1 | 0 | 0 | 3 | 456 | 12359 |
| 2026-02-02 | Baseline RAG | NO | 4 | 0 | 0 | 2 | 193 | 2348 |
| 2026-04-27 | Agente | sì | 3 | 0 | 0 | 2 | 342 | 18090 |
| 2026-04-27 | Baseline RAG | sì | 2 | 0 | 3 | 3 | 71 | 1834 |
| 2026-04-30 | Agente | sì | 1 | 0 | 0 | 2 | 349 | 11826 |
| 2026-04-30 | Baseline RAG | sì | 2 | 0 | 4 | 3 | 234 | 3095 |
| 2026-06-01 | Agente | sì | 5 | 0 | 0 | 0 | 172 | 11714 |
| 2026-06-01 | Baseline RAG | sì | 2 | 0 | 0 | 2 | 111 | 2607 |
| 2026-07-15 | Agente | NO | 0 | 3 | 0 | 0 | 223 | 4262 |
| 2026-07-15 | Baseline RAG | NO | 3 | 0 | 0 | 2 | 289 | 2901 |
| 2026-07-27 | Agente | NO | 0 | 0 | 0 | 3 | 378 | 13081 |
| 2026-07-27 | Baseline RAG | sì | 5 | 0 | 2 | 1 | 210 | 2877 |
| 2026-08-27 | Agente | sì | 4 | 0 | 0 | 2 | 275 | 7415 |
| 2026-08-27 | Baseline RAG | sì | 4 | 0 | 1 | 2 | 110 | 1877 |
| 2026-11-04 | Agente | sì | 4 | 0 | 1 | 0 | 302 | 12006 |
| 2026-11-04 | Baseline RAG | NO | 3 | 0 | 0 | 1 | 173 | 2060 |
| 2026-11-09 | Agente | sì | 3 | 0 | 0 | 0 | 336 | 11884 |
| 2026-11-09 | Baseline RAG | sì | 6 | 0 | 1 | 3 | 76 | 1981 |

### Come leggere la tabella

- **Causa trovata**: la risposta contiene i termini chiave della causa vera (`data/ground_truth.yaml`). Controllo lessicale, quindi può sbagliare: va riletto a mano.
- **Citazioni inventate**: la risposta cita notizie mai recuperate, o pubblicate dopo la chiusura del giorno dell'evento.
- **Numeri non verificati**: percentuali scritte nel testo che non corrispondono a nessun valore calcolato dall'event study (tolleranza 0,5 punti). Per la baseline RAG, che non misura nulla, ogni percentuale citata rientra per costruzione in questa colonna: è esattamente il punto del confronto.

## Da controllare a mano

- **2026-02-02 · Agente** — termini della causa vera non citati: openai; pause / paused / snag / doubt / stall / delay / concern / funding
- **2026-02-02 · Baseline RAG** — termini della causa vera non citati: pause / paused / snag / doubt / stall / delay / concern / funding
- **2026-04-27 · Baseline RAG** — percentuali non verificate: 4%, 73.2%, 4%
- **2026-04-30 · Baseline RAG** — percentuali non verificate: 4%, 27%, 27%, 4%
- **2026-07-15 · Agente** — termini della causa vera non citati: no specific / no clear / no single / no evidence / no identifiable / no direct / nothing specific / does not explain / do not explain / not anomalous / not unusual / insufficient evidence / no significant / within its normal range; 3 citazioni non valide
- **2026-07-15 · Baseline RAG** — termini della causa vera non citati: no specific / no clear / no single / no evidence / no identifiable / no direct / nothing specific / does not explain / do not explain / not anomalous / not unusual / insufficient evidence / no significant / within its normal range
- **2026-07-27 · Agente** — termini della causa vera non citati: circular / backstop / financing / balance sheet / spending / openai
- **2026-07-27 · Baseline RAG** — percentuali non verificate: 5%, 4%
- **2026-08-27 · Baseline RAG** — percentuali non verificate: 4.7%
- **2026-11-04 · Agente** — percentuali non verificate: 20%
- **2026-11-04 · Baseline RAG** — termini della causa vera non citati: china / shortfall / export / licence / license / regulation / election / data center / data centre / crackdown
- **2026-11-09 · Baseline RAG** — percentuali non verificate: 5%
