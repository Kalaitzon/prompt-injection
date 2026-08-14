# Bank Support Assistant — Prompt-Injection Attack & Defense Benchmark

This project implements a small LLM-based banking support assistant, attacks it
with a structured prompt-injection benchmark, hardens it with three independent
defense layers, and evaluates the result with an ablation study, an error
analysis, an interpretability dashboard, and two optional extensions (an
independent judge and a cross-model comparison).

The full documentation of the design, threat model, methodology, and results is
provided separately in the Report. This README is the map: what each file does,
how the files map to the assignment tasks, and exactly how to run everything.

All source files are documented: each module begins with a docstring explaining
its purpose and design choices, and contains inline comments justifying the
non-obvious decisions (for example, why leak detection is kept independent of
the defenses, and how reproducibility is handled honestly).

---

## 1. Folder structure

The project uses a clean folder layout. Place the files exactly as below. Paths
are resolved relative to the source files, so the scripts work regardless of the
folder you launch them from (from the project root as `python src/runner.py`, or
inside `src/` as `python runner.py`). The `benchmark/`, `logs/`, `results/`,
`transcripts/`, and `figures/` folders are created automatically when you run the
scripts. You only need to place the `.py` files.

```
prompt-injection/                 <- project root (any name)
|
+-- src/                          ALL Python source files here:
|     config.py                   central configuration, canary registry
|     providers.py                model backends: replay / anthropic / ollama
|     logging_utils.py            structured JSONL interaction logging
|     canary.py                   independent leak detection (canary tokens)
|     assistant_baseline.py       baseline assistant + system prompt (task 1)
|     defenses.py                 the three defense layers (task 4)
|     assistant_defended.py       baseline + defenses, with toggleable layers
|     benchmark_generator.py      builds the test corpus (task 2)
|     runner.py                   runs the benchmark, computes metrics (task 3)
|     ablation.py                 8-configuration ablation study (task 5)
|     error_analysis.py           confusion matrix + FP/FN causes (task 6)
|     dashboard.py                per-request risk/defense dashboard (task 7)
|     judge.py                    independent LLM judge (optional)
|     cross_model.py              hosted-vs-local comparison (optional)
|
+-- benchmark/                    prompt_corpus.json , prompt_corpus.csv
+-- logs/                         baseline/defended _interactions.jsonl
+-- results/                      *_results.json, ablation.json,
|                                 error_analysis.json, judge_cross_check.json,
|                                 cross_model.json
+-- transcripts/                  saved model calls (auto, for replay)
+-- figures/                      figures for the report (auto)
+-- screenshots/                  execution screenshots for the report
|
+-- README.md                     this file
+-- REPORT                        full report (separate document)
+-- requirements.txt              Python dependencies
+-- .env.example                  template for the API key (live runs only)
```

---

## 2. Requirements

Python 3.10 or newer is recommended.

The project runs OFFLINE by default in "replay" mode: no API key, no network,
fully deterministic. This lets the whole pipeline be executed and reproduced with
no setup.

For the actual results in the report, a live local model was used via Ollama. To
reproduce them you need:

```bash
pip install ollama              # the python package
# plus an installed and running Ollama server (https://ollama.com)
ollama pull llama3.1:8b         # the primary model
ollama pull mistral             # the second model (for the cross-model study)
```

(Alternatively, for a hosted model: `pip install anthropic`, set
`ANTHROPIC_API_KEY`, and `set BANK_PROVIDER=anthropic`.)

---

## 3. How to run (in order)

All commands run from the project root (the folder containing `src/`). On Windows
use `python`. First set the provider to ollama:

```bash
set BANK_PROVIDER=ollama
```

**Step 0 — Ollama setup**
```bash
ollama --version
ollama pull llama3.1:8b
ollama pull mistral
pip install ollama
```

**Step 1 — Generate the corpus (task 2)**
```bash
python src/benchmark_generator.py
# produces benchmark/prompt_corpus.json and .csv (60 cases)
```

**Step 2 — Baseline evaluation (task 3)**
```bash
python src/runner.py --system baseline
# results/baseline_results.json
# expected: attack success ~7%, benign ~100%, false refusals 0%
```

**Step 3 — Defended evaluation (task 4 & 5)**
```bash
python src/runner.py --system defended
# results/defended_results.json
# expected: attack success 0%, benign 100%, false refusals 0%
```

**Step 4 — Error analysis (task 6)**
```bash
python src/error_analysis.py
# results/error_analysis.json — confusion matrix, precision/recall/F1, FN/FP causes
```

**Step 5 — Ablation study (task 5)**
```bash
set BANK_ABLATION_RUNS=1
python src/ablation.py
# results/ablation.json — 8 configurations, marginal contribution of each layer
```

**Step 6 — Interpretability dashboard (task 7)**
```bash
python src/dashboard.py --demo
# shows input/output risk, defense trace, decision, per-request latency
```

**Step 7 — Independent judge (optional)**
```bash
python src/judge.py
# results/judge_cross_check.json — judge vs canary cross-check, agreement rate
```

**Step 8 — Cross-model comparison (optional)**
```bash
python src/cross_model.py --models ollama:llama3.1:8b ollama:mistral
# results/cross_model.json — baseline vs defended for both models,
# per-family defense transfer
```

Note: the security rates are deterministic (temperature=0), so they do not change
between runs. Only latency may vary slightly.

---

## 4. Task-to-file map

**Task 1 — Assistant and threat model.** The assistant ("Bank Support Assistant")
answers public-FAQ questions and classifies customer messages while holding
hidden protected items.
Files: `assistant_baseline.py` (assistant + hidden system prompt), `config.py`
(the protected "canary" items and thresholds), `providers.py` (model backend),
`logging_utils.py` (logging).

**Task 2 — Benchmark construction.** 60 cases: 30 benign (15 Q&A + 15
classification) and 30 adversarial across 5 families (direct override, role-play,
indirect injection via document, encoded/obfuscated, multi-turn decomposition).
The multi-turn cases are genuine multi-step sequences. Attack success is defined
as a canary-token leak, checked independently of the defenses.
Files: `benchmark_generator.py` → `prompt_corpus.json`, `prompt_corpus.csv`;
`canary.py` (independent success criterion).

**Task 3 — Baseline evaluation.** The runner executes each case (with correct
multi-turn state) and reports attack success overall / per family / per
difficulty, benign success, false-refusal rate, and latency.
Files: `runner.py` → `baseline_results.json`, `baseline_interactions.jsonl`.

**Task 4 — Defense system (three layers).** Three independent layers acting at
different stages:
- Layer 1 — input validation (pre-generation, decodes obfuscations)
- Layer 2 — context separation (isolates untrusted document content)
- Layer 3 — output filtering (post-generation, blocks/redacts leakage)

Files: `defenses.py`, `assistant_defended.py` → `defended_results.json`,
`defended_interactions.jsonl` (from `runner.py --system defended`).

**Task 5 — Comparative evaluation / ablation.** All 8 layer combinations are
tested with repetition and reported as mean +/- standard deviation, plus a
marginal-contribution analysis quantifying each layer's mean security gain and
usability cost.
Files: `ablation.py` → `ablation.json`.

**Task 6 — Error analysis.** Builds a confusion matrix with precision / recall /
F1, lists false negatives (attacks that leaked) with per-family causes, and
splits false positives by cause (defense block vs model refusal vs accidental
leak), separating genuine defense failures from task-classifier quirks.
Files: `error_analysis.py` → `error_analysis.json`.

**Task 7 — Interpretable feature.** A per-request dashboard showing the real
input/output risk scores, which defense layers triggered and why, the final
decision, and the actually-measured latency.
Files: `dashboard.py`.

---

## 5. Optional extensions

**Optional 1 — Independent judge.** A second LLM acts as an independent leakage
oracle, complementing the exact canary detector by catching paraphrased
disclosure. The two are cross-checked and their agreement / disagreement is
reported, so the success metric is never collapsed into the defense logic.
Files: `judge.py` → `judge_cross_check.json`.

**Optional 2 — Cross-model comparison (hosted vs local).** Runs baseline and
defended under two model backends and measures whether the defenses transfer
across models (per-family "defense transfer" delta).
Files: `cross_model.py` → `cross_model.json`.

---

## 6. Documentation

The Report (separate document) contains the full account: the threat model and
assumptions, the design of the assistant and the protected items, the benchmark
construction and labelling, baseline and defended results with tables and
figures, the ablation study and its trade-off discussion, the error analysis,
the two optional extensions, an ethics / responsible-AI discussion, and an
explicit statement of which security claims are evidenced and which remain
untested.
