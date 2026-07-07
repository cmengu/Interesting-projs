# API-Only Frontier-RE Project Ideas

*Projects buildable with **no GPU** — just model APIs (chat + logprobs + embeddings, occasionally hosted fine-tuning). Curated for **frontier research-engineer credibility**: empirical taste + a clean, shareable result. Each is designed to end in a postable artifact — a scaling curve, a benchmark, a negative result, an open eval harness. Aim each at an arXiv note / lab-quality blog + open-sourced code.*

*Context: these grew out of the mid-2026 frontier survey in [`frontier-2026-beyond-world-models.md`](./frontier-2026-beyond-world-models.md). North-star thesis: **own the evals** — verifiers/reward-models/measurement are the real bottleneck, and you can attack all of them with an API alone.*

⭐ = strongest opening bets.

---

## A. Reward modeling & verifiers — the #1 frontier-RE bottleneck, and the core thesis

### 1. Verifier-guided scaling where verification is *hard* ⭐
The whole field's test-time-scaling results live in math/code because checking answers is free there. **Open question:** does the "verifier beats majority-vote" lever survive where verification is hard?
- **Build:** pick a domain with no cheap checker (medical QA, multi-step open reasoning, summary faithfulness). Generate N candidates via API; compare selection by (a) self-consistency, (b) a prompted *generative* verifier, (c) an **RLPR-style self-logprob reward** (use the API's logprobs), (d) oracle. Plot accuracy vs. N; find where each plateaus.
- **RE signal:** reward-model design is *the* bottleneck; extending the result off-math is genuinely unsolved.
- **Effort:** ~2 weeks. **Deliverable:** scaling-curve write-up + open eval harness.

### 2. LLM-judge stress-test + robust aggregation ⭐ *(build on the Council harness)*
- **Build:** construct adversarial cases that fool single judges (sycophancy, verbosity/position bias, confidently-wrong). Measure error rates, then test recovery via panel-of-N, debate, and judge-with-verification. Quantify how much each fixes.
- **RE signal:** LLM-as-judge underpins all of RLAIF + eval; its failure modes are a live safety problem. Reuse the existing cross-family adversarial infra (Claude-proposes / Codex-critiques) — turn Council into a *measured* judge-robustness study.
- **Effort:** ~2 weeks.

### 3. Black-box CoT faithfulness suite — *interp without weights*
- **Build:** reproduce Anthropic's faithful-vs-unfaithful distinction using only prompting — inject a hint and check if the model *uses but doesn't admit* it; perturb reasoning steps and see if the answer follows. Quantify unfaithfulness rates across 5 models.
- **RE signal:** CoT faithfulness is central to trusting reasoning models; everyone assumes you need internals — a rigorous *behavioral* suite is a fresh angle.
- **Effort:** ~2 weeks.

## B. Test-time compute — hot, cheap, API-native

### 4. Adaptive compute-allocation router ⭐
- **Build:** train a tiny difficulty predictor (on API embeddings) that decides *per query* how many samples to spend; show you match fixed-large-N accuracy at a fraction of average cost on a mixed-difficulty set.
- **RE signal:** "efficient test-time compute" is hot and almost everyone spends *uniform* compute — per-query allocation matching accuracy at lower cost is exactly the shape of result labs like.
- **Effort:** ~2–3 weeks.

### 5. TTRL-lite: inference-time self-improvement
- **Build:** reproduce AlphaProof's "learn from problem variants at inference" at toy scale via API — for a target problem, auto-generate related variants, solve them, feed *verified* solutions back in-context, measure lift on the target. No weight updates, pure in-context.
- **RE signal:** cleanly *isolates the self-improvement mechanism from training* — nobody's done that crisply for general tasks.
- **Effort:** ~2–3 weeks.

## C. Evals & measurement — a good benchmark is a classic RE launchpad

### 6. A benchmark for something the field can't measure
- **Build:** pick a capability you've personally hit as a harness builder — long-horizon *self-consistency* (does the agent contradict itself over 50 turns?), or faithfulness-under-distraction. 150–300 items + an automatic scorer; baseline 5 frontier models.
- **RE signal:** people cite and *use* benchmarks; the capability you choose is the novelty, and it shows research taste.
- **Effort:** ~3 weeks.

### 7. Contamination / memorization audit — *slightly subversive, very useful*
- **Build:** take popular benchmarks, perturb items (rename entities, swap numbers, paraphrase), measure the accuracy drop per model. Big drop = memorization, not reasoning.
- **RE signal:** eval integrity is a real crisis; a clean contamination probe gets attention and is a genuine service to the field.
- **Effort:** ~2 weeks.

## D. Data-centric — a direct audition for the GLOW.AI / Bryan Low lab

### 8. Data-selection beats more data ⭐ *(on-thesis for data-selection-for-LLM-training)*
- **Build:** take a fine-tuning set, build a selection method (dedup + difficulty + diversity, or an influence proxy), then **fine-tune via a hosted API** (no GPU) on your selected 20% vs. random 20% vs. 100%. Measure.
- **RE signal:** data-selection-for-LLM-training *is* the GLOW.AI direction — the most literal possible audition for that lab.
- **Effort:** ~2–3 weeks. *(Costs some $ for hosted fine-tunes, but zero GPU.)*

---

## Recommended opening pair
- **#1 (verifier-scaling off-math)** — cleanest expression of the own-the-evals thesis, and the result is genuinely open, so it can't be a boring reproduction.
- **#2 (judge stress-test on Council)** — reuses existing infra; reward-model robustness is a hot, safety-flavored result shippable in two weeks.

Then **#8** to line up the Bryan Low conversation with a real artifact in hand.

---

## Appendix — broader set (includes GPU-dependent projects, for reference)

| # | Project | Frontier | Compute |
|---|---------|----------|---------|
| A1 | Verifier-scaling harness (majority-vote vs. trained reranker vs. oracle) | Test-time compute | API-only |
| A2 | Mini-Robin: autonomous drug-repurposing agent with a real verification loop | AI-for-science / agents | API + open bio models |
| A3 | "Make a Transformer sub-quadratic in a weekend" — scoped HALO linearization | Post-Transformer / kernels | 1 GPU |
| A4 | GRPO/DAPO from scratch on a small multi-turn agent (WebAgent-R1-style) | Agentic RL | 1–2 GPUs |
| A5 | Attribution-graph interp reproduction on Gemma-2-2b | Mech interp | Local small model |
| A6 | Continual-learning / TTT toy: fold a long doc into weights at inference | Continual learning / TTT | 1 GPU |

*A1 is the same as #1 above; A2 doubles as healthtech + Hummingbird Bioscience interview material.*
