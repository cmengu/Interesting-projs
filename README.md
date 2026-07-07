# Interesting Projects

Frontier-research project ideas, plus the survey that motivated them. Bias toward **frontier research-engineer credibility**: small, sharp, API-only studies that end in a shareable result (a scaling curve, a benchmark, a negative result, an open eval harness).

## Contents

- **[api-only-frontier-re-projects.md](./api-only-frontier-re-projects.md)** — 13 project ideas buildable with **no GPU** (just model APIs), each with a full spec: research question, method, exact models/datasets, baselines, the one plottable artifact to ship, stretch goals, and the "what makes it boring" trap. Grouped by: reward modeling & verifiers, test-time compute, evals & measurement, data-centric, and **VLM/VLA** (vision-language(-action), API-only). Includes a practical-setup section (models, the Anthropic-logprobs gotcha, cost discipline, stats hygiene) and an appendix with the broader GPU-dependent set.

- **[frontier-2026-beyond-world-models.md](./frontier-2026-beyond-world-models.md)** — a mid-2026 survey of the AI/ML frontier *beyond* world models and harnesses: RL-for-agents, data-free self-improvement, test-time compute (and where it's hype), post-Transformer architectures, native multimodal, embodied/VLAs, AI-for-science, mech-interp correcting itself, and continual learning / test-time training. Compiled from a deep-research harness run (78 search/fetch/verify agents → 136 claims → 42 adversarial verdicts; 4 refuted claims corrected inline).

## Guiding thesis

**Own the evals.** Verifiers, reward models, and measurement are the real bottleneck at the frontier — and all of them can be attacked with an API alone. Test-time compute keeps scaling with a *good verifier* and plateaus without one; the lever is verification quality, not sample count. These projects are ways to build that muscle in public.

## Suggested starting pair

1. **The Verifier Wall** (verifier-guided scaling where verification is hard) — cleanest expression of the thesis; genuinely open result.
2. **How to Break an LLM Judge** — reuses existing cross-family adversarial infra (Council); shippable in ~2 weeks.

...and the exciting new lane: **"Can Frontier VLMs Actually Reason About Space?"** — an embodied-reasoning benchmark that's the VLM core of every VLA, with a spicy negative result.
