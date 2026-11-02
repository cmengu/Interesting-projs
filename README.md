<h1 align="center">Interesting Projects</h1>

<p align="center">Research notes on frontier-AI project ideas that can be built with model APIs alone, plus the survey that motivated them.</p>

## Why

Verifiers, reward models and measurement are a main bottleneck at the frontier, and an API is enough to work on all of them. The notes argue that test-time compute keeps scaling with a good verifier and plateaus without one. Each project is specified to end in one shareable result: a scaling curve, a benchmark, a negative result or an open eval harness.

This repo holds documents only; there is no code to run.

## Contents

- [api-only-frontier-re-projects.md](./api-only-frontier-re-projects.md): 13 project ideas that need no GPU. Each has a research question, method, models and datasets, baselines, the one artifact to ship, stretch goals and a failure mode to avoid. Groups: reward modeling and verifiers, test-time compute, evals and measurement, data-centric, and vision-language(-action). It also has a practical-setup section (models, the fact that Anthropic does not expose token logprobs, cost, statistics) and an appendix of GPU-dependent projects.
- [frontier-2026-beyond-world-models.md](./frontier-2026-beyond-world-models.md): a mid-2026 survey of AI/ML beyond world models and harnesses: RL for agents, data-free self-improvement, test-time compute, post-Transformer architectures, multimodal, embodied models, AI for science, interpretability and continual learning. Its header says it was compiled from a deep-research run of 78 agents, 136 claims and 42 adversarial verdicts.
- [reasoning-not-recall.md](./reasoning-not-recall.md): an eval design for comparing locally served open-weight models with a frontier API as research agents, built to measure reasoning on unseen problems rather than recall. Uses three recency buckets, perturbed-real and synthetic datasets, and tool-call reliability as a result. The full task cards, prompts and answer keys are in [reasoning-not-recall-tasks.md](./reasoning-not-recall-tasks.md).
- [open-science-feedback-11july.md](./open-science-feedback-11july.md) and [nebius-llm-reasoning-per-dollar-exec-brief.pdf](./nebius-llm-reasoning-per-dollar-exec-brief.pdf): notes on running the open-science harness with a local model, and an executive brief on open-weight LLM reasoning per dollar of Nebius GPU time.

## Suggested starting pair

1. The Verifier Wall: verifier-guided scaling where verification is hard.
2. How to Break an LLM Judge: reuses the Council cross-family adversarial harness; estimated at about two weeks.

A third option is "Can Frontier VLMs Actually Reason About Space?", an embodied-reasoning benchmark.
