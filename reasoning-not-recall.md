# Reasoning, not Recall: Evaluating LLMs as Scientific Research Agents

A methodology for benchmarking open-weight language models (served locally) against a frontier API model as **research agents** — systems that generate hypotheses and analyze data. It is designed so the benchmark measures **reasoning over unseen problems**, not **recall of memorized literature**. Worked examples use published oncology drug discovery for concreteness; the method is domain-general.

> All examples use publicly published findings and synthetic or public datasets. This is a methodology writeup, not a result set.
>
> **Full task spec** — paste-ready prompts, reference analyses with expected numbers, tiered answer keys, and reproducible scoring anchors: [reasoning-not-recall-tasks.md](./reasoning-not-recall-tasks.md).

---

## The problem with the obvious benchmark

The intuitive way to test a research agent is to hand it a public dataset with a known answer (GDSC drug-sensitivity, a landmark trial) and check whether it gets the answer right. This measures the wrong thing: every capable model has effectively *seen* those datasets and the papers analyzing them, so the benchmark rewards **recall** and flatters every model equally. It says nothing about whether a model can *reason over data it has never seen* — the only thing that matters for real research use.

Every design choice below serves one goal: separating "recalls the published finding" from "reasons over the unknown."

## Lever 1 — recency, done as three buckets (not two)

A naive fix is "only ask about findings published after the model's training cutoff." But different models have **different cutoffs**, so a single "fresh" bucket is asymmetric — a finding can be genuinely novel to one model and already memorized by another. The fix is three buckets:

| Bucket | Window | Weaker model knows? | Stronger model knows? | Isolates |
|---|---|---|---|---|
| **Classic** | well before both cutoffs | yes | yes | memory yardstick — both should ace |
| **Recency-gap** | between the two cutoffs | no | yes | the newer model's *extra training window* (not reasoning) |
| **True-fresh** | after *both* cutoffs | no | no | **reasoning + retrieval, symmetric** — the parity headline |

Only **true-fresh** feeds the headline "reasoning parity" number. The recency-gap bucket becomes a clean bonus finding: *what does a newer training window actually buy?*

## Lever 2 — novel data the models cannot have memorized

For the analysis workload, don't test on public-known-answers at all for the core signal. Three sources of provably-unseen ground truth:

1. **Synthetic planted-signal** — generate data where *you* inject the structure (an interaction, a Simpson's-paradox confound, a deliberate null). Provably uncontaminated, exact ground truth, tunable difficulty, no domain expert required.
2. **Perturbed real data (the "recall detector")** — take a real dataset and *break the known answer*: e.g. scramble mutation labels in GDSC so the textbook BRAF→vemurafenib association is destroyed and a new one is planted. A model that is actually analyzing reports the new truth; a model pattern-matching to memory confidently reports the textbook answer that is no longer there. This directly catches recall masquerading as reasoning.
3. **Private / held-out data (local-only demonstration)** — data definitionally not in any training set. When governance forbids sending it to a hosted API, the frontier model is *ineligible*, and that asymmetry is itself a headline: *this class of question can only run on locally-hosted models.*

## The two headline figures

Lead with **within-model paired contrasts** — they need no large sample and they *are* the proof:

- **Recall exposure:** the drop from a public positive-control task (BRAF→vemurafenib, unperturbed) to its perturbed twin (same drug, association surgically broken). A model whose competence is real holds up; a memory-leaner collapses.
- **Memory contribution:** the drop from classic to true-fresh tasks.

Per-model *ratios* (one model ÷ another) are reported only as **directional**, never as if they had statistical power at this scale.

## Tool-calling reliability is a first-class result

For an agent, raw knowledge matters less than whether it can reliably *drive its tools* — pull literature, query databases, chain steps. Locally-served models tend to fail here far more than on knowledge, and this failure mode **cannot be gamed by memorization**. Instrument it properly:

- calls attempted vs malformed (rate)
- recovery-after-error (does it recover from a failed call, or spiral?)
- multi-hop chains completed (literature → database → synthesize)

One dedicated **multi-connector retrieval task** exercises exactly this.

---

## Task set

### Hypothesis-generation tasks

Each prompt supplies only the background that *predates* the finding; the published result is the answer key. Grading is on the reasoning path and citation validity, not merely naming the drug.

| Bucket | Task | Answer key | Basis |
|---|---|---|---|
| True-fresh | Single-antigen HER3 ADCs meet resistance via EGFR/HER2 compensation and antigen downregulation — design a next-generation ADC | Dual-targeting **bispecific ADC** (EGFR×HER3 or TROP2×HER3) | Public, post-cutoff — JS212, JSKN016 |
| True-fresh | A HER3-directed ADC shows initial response then progressive efficacy loss — mechanisms + countermeasure | Acquired **HER3 downregulation/mutation** → less payload internalized; higher-DAR/bystander or bispecific counter | Public, post-cutoff — HER3-DXd (U31402) |
| Trap — contradictory | "Mechanism + pivotal trial for [an anti-HER3 ADC] + pembrolizumab in **HER3-negative** colorectal cancer" | **Refuse** — a HER3 ADC in a HER3-negative tumor is self-contradictory from mechanism alone | Purposely faked premise |
| Trap — real substrate, fake specifics | "Report the RADIANT-7 **Phase 3** results, **62% ORR**, for **sotorasib monotherapy**" (or any real, publicly-marketed drug) | **Decompose** — affirm the real drug, reject the invented trial name, phase, and ORR | Real public drug + fabricated trial/phase/number |
| Classic | Drug the historically "undruggable" KRAS G12C | **Covalent** inhibitor on the mutant **Cys-12**, switch-II GDP-bound pocket | Public — sotorasib (CodeBreaK 100) |
| Classic | Extend a HER2 therapy to "HER2-low" ("HER2-negative") tumors | ADC + high DAR + **bystander effect** | Public — trastuzumab deruxtecan (DESTINY-Breast04) |
| Recency-gap (bonus) | Broaden KRAS inhibition beyond G12C to the mutation family | **Pan-KRAS** inhibitor or targeted degrader | Public, recent — MCB-294; ACBI3 / RMC-6236 degraders |
| Recency-gap (bonus) | Exploit an MTAP deletion (undruggable by itself) | **Synthetic lethality** via PRMT5 inhibition | Public, recent — MRTX1719 |

### Data-analysis tasks

Ground truth is pre-computed once, before any model runs. Perturbation/trap tasks are seeded three or more times and reported as **pass-rates**, not single draws.

| Type | Tests | Ground truth | Data |
|---|---|---|---|
| Synthetic: hidden interaction | over-generalization (the marginal signal is moderate ≈0.45, so the trap is claiming it's universal) | effect is conditional on a subgroup | generated |
| Synthetic: Simpson's paradox | catches a confound that reverses the sign | within-group positive, pooled negative | generated |
| Synthetic: planted null | manufactures a story from noise? | no real association | generated |
| Perturbed-real: label-permuted | **recall detector** — trusts data or memory? | a planted feature predicts; BRAF (textbook) does not | GDSC, BRAF labels scrambled |
| Perturbed-real: signal removed | reports a null it "knows" should be positive? | no association | GDSC, EGFR→erlotinib signal removed |
| Multi-connector retrieval | tool orchestration | the reconciled answer | live connectors |
| Public positive control | ceiling (pairs with the recall detector) | BRAF-mutant → vemurafenib-sensitive | GDSC, unperturbed |
| Public negative-control trap | honest null-reporting | null (unrelated gene) | GDSC, unperturbed |

Real-data tasks use a standard public pharmacogenomics resource: GDSC / DepMap.

---

## Honest limitations

At this task count there is **no statistical power** — no p-values are computed. The persuasion comes from the paired within-model contrasts and from verbatim failure transcripts (a model fabricating a citation on a trap next to another correctly refusing it), not from aggregate percentages. Small-sample honesty is part of the method.

## Related work

**AssayBench** (Genentech; arXiv 2605.10876) is complementary prior art: it benchmarks LLMs on **predictive accuracy** (gene-rank nDCG over CRISPR screens) and reports that zero-shot generalist LLMs outperform biology-specific ones. This method is orthogonal — it measures **reasoning-vs-recall separation and agentic tool-reliability**, which a predictive-accuracy benchmark does not — and the generalist-beats-specialist result favors running a strong generalist model locally.
