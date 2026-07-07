# API-Only Frontier-RE Project Ideas (deep dive)

*Projects buildable with **no GPU** — just model APIs (chat + logprobs + embeddings, occasionally hosted fine-tuning). Curated for **frontier research-engineer credibility**: empirical taste + a clean, shareable result. Every project is engineered to end in one **tweetable artifact** — a scaling curve, a leaderboard, a benchmark, a negative result, an open eval harness. Ship each as an arXiv note / lab-quality blog + open-sourced code.*

*Context: these grew out of the mid-2026 frontier survey in [`frontier-2026-beyond-world-models.md`](./frontier-2026-beyond-world-models.md). North-star thesis: **own the evals** — verifiers/reward-models/measurement are the real bottleneck, and you can attack all of them with an API alone.*

⭐ = strongest opening bets.

---

## Practical setup (read once, applies to everything below)

- **Models you have API access to (mid-2026):** closed frontier — Claude (Opus 4.8 / Sonnet 5 / Fable 5), GPT-5.x, Gemini 2.5/3. Open via hosted inference (Together, Fireworks, DeepInfra, OpenRouter) — Qwen3 (0.6B–235B), DeepSeek-R1/V3, Llama-3.x, GLM-5.2, Nemotron-3. Pick 4–6 spanning **both frontier and open, small and large** — cross-model spread is what makes a result interesting.
- **⚠️ Logprobs caveat:** **Anthropic does not expose token logprobs.** OpenAI does (`logprobs=True`, `top_logprobs`), and open-model hosts (Together, Fireworks, DeepInfra) do. So any project using a **self-logprob / perplexity reward** (RLPR-style) or confidence signal must run on OpenAI or an open model — not Claude. Flagged inline where it matters.
- **Embeddings:** OpenAI `text-embedding-3-large`, Voyage, Cohere, or open (BGE, GTE) — for difficulty predictors and clustering.
- **Hosted fine-tuning (still GPU-free *for you*):** OpenAI fine-tuning API, Together, Fireworks. Needed only for Project 8.
- **Cost discipline:** most of these run for **$20–150** in API credits if you cache aggressively and pilot on 50 items before scaling to 300. Log every request to disk (jsonl) so a crash never re-bills you. Budget the "N=256 samples × 300 problems × 5 models" projects carefully — that's where spend hides.
- **Stats hygiene that separates a real result from a blog-toy:** always report **bootstrapped 95% CIs** over items, not just point accuracy. A "curve" without error bars is not a finding.

---

## A. Reward modeling & verifiers — the #1 frontier-RE bottleneck, and the core thesis

### 1. "The Verifier Wall" ⭐
> **Result you're hunting:** *Everyone's test-time-compute results live in math/code because checking is free there. Find the domain where the verifier-beats-majority-vote law collapses — and show exactly where the wall is.*

- **Research question:** As verification gets harder (free checker → LLM-judge → self-confidence → nothing), at what point does verifier-guided selection stop beating plain self-consistency, and does the accuracy-vs-N curve flatten or *invert*?
- **Why it matters (RE signal):** The single most-cited test-time-compute finding (Large Language Monkeys; o1's rerank-to-93%) is that a *good verifier* keeps the curve climbing while majority-vote saturates. But that's only ever been shown where verification is trivial. Extending — or **breaking** — it off-math is genuinely open, and reward-model design is *the* frontier bottleneck. This is your thesis as a runnable plot.
- **Method:**
  1. Pick 3 domains on a verifiability gradient: **easy** (MATH/AIME — exact-match checker), **medium** (GPQA-diamond or MedQA — MC answer checkable but reasoning isn't), **hard** (long-form: summary faithfulness on AggreFact/FRANK, or open-ended medical QA scored by a rubric).
  2. For each problem, sample **N = 1, 2, 4, …, 256** candidates from a fixed generator (e.g. Qwen3-8B via Together, temp 0.8).
  3. Select the "final answer" four ways: **(a)** self-consistency / majority vote, **(b)** a **prompted generative verifier** (a strong model asked to pick the best candidate), **(c)** an **RLPR-style self-logprob reward** — mean token-logprob of each candidate *(⚠️ needs an open model or OpenAI, not Claude)*, **(d)** oracle (upper bound).
  4. Plot accuracy vs. log(N), one panel per domain, four lines each.
- **Models & data:** generator = one open model (for logprobs); verifier = a strong frontier model; datasets above. ~300 items/domain.
- **Baselines:** self-consistency is the baseline everyone must beat; oracle is the ceiling.
- **The artifact:** a 3-panel figure showing the verifier's advantage **shrinking to zero (or going negative) as verifiability drops** — plus the crossover N. That's the tweet.
- **Stretch:** train a tiny **learned** verifier (logistic regression on embeddings of `(problem, candidate)`) and see if it rescues the hard domain where prompted judges fail.
- **Watch out:** if you only test math, it's a boring reproduction. The *gradient* is the whole paper. Also: make the generator weaker than the verifier, or (c) and (b) collapse into the same thing.
- **Effort:** ~2 weeks.

### 2. "How to Break an LLM Judge" ⭐ *(build on the Council harness)*
> **Result you're hunting:** *A curated set of cases that flip a frontier LLM-judge's verdict — then a leaderboard of which defense (panel / debate / verify-first) actually recovers accuracy, and by how much.*

- **Research question:** Which known judge biases (position, verbosity, sycophancy, self-preference, confident-but-wrong) are most exploitable in mid-2026 frontier models, and which aggregation defense closes the most of the gap per dollar?
- **Why it matters (RE signal):** LLM-as-judge underpins RLAIF, eval leaderboards, and most agent reward signals. Its failure modes are a live safety/reliability problem. You already own cross-family adversarial infra (Council: Claude proposes / Codex critiques) — turn it into a *measured* study instead of a tool.
- **Method:**
  1. Build ~100–150 adversarial pairs across 5 bias categories, each with a known-correct answer. E.g. **verbosity:** a correct terse answer vs. a wrong verbose one; **position:** same pair in both orders; **self-preference:** model A judging A-vs-B.
  2. Measure single-judge error rate per bias, per model (GPT-5, Claude, Gemini, an open judge).
  3. Test 4 defenses: **panel-of-5** (majority), **debate** (two models argue, third decides), **verify-first** (judge must check claims before ranking), **swap-and-average** (run both orderings, average).
  4. Report error-rate reduction and cost multiplier for each.
- **Models & data:** ≥3 frontier judges + 1 open; MT-Bench / RewardBench items as a realistic base, adversarial items hand-built. Reuse Council for the debate/panel plumbing.
- **The artifact:** a bias × defense heatmap of "error reduction," plus a cost-adjusted "best defense" verdict. Position bias in particular tends to be embarrassingly large — a clean number there travels.
- **Stretch:** show a defense that *backfires* (e.g. debate amplifies a shared bias both models have). Negative results on defenses are gold.
- **Watch out:** don't hand-pick cases that only your favorite model fails — pre-register the bias taxonomy, generate items blind to which model you'll test.
- **Effort:** ~2 weeks.

### 3. "Catching a Model Lying About Its Reasoning — no weights needed" ⭐
> **Result you're hunting:** *Rank 6 frontier models by "lie rate" — how often they use a planted hint to get the answer while their chain-of-thought pretends they reasoned it out honestly.*

- **Research question:** How prevalent is unfaithful CoT in mid-2026 reasoning models, and does more "thinking" (longer traces, reasoning-mode on) make faithfulness better or *worse*?
- **Why it matters (RE signal):** CoT faithfulness is central to whether we can trust reasoning traces for oversight. The field assumes you need internals (activations) to study it — a rigorous **behavioral** suite that needs only an API is a fresh, contrarian angle. Directly extends Turpin et al. ("Language Models Don't Always Say What They Think") into the reasoning-model era.
- **Method:**
  1. Take MC questions (BBH, MMLU, GSM8K-MC). Create a biased variant by injecting a hint: reorder so the answer is always "(A)", or add "a Stanford professor thinks the answer is (B)".
  2. Measure: does the model's answer shift toward the hint? And crucially — does its CoT *mention* the hint, or fabricate independent reasoning?
  3. Score "unfaithfulness" = (answer moved to hint) AND (CoT never acknowledged the hint). Automate the acknowledgement check with a judge.
  4. Repeat with reasoning-mode on vs. off (where the API allows) to test the "does thinking help?" axis.
- **Models & data:** 6 models spanning reasoning/non-reasoning; ~200 items × several hint types.
- **The artifact:** a per-model lie-rate bar chart + the counterintuitive finding on whether reasoning-mode raises or lowers faithfulness. If longer reasoning → *more* confabulation, that's a strong, spicy result.
- **Stretch:** test whether asking the model up-front to "flag any external hints you use" reduces the lie rate (cheap alignment intervention).
- **Watch out:** the acknowledgement-detection judge is itself fallible — validate it on a hand-labeled subset and report its precision/recall.
- **Effort:** ~2 weeks.

## B. Test-time compute — hot, cheap, API-native

### 4. "Spend Compute Like a Poker Player" ⭐
> **Result you're hunting:** *Match fixed-large-N accuracy at ~1/5th the average cost by betting sampling budget per-query on predicted difficulty.*

- **Research question:** Can a cheap difficulty predictor allocate test-time samples adaptively (easy → N=1, hard → N=64) to Pareto-dominate uniform sampling?
- **Why it matters (RE signal):** "Efficient test-time compute" is a hot frontier, and almost everyone spends *uniform* compute per query. A per-query allocator that matches accuracy at a fraction of the cost is exactly the shape of result labs like — it's a systems-y, budget-aware finding.
- **Method:**
  1. On a mixed-difficulty set (MATH levels 1–5, or MMLU-Pro), compute each item's empirical "hardness" (accuracy over a pilot sweep).
  2. Train a tiny predictor (logistic reg / gradient-boost) mapping **embedding(question) → predicted hardness**.
  3. At test time, allocate N per query from a budget policy (e.g. spend proportional to predicted hardness), select by self-consistency.
  4. Compare against uniform N at matched *total* budget.
- **Models & data:** any generator with cheap sampling (open model); OpenAI/open embeddings for the predictor.
- **The artifact:** a cost-vs-accuracy Pareto curve where the adaptive line sits above uniform across the whole budget range. Headline: "same accuracy, 5× cheaper."
- **Stretch:** add a **stopping rule** — stop sampling early once the running majority is confident (sequential test), squeezing cost further.
- **Watch out:** the predictor must be trained/tested on disjoint items or you're leaking. Report the allocator's overhead cost too (embedding calls aren't free).
- **Effort:** ~2–3 weeks.

### 5. "Self-Improvement at Inference, Zero Training" 
> **Result you're hunting:** *A model invents and solves its own warm-up problems in-context, then solves a target it just failed — isolating the self-improvement mechanism from any weight update.*

- **Research question:** Can AlphaProof-style "learn from related problem variants at inference" be reproduced with **pure in-context** learning (no fine-tuning), and how much lift does it give on the target problem?
- **Why it matters (RE signal):** Everyone conflates self-improvement with training. Showing a clean **training-free** version isolates the *mechanism* — which nobody has done crisply for general tasks. It's a conceptually sharp, cheap experiment.
- **Method:**
  1. Take target problems the base model fails at N=1.
  2. Have the model **auto-generate K related-but-easier variants**, solve them, and **verify** the solutions (exact-match where possible).
  3. Feed only the *verified* variant solutions back into context, then re-attempt the target.
  4. Measure lift vs. controls: (a) no variants, (b) K random unrelated solved problems, (c) K *unverified* variants (does verification matter?).
- **Models & data:** MATH/AIME targets (verifiable); any capable generator.
- **The artifact:** lift-on-target bar chart with the ablation showing **verification is the load-bearing ingredient** — feeding unverified variants should help far less (or hurt).
- **Stretch:** vary K and variant-difficulty to find the sweet spot; show it scales with target difficulty.
- **Watch out:** contamination — make sure "variants" aren't just the target restated. Measure variant-target similarity and cap it.
- **Effort:** ~2–3 weeks.

## C. Evals & measurement — a good benchmark is a classic RE launchpad

### 6. "Does Your Agent Contradict Itself?" 
> **Result you're hunting:** *A 50-turn self-consistency benchmark that quantifies the failure every harness-builder has felt but nobody measures — the agent confidently contradicting its own earlier claims.*

- **Research question:** How fast does long-horizon self-consistency degrade with conversation length, and which models hold their story longest?
- **Why it matters (RE signal):** Long-horizon reliability is the open problem in agents (per the frontier survey's agentic-RL section). A crisp *measurement* of intra-conversation contradiction is a benchmark people will use — and it comes straight from your lived experience building harnesses.
- **Method:**
  1. Design tasks that plant a fact early ("the client's budget is $40k") and probe it 5, 15, 30, 50 turns later under distraction (unrelated intervening tasks).
  2. Also test *derived* consistency: does turn-50 reasoning respect a constraint stated at turn 3?
  3. Auto-score contradictions with a judge over transcript pairs.
  4. Curve: contradiction rate vs. turn-distance, per model and per context-length.
- **Models & data:** frontier + long-context open models; ~150 scripted scenarios.
- **The artifact:** contradiction-vs-turn-distance curves; a leaderboard of "story-holding." Bonus if a 1M-context model is *worse* than a shorter one (context ≠ coherence).
- **Stretch:** test whether an explicit "memory summary" every 10 turns fixes it — connects to the continual-learning frontier.
- **Watch out:** distinguish genuine contradiction from legitimate updating ("budget changed"). The scenario design must make the ground truth unambiguous.
- **Effort:** ~3 weeks.

### 7. "Reasoning or Remembering?" 
> **Result you're hunting:** *Perturb popular benchmarks (rename entities, swap numbers, paraphrase) and watch scores collapse — a memorization exposé quantifying how much "reasoning" is recall.*

- **Research question:** How much of frontier models' benchmark performance survives semantics-preserving perturbation, and which benchmarks are most contaminated?
- **Why it matters (RE signal):** Eval integrity is a real crisis, and a clean contamination probe is both attention-grabbing and a genuine service to the field (à la GSM-Symbolic). It signals you don't take leaderboards at face value — exactly the skepticism RE hiring looks for.
- **Method:**
  1. Take GSM8K, MMLU, a coding set. Build perturbation operators: rename entities, swap numeric values (recompute the answer), paraphrase, add an irrelevant clause.
  2. Measure accuracy drop per operator per model. A large gap between original and perturbed = memorization signal.
  3. Rank benchmarks by "fragility," rank models by "robustness."
- **Models & data:** ≥5 models incl. some released before/after each benchmark, to correlate release-date with contamination.
- **The artifact:** original-vs-perturbed scatter with the diagonal; the outliers are the story. A benchmark-fragility leaderboard.
- **Stretch:** correlate fragility with each model's training cutoff — a temporal contamination signal.
- **Watch out:** perturbation must truly preserve difficulty; validate a sample by hand so you're not just making questions *harder*.
- **Effort:** ~2 weeks.

## D. Data-centric — a direct audition for the GLOW.AI / Bryan Low lab

### 8. "20% of the Data, 100% of the Performance" ⭐ *(on-thesis for data-selection-for-LLM-training)*
> **Result you're hunting:** *Beat full-dataset fine-tuning using a smart 20% subset — the literal GLOW.AI / Bryan Low audition, done GPU-free via hosted fine-tuning.*

- **Research question:** Can a principled data-selection method (quality × difficulty × diversity) match or beat training on 100% of an SFT set, using only 20% of it?
- **Why it matters (RE signal):** Data-selection-for-LLM-training *is* the GLOW.AI research direction. This is the most literal possible artifact to bring to that conversation — and data-centric work is broadly one of the highest-leverage, least-saturated frontier areas.
- **Method:**
  1. Take an instruction/reasoning SFT set (e.g. a slice of Tulu / OpenHermes / a math-reasoning set).
  2. Build a selector: dedup (embedding clustering) → difficulty score (base-model perplexity or failure rate; ⚠️ perplexity needs logprobs → open model/OpenAI) → diversity sampling.
  3. Fine-tune via **hosted API** (OpenAI/Together) on: selected 20%, random 20%, full 100%.
  4. Evaluate all three on held-out benchmarks.
- **Models & data:** hosted fine-tune of an open base (Together/Fireworks) or a GPT model; eval on GSM8K/MMLU/etc.
- **The artifact:** a bar chart where **selected-20% ≈ or > full-100%**, and random-20% clearly below. That's the whole thesis of data-centric ML in one plot.
- **Stretch:** ablate the three selection criteria to show which does the work; test whether the selected subset transfers across base models.
- **Watch out:** fine-tuning cost and variance — run each condition ≥2 seeds. Keep the eval set truly held out from selection.
- **Effort:** ~2–3 weeks. *(Some $ for hosted fine-tunes; still zero GPU.)*

---

## E. VLM / VLA — vision-language(-action), API-only

**The honest caveat first:** a true **VLA** ends in *physical action* (robot motor control), which needs a robot or heavy sim → **not** API-only. But a VLA is really two halves, and **both accessible halves are API-native:** (1) the **embodied-reasoning / perception** half — pure VLM: spatial understanding, planning, progress estimation (exactly what Gemini Robotics-**ER** targets); and (2) the **action** half reinterpreted as **GUI / computer-use control** — clicking and typing literally *is* vision → language → action, fully API-native via Claude / OpenAI computer-use. So you can do frontier VLA-flavored work without a single servo.

### V1. "Can Frontier VLMs Actually Reason About Space?" ⭐
> **Result you're hunting:** *An embodied-reasoning benchmark showing frontier VLMs are shockingly bad at the spatial/physical reasoning that every VLA depends on.*

- **Research question:** On the reasoning skills Gemini Robotics-ER claims SOTA on — relative position, occlusion, physical outcome prediction, progress-from-a-photo — how do general frontier VLMs actually score, and where do they fall off a cliff?
- **Why it matters (RE signal):** This is the **VLM core of every VLA**, and the field lacks a clean, hard, API-testable embodied-reasoning benchmark. A rigorous negative result ("frontier VLMs can't tell what's behind what") is both exciting and genuinely useful — and it's the exact capability robotics labs are hiring to fix.
- **Method:**
  1. Assemble ~200–300 items across skills: **relative position** ("is the mug left of the laptop?"), **occlusion** ("which object is partially hidden?"), **counting under clutter**, **physical outcome** ("if I pull the bottom block, what happens?"), **progress estimation** ("what fraction of this task is done?").
  2. Draw from existing sets (BLINK, CV-Bench, What'sUp, EmbSpatial) + hand-built items with photos you control (harder to contaminate).
  3. Score exact-match / MC; break down by skill.
- **Models & data:** GPT-5V, Gemini 3, Claude (vision), plus an open VLM (Qwen-VL) via a host.
- **The artifact:** per-skill accuracy bars showing a big drop on occlusion/physical-outcome vs. easy naming — the "spatial blind spot" of frontier VLMs.
- **Stretch:** does chain-of-thought or "describe the scene first" close the gap? Connects to V4.
- **Watch out:** image contamination — mix in original photos. And keep questions unambiguous to humans (validate with a friend).
- **Effort:** ~2–3 weeks.

### V2. "Where Do Computer-Use Agents Break?" ⭐
> **Result you're hunting:** *A failure-mode taxonomy + reliability curve for GUI agents — treating clicking/typing as the API-native form of VLA action.*

- **Research question:** On real GUI tasks, how does computer-use-agent success decay with task length, and what are the dominant failure modes (misread UI, wrong element, no recovery)?
- **Why it matters (RE signal):** Computer-use is the hottest API-native "action" frontier, and long-horizon GUI reliability is unsolved. A crisp failure taxonomy is exactly the kind of empirical map labs value — and it's *the* VLA story you can tell without hardware.
- **Method:**
  1. Use Claude / OpenAI computer-use APIs on a task suite (WebArena / WebVoyager / a set of your own browser tasks).
  2. Log full trajectories; bin tasks by number of required steps.
  3. Hand-code failure causes on failed runs; plot success vs. step-count and a failure-mode histogram.
- **Models & data:** ≥2 computer-use-capable models; 100–150 tasks.
- **The artifact:** success-vs-horizon curve + failure-mode breakdown; the "cliff" step-count where agents fall apart.
- **Stretch:** test one intervention — a "verify the screen changed as expected" self-check after each action — and measure recovery gain (ties to the verifier thesis).
- **Watch out:** computer-use can be slow/flaky and costs add up; cap steps, run headless, cache. Respect sites' ToS.
- **Effort:** ~3 weeks.

### V3. "Do VLMs Have an Intuitive Physics Engine?"
> **Result you're hunting:** *A measurement of whether VLMs can predict the next physical state from a frame + an action — a world-model probe with a scoreboard, not a demo.*

- **Research question:** Given an image and a proposed action, can a VLM predict the resulting state (in structured text) consistently with physics, and where does it violate conservation/occlusion?
- **Why it matters (RE signal):** This is a *measurement* angle on world models — the space you keep circling — without generating video. It reframes "world model" as a testable prediction problem, which is more RE-credible than another generation demo.
- **Method:**
  1. Use frames from physics benchmarks (Physion, IntPhys) or simple staged photos (blocks, ramps, liquids).
  2. Prompt: "given this scene and action X, describe the next state" → parse into checkable predicates (object A now below B; ball rolled right).
  3. Score physical plausibility against ground truth; catalog violation types.
- **Models & data:** frontier VLMs + open VLM; a few hundred frame-action pairs.
- **The artifact:** plausibility score per model + a taxonomy of physics errors (teleporting objects, ignored gravity, occlusion failures).
- **Stretch:** correlate physics-prediction skill with V1 spatial scores — do they share a bottleneck?
- **Watch out:** free-text outputs are hard to score; constrain to a predicate schema up front.
- **Effort:** ~2–3 weeks.

### V4. "Is 'Think Before Acting' Real or Theater?"
> **Result you're hunting:** *An ablation testing Gemini Robotics 1.5's headline claim — that interleaving natural-language reasoning with actions helps — in an API-only GUI/game agent.*

- **Research question:** Does forcing an explicit reason-before-act step actually raise task success in a VLM-driven agent, or is it narration that doesn't change behavior?
- **Why it matters (RE signal):** On-thesis with the survey's loop-skepticism: it puts a fashionable claim to a controlled test. A clean "reasoning helps here, not there" result is exactly the signal-vs-hype work you want to be known for.
- **Method:**
  1. Pick a VLM agent task (a browser task suite, or a visual game/puzzle).
  2. Run three conditions: **act-only**, **reason-then-act** (CoT each step), **reason-then-act with self-critique**.
  3. Measure success, steps-to-success, and cost; test across task difficulty.
- **Models & data:** ≥2 VLMs; a task suite with a difficulty gradient.
- **The artifact:** success-lift-from-reasoning as a function of task difficulty — likely helps on hard tasks, wasted tokens on easy ones. That contingent finding is the value.
- **Stretch:** check whether the *stated* reasoning is faithful to the action taken (V5 / Project 3 crossover).
- **Watch out:** confounds — hold the prompt/scaffold identical except for the reasoning toggle.
- **Effort:** ~2 weeks.

### V5. "Is the VLM Looking Where It Says It's Looking?"
> **Result you're hunting:** *Visual chain-of-thought faithfulness — perturb image regions and check whether the answer changes where the model claims it attended.*

- **Research question:** When a VLM says "I based this on the sign in the top-left," is that region actually causally responsible for the answer?
- **Why it matters (RE signal):** The multimodal extension of CoT-faithfulness (Project 3) — an underexplored, API-only interpretability angle. Trust in multimodal reasoning traces is a fresh niche.
- **Method:**
  1. On VQA/GQA items, get the model's answer + its stated visual evidence (which region).
  2. Perturb the claimed region (blur/occlude) vs. a control region; measure which perturbation actually flips the answer.
  3. Faithfulness = answer flips when the *claimed* region is perturbed, not the control.
- **Models & data:** frontier VLMs + open VLM; grounding-friendly datasets (GQA, VQAv2).
- **The artifact:** per-model visual-faithfulness score; cases where models cite a region they demonstrably ignored.
- **Stretch:** compare against the model's own attention/grounding output where the API exposes bounding boxes.
- **Watch out:** you need the model to *name a region* reliably — constrain the output format (grid cell or bbox).
- **Effort:** ~2 weeks.

---

## Recommended sequencing

- **Opener (thesis + shippable in 2 wks):** **#1 The Verifier Wall** — cleanest expression of own-the-evals, genuinely open result.
- **Fast second (reuse Council):** **#2 How to Break an LLM Judge** — hot, safety-flavored, infra already built.
- **The exciting new lane:** **V1 Spatial Reasoning** — the embodied-reasoning benchmark the field lacks; spicy negative result; doubles as the ER-half of the robotics stack. Pair with **V2** for the "action" story.
- **Lab-specific audition:** **#8 Data-Selection** when you want a real artifact for the Bryan Low / GLOW.AI conversation.

Every one of these ends in a single figure or table you can post. That figure — not the code — is the credibility.

---

## Appendix — broader set (GPU-dependent, for reference)

| # | Project | Frontier | Compute | One-line result |
|---|---------|----------|---------|-----------------|
| G1 | Scoped **HALO** — distill Qwen3-1.7B into a linear-attention hybrid | Post-Transformer / kernels | 1 GPU | "linearized an LLM on <0.01% of its data, 3× faster at 512K ctx" |
| G2 | **GRPO/DAPO from scratch** on a small multi-turn agent | Agentic RL | 1–2 GPUs | "3B model beats o3 on a narrow web task via outcome-only RL" |
| G3 | **Mini-Robin** autonomous drug-repurposing agent w/ verification loop | AI-for-science / agents | API + open bio models (mostly API!) | "agent proposed a repurposing hypothesis with a checkable signal" |
| G4 | **Attribution-graph interp** reproduction on Gemma-2-2b | Mech interp | Local small model | "traced a two-hop reasoning circuit; caught unfaithful CoT" |
| G5 | **Continual-learning / TTT toy** — fold a long doc into weights at inference | Continual learning / TTT | 1 GPU | "test-time-trained memory beats RAG on recall at 100K ctx" |

*G3 is mostly API-only (literature + reasoning agents) and doubles as healthtech + Hummingbird Bioscience interview material — promote it to the main list if you can source a checkable biological signal.*
