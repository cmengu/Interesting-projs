# The AI/ML Frontier — mid-2026, beyond World Models & Harnesses

*Compiled 2026-07-07 from a deep-research harness run: 78 completed search/fetch/verify agents → 136 extracted claims → 42 adversarial verdicts (3-vote refutation). 4 claims were refuted in verification and are corrected inline. Ratings: 🟢 real, results-backed signal · 🟡 real but load-bearing caveat · 🔴 where the hype is.*

Scope: deliberately **excludes** world models, harnesses/meta-harnesses, and "loop engineering" (per the brief). Audience: an expert builder of agentic harnesses.

---

## The throughline (the single most decision-relevant finding)

**RL over verifiable/agentic rewards — not scaffolding, not loops — is the mechanism converting capability into reliable agent behavior.** The agentic-RL survey (*The Landscape of Agentic Reinforcement Learning for LLMs*, arXiv **2509.02547**, TMLR 2026, 500+ works) states it directly: RL is "the critical mechanism for transforming [planning, tool-use, memory, reasoning, self-improvement, perception] from **static, heuristic modules** into adaptive, robust agentic behavior." It reframes the training problem from degenerate single-step MDPs (RLHF) into temporally-extended **POMDPs** — which is what enables long-horizon autonomy rather than one-shot response tuning. This is the evidence base for the "loop-engineering-is-hype" prior: the reliability you cannot prompt your way to is being *trained in*.

---

## 1. RL for agents (successor to RL-for-reasoning) — 🟢 strongest signal

The field **bifurcated**: reasoning-RL is consolidating (GRPO-family + process rewards), while the live frontier is **agentic RL** — credit assignment across 100+ turn, 100K–1M-token horizons with stochastic transitions and partial observability, where episode-level reward is uninformative.

- **The GRPO explosion is substance, not naming theater.** DAPO, GSPO, Dr.GRPO, ProRL, StarPO, TreePO, DARS + ~15 more, differentiated by *optimization granularity* (GSPO = sequence-level ratio optimization vs. GRPO's token-level). GRPO removes the critic by normalizing rewards within a sampled group; claimed asymptotically equivalent to an oracle value function.
  - Concrete: **DAPO** trained Qwen2.5-32B → **50 on AIME 2024**, beating DeepSeek-R1-Zero with **50% fewer steps** (Clip-Higher, dynamic sampling, token-level PG loss, overlong-reward shaping).
- **Agentic credit assignment is genuinely new math**, no precedent in reasoning-RL: hindsight counterfactual analysis, privileged asymmetric critics, turn-level MDP reformulations. Survey: *From Reasoning to Agentic: Credit Assignment in RL for LLMs*, arXiv **2604.09459** — **47 CA methods** (2024–early 2026), taxonomized by granularity (token→multi-agent) × methodology (MC, TD, model-based, game-theoretic, information-theoretic).
- **Outcome-only RL beats scaffolding.** **WebAgent-R1** (arXiv **2505.16421**): binary task-success reward, async trajectory generation on live web → Qwen-2.5-3B **6.1%→33.9%**, Llama-3.1-8B **8.5%→44.8%** on WebArena-Lite; the 3B/8B models **beat OpenAI o3**. No process rewards, no step supervision.
- **Recipe knowledge is forming.** *Demystifying RL for Long-Horizon Tool-Using Agents* (arXiv **2603.21972**, TravelPlanner testbed): optimal config is **scale-dependent** (small models need staged/curriculum rewards + exploration; large models converge under simple dense rewards); ~**1K balanced trajectories** is the data sweet spot; environment determinism during rollouts is critical.
- **Watch the layer:** RL environments / training-verification are being called the value-capture layer of the agent stack (Wing VC, "who wins the training & verification layer by 2030").

**Builder takeaway:** the durable capability lives in the *policy*; harnesses are increasingly the environment/reward plumbing around an RL-trained agent. This is the frontier most adjacent to what you build, and the one most worth going deep on.

---

## 2. Data-free self-improvement (self-play without human data) — 🟢 signal, 🟡 safety caveat

- **Absolute Zero / AZR** (arXiv **2505.03335**, NeurIPS 2025; LeapLabTHU): one model proposes *and* solves its own code-reasoning tasks, code executor as sole verifiable reward. Beats "zero-setting" models trained on tens of thousands of human-curated examples (+1.8 avg; AZR-Coder-7B AVG 50.4, +10.2 over base). **Code self-play transfers OOD to math** (+10.9/+15.2 pts vs. ~0.65 for conventional RLVR on expert code models). **Gains scale with size** (3B/7B/14B → +5.7/+10.2/+13.2) — the loop keeps paying off.
- **SPIRAL** (arXiv **2506.24119**, ICLR 2026): self-play on zero-sum *games* → +up to 10% across 8 reasoning benchmarks; beats SFT on 25K expert trajectories. Games map to skills (Poker→probabilistic, Negotiation→strategic). Key trick: **Role-conditioned Advantage Estimation** — ablate it and reasoning collapses to empty CoT after ~200 steps. Complementary to RLVR (a R1-distilled model still gains).
- **RLPR** (arXiv **2506.18254**, OpenBMB): **verifier-free** RL — uses the model's own **token-probability of the reference answer** as reward, extending RLVR beyond math/code into general domains. Beats concurrent VeriFree (+7.6 TheoremQA / +7.5 Minerva) and even a verifier-dependent method (+1.6 avg over 7 benchmarks). Requires variance control (mean-token prob + debiasing baseline + std curriculum filter).
- 🟡 **Caveat is safety, not efficacy:** AZR on Llama-3.1-8B occasionally emits alarming CoT (authors' literal "uh-oh moment"). Self-proposed-task training can drift.

---

## 3. Test-time compute — the verifier is the lever, not the sample count — 🔴→🟢 (the key hype correction)

**Verification killed the naive version of this.** A claim that test-time scaling "universally, predictably plateaus" (*Scaling over Scaling / TTSPM*, arXiv **2505.20522**) was **REFUTED by 3 independent verifiers.** Corrected truth:

- **Verifier-*free* aggregation saturates** — majority vote / self-rethink hits diminishing returns fast. (The TTSPM paper *does* exist and derives a real stopping formula N* = ⌈ln(ε/(F_max·p_x))/ln(1−p_x)⌉ — but only for that regime.)
- **Verifier-*guided* scaling does not.** *Large Language Monkeys* (Brown et al., Stanford, arXiv **2407.21787**): coverage/pass@k scales **log-linearly over 4 orders of magnitude** (SWE-bench Lite 15.9%→56%, 1→250 samples). o1: **74%→83%@64→93%** with a learned reranker on AIME. o3 used 1,024 parallel samples on ARC-AGI. **The lever is reward-model/verifier quality, not raw N.** Spending inference compute without a discriminator = the flat part of the curve. *This is the "own the evals" thesis in mechanism form.*
- **New mechanism — TTRL (test-time RL):** AlphaProof generates and *learns from millions of problem variants at inference* for problem-specific adaptation (perfect miniF2F-valid). Beyond best-of-N.
- **Inference-systems angle — speculative decoding for reasoning** (*Scaling Up, Speeding Up*, arXiv **2509.04474**, ICLR 2026): reasoning traces are repetitive, so training-free n-gram drafting (SAM) is competitive — **but** the "n-gram beats trained EAGLE-3" framing was **REFUTED** as cherry-picked (SAM wins 1 of 4 model×temp cells; that cell is an anomalously weak EAGLE-3 checkpoint). Real result: **hybrid SAM[EAGLE-3] is SOTA**; trained draft heads are brittle across targets (−34% Qwen3-8B→R1-distill). Combine, don't replace.

---

## 4. Post-Transformer architectures & inference systems — 🟢 hybrids won at scale

- **Production hybrid SSM-Transformer-MoE shipped:** NVIDIA **Nemotron 3 Super** (NVIDIA dev blog, 11 Mar 2026; tech report arXiv **2604.12374**; open weights `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`) — 120B-total/**12B-active**, repeating **Mamba-2 → Latent MoE → Mamba-2 → Attention** pattern; **1M context**; **>5× throughput** vs. prior Nemotron; **4× on B200**. SSM layers give linear-time scan (what makes 1M *practical*); attention at key depths preserves needle-in-haystack recall. Two sub-signals: **Latent MoE** (compress tokens pre-routing → 4× experts at same cost) and **native NVFP4 4-bit-float pretraining** (majority of MACs in 4-bit — low precision moved *into* pretraining). Broader family: IBM Granite 4, Qwen3-Next, Jamba, Falcon-H1 — hybrids are now the mainstream long-context design.
- **Linearizing existing LLMs is now cheap:** **HALO / HypeNet** (Tsinghua THUNLP, arXiv **2601.22156**, Jan 2026; code released) converts Qwen3 → RNN-attention hybrid with **2.3B tokens = <0.01% of pretraining** (vs. 20B–400B prior: Mamba-in-Llama 20B, Jet-Nemotron 400B). **3.0×/3.4× decode/prefill at 512K**, runs at 1M where source OOMs, and *fixes* hybrids' historical long-context weakness (256K NIAH far above Jet-Nemotron's ~0%). **Counter-signal to Mamba maximalism:** simple **Lightning Attention beat Mamba2 and Gated DeltaNet** as the recurrent mixer. (Skeptic's counterweight worth reading: LessWrong "Debunking Claims About Subquadratic Attention.")
- **Diffusion LLMs — 🟡 real inference-efficiency frontier, quality caveat.** Compute-bound rather than memory-bandwidth-bound (AR decode ≈ 1 FLOP/byte on hardware built for 100+). LLaDA-8B ≈ LLaMA3-8B (MMLU 65.9 vs 65.4); Mercury Coder (~1,109 tok/s), Gemini Diffusion (1,479 tok/s), ByteDance **Seed Diffusion** (arXiv **2508.02193**; **2,146 tok/s on H20**). "Block diffusion" restores KV-caching. Caveats: authors concede diffusion still **lags AR on quality**; headline speed comparisons are cross-hardware.

---

## 5. Native multimodal & unified understanding+generation — 🟢 nascent but architecturally real

- **Emu3.5** (BAAI, arXiv **2510.26583**, *Native Multimodal Models are World Learners*): single **next-token** objective over vision+language, pretrained on **10T tokens of sequential video frames + transcripts**; parity with Gemini 2.5 Flash Image ("Nano Banana"). **DiDA** converts AR image decoding → bidirectional-parallel for **~20× per-image speedup**. Training on video yields emergent embodied/spatiotemporal capability (bridge to VLAs).
- **Unification survey** (arXiv **2505.02567**, v6 Jan 2026): 3 paradigms (diffusion / autoregressive / hybrid); the *actual* open problems are **tokenization strategy, cross-modal attention, and data — explicitly not scaling**. GPT-4o image-gen is the exemplar that triggered the push. Still early-stage.

---

## 6. Embodied AI / Vision-Language-Action models — 🟢 cross-embodiment is the live problem

- **Gemini Robotics 1.5** (DeepMind, arXiv **2510.03342**): multi-embodiment VLA with a **Motion Transfer** mechanism (learn across robot bodies), and it **interleaves actions with natural-language CoT ("think before acting")** — test-time reasoning reached robot control. Paired with a SOTA **Embodied Reasoning** model in a two-model "physical agents" architecture.
- **π0.5** (Physical Intelligence, arXiv **2504.16054**): first end-to-end learned system to do **long-horizon dexterous manipulation in entirely unseen homes** (clean a kitchen). Lever = **co-training on heterogeneous data** (multiple robots + web + semantic subtask prediction); ablations show cross-source transfer is *necessary*. Open-world generalization explicitly still contested.
- **Open weights exist:** NVIDIA **Isaac GR00T N1.5-3B** (open foundation VLA for humanoids). Comparative architecture teardown: Pebblous "Three Teams, Three Robot Brains" (GR00T vs Gemini Robotics vs π).

---

## 7. AI-for-science with real verification loops — 🟢 where "pure autonomous agents" produce checkable output

The honest home of autonomous agents — reward = proof checker or wet lab.

- **AlphaProof** (DeepMind, *Nature* **s41586-025-09833-y**): AlphaZero-style RL searching proofs *inside Lean* → IMO **silver (2024)** → **gold-level (2025, 35/42)**. Machine-checked = **irrefutable**. Enabled by autoformalizing ~1M NL problems → **~80M Lean problems** (~80K TPU-days, a 3B model + AND–OR tree search — lever is *search+RL over a verifier*, not raw scale). 🟡 Wildly uneven: number theory 75.7%, algebra 72.6%, **combinatorics 20.3%**.
- **Robin** (FutureHouse, *Nature 2026* **s41586-026-10652-y**): multi-agent discovery (PaperQA2 literature agents + autonomous data-analysis agent Finch) **autonomously proposed ripasudil + KL001 for dry AMD — both wet-lab-validated** (1.89× phagocytosis); KL001 novel to literature. ~**200× throughput** (551 papers/30 min vs. 294 human-hours); 7.25/10 top hypotheses matched domain experts.
- **The mature, real stack:** AlphaFold 3 (diffusion architecture, first to beat physics-based tools, PoseBusters +50%; Isomorphic Labs + DeepMind); **de novo protein binder design** (*Structure*/Cell Press, S0969-2126(25)00311-9: RFdiffusion + Chroma + ProteinMPNN + BindCraft → **~46% hit rate on GPCRs, ~30% on TNFR** vs. <0.1% traditional); **GNoME + Berkeley A-Lab** (2.2M crystals, 380K stable, **41 autonomously robot-synthesized**, 736 externally validated); **AlphaGenome** (DeepMind, regulatory genomics, open-sourced Jan 2026, ~1M API calls/day). 🟡 Universal caveat: **in-silico scores still don't predict real binding — wet-lab validation is the rate-limiter.**

---

## 8. Mechanistic interpretability — 🟡 a field publicly correcting itself

- **SAEs are cooling** (the 2024 darling). DeepMind **deprioritized fundamental SAE research** (Mar 2025) after **dense linear probes beat k-sparse SAE probes OOD** on jailbreak detection (SAEs discard residual-stream information). Then *Sanity Checks for Sparse Autoencoders* (arXiv **2602.14111**, Feb 2026): **random baselines match trained SAEs** (interp 0.87 vs 0.90, probing 0.69 vs 0.72, editing 0.73 vs 0.72), and on synthetic ground truth **SAEs recover only 9% of true features despite 71% explained variance** — reconstruction ≠ recovery.
- **Live direction: attribution graphs / circuit tracing** (Anthropic, *On the Biology of a Large Language Model*, transformer-circuits.pub Mar 2025; tools open-sourced May 2025 on Neuronpedia; cross-layer transcoder, ~30M features). Concrete wins: model **plans rhyme words ahead**; genuine **two-hop reasoning** (Dallas→Texas→capital); can **mechanistically distinguish faithful CoT from confabulation vs. working-backward-from-a-hint** (directly relevant to trusting reasoning traces). Supports causal interventions. 🟡 Still early: satisfying insight on **~25% of prompts**; frontier moving to model-diffing, interpreting reasoning models, deception. (See also Matryoshka SAEs arXiv 2503.17547; Anthropic "Emotion Concepts" arXiv 2604.07729.)

---

## Where the hype is (so you can ignore it)

- **"More inference samples ⇒ monotonically better."** False without a verifier — you saturate on majority vote. Verifier quality is the axis.
- **"Training-free speculative decoding beats trained draft models."** No — hybrids win.
- **"SAEs will crack interpretability."** The field itself is walking this back.
- **"Mamba/SSMs strictly dominate attention."** No — winners are *hybrids*; simple Lightning Attention often beats Mamba2 as the mixer.
- **"Loop engineering"** — corroborated as low-signal: capability is *trained* into policies via RL, not orchestrated in via scaffolds.

## If picking what to go deep on (as an agentic-harness builder)

1. **Agentic credit assignment** (arXiv 2604.09459) — the theory of *why* your harnesses are/aren't reliable; genuinely new.
2. **Verifier / reward-model design for test-time compute** — highest-leverage, least-hyped; the "own the evals" thesis as mechanism.
3. **Hybrid architectures + cheap linearization (HALO, arXiv 2601.22156)** — 1M-context linear-time agents change what one episode can hold.

---

## Citations (verified in corpus)

**RL / agents:** Landscape of Agentic RL 2509.02547 · Credit Assignment survey 2604.09459 · WebAgent-R1 2505.16421 · Long-Horizon Tool-Use recipe 2603.21972 · Turing Post "Reasoning RL in 2026" · LLM-Stats "Post-Training in 2026"
**Self-improvement:** Absolute Zero 2505.03335 · SPIRAL 2506.24119 · RLPR 2506.18254
**Test-time compute:** TTSPM 2505.20522 (plateau claim refuted) · Large Language Monkeys 2407.21787 · Spec-decode benchmark 2509.04474
**Architectures/systems:** Nemotron 3 Super (NVIDIA blog + 2604.12374) · HALO/HypeNet 2601.22156 · Seed Diffusion 2508.02193 · Jet-Nemotron 2508.15884 · Mamba-in-Llama 2408.15237 · Jamba 2403.19887 · LessWrong "Debunking Subquadratic Attention"
**Multimodal:** Emu3.5 2510.26583 · Unified survey 2505.02567
**Embodied:** Gemini Robotics 1.5 2510.03342 · π0.5 2504.16054 · NVIDIA GR00T N1.5-3B (HF)
**AI-for-science:** AlphaProof Nature s41586-025-09833-y · Robin Nature s41586-026-10652-y · AlphaFold 3 (Isomorphic Labs) · Protein binders Structure S0969-2126(25)00311-9 · GNoME+A-Lab (DeepMind) · AlphaGenome (StatNews Jan 2026)
**Interpretability:** SAE sanity checks 2602.14111 · DeepMind SAE deprioritization (Medium, Mar 2025) · Anthropic Biology of an LLM (transformer-circuits.pub) · Anthropic circuit-tracing open-source · Matryoshka SAEs 2503.17547
**Memory:** LLM Agent Memory survey 2605.06716

*4 refuted claims: (1) test-time scaling "universal plateau" — corrected to verifier-free-only; (2) "n-gram beats trained EAGLE-3" general framing — corrected to hybrid-wins. Both counted twice/thrice across verifiers.*
