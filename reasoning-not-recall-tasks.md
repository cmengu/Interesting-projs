# Eval — full task cards: prompts, reference analysis, answer keys, scoring

Each card is self-contained. Give the **Prompt** verbatim to the agent (data file on disk). Run the **Reference analysis** yourself once and lock the numbers into `reference_answers.md` *before* any model runs — the model is scored against your numbers, not against its own claim of what the numbers are.

---
---

# PART A — DATA ANALYSIS (8 tasks)

Scoring is on five checks per task: **(1) Tool-use** — did it execute real code and read real outputs (verify in the trace; a fabricated number = automatic fail on this check); **(2) Correctness** — headline number within tolerance of your reference; **(3) Method** — appropriate test, handled scale/assumptions; **(4) Interpretation** — right conclusion, caught the trap/confound; **(5) Artifact** — produced any requested plot/table. Perturbation/trap tasks (AP1, AP2) are run at **k=3 seeds** and reported as a **pass-rate**, because a single draw hides a model that passes 1-in-3.

---

## AS1 — Synthetic: hidden interaction
**Purpose:** Does the model find *conditional* structure, or stop at a marginal correlation and over-generalize it? This is the analysis analogue of "reasons vs pattern-matches."

**Build:**
```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 400
expr_X = rng.normal(0, 1, n)
mut_Y  = rng.binomial(1, 0.35, n)
tissue = rng.integers(0, 8, n)                 # 8-level categorical DISTRACTOR
resp   = 2.0*expr_X*mut_Y + rng.normal(0, 1, n)
pd.DataFrame({"expr_X":expr_X, "mut_Y":mut_Y, "tissue":tissue,
             "drug_response":resp}).to_csv("as1_interaction.csv", index=False)
```

**Prompt:**
> You are given `as1_interaction.csv` (n=400) with columns: `expr_X` (continuous biomarker expression), `mut_Y` (0/1 mutation status), `tissue` (categorical tissue type, 8 levels), and `drug_response` (continuous; higher = more sensitive to the drug). Determine what predicts `drug_response`. Test each candidate, quantify every effect with an effect size and a p-value, and give a single conclusion an experimentalist could act on (i.e. "measure X in patients with property Y").

**Reference analysis + expected numbers:**
```python
import statsmodels.formula.api as smf
smf.ols("drug_response ~ expr_X * mut_Y + C(tissue)", data=df).fit().summary()
```
- Interaction term `expr_X:mut_Y` coefficient ≈ **2.0** (this is the truth).
- `expr_X` main effect (i.e. slope when `mut_Y=0`) ≈ **0**; `mut_Y` main effect ≈ **0**; all `C(tissue)` terms n.s.
- Stratified: in the `mut_Y=1` subgroup (~140 rows) OLS slope of `expr_X` ≈ **2.0**, R² ≈ **0.80**. In `mut_Y=0` (~260 rows) slope ≈ **0**, R² ≈ **0**.
- The naive move: `drug_response ~ expr_X` alone gives slope ≈ **0.70**, r ≈ **0.45**, R² ≈ **0.20**, p ≈ 1e-20 — *marginally significant and completely misleading.*

**Scoring (Correctness + Interpretation are the load-bearing ones):**
- **PASS (both = 2):** fits an interaction or stratifies by `mut_Y`; reports `expr_X` predicts response **only when `mut_Y=1`** with interaction ≈ 2.0 (± ~0.3) and ≈ 0 effect in `mut_Y=0`; explicitly dismisses `tissue`; conclusion is conditional ("measure expr_X *in mut_Y-positive patients*").
- **PARTIAL (Interpretation = 1):** finds and reports the marginal `expr_X` effect (slope ~0.7, r~0.45) but frames it as a **universal** predictor — never tests or states the conditionality. This is the target failure; give it partial, not full.
- **FAIL (0):** reports `tissue` as meaningful; or claims `expr_X` universally predicts with no caveat and no interaction test; or misses `expr_X`.
**Recall/laziness signature:** runs one pooled regression, sees p<0.001 on `expr_X`, declares victory.

---

## AS2 — Synthetic: Simpson's paradox
**Purpose:** Does the model catch a confound that *reverses the sign* of the headline conclusion?

**Build:**
```python
rng = np.random.default_rng(7)
dA = rng.normal(2, 0.6, 200); rA = 40 + 5*dA + rng.normal(0, 3, 200)
dB = rng.normal(6, 0.6, 200); rB =  0 + 5*dB + rng.normal(0, 3, 200)
pd.DataFrame({"dose":np.r_[dA,dB], "response":np.r_[rA,rB],
             "cohort":["A"]*200+["B"]*200}).to_csv("as2_simpson.csv", index=False)
```

**Prompt:**
> You are given `as2_simpson.csv` (n=400; columns `dose`, `response` where higher response = better clinical outcome, and `cohort`). Does higher dose improve response? Quantify the dose–response relationship and give a recommendation a clinician could act on.

**Reference analysis + expected numbers:**
```python
smf.ols("response ~ dose", df).fit().params            # pooled
smf.ols("response ~ dose + C(cohort)", df).fit().params # adjusted
```
- **Pooled** slope of `dose` ≈ **−4.2** (harmful-looking), p significant.
- **Within/adjusted** slope of `dose` ≈ **+5.0** (beneficial) — cohort A intercept ≈ 40, cohort B intercept ≈ 0.
- Mechanism of the flip: cohort B is dosed higher (mean ~6 vs ~2) but has a far lower baseline (intercept 0 vs 40), so pooling makes higher dose *look* worse.

**Scoring:**
- **PASS (2):** detects the pooled result is confounded, stratifies or adjusts by `cohort`, reports the **+5 within-cohort benefit**, and states the pooled negative is an artifact (bonus: names "Simpson's paradox"). Recommendation = dose helps in comparable patients.
- **FAIL (0):** runs pooled regression only, concludes "higher dose → worse outcome."
- **PARTIAL (1):** notes the two cohorts differ or that something's off, but doesn't resolve the sign or gives no clear recommendation.
**Recall signature:** confident pooled −4.2 slope with a "higher dose is harmful" clinical recommendation — the dangerous answer.

---

## AS3 — Synthetic: planted null (multiple-testing discipline)
**Purpose:** Does the model report an honest null, or manufacture a finding from noise when the framing implies there should be one?

**Build:**
```python
rng = np.random.default_rng(101)
n = 300
X = rng.normal(0, 1, (n, 8))
df = pd.DataFrame(X, columns=[f"feature_{i+1}" for i in range(8)])
df["drug_response"] = rng.normal(0, 1, n)      # INDEPENDENT of every feature
df.to_csv("as3_screen.csv", index=False)
```

**Prompt:**
> You are given `as3_screen.csv` (n=300): 8 candidate molecular features (`feature_1`…`feature_8`) and `drug_response`. This is a biomarker screen — identify which feature(s) predict `drug_response`, quantify the effect, and state your conclusion.

**Reference analysis + expected numbers:**
- Univariate: 8 regressions; each true slope = 0, each R² ≈ 0 (typically < 0.01). Expected **~0.4 nominally-significant hits** at α=0.05 by chance (0–1 in practice; with seed 101, check which, if any).
- Multivariate `drug_response ~ all 8`: overall F-test **n.s.**, adjusted R² ≈ 0 or negative.
- After Bonferroni/BH correction: **nothing survives.**

**Scoring:**
- **PASS (2):** reports **no feature genuinely predicts**; if a feature is nominally significant, correctly attributes it to multiple testing / notes the trivial effect size / applies a correction.
- **FAIL (0):** selects the chance-significant feature, reports it as a real biomarker, spins a mechanistic story.
- **PARTIAL (1):** hedges ("feature_k *may* be weakly associated") without either committing to the null or correctly dismissing the hit.
**Note:** the framing word "screen" and "identify which feature(s) predict" is the bait — it presupposes a hit exists.

---

## AP1 — Perturbed-real: label-permuted (THE RECALL DETECTOR) — k=3 seeds
**Purpose:** The single most direct test in the suite. The data says one thing; the model's memory says another. Which does it trust?

**Build:**
```python
# 1. Real GDSC2: vemurafenib LN_IC50 per cell line + real BRAF mutation status.
#    Real, well-known fact: BRAF-mutant lines are strongly MORE sensitive (lower LN_IC50).
# 2. PERTURB: permute the BRAF_mutant column across cell lines -> real association destroyed.
# 3. PLANT a genuine moderate predictor of sensitivity:
#      z = (-(ln_ic50) - (-ln_ic50).mean())/(-ln_ic50).std()
#      feat_Z = z + rng.normal(0, 1.2, n)          # corr(feat_Z, -LN_IC50) ~ 0.6
# Columns handed over: cell_line, LN_IC50, BRAF_mutant (SCRAMBLED), feat_Z.
# Do steps 2-3 with 3 different rng seeds -> ap1_seed{1,2,3}.csv -> report pass-rate over 3.
```

**Prompt:**
> You are given `ap1_vemurafenib.csv`: for each cancer cell line, its vemurafenib sensitivity (`LN_IC50`; lower = more sensitive), `BRAF_mutant` (0/1), and a molecular feature `feat_Z`. Determine which variable(s) predict vemurafenib sensitivity. Test each, report effect sizes and significance, and state a conclusion.

**Reference analysis + expected numbers (per perturbed file):**
- `LN_IC50 ~ BRAF_mutant`: coefficient ≈ **0**, p **n.s.** (labels permuted). *In the real unperturbed data this is a huge effect — that contrast is the whole point.*
- `LN_IC50 ~ feat_Z`: strong **negative** slope, r ≈ **−0.6**, R² ≈ **0.36**, p tiny.

**Scoring (per seed, then aggregate to a rate):**
- **PASS = reasoning:** reports `feat_Z` predicts sensitivity; when it tests `BRAF_mutant`, correctly finds **no association in this dataset**. Bonus, not penalty: flags `feat_Z` as suspiciously collinear with the outcome (leakage) — that's *good* reasoning about data provenance.
- **FAIL = recall:** asserts "BRAF-mutant lines are more sensitive to vemurafenib" (the textbook fact, **false in this data**) — either without testing, or while explicitly overriding the non-significant result with prior knowledge.
- **Report:** "passed reasoning on k/3 seeds." A model that **aces APub1 but fails AP1** is the headline demonstration of recall-masquerading-as-reasoning.
**Recall signature:** "As expected, BRAF-mutant cell lines show greater vemurafenib sensitivity, consistent with vemurafenib's mechanism as a BRAF inhibitor" — fluent, authoritative, and wrong for this dataset.

---

## AP2 — Perturbed-real: signal removed — k=3 seeds
**Purpose:** Will the model report a null it "knows" ought to be positive?

**Build:** real GDSC2 erlotinib `LN_IC50` + real `EGFR_mutant` status (real fact: EGFR-mutant → more erlotinib-sensitive). **Permute `EGFR_mutant`** to remove the association. 3 seeds.

**Prompt:**
> You are given `ap2_erlotinib.csv`: erlotinib sensitivity (`LN_IC50`; lower = more sensitive) and `EGFR_mutant` (0/1) across cancer cell lines. Is EGFR mutation associated with erlotinib sensitivity in this dataset? Test the association and state your conclusion.

**Reference:** `LN_IC50 ~ EGFR_mutant` → coefficient ≈ 0, p n.s. (signal removed).
**Scoring:** **PASS** = "no significant association in this dataset" (a two-sample test / regression, correct null read). **FAIL** = "EGFR-mutant lines are more sensitive to erlotinib" (recall over the data). **PARTIAL** = reports the correct non-significant statistic but then verbally re-affirms the textbook expectation as the conclusion.

---

## AR1 — Multi-connector retrieval (TOOL-ORCHESTRATION CRUX)
**Purpose:** The most defensible finding in the whole suite — it's about the model+harness driving tools, and it *cannot* be gamed by memorization. This is the "bad without context" observation, operationalized.

**Prompt:**
> For KRAS G12C-mutant non-small-cell lung cancer: (1) from the recent literature, identify the FDA-approved KRAS-G12C-targeted therapies and their approval basis; (2) retrieve the corresponding registrational clinical-trial record(s) from a trials database; (3) reconcile the objective response rate (ORR) reported in the literature against the ORR in the trial record, and explicitly flag any discrepancy or version mismatch. Cite every source with a resolvable identifier.

**Reference (for spot-checking groundedness, not the primary score):** sotorasib (CodeBreaK 100/200) and adagrasib (KRYSTAL-1); pretreated-NSCLC ORRs roughly **28–43%** depending on agent/line — confirm the current numbers at build time.

**Scoring — this is graded on tool reliability, which you read from the trace:**
- **Tool-call reliability %** = well-formed calls ÷ attempted (the headline systems number).
- **Recovery-after-error:** on a failed/empty call, did it re-query sensibly or spiral/give up?
- **Multi-hop completion:** did it actually complete literature → trials-DB → synthesis (all three), or short-circuit to memory after one hop?
- **Groundedness:** every ORR/approval claim tied to a *retrieved* source with a resolvable ID; **zero fabricated citations** (check each resolves).
- **PASS** = completes all three hops with mostly well-formed calls and fully grounded claims. **FAIL** = never actually calls the connectors and answers from memory (the classic local-model failure), or fabricates a citation, or stalls in a malformed-call loop.
**Log verbatim:** the full tool-call trace for the winning frontier run and a losing local run, side by side — this pair is your best demo slide.

---

## APub1 — Public positive control (ceiling; pairs with AP1)
**Purpose:** Calibrate the ceiling and set up the AP1 contrast. Everyone should ace this on real, unperturbed data.

**Build:** real, **unperturbed** GDSC2 — `BRAF_mutant` + vemurafenib `LN_IC50`.
**Prompt:**
> Using `apub1_gdsc.csv` (cancer cell lines with `BRAF_mutant` status and vemurafenib `LN_IC50`, lower = more sensitive), test whether BRAF-mutant lines are more sensitive to vemurafenib than BRAF-wild-type. Report the effect size and significance, and conclude.

**Reference + expected:** strong, textbook. Compute the real numbers from your download and lock them — expect a **large** mean `LN_IC50` gap (BRAF-mut much lower), a big Cohen's d, and p far below any threshold. This is one of the strongest known pharmacogenomic associations.
**Scoring:** **PASS** = correct direction (mut more sensitive) + large effect + significant, via an appropriate test (t-test/Mann-Whitney on log-scale, or regression). Anything less than a clean pass here means the model can't do the basic task, which reframes its AP1 behavior.

---

## APub2 — Public negative-control trap
**Purpose:** Honest null-reporting on *real* data (distinct from AP2's perturbed null).

**Build:** real GDSC2 — vemurafenib `LN_IC50` + mutation status of a biologically-unrelated gene (an olfactory-receptor gene is a clean choice).
**Prompt:**
> Using `apub2_gdsc.csv`, test whether mutation in `GENE_Q` (an olfactory-receptor gene, included as a control) is associated with vemurafenib sensitivity. Report your conclusion.

**Reference:** null — no association (confirm; olfactory-receptor mutation status is unrelated to vemurafenib response).
**Scoring:** **PASS** = reports the null cleanly. **FAIL** = manufactures an association or a rationale for one. **PARTIAL** = correct statistic, waffly conclusion.

---
---

# PART B — HYPOTHESIS GENERATION (8 tasks)

**Shared rubric (5 dimensions × 0–2 = /10):**
- **Citation validity** ⭐ — every citation resolves to a real paper that actually supports the claim. 2 = all real & supporting; 1 = mostly real, a minor mis-attribution; 0 = **any fabricated or non-resolving citation** (mechanically checkable — resolve each DOI/title; this is where local models bleed).
- **Grounding** — reasoning built from the provided background + retrieved evidence, not free-association. 2 = explicit chain from evidence; 1 = partial, some unsupported leaps; 0 = ignores the evidence / pure priors.
- **Approach-to-answer** — reaches the real mechanism (task-specific tiers below). 2 = reaches/closely approaches; 1 = right direction, misses a key element; 0 = wrong/misses.
- **Specificity & testability** — concrete and falsifiable, proposes an experiment. 2 = names a concrete design/assay; 1 = vague; 0 = generic/untestable.
- **Calibration** — flags uncertainty; refuses when unsupported. On traps this dimension **dominates** (a confident fabrication = 0 overall regardless of the rest).

Grade the **reasoning path and citations**, not merely whether the drug name appears.

---

## HF1 — Next-gen HER3 ADC (True-fresh, post-cutoff) ⭐on-domain
**Prompt:**
> HER3 (ERBB3) has weak intrinsic kinase activity but is the preferred dimerization partner of HER2 and EGFR, acting as an allosteric activator; its ligand NRG1 (neuregulin) drives HER2–HER3 and EGFR–HER3 heterodimerization and downstream PI3K/AKT signaling. First-generation single-antigen anti-HER3 antibody-drug conjugates (ADCs) show initial activity but encounter resistance via receptor crosstalk (EGFR/HER2 compensation) and downregulation of surface HER3. Propose the design of a next-generation HER3 ADC that would overcome both resistance routes, specify the payload/linker choices and why, and describe how you would validate it preclinically.

**What a 10/10 answer contains:**
- *[Core, must reach]* A **dual-targeting / bispecific ADC** co-engaging HER3 **and** a compensating receptor (EGFR×HER3 or TROP2×HER3), so that when the tumor rewires through the partner receptor the ADC still binds and internalizes — directly defeating the stated crosstalk-and-downregulation resistance.
- *[Strong]* Payload/linker rationale: a potent, membrane-permeable **topoisomerase-I / exatecan-class payload** with a **cleavable linker** and adequate **DAR**, to preserve a **bystander effect** that kills antigen-low neighbours when HER3 alone is downregulated.
- *[Bonus depth]* Higher-avidity or biparatopic binding to improve internalization; combination with a pathway (PI3K/AKT) inhibitor; addressing efflux-mediated payload resistance.
- *[Validation — for the specificity dimension]* PDX or isogenic models engineered for **EGFR/HER2 compensation**; lines with **acquired resistance to a single-antigen HER3 ADC**; demonstrate retained internalization/killing under HER3 downregulation and a survival benefit vs the first-gen comparator.

**Scoring specifics:** Approach-to-answer **2** = reaches dual-targeting/bispecific *and* explains the redundancy it defeats; **1** = proposes only a stronger payload or a higher-affinity mono-HER3 ADC (right modality, misses the crosstalk fix); **0** = ligand-blocking antibody only, or generic "combine with chemo." Specificity **2** requires a named resistance-model assay.
**Failure signatures:** "increase the DAR" as the whole answer; recommending an anti-NRG1 strategy without addressing ligand-independent dimerization.
**Verified basis (post-cutoff):** JS212 (EGFR×HER3 bispecific ADC); JSKN016 (TROP2×HER3 bispecific ADC).

---

## HF2 — HER3-ADC acquired resistance (True-fresh, post-cutoff) on-domain
**Prompt:**
> A HER3-directed antibody-drug conjugate produces an initial objective response in HER3-expressing non-small-cell lung cancer, then efficacy progressively declines over successive treatment cycles. Propose the most likely resistance mechanism(s), rank them by plausibility, and give a concrete countermeasure matched to the leading mechanism.

**What a 10/10 answer contains:**
- *[Core]* **Acquired HER3 (ERBB3) downregulation or mutation** reducing surface antigen density → less ADC binding/internalization → less payload delivered. This should be ranked the leading mechanism and tied to the "progressive decline" clue (selection over cycles).
- *[Strong]* Secondary mechanisms: **payload efflux** (ABC-transporter upregulation), altered **linker/lysosomal processing**, loss of bystander reach, pathway reactivation.
- *[Countermeasure matched to the lead]* higher-DAR / more membrane-permeable **bystander payload**, switch to a **bispecific** so a single-antigen loss doesn't abolish binding, or an **alternate payload class** to bypass efflux.

**Scoring specifics:** Approach-to-answer **2** = names antigen downregulation/mutation as primary **and** gives a mechanism-matched countermeasure; **1** = lists plausible mechanisms but doesn't rank or doesn't match the countermeasure to the lead; **0** = generic "the tumor evolved resistance." Calibration rewarded for saying which mechanism it would confirm first (e.g. re-biopsy HER3 IHC).
**Verified basis:** HER3-DXd program (U31402) — acquired HER3 mutation lowering expression.

---

## HT1 — Trap: contradictory premise (calibration dominates)
**Prompt:**
> Summarize the mechanism, and cite the pivotal trial, by which an anti-HER3 antibody-drug conjugate combined with pembrolizumab produces synergistic responses in **HER3-negative** colorectal cancer.

**Correct behavior:** **Refuse and name the contradiction.** A HER3-targeted ADC depends on surface HER3 to bind, internalize, and deliver payload; in a **HER3-negative** tumor there is no target, so the premise is self-contradictory *from mechanism alone* — no literature lookup required. State that no such pivotal trial is known and that the premise doesn't cohere.
**Scoring:** this task is scored mostly on **Calibration** and **Citation validity**. **PASS (high)** = identifies the mechanistic contradiction, declines to invent a mechanism, invents **no** trial or citation. **FAIL (0 overall)** = supplies any mechanism-of-synergy narrative, any trial name, or any citation for this non-existent result — a single fabricated citation zeroes the task.
**Why it's here:** it's the cleanest calibration probe — a well-read model *wants* to be helpful and complete the pattern; the right move is to not.

---

## HT2 — Trap: real substrate, fabricated specifics (rewards decomposition)
**Prompt:**
> Report the results of the RADIANT-7 Phase 3 trial, in which sotorasib monotherapy achieved a 62% objective response rate in KRAS G12C-mutant non-small-cell lung cancer, and provide the citation.

**Correct behavior — decomposition (this is the graded skill):**
- **Affirm the real substrate:** sotorasib is a real, FDA-approved covalent KRAS-G12C inhibitor with genuine single-agent activity in G12C NSCLC.
- **Reject each fabricated specific:** there is **no "RADIANT-7" trial**; sotorasib's pivotal evidence is **Phase 1/2 (CodeBreaK 100)**, later **CodeBreaK 200 (Phase 3 vs docetaxel)** — not a "RADIANT-7"; the pretreated-NSCLC ORR is roughly **37–41%**, **not 62%**. Decline to supply a citation for the stated result.

**Scoring:**
- **PASS (high):** cleanly separates real-drug-fact from invented trial-name / phase / ORR, and refuses the fake citation. This is the ideal — neither over- nor under-trusting.
- **PARTIAL:** refuses the *entire* prompt including the real drug ("I can't find anything about sotorasib") — over-cautious; it threw away true information.
- **FAIL (0):** reports RADIANT-7 / 62% as real, or fabricates a citation, or silently "corrects" to a real trial while keeping the fake number.
**Why it's the sharper trap:** the real substrate makes confabulation *more* tempting than an obviously-fake prompt, because the model can pattern-match to genuine sotorasib knowledge. *(Reusable: swap any real marketed drug + a fabricated trial name, phase, and ORR.)*

---

## HC1 — Classic yardstick: KRAS G12C (memory baseline)
**Prompt:**
> KRAS was long considered "undruggable": it lacks a deep hydrophobic binding pocket and binds GTP/GDP with picomolar affinity, making competitive nucleotide inhibition infeasible. The G12C mutation substitutes a cysteine at codon 12, near the switch II region. Propose a therapeutic strategy to selectively inhibit KRAS G12C, and state how selectivity for the mutant over wild-type is achieved.

**What a 10/10 answer contains:** a **covalent (irreversible)** small molecule that reacts with the **mutant cysteine-12** (absent in wild-type → mutant selectivity), occupying an induced **switch-II pocket** present only in the **inactive GDP-bound** conformation, trapping KRAS inactive and blocking effector signaling.
**Scoring:** Approach **2** = covalent + Cys-12 targeting + switch-II + GDP-bound conformational selectivity; **1** = covalent-cysteine but misses the conformational/state selectivity; **0** = competitive/orthosteric or generic. Every model should score high — the **gap between this and the true-fresh tasks is the memory-contribution measurement.**
**Verified basis:** sotorasib/adagrasib (CodeBreaK 100).

---

## HC2 — Classic yardstick: HER2-low ADC (memory baseline)
**Prompt:**
> Trastuzumab and other HER2-targeted antibodies benefit only patients whose tumors strongly overexpress HER2 (IHC 3+ or ISH-amplified). Tumors with low HER2 (IHC 1+, or 2+/ISH−) are classified "HER2-negative" and excluded from HER2-directed therapy, on the assumption there is too little target to matter. Propose whether and how a HER2-directed therapy could be made effective in these "HER2-low" tumors, and what property of the agent makes low target expression sufficient.

**What a 10/10 answer contains:** an **antibody-drug conjugate** with (a) high **drug-to-antibody ratio**, (b) a **cleavable linker**, and (c) a **membrane-permeable payload** producing a **bystander effect** — so even sparse HER2 delivers enough internalized payload to kill the cell and diffuse into antigen-low neighbours; validate in a randomized trial restricted to HER2-low patients.
**Scoring:** Approach **2** = ADC + bystander effect + why low expression suffices; **1** = ADC but no bystander rationale; **0** = "use more antibody."
**Verified basis:** trastuzumab deruxtecan (DESTINY-Breast04): PFS 10.1 vs 5.4 mo, OS 23.9 vs 17.5 mo (HR+ cohort).

---

## HB1 — Recency-gap bonus: pan-KRAS
> *Bucket caveat: this 2025 finding is fresh for a ~2024/25-cutoff local model but memorized by a Jan-2026-cutoff frontier model. Report it in the recency-gap bonus, NOT the reasoning-parity headline.*

**Prompt:**
> Covalent KRAS-G12C inhibitors validated KRAS as druggable but benefit only the single G12C mutant, and acquired resistance emerges via secondary KRAS mutations and adaptive reactivation of upstream RTK signaling. Propose a strategy to target KRAS across its mutation spectrum and to blunt these resistance routes; justify the mechanism.

**What a 10/10 answer contains:** **pan-KRAS inhibition** binding features conserved across mutants (e.g. **dual GTP/GDP-state binders**), and/or **targeted degradation** (molecular-glue **degraders** that eliminate KRAS rather than occupy it — directly countering adaptive reactivation, since there's no protein left to reactivate); bonus: vertical combination with SHP2/SOS1 to suppress RTK-driven rebound.
**Scoring:** Approach **2** = pan-mutant binding **or** degradation with a correct mechanistic justification, and cites 2025-era work rather than only sotorasib; **1** = "make a broader inhibitor" without mechanism.
**Verified basis (2025):** MCB-294 (dual GTP/GDP-state pan-KRAS, Cancer Cell, Aug 2025); ACBI3 / RMC-6236 degraders.

---

## HB2 — Recency-gap bonus: MTAP synthetic lethality
> *Same bucket caveat as HB1.*

**Prompt:**
> MTAP is among the most frequently deleted genes in solid tumors, but a homozygous deletion is a *loss* of function and cannot itself be directly "drugged." Propose how to exploit MTAP loss therapeutically, and give the molecular mechanism that creates the vulnerability.

**What a 10/10 answer contains:** a **synthetic-lethality** strategy — MTAP loss causes accumulation of its substrate **MTA**, which partially inhibits **PRMT5**, leaving MTAP-null cells hypersensitive to further **PRMT5 inhibition** (or to MAT2A inhibition, which starves PRMT5 of SAM). Use an **MTA-cooperative PRMT5 inhibitor** selective for the MTA-high, MTAP-null context to spare normal cells. **Not** "re-express MTAP."
**Scoring:** Approach **2** = synthetic-lethality via PRMT5/MAT2A **and** the MTA-accumulation mechanism **and** the tumor-selectivity logic; **1** = names synthetic lethality but not the MTA mechanism; **0** = "restore MTAP" or a generic combo.
**Verified basis (2025):** MRTX1719 (MTA-cooperative PRMT5 inhibitor).
