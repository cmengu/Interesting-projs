# open-science + ToolUniverse — setup feedback (11 Jul 2026)

Playbook for running the [open-science](https://github.com/ai4s-research/open-science) harness with a local model (Qwen3.6-35B-A3B on 80 GB VRAM, 2×A100 40GB), distilled from a working session. Core finding: **the bottleneck is harness design, not VRAM** — ~70k tokens of MCP tool schemas were being loaded into every turn.

## Mental model

Reasoning quality of a local model on agentic tasks ≈
**(fraction of context that's signal) × (tokens left for thinking) × (number of verified attempts)**.

Every change below pushes one of those three numbers up. Token budget target for tool schemas: **≤ ~5k tokens, ≤ ~30 visible tools, exactly one tool per capability.**

## 1. ToolUniverse: enable compact mode (5-minute fix)

600+ tools ≈ 70k tokens of schemas. [Compact mode](https://zitniklab.hms.harvard.edu/ToolUniverse/guide/building_ai_scientists/compact_mode.html) replaces them with 4–5 meta-tools:

```json
{
  "mcpServers": {
    "tooluniverse": {
      "command": "tooluniverse-smcp-stdio",
      "args": ["--compact-mode"],
      "env": {"PYTHONIOENCODING": "utf-8"}
    }
  }
}
```

Model then discovers tools on demand: `list_tools(mode="names")` → `grep_tools(pattern=...)` → `get_tool_info(tool_names=[...])` → `execute_tool(tool_name=..., arguments={...})`.

If the Settings UI has no args field, wrap it: a two-line shell script `exec tooluniverse-smcp-stdio --compact-mode`, registered as the command.

**A ~35B won't invent the discipline itself** — pin it in the workspace `AGENTS.md` (read into every session):

```markdown
# Tool use protocol (ToolUniverse)
1. grep_tools(pattern=...) to find tools for the capability you need.
2. get_tool_info on at most 5 candidates — never more.
3. execute_tool(tool_name=..., arguments={...}) to run your pick.
Never ask for the full tool list in one shot. If a call fails,
read the error, fix the arguments once, then try a different tool.
```

## 2. Stronger option: code-as-actions (skip MCP for ToolUniverse)

ToolUniverse is a Python package first. open-science bundles Jupyter via `uv`, so tools can be called from code cells — zero schemas in context, chains collapse into one cell, intermediate results stay in variables. This is the pattern behind Anthropic's [98.7% token reduction measurement](https://www.anthropic.com/engineering/code-execution-with-mcp).

Setup: `uv pip install tooluniverse` into the workspace env, disable the ToolUniverse MCP connector, add a local skill at `.opencode/skills/tooluniverse-code/SKILL.md`:

```markdown
---
name: tooluniverse-code
description: Use when a task needs scientific databases or biomedical
  tools — genes, proteins, drugs, clinical trials, literature lookups.
  Call ToolUniverse from Python code cells instead of MCP tools.
---

# ToolUniverse via Python (not MCP)

## Setup (once per notebook)
    from tooluniverse import ToolUniverse
    tu = ToolUniverse()
    tu.load_tools()

## Find tools (max 5 results, keyword first)
    hits = tu.run({"name": "Tool_Finder_Keyword",
                   "arguments": {"query": "protein structure", "limit": 5}})

## Inspect one tool's schema
    spec = tu.tool_specification("UniProt_get_protein_info")

## Run a tool
    r = tu.run({"name": "UniProt_get_protein_info",
                "arguments": {"gene_symbol": "BRCA1"}})

## Rules
- Batch related calls in ONE cell with a loop; join/filter with pandas.
- Intermediate results stay in variables. Print only compact summaries.
- Persist anything reusable to a file and reference the path.
```

Rollout: compact-mode MCP for chat/survey stages, code-as-actions for the experiment-suite stage (where long tool chains live). They coexist fine.

## 3. Other MCP servers: consolidate, don't compact

The real killer across a multi-server stack is **overlap**, not size. Four servers each exposing a "search papers" tool = near-duplicate tools with different schemas → wasted selection steps, inconsistent sources mid-chain, more malformed calls.

| Server | Size | Verdict |
|---|---|---|
| ToolUniverse | 600+ tools | Compact mode — the single catalog |
| [BioMCP](https://github.com/genomoncology/biomcp) | 21 tools | Keep as-is **only** for its unique parts (MyVariant, PubTator3 annotations, `think` tool) |
| [BioContext knowledgebase](https://docs.kb.biocontext.ai/latest/) | ~20+ DBs | Off — self-described proof-of-concept, fully shadowed by the two above |
| Paper-search | handful | Off — open-science ships arXiv/PubMed connectors natively |

Rule of thumb: retrieval-gate catalogs (>~50 tools); load small servers (≤25 tools) directly; assign one owner per capability.

**Result hygiene matters more than schemas for small servers**: one PubMed search returning 20 abstracts injects 10k+ tokens permanently. Always pass `limit`-style params (5, not default); summarize-then-move-on. Code-as-actions solves this structurally (results land in variables; any MCP server is callable from Python via a `fastmcp` client too).

## 4. Model & serving notes (Qwen3.6-35B-A3B on 80 GB)

- **Weights**: AWQ 4-bit (~19 GB, [community quant](https://huggingface.co/cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit)) — big headroom. BF16 ≈ 70 GB, too tight. Skip the FP8 checkpoint on Ampere (no native FP8 compute).
- **KV cache**: the hybrid Gated-DeltaNet architecture keeps real KV in only ~1 in 4 layers — long context is cheap in VRAM. The 70k tool context was an *accuracy* problem, not a memory one.
- **Integration risk**: the hybrid architecture needs a recent vLLM — verify the version supporting Qwen3.6 still ships wheels for your CUDA/driver pin. llama.cpp is the known-good fallback.
- **Test-time compute is the local advantage**: 3B active params → fast decode → sample 5–8 candidates, majority-vote, verify-then-retry on every stage. Local tokens are flat-rate; spend them. A ~35B closes reasoning gaps through more *verified* attempts, not a bigger brain.

## 5. Measure before upgrading GPUs

Before/after every harness change, on a fixed task set: **tokens-per-turn, malformed-call rate, wrong-tool-choice rate, task completion.** Expected from this playbook: ~70k → 3–5k (compact mode) or <1k (code-as-actions) tokens of tool context; the dedup likely moves error rates more than the compact mode does.

Scaling rule: preliminary results reflect harness quality more than VRAM. Prove the free levers first; then an 80→160→320 GB upgrade ask is backed by an accuracy-vs-model-size curve with the harness held constant — which is also the evidence that gets budgets approved.

## Sources

[ToolUniverse paper](https://arxiv.org/html/2509.23426v1) · [compact mode guide](https://zitniklab.hms.harvard.edu/ToolUniverse/guide/building_ai_scientists/compact_mode.html) · [getting started](https://zitniklab.hms.harvard.edu/bioagent/getting_started.html) · [Anthropic: advanced tool use](https://www.anthropic.com/engineering/advanced-tool-use) · [Anthropic: code execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp) · [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) · [open-science](https://github.com/ai4s-research/open-science) · [ai4s-skills format](https://github.com/ai4s-research/ai4s-skills) · [BioMCP](https://biomcp.org/)
