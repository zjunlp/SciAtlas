# GRPO Family Paper Locator

This note records the paper-level provenance for the post-GRPO branch in the
RL case-study timeline, so the timeline spec can use stable titles instead of
community shorthand alone.

## Included in `M6 RL for LLMs / Alignment`

| Label in spec | Canonical paper title | First author | Year | arXiv |
|---|---|---|---:|---|
| `GRPO` | *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* | Shao et al. | 2024 | `2402.03300` |
| `Dr. GRPO` | *Understanding R1-Zero-Like Training: A Critical Perspective* | Liu et al. | 2025 | `2503.20783` |
| `DAPO` | *DAPO: An Open-Source LLM Reinforcement Learning System at Scale* | Yu et al. | 2025 | `2503.14476` |
| `VAPO` | *VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks* | Yue et al. | 2025 | `2504.05118` |
| `ARPO (Replay)` | *ARPO: End-to-End Policy Optimization for GUI Agents with Experience Replay* | Lu et al. | 2025 | `2505.16282` |
| `ARPO (Agentic)` | *Agentic Reinforced Policy Optimization* | Dong et al. | 2025 | `2507.19849` |
| `GiGPO` | *Group-in-Group Policy Optimization for LLM Agent Training* | Feng et al. | 2025 | `2505.10978` |
| `DeepSeek-R1` | *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* | DeepSeek-AI | 2025 | `2501.12948` |

## Naming notes

- `GiGRPO` is a community shorthand I saw informally, but the paper title is
  **GiGPO** (`Group-in-Group Policy Optimization`), not `GiGRPO`. The spec now
  uses the paper's own acronym.
- `ARPO` is ambiguous. At least two relevant 2025 papers use the acronym:
  `Agentic Reinforced Policy Optimization` and `ARPO: End-to-End Policy
  Optimization for GUI Agents with Experience Replay`. Both are post-GRPO
  agent-training variants, but they solve different problems.
- `VAPO` is not literally a GRPO variant. It is included because it is part of
  the same 2025 reasoning-RL branch and is repeatedly compared against
  GRPO/DAPO-style methods in that literature.

## Excluded for now

- Later 2025-2026 GRPO-derivative papers such as `Scaf-GRPO`, `SGPO`,
  `lambda-GRPO`, `GRPO-VPS`, `N-GRPO`, and many multimodal variants were not
  added to the timeline because the current RL figure would become too crowded.
- If we want a denser "reasoning-RL micro-timeline", the clean next step is to
  split `M6` into `preference/alignment` and `reasoning-RL / agent RL`.
