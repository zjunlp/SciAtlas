#!/usr/bin/env python3
"""Build method-timeline specs from Claude Code baseline paper lists.

The Claude Code baseline returns a flat list of Semantic Scholar papers.  This
script turns that list into the human-editable YAML format consumed by
``render_method_timeline.py``.  It intentionally uses a curated, conservative
selection from the retrieved papers instead of injecting missing canonical
anchors, so the resulting figure reflects what the baseline actually found.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DNN_SELECTION = {
    "topic": {
        "title": "Claude Code Baseline: Deep Neural Network Architectures",
        "subtitle": "Anchors selected only from the Claude Code baseline paper list",
        "time_range": [1990, 2026],
        "current_year": 2026,
        "row_height": 145,
        "time_segments": [
            {"start": 1990, "end": 2005, "weight": 0.13},
            {"start": 2005, "end": 2012, "weight": 0.10},
            {"start": 2012, "end": 2016, "weight": 0.15},
            {"start": 2016, "end": 2020, "weight": 0.18},
            {"start": 2020, "end": 2023, "weight": 0.22},
            {"start": 2023, "end": 2026, "weight": 0.22},
        ],
        "future_work_width": 360,
    },
    "future_work": [
        {
            "title": "Coverage Gap",
            "text": "The baseline often retrieves reviews and applications before the original architecture papers; downstream organization should separate retrieval recall from method taxonomy quality.",
            "lane_ids": ["D1", "D2", "D3"],
            "lane_labels": ["MLP / Backprop", "CNN Architectures", "RNN / Sequence"],
        },
        {
            "title": "Vision-Transformer Bias",
            "text": "The strongest canonical hit is ViT, while many CNN and transformer-language anchors are absent from the returned list.",
            "lane_ids": ["D4", "D5", "D6"],
            "lane_labels": ["Seq2Seq / Attention", "Transformers", "Vision Transformers"],
        },
        {
            "title": "Generative Tail",
            "text": "Diffusion-related retrieval is active but dominated by recent variants, applications, and acceleration papers rather than the earliest foundations.",
            "lane_ids": ["D7"],
            "lane_labels": ["Diffusion / Score-Based"],
        },
    ],
    "clusters": [
        {
            "id": "D1",
            "name": "MLP / Backprop",
            "color": "#5B6770",
            "papers": [
                ("Multilayer Perceptron Learning Optimized for On-Chip Implementation: A Noise-Robust System", "MLP hardware"),
                ("Backpropagation: the basic theory", "Backprop theory"),
                ("Unsupervised Discovery of Non-linear Structure Using Contrastive Backpropagation", "Contrastive backprop"),
            ],
        },
        {
            "id": "D2",
            "name": "CNN Architectures",
            "color": "#1E88E5",
            "papers": [
                (
                    "Advances in Convolutional Neural Networks for Image Classification : Architecture Evolution, Transformer Fusion, and Application Expansion",
                    "CNN evolution",
                ),
            ],
        },
        {
            "id": "D3",
            "name": "RNN / Sequence",
            "color": "#43A047",
            "papers": [
                ("Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling", "Gated RNNs"),
                ("A Review of Recurrent Neural Network Architecture for Sequence Learning: Comparison between LSTM and GRU", "RNN review"),
                ("RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series Tasks", "RWKV-TS"),
            ],
        },
        {
            "id": "D4",
            "name": "Seq2Seq / Attention",
            "color": "#00897B",
            "papers": [
                ("Interactive Attention for Neural Machine Translation", "Interactive attention"),
                ("Learning When to Concentrate or Divert Attention: Self-Adaptive Attention Temperature for Neural Machine Translation", "Attention temp."),
                ("Neural machine translation with Gumbel Tree-LSTM based encoder", "Gumbel Tree-LSTM"),
            ],
        },
        {
            "id": "D5",
            "name": "Transformers",
            "color": "#8E24AA",
            "papers": [
                ("A Comparative Analysis of Transformers for Multilingual Neural Machine Translation", "Multilingual Transformers"),
                ("English-to-Malayalam Machine Translation Framework using Transformers", "NMT Transformers"),
            ],
        },
        {
            "id": "D6",
            "name": "Vision Transformers",
            "color": "#3949AB",
            "papers": [
                ("An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", "ViT"),
                ("CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification", "CrossViT"),
                ("Do Vision Transformers See Like Convolutional Neural Networks?", "ViT analysis"),
                ("Funnel Vision Transformer for image classification", "Funnel ViT"),
                ("Hyperspectral Image Classification Using Groupwise Separable Convolutional Vision Transformer Network", "GSC-ViT"),
                ("Vision Transformer (ViT)-based Applications in Image Classification", "ViT applications"),
            ],
        },
        {
            "id": "D7",
            "name": "Diffusion / Score-Based",
            "color": "#D81B60",
            "papers": [
                ("Structured Denoising Diffusion Models in Discrete State-Spaces", "Discrete diffusion"),
                ("Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference", "LCM"),
                ("Adversarial Diffusion Distillation", "ADD"),
                ("SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion", "SV3D"),
                ("A Gradient Flow Approach to Solving Inverse Problems with Latent Diffusion Models", "Latent gradient flow"),
                ("Pixel-Perfect Depth with Semantics-Prompted Diffusion Transformers", "Diffusion transformer"),
            ],
        },
    ],
}


RL_SELECTION = {
    "topic": {
        "title": "Claude Code Baseline: Reinforcement Learning Algorithms",
        "subtitle": "Anchors selected only from the Claude Code baseline paper list",
        "time_range": [2000, 2026],
        "current_year": 2026,
        "row_height": 145,
        "time_segments": [
            {"start": 2000, "end": 2013, "weight": 0.13},
            {"start": 2013, "end": 2017, "weight": 0.16},
            {"start": 2017, "end": 2020, "weight": 0.17},
            {"start": 2020, "end": 2022, "weight": 0.16},
            {"start": 2022, "end": 2024, "weight": 0.18},
            {"start": 2024, "end": 2026, "weight": 0.20},
        ],
        "future_work_width": 390,
    },
    "future_work": [
        {
            "title": "Canonical Misses",
            "text": "The baseline retrieves several related papers but misses many original anchors such as DQN, TRPO, SAC, MuZero, InstructGPT, and DPO.",
            "lane_ids": ["R1", "R2", "R3", "R4", "R6", "R8"],
            "lane_labels": [
                "Classical Foundations",
                "Deep Value-Based RL",
                "Actor-Critic / Continuous Control",
                "Trust Region / PPO",
                "Model-Based RL",
                "RLHF / LLM Alignment",
            ],
        },
        {
            "title": "Application Drift",
            "text": "Many high-ranked hits are robotics, wireless, UAV, medical, or traffic-control applications that mention a method without introducing it.",
            "lane_ids": ["R2", "R3", "R5", "R7"],
            "lane_labels": ["Deep Value-Based RL", "Actor-Critic", "Offline RL", "Sequence RL"],
        },
        {
            "title": "Sequence-RL Strength",
            "text": "Decision Transformer and later sequence-modeling variants are the clearest method-family coverage in this baseline list.",
            "lane_ids": ["R7"],
            "lane_labels": ["Decision Transformer / Sequence RL"],
        },
    ],
    "clusters": [
        {
            "id": "R1",
            "name": "Classical Foundations",
            "color": "#6B4F3F",
            "papers": [
                ("Off-Policy Temporal Difference Learning with Function Approximation", "Off-policy TD"),
                ("QV(λ)-learning: A New On-policy Reinforcement Learning Algorithm", "QV-learning"),
                ("Convergent Temporal-Difference Learning with Arbitrary Smooth Function Approximation", "Convergent TD"),
                ("Gradient temporal-difference learning algorithms", "GTD"),
            ],
        },
        {
            "id": "R2",
            "name": "Deep Value-Based RL",
            "color": "#1565C0",
            "papers": [
                ("Dynamic path planning via Dueling Double Deep Q-Network (D3QN) with prioritized experience replay", "D3QN + PER"),
                ("An Improved Dueling Deep Double-Q Network Based on Prioritized Experience Replay for Path Planning of Unmanned Surface Vehicles", "Dueling DDQN"),
                ("Task Offloading via Prioritized Experience-Based Double Dueling DQN in Edge-Assisted IIoT", "Double Dueling DQN"),
                ("Dual-Priority Delayed Deep Double Q-Network (DPD3QN): A Dueling Double Deep Q-Network with Dual-Priority Experience Replay for Autonomous Driving Behavior Decision-Making", "DPD3QN"),
            ],
        },
        {
            "id": "R3",
            "name": "Actor-Critic / Continuous Control",
            "color": "#EF6C00",
            "papers": [
                ("M-A3C: A Mean-Asynchronous Advantage Actor-Critic Reinforcement Learning Method for Real-Time Gait Planning of Biped Robot", "M-A3C"),
                ("Distributional Soft Actor-Critic With Three Refinements", "DSAC-T"),
                ("A Strategy-Oriented Bayesian Soft Actor-Critic Model", "Bayesian SAC"),
                ("Broad Critic Deep Actor Reinforcement Learning for Continuous Control", "Broad critic actor"),
            ],
        },
        {
            "id": "R4",
            "name": "Trust Region / PPO",
            "color": "#8E24AA",
            "papers": [
                ("Proximal Policy Optimization Algorithms", "PPO"),
                ("Neural Trust Region/Proximal Policy Optimization Attains Globally Optimal Policy", "Neural TRPO/PPO"),
                ("Neural Proximal/Trust Region Policy Optimization Attains Globally Optimal Policy", "Neural PPO/TRPO"),
                ("Accelerating Proximal Policy Optimization Learning Using Task Prediction for Solving Environments with Delayed Rewards", "PPO task prediction"),
                ("Trust Regions Sell, But Who's Buying? Overlap Geometry as an Alternative Trust Region for Policy Optimization", "Overlap trust region"),
            ],
        },
        {
            "id": "R5",
            "name": "Offline RL",
            "color": "#00897B",
            "papers": [
                ("Offline Reinforcement Learning for Autonomous Driving with Safety and Exploration Enhancement", "Safe offline RL"),
                ("Offline Reinforcement Learning for Wireless Network Optimization with Mixture Datasets", "Mixture offline RL"),
                ("Adaptive Neighborhood-Constrained Q Learning for Offline Reinforcement Learning", "ANCQ"),
                ("Coordinating Ride-Pooling with Public Transit using Reward-Guided Conservative Q-Learning: An Offline Training and Online Fine-Tuning Reinforcement Learning Framework", "Reward-guided CQL"),
                ("Physics-informed Koopman-constrained implicit Q-learning for safe offline reinforcement learning in mechanical ventilation", "Koopman IQL"),
            ],
        },
        {
            "id": "R6",
            "name": "Model-Based RL",
            "color": "#2E7D32",
            "papers": [
                ("On the role of planning in model-based deep reinforcement learning", "Planning in MBRL"),
                ("The Value Equivalence Principle for Model-Based Reinforcement Learning", "Value equivalence"),
                ("Planning with Uncertainty: Deep Exploration in Model-Based Reinforcement Learning", "Uncertain planning"),
                ("Bayes Adaptive Monte Carlo Tree Search for Offline Model-based Reinforcement Learning", "Bayes MCTS"),
                ("Efficient Multi-agent Reinforcement Learning by Planning", "MARL planning"),
                ("Multimodal Dreaming: A Global Workspace Approach to World Model-Based Reinforcement Learning", "Multimodal Dreaming"),
            ],
        },
        {
            "id": "R7",
            "name": "Decision Transformer / Sequence RL",
            "color": "#3949AB",
            "papers": [
                ("Decision Transformer: Reinforcement Learning via Sequence Modeling", "Decision Transformer"),
                ("Decision Mamba: Reinforcement Learning via Sequence Modeling with Selective State Spaces", "Decision Mamba"),
                ("Decision Mamba: Reinforcement Learning via Hybrid Selective Sequence Modeling", "Hybrid Decision Mamba"),
                ("HarmoDT: Harmony Multi-Task Decision Transformer for Offline Reinforcement Learning", "HarmoDT"),
                ("Graph decision transformer for offline reinforcement learning", "Graph DT"),
                ("PCDT: Pessimistic Critic Decision Transformer for Offline Reinforcement Learning", "PCDT"),
            ],
        },
        {
            "id": "R8",
            "name": "RLHF / LLM Alignment",
            "color": "#D81B60",
            "papers": [
                ("RLHF in an SFT Way: From Optimal Solution to Reward-Weighted Alignment", "Reward-weighted alignment"),
                ("A Technical Survey of Reinforcement Learning Techniques for Large Language Models", "LLM RL survey"),
                ("Optimizing Safe and Aligned Language Generation: A Multi-Objective GRPO Approach", "Multi-objective GRPO"),
                ("Noise-Aware Direct Preference Optimization for RLAIF", "Noise-aware DPO"),
                ("BiasGRPO: Stabilizing Bias Mitigation in High-Variance Reward Landscapes via Group-Relative Policy Optimization", "BiasGRPO"),
            ],
        },
    ],
}


def load_papers(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    papers = payload.get("papers")
    if not isinstance(papers, list):
        raise SystemExit(f"{path} must contain a top-level papers list")
    return papers


def normalize_title(title: str) -> str:
    return " ".join(title.casefold().split())


def first_author(authors: Any) -> str:
    if not authors:
        return ""
    author = authors[0]
    if isinstance(author, dict):
        return str(author.get("name") or "")
    return str(author)


def author_text(authors: Any) -> str:
    if not authors:
        return ""
    names = [a.get("name", "") if isinstance(a, dict) else str(a) for a in authors]
    names = [n for n in names if n]
    if len(names) <= 3:
        return ", ".join(names)
    return f"{', '.join(names[:3])} et al."


def make_anchor(paper: dict[str, Any], *, label: str, source_index: int) -> dict[str, Any]:
    year = paper.get("year") or paper.get("publication_year")
    if not isinstance(year, int):
        raise SystemExit(f"Selected paper lacks integer year: {paper.get('title')}")
    notes = [f"Claude Code paper index: {source_index}"]
    url = paper.get("url")
    if url:
        notes.append(f"URL: {url}")
    retrieval = paper.get("retrieval") or {}
    if retrieval.get("queries"):
        notes.append("Matched query: " + " | ".join(map(str, retrieval["queries"])))
    return {
        "label": label,
        "full_title": paper.get("title") or label,
        "first_author": first_author(paper.get("authors")),
        "authors": author_text(paper.get("authors")),
        "year": year,
        "citation_count": int(paper.get("citation_count") or 0),
        "venue": paper.get("venue") or "",
        "in_corpus": True,
        "corpus_paper_id": f"CC{source_index:03d}",
        "paper_id": paper.get("paper_id") or paper.get("paperId"),
        "corpus_id": paper.get("corpus_id") or paper.get("corpusId"),
        "notes": " | ".join(notes),
    }


def build_spec(selection: dict[str, Any], papers: list[dict[str, Any]]) -> dict[str, Any]:
    by_title = {normalize_title(str(p.get("title") or "")): (idx, p) for idx, p in enumerate(papers, 1)}
    clusters = []
    missing: list[str] = []
    for cluster in selection["clusters"]:
        anchors = []
        for title, label in cluster["papers"]:
            found = by_title.get(normalize_title(title))
            if not found:
                missing.append(title)
                continue
            idx, paper = found
            anchors.append(make_anchor(paper, label=label, source_index=idx))
        anchors.sort(key=lambda item: (item["year"], item["label"]))
        clusters.append(
            {
                "id": cluster["id"],
                "name": cluster["name"],
                "color": cluster["color"],
                "anchors": anchors,
            }
        )
    if missing:
        raise SystemExit("Selected titles missing from Claude output:\n" + "\n".join(missing))
    return {
        "topic": selection["topic"],
        "future_work": selection["future_work"],
        "clusters": clusters,
    }


def dump_yaml(data: dict[str, Any], path: Path) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyYAML is required to write YAML specs") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )


PALETTES = {
    "dnn": {
        "C_MLP": "#5B6770",
        "C_CNN": "#1E88E5",
        "C_RNN": "#43A047",
        "C_SEQATTN": "#00897B",
        "C_TRANSFORMER": "#8E24AA",
        "C_VIT": "#3949AB",
        "C_DIFFUSION": "#D81B60",
    },
    "rl": {
        "M1": "#6B4F3F",
        "M2": "#1565C0",
        "M3": "#EF6C00",
        "M4": "#8E24AA",
        "M5b": "#00897B",
        "M5a": "#2E7D32",
        "M7": "#3949AB",
        "M6": "#D81B60",
    },
}


def topic_for_candidate_json(case: str) -> dict[str, Any]:
    if case == "dnn":
        return {
            "title": "Claude Code Baseline: Deep Neural Network Architectures",
            "subtitle": "Method-family anchors selected from the Claude Code baseline paper list",
            "time_range": [1990, 2026],
            "current_year": 2026,
            "row_height": 150,
            "time_segments": [
                {"start": 1990, "end": 2005, "weight": 0.13},
                {"start": 2005, "end": 2012, "weight": 0.10},
                {"start": 2012, "end": 2016, "weight": 0.14},
                {"start": 2016, "end": 2020, "weight": 0.17},
                {"start": 2020, "end": 2023, "weight": 0.22},
                {"start": 2023, "end": 2026, "weight": 0.24},
            ],
            "future_work_width": 360,
        }
    return {
        "title": "Claude Code Baseline: Reinforcement Learning Algorithms",
        "subtitle": "Method-family anchors selected from the Claude Code baseline paper list",
        "time_range": [2000, 2026],
        "current_year": 2026,
        "row_height": 150,
        "time_segments": [
            {"start": 2000, "end": 2013, "weight": 0.12},
            {"start": 2013, "end": 2017, "weight": 0.14},
            {"start": 2017, "end": 2020, "weight": 0.16},
            {"start": 2020, "end": 2022, "weight": 0.15},
            {"start": 2022, "end": 2024, "weight": 0.19},
            {"start": 2024, "end": 2026, "weight": 0.24},
        ],
        "future_work_width": 390,
    }


def future_work_for_candidate_json(case: str) -> list[dict[str, Any]]:
    if case == "dnn":
        return [
            {
                "title": "Retrieval Recall Gap",
                "text": "The baseline misses many original architecture anchors and often returns reviews or applications instead, especially for CNNs and language Transformers.",
                "lane_ids": ["C_MLP", "C_CNN", "C_TRANSFORMER"],
                "lane_labels": ["MLP / Backprop", "CNN Architectures", "Transformers"],
            },
            {
                "title": "Strong ViT Coverage",
                "text": "Vision Transformer retrieval is the strongest part of the list, with the original ViT paper and several follow-on variants.",
                "lane_ids": ["C_CNN", "C_VIT"],
                "lane_labels": ["CNN Architectures", "Vision Transformers"],
            },
            {
                "title": "Recent Diffusion Bias",
                "text": "Diffusion papers are present, but the list emphasizes recent variants, distillation, and applications rather than DDPM, Score-SDE, or latent diffusion foundations.",
                "lane_ids": ["C_DIFFUSION"],
                "lane_labels": ["Diffusion / Score-Based"],
            },
        ]
    return [
        {
            "title": "Canonical Misses",
            "text": "The baseline retrieves method-adjacent papers but misses many original anchors such as DQN, TRPO, SAC, MuZero, InstructGPT, DPO, and GRPO.",
            "lane_ids": ["M1", "M2", "M3", "M4", "M5a", "M6"],
            "lane_labels": [
                "Classical Foundations",
                "Deep Value-Based RL",
                "Actor-Critic / Continuous Control",
                "Trust Region / PPO",
                "Model-Based RL",
                "RLHF / LLM Alignment",
            ],
        },
        {
            "title": "Application Drift",
            "text": "Many high-ranked hits are robotics, wireless, UAV, transportation, or healthcare applications that mention algorithms without introducing them.",
            "lane_ids": ["M2", "M3", "M5b", "M7"],
            "lane_labels": ["Deep Value-Based RL", "Actor-Critic", "Offline RL", "Decision Transformer / Sequence RL"],
        },
        {
            "title": "Sequence-RL Coverage",
            "text": "Decision Transformer and later sequence-modeling variants are the clearest method-family coverage in this baseline list.",
            "lane_ids": ["M7"],
            "lane_labels": ["Decision Transformer / Sequence RL"],
        },
    ]


def label_from_title(title: str) -> str:
    replacements = [
        ("Reinforcement Learning", "RL"),
        ("Decision Transformer", "DT"),
        ("Proximal Policy Optimization", "PPO"),
        ("Temporal-Difference", "TD"),
        ("Temporal Difference", "TD"),
        ("Model-Based Reinforcement Learning", "MBRL"),
        ("Deep Q-Network", "DQN"),
        ("Deep Double-Q Network", "DDQN"),
        ("Soft Actor-Critic", "SAC"),
        ("Direct Preference Optimization", "DPO"),
        ("Group-Relative Policy Optimization", "GRPO"),
    ]
    label = title
    for old, new in replacements:
        label = label.replace(old, new)
    label = label.split(":")[0].split(";")[0]
    words = label.split()
    if len(label) > 34:
        label = " ".join(words[:4])
    return label[:34].rstrip()


def anchor_from_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    title = str(candidate.get("title") or "")
    paper_index = candidate.get("paper_index") or candidate.get("rank")
    notes = str(candidate.get("notes") or "")
    if paper_index:
        notes = f"Claude Code paper index: {paper_index}" + (f" | {notes}" if notes else "")
    return {
        "label": candidate.get("label") or label_from_title(title),
        "full_title": title,
        "first_author": candidate.get("first_author") or "",
        "authors": candidate.get("authors") or candidate.get("authors_string") or "",
        "year": candidate.get("year"),
        "citation_count": int(candidate.get("citation_count") or 0),
        "venue": candidate.get("venue") or "",
        "in_corpus": True,
        "corpus_paper_id": f"P{int(paper_index):03d}" if isinstance(paper_index, int) else str(paper_index or ""),
        "paper_index": paper_index,
        "url": candidate.get("url") or "",
        "corpus_id": candidate.get("corpus_id"),
        "paper_id": candidate.get("paper_id") or candidate.get("s2_paper_id"),
        "s2_paper_id": candidate.get("s2_paper_id") or candidate.get("paper_id"),
        "notes": notes,
    }


def build_spec_from_candidate_json(case: str, candidate_json: Path) -> dict[str, Any]:
    payload = json.loads(candidate_json.read_text(encoding="utf-8"))
    raw_clusters = payload.get("clusters")
    if not isinstance(raw_clusters, list):
        raise SystemExit(f"{candidate_json} must contain a top-level clusters list")

    clusters = []
    for raw_cluster in raw_clusters:
        cluster_id = raw_cluster.get("lane_id") or raw_cluster.get("id")
        anchors = raw_cluster.get("anchors")
        candidates = raw_cluster.get("candidates")
        if anchors is None and isinstance(candidates, list):
            anchors = [anchor_from_candidate(candidate) for candidate in candidates]
        if not cluster_id or not isinstance(anchors, list):
            raise SystemExit(f"Malformed cluster in {candidate_json}: {raw_cluster!r}")
        cleaned_anchors = []
        for anchor in anchors:
            cleaned = dict(anchor)
            if "s2_paper_id" in cleaned and "paper_id" not in cleaned:
                cleaned["paper_id"] = cleaned["s2_paper_id"]
            cleaned["in_corpus"] = True
            cleaned_anchors.append(cleaned)
        cleaned_anchors.sort(key=lambda item: (item.get("year") or 9999, item.get("label") or ""))
        clusters.append(
            {
                "id": cluster_id,
                "name": raw_cluster.get("lane_name") or raw_cluster.get("name") or cluster_id,
                "color": PALETTES[case].get(cluster_id, "#5B6770"),
                "anchors": cleaned_anchors,
                "notes": raw_cluster.get("rationale", ""),
            }
        )

    return {
        "topic": topic_for_candidate_json(case),
        "future_work": future_work_for_candidate_json(case),
        "clusters": clusters,
        "source": {
            "baseline": "claude_code",
            "candidate_json": str(candidate_json),
            "selection_policy": payload.get("selection_policy", ""),
            "noise_summary": payload.get("noise_summary", {}),
            "notable_missing_canonical_anchors": payload.get("notable_missing_canonical_anchors", []),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["dnn", "rl"], required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path)
    parser.add_argument("--output-spec", type=Path, required=True)
    args = parser.parse_args()

    if args.candidate_json:
        spec = build_spec_from_candidate_json(args.case, args.candidate_json)
    else:
        selection = DNN_SELECTION if args.case == "dnn" else RL_SELECTION
        spec = build_spec(selection, load_papers(args.input_json))
    dump_yaml(spec, args.output_spec)
    counts = {cluster["name"]: len(cluster["anchors"]) for cluster in spec["clusters"]}
    print(json.dumps({"output_spec": str(args.output_spec), "counts": counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
