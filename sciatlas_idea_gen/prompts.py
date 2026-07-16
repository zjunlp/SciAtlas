"""Prompt templates for every LLM step.

Each template is a ``str.format``-style string; the calling step is responsible
for supplying the named fields. Templates are transcribed from
``docs/sciatlas-ar-pipeline.md`` with only minor cleanup so the pipeline stays
faithful to the documented design.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# Step 1 — query refinement
# --------------------------------------------------------------------------- #
QUERY_REFINEMENT = """\
You are a scientific research analyst. The user provides a raw research topic or question.

Raw query:
{raw_query}

Your task:
Refine the raw query into a precise, well-scoped research framing that can drive literature retrieval.

Requirements:
1. `research_problem`: one or two sentences stating the concrete research problem behind the raw query.
2. `keywords`: a list of 4-8 short technical search keywords or phrases (no full sentences).
3. `refined_research_question`: a single, sharper research question that captures the intent of the raw query, suitable as a search query.
4. Stay faithful to the raw query. Do not invent a different topic. Do not add facts you cannot infer.
5. Output strict JSON only with this schema and nothing else:
{{
  "research_problem": "...",
  "keywords": ["...", "..."],
  "refined_research_question": "..."
}}
"""

# --------------------------------------------------------------------------- #
# Step 1 — diverse query generation (multi-query seed retrieval)
# --------------------------------------------------------------------------- #
SEED_QUERY_DIVERSIFICATION = """\
You are helping retrieve a DIVERSE set of seed papers for a research-idea-generation pipeline.
Everything downstream is built outward from these seeds, so the retrieval must cover the
research problem from several different angles instead of returning near-duplicate papers.

Research problem:
{research_problem}

Refined research question:
{refined_query}

Keywords:
{keywords}

Your task:
Generate exactly {k} DISTINCT search queries that together span the breadth of this research
problem. The queries are run independently and their results are pooled, so they should be as
complementary as possible.

Requirements:
1. Each query is a concise search query (a short phrase or single question), not a paragraph.
2. Every query targets a DIFFERENT facet, sub-direction, method family, application, problem
   formulation, or perspective of the research problem — minimize overlap between queries.
3. Stay faithful to the research problem. Do not drift into an unrelated topic.
4. Make the queries specific enough to retrieve focused primary-research papers (avoid vague,
   one-word queries).

Output strict JSON only with this schema and nothing else:
{{
  "queries": ["query 1", "query 2", "..."]
}}
Return exactly {k} queries.
"""

# --------------------------------------------------------------------------- #
# Step 1 — seed-paper selection (topic relevance + influence)
# --------------------------------------------------------------------------- #
SEED_SELECTION = """\
You are selecting the seed papers that will anchor a research-idea-generation pipeline for a specific research topic. Everything downstream (the citation graph, the trend analysis, the final idea) is built outward from these seeds, so an off-topic seed contaminates the whole run.

Research topic:
{research_topic}

Candidate papers (already retrieved and pre-ranked; each has a citation count and year):
{candidate_list}

Your task:
Select up to {k} seed papers that BEST anchor work on the research topic. A good seed must satisfy BOTH criteria:
1. Strong topical relevance: the paper is directly and centrally about the research topic, not merely sharing a broad theme (e.g. "LLMs", "deep learning", "neural networks") and not about an adjacent application area that only borrows similar techniques.
2. Influence / foundational value: prefer well-cited, foundational, or methodologically pivotal papers over incremental or niche ones.

Rules:
- Reject papers that are off-topic or only loosely related, EVEN IF they are highly cited. A high citation count never compensates for being off-topic.
- Prefer primary research (methods / empirical) over surveys, reviews, or tutorials.
- Favor diversity across the core sub-directions of the topic over several near-duplicates of the same approach.
- If fewer than {k} candidates are strongly relevant, return FEWER rather than padding the list with off-topic papers.

Output strict JSON only with this schema and nothing else:
{{
  "selected_paper_ids": ["paper_id_1", "paper_id_2"],
  "rationale": "brief explanation of why each chosen paper is both strongly on-topic and influential"
}}
Every selected id must come from the candidate list. Do not return more than {k} ids.
"""

# --------------------------------------------------------------------------- #
# Step 2 — merge_search-style relevance rerank (survey papers get 0)
# --------------------------------------------------------------------------- #
MERGE_RERANK_RELEVANCE = """\
Evaluate how substantively relevant each candidate paper is to the query.

Scoring target:
- Judge each paper independently, not comparatively.
- Focus on overlap in the core research problem, evaluation target, inputs/outputs, and methodological purpose.
- A paper can score high even if the method differs, as long as it directly addresses the same problem.
- Penalize papers that only share broad themes such as LLMs, scientific discovery, benchmarks, or evaluation.
- Ignore citation counts, venue prestige, popularity, and writing polish.
- If the candidate is a survey, review, overview, literature review, systematic review, meta-analysis, tutorial, or primer paper, give it a score of 0.

Rubric:
- 9-10: Extremely close neighbor; essentially the same problem or an immediately adjacent paper you would definitely cite.
- 7-8: Directly relevant; strong task or evaluation overlap.
- 5-6: Meaningful partial overlap, but not the same core problem.
- 3-4: Weak topical relation only.
- 0-2: Effectively irrelevant, or a survey/review/overview/tutorial paper.

Return ONLY JSON in this format:
{{"papers":[{{"paper_index":1,"score":8.7,"reason":"short reason"}}]}}

Requirements for output:
- Include every paper exactly once.
- score must be a number in [0, 10].
- reason must be concise and mention the main match or mismatch (or that it is a survey).

Query type: {query_type}
Query paper title: {query_title}
Query text:
{query_text}

Candidate papers:
{candidate_blocks}
"""

# --------------------------------------------------------------------------- #
# Step 1 — final seed selection from the pooled multi-query candidates.
# Same scoring rubric as MERGE_RERANK_RELEVANCE, but with an explicit emphasis on
# DIVERSITY so the final seed set spans different sub-directions of the topic
# instead of collapsing onto several near-duplicate papers.
# --------------------------------------------------------------------------- #
MERGE_RERANK_RELEVANCE_DIVERSE = """\
Evaluate how substantively relevant each candidate paper is to the query.

Scoring target:
- Judge each paper on its relevance to the query.
- Focus on overlap in the core research problem, evaluation target, inputs/outputs, and methodological purpose.
- A paper can score high even if the method differs, as long as it directly addresses the same problem.
- Penalize papers that only share broad themes such as LLMs, scientific discovery, benchmarks, or evaluation.
- Ignore citation counts, venue prestige, popularity, and writing polish.
- If the candidate is a survey, review, overview, literature review, systematic review, meta-analysis, tutorial, or primer paper, give it a score of 0.

IMPORTANT — emphasize diversity:
- These papers will become the seeds that anchor the whole pipeline, so the final set must be DIVERSE and cover different sub-directions, methods, and perspectives of the query rather than several near-duplicates of the same approach.
- When two candidates are strongly relevant but nearly redundant (same method/finding/sub-topic), keep the strongest one scored high and score the redundant one(s) LOWER so the high-scoring set stays diverse.
- Reward a paper that is relevant AND brings a distinct angle the other high-scoring candidates do not already cover.

Rubric:
- 9-10: Extremely close neighbor; essentially the same problem or an immediately adjacent paper you would definitely cite, AND it adds a distinct direction not already covered.
- 7-8: Directly relevant; strong task or evaluation overlap.
- 5-6: Meaningful partial overlap, or relevant but largely redundant with a higher-scored paper.
- 3-4: Weak topical relation only.
- 0-2: Effectively irrelevant, or a survey/review/overview/tutorial paper.

Return ONLY JSON in this format:
{{"papers":[{{"paper_index":1,"score":8.7,"reason":"short reason"}}]}}

Requirements for output:
- Include every paper exactly once.
- score must be a number in [0, 10].
- reason must be concise and mention the main match or mismatch (or that it is a survey / redundant with another candidate).

Query type: {query_type}
Query paper title: {query_title}
Query text:
{query_text}

Candidate papers:
{candidate_blocks}
"""

# --------------------------------------------------------------------------- #
# Step 2 — predecessor selection for a graph node
# --------------------------------------------------------------------------- #
PREDECESSOR_SELECTION = """\
You are selecting predecessor papers for a node in a citation research graph.

Overall research topic (the graph is being grown to generate ideas for THIS topic — keep it centered here):
{research_topic}

Node paper:
- Title: {node_title}
- Abstract: {node_abstract}

Candidate predecessors (already reranked for relevance to the node; each has a relevance score in [0,10] and a short reason):
{candidate_list}

Your task:
1. Choose exactly up to {num_candidate} candidates to keep as predecessors of the node paper.
2. Prefer candidates relevant to BOTH the node paper AND the overall research topic. A candidate that informed the node but drifts away from the research topic should be deprioritized in favor of one that keeps the graph centered on the topic. Do not let the graph wander into an adjacent application area just because a node happened to cite it.
3. Weigh BOTH the relevance score AND the reason. A high score with a weak/irrelevant reason should be deprioritized.
4. Strictly exclude survey / review / overview / tutorial / primer papers (they should already be scored 0).
5. Prefer papers that genuinely informed or motivated the node paper's problem or method.

Requirements:
1. Output strict JSON only with this schema and nothing else:
{{
  "selected_paper_ids": ["paper_id_1", "paper_id_2"],
  "rationale": "brief explanation"
}}
2. Every selected paper id must come from the candidate list.
3. Do not include more than {num_candidate} selected ids.
"""

PHASE_ASSIGNMENT = """\
You are organizing a local research graph into a small number of research phases.

Seed Paper:
- Title: {seed_paper_title}
- Abstract: {seed_paper_abstract}

Candidate Papers:
{paper_list}

Known Structural Edges:
{edge_list}

Your task:
1. Group the candidate papers into at least 3-6 coherent phases.
2. Each phase should represent a meaningful stage such as foundations, early problem framing, method development, scaling, or recent extensions.
3. Use the paper metadata and structural edges as hints, but do not mechanically follow topology. Use your own judgment to form sensible phases.
4. Every paper id must appear exactly once across all phases.
5. Respect temporal and citation precedence: if paper A is a predecessor of paper B, then A must appear in an earlier phase or the same phase, never a later phase.

Requirements:
1. Output strict JSON only:
{{
  "phases": [
    {{
      "name": "brief phase name",
      "paper_ids": ["paper_id_1", "paper_id_2"]
    }}
  ]
}}
2. Do not include the seed paper id in the phases.
3. Order phases from earlier / foundational work to later / more specialized work.
4. Do not include any text outside the JSON.
"""

IDEA_FIELDS_EXTRACTION = """\
You are a scientific paper idea extractor.

Paper:
- Title: {paper_title}
- Abstract: {paper_abstract}

Extract the core idea of this paper in exactly five fields. Output strict JSON only.

{{
  "Problem": "What problem does this paper address? (1-2 sentences)",
  "Existing Methods": "What are the main existing approaches and their limitations? (1-2 sentences)",
  "Motivation": "Why is a new approach needed? What gap does this paper fill? (1-2 sentences)",
  "Proposed Method": "What method does this paper propose? (1-2 sentences)",
  "Experiment Plan": "How does the paper validate its approach? (1-2 sentences)"
}}
"""

# --------------------------------------------------------------------------- #
# Step 3 — GoR-style serialization and trend summary
# --------------------------------------------------------------------------- #
TREND_SUMMARY = """\
You are a scientific research expert tasked with summarizing the historical progression of research related to our current topic, based on the structured citation evolution graph we have constructed.

Research Topic: {research_topic}

Citation Evolution Graph:
{serialized_graph}

Your objective is to analyze the research evolution using the structural signals in the graph. Pay special attention to:

1. Explicit predecessor chains: Papers connected by `explicit_pred` edges form the main research lineage. Trace how methods evolve along these chains.
2. Parallel work patterns: Papers connected by `parallel_pred` edges represent competing approaches in the same period. Identify their differences and which direction prevailed.
3. Foundational papers: Papers with high `cited_by_subgraph` values are shared foundations. Explain why they matter.
4. Structural gaps: Look for *scientific* open problems the field itself has not solved — capabilities or mechanism combinations that no paper in the graph achieves, recurring limitations no method fully removes, or tensions left unresolved across the lineage. These are gaps in the SCIENCE, not gaps in this graph's bookkeeping. The graph is only an imperfect, incomplete sample of the literature: if well-known papers appear absent, mis-linked, or attached to the wrong predecessor, that is a limitation of THIS artifact, not a research gap — never report "paper X is missing", "the lineage is incomplete", "the canonical milestone is absent", or "predecessor edges are mis-typed" as a gap. A reader who already possessed a perfect citation graph must still find every gap you list worth solving.

Requirements:
1. Analysis of Published Viewpoints: Examine the progression of ideas across the identified papers. Detail how each paper transitions to the next. Focus on understanding how later papers build upon earlier concepts, what concrete advances they add, and why those advances matter.
2. Identify key turning points where the research direction shifted.
3. Identify recurring limitations that multiple papers acknowledge but fail to fully resolve.

Please present your findings in the following format:

Trends:
Paper 0 to Paper 1: ...
Paper 1 to Paper 2: ...

Key Turning Points: ...

Recurring Limitations: ...

Structural Gaps (open SCIENTIFIC problems in the field that remain unsolved — NOT missing papers, absent milestones, or graph-wiring issues): ...
"""

# --------------------------------------------------------------------------- #
# Step 4 — Research Structure Signature (RSS) extraction
# --------------------------------------------------------------------------- #
RSS_EXTRACTION = """\
You are an expert research-structure analyst.

Your task is to convert a GoR-style Research Graph into a Research Structure Signature (RSS).

The RSS is NOT a research idea.
The RSS is an abstract structural representation of the target research area.
It should summarize the research evolution, identify unresolved gaps, extract important objects and relations, and describe what kind of solution mechanism may be needed.

You must use only the provided GoR graph information.
Do not invent papers, methods, datasets, claims, or experimental results that are not present in the input.
Do not propose final research ideas.
Do not search externally.
Return strict JSON only.

Relevance constraint (MUST):
Stay as on-topic as possible. The RSS must remain tightly relevant to the original research topic provided below and must not drift into tangential, adjacent, or unrelated research areas. The target_domain, core_gap, and all extracted structure must clearly serve the original research topic.

Scientific-gap constraint (MUST):
The RSS describes the SCIENCE of the target domain, not this knowledge graph. The `core_gap` MUST be an unsolved scientific or methodological problem in the field itself — a capability no existing method delivers, a mechanism that is missing, a trade-off no approach resolves. It MUST NOT be a deficiency of the provided graph (e.g. "canonical papers are missing", "the lineage is incomplete", "predecessor edges are mis-typed", "coverage is sparse", "the main trunk is discontinuous"). The graph is only an imperfect sample of the literature; treat any missing or mislabeled node as an artifact limitation to silently work around, never as the gap. Likewise, `possible_mechanism_types` and `inspiration_search_hints` MUST aim at SOLVING the scientific problem in the domain, not at reconstructing, completing, curating, typing, scoring, or analyzing the citation/lineage graph itself. If you catch yourself proposing to "reconstruct the lineage", "complete the graph", "type the edges", "separate inheritance from benchmark/citation influence", or "detect missing milestones", you have mistaken the artifact for the science — discard it and state the real domain gap instead.

Input:

Original research topic (the user's query — anchor everything to this):
{research_topic}

Seed paper or target direction:
{seed_paper_or_target_direction}

GoR-style Research Graph (serialized with [EDGE], [PREDECESSORS], [IDEA], [ABSTRACT] blocks):
{gor_graph}

Current Trend:
{trend}

Your task:

1. Understand the research evolution from the GoR graph. Identify how the field develops over time. Pay attention to predecessor chains, parallel branches, method evolution, and repeated limitations.
2. Identify the main research gap. The gap should be grounded in the GoR graph. It should explain what existing methods fail to solve, why this failure matters, and which part of the graph supports this gap.
3. Extract the Research Structure Signature, describing the target problem in an abstract relational form.
4. Keep the RSS general enough for analogy or mechanism transfer, but do not over-generalize: preserve the concrete research context.
5. Distinguish between evidence (directly from the GoR graph) and inference (patterns supported by graph evidence).

Output strict JSON with the following schema:
{{
  "target_domain": "Research field and target domain of this research work",
  "core_gap": "Core research deficiencies of current mainstream technologies in this domain",
  "objects": ["core research objects and key entities"],
  "relations": ["logical associations between the core objects"],
  "bottlenecks": ["key technical bottlenecks and limitations"],
  "unresolved_tensions": ["important trade-offs or unresolved tensions in the field"],
  "desired_solution_properties": ["expected attributes the final solution must satisfy"],
  "possible_mechanism_types": ["abstract mechanism families that may help address the gap"],
  "inspiration_search_hints": ["hints for what kinds of external mechanisms or domains to search next"]
}}
"""

# --------------------------------------------------------------------------- #
# Step 5 — Gate LLM decides inspiration radius
# --------------------------------------------------------------------------- #
INSPIRATION_RADIUS_GATE = """\
You are a research inspiration planning agent.

Inputs:
- Research Structure Signature (RSS): {rss_json}
- Trend Analysis: {trend}
- Default same-field top-k: {same_field_top_k}
- Default number of cross domains: {num_cross_domains}
- Default cross-domain top-k per domain: {cross_domain_top_k_per_domain}

Your task:
Decide how broad the inspiration search should be for the current problem. Use the RSS bottlenecks, unresolved tensions, desired solution properties, and trend analysis to decide whether to search same-field inspirations, cross-domain inspirations, or both.

Requirements:
1. Be conservative and grounded in the provided RSS and trend.
2. Output strict JSON only using this schema:
{{
  "use_same_field": true,
  "use_cross_domain": true,
  "same_field_top_k": {same_field_top_k},
  "num_cross_domains": {num_cross_domains},
  "cross_domain_top_k_per_domain": {cross_domain_top_k_per_domain},
  "rationale": "brief explanation"
}}
3. If you disable a radius, set its corresponding count to 0.
4. Prefer same-field retrieval when feasibility is the main concern, and add cross-domain retrieval when structural bottlenecks suggest mechanism transfer is needed.
"""

# --------------------------------------------------------------------------- #
# Step 6 — multi-radius inspiration probing
# --------------------------------------------------------------------------- #
DOMAIN_QUERY_GEN = """\
You are analyzing a research problem to identify relevant domains for cross-domain transfer.

Problem:
{problem_text}

Your task:
1. Identify {num_domains} other domains (e.g., "computer_science", "logistics") that might have relevant methods for addressing this problem.
2. Generate a SciAtlas Query for each, containing query text and a keyword list.

Requirements:
1. Domain selection: choose {num_domains} broad scientific or engineering domains plausibly relevant to the problem. Use high-level disciplines, not niche topics.
2. One query per domain. Each query is a JSON object with `query_text` and `keywords`.
3. `query_text` should capture a transferable method, principle, or perspective from the target domain that could help address the original problem. It must not merely repeat the original problem description.
4. `keywords` is a list of 3-5 dictionaries, each with a `text` field and an integer `score` field (1-10).
5. Output structure: return a single JSON object with a top-level field `domain_queries` (array). Each object includes `domain`, `query_text`, and `keywords`. No prose outside the JSON.
6. Ground suggestions in the provided problem and general domain knowledge. Do not invent facts.
"""

# --------------------------------------------------------------------------- #
# Step 7 — analogy / mechanism extraction
# --------------------------------------------------------------------------- #
COMBINATION_CHECK = """\
You are a cross-domain synergy analyst. You need to identify if a candidate paper can inspire the current problem.

Original Problem:
{problem_text}

Candidate Paper:
- Title: {title}
- Abstract: {abstract}

Your task:
1. This candidate was retrieved BY DESIGN from a different field, to serve as a cross-domain ANALOGY. Judge it as an analogy, not as a drop-in method. Its surface topic, data, and terminology WILL differ from the problem — that is expected and is NOT, by itself, a reason to reject.
2. Look for a transferable ABSTRACTION: a mechanism, principle, mathematical structure, dynamical behaviour, control/stability strategy, organizing scheme, conservation law, or trade-off the candidate handles that maps onto the problem's bottlenecks. A mechanistic / structural / mathematical analogy COUNTS as plausible even when the application domain is unrelated (e.g. a control-theory stability result, a statistical-physics phase transition, or a neuroscience routing scheme informing a deep-learning architecture).
3. If such an abstraction exists, set is_combination_plausible=true, name the transferable concept(s) in combination_points, and give a brief combination_plan that adapts the mechanism to the original problem.
4. Only set is_combination_plausible=false when there is genuinely NO transferable principle, mechanism, or structure — i.e. any analogy would be purely superficial or forced. Do NOT reject merely because the candidate is "from a different field", "biological/physical, not deep learning", "a survey/review", or "not directly applicable" — direct applicability is not required for an analogy.

Requirements:
1. Base your judgment solely on the provided problem and candidate paper information.
2. Use clear reasoning and avoid vague statements. Bias toward finding a usable analogy when a real structural/mechanistic mapping exists.
3. Output strict JSON only with this schema:
{{
  "is_combination_plausible": true,
  "combination_points": ["concepts or methods from the candidate paper that could be combined"],
  "combination_plan": "one or two sentences describing how to integrate these concepts into the original problem solution",
  "justification": "brief explanation of your judgment"
}}
4. If no combination is plausible, set `combination_points` to an empty list and `combination_plan` to an empty string.
"""

# --------------------------------------------------------------------------- #
# Step 8 — Gate LLM filters and ranks inspirations
# --------------------------------------------------------------------------- #
BEST_INSPIRATION_SELECTION = """\
You are a scientific inspiration evaluator.

Inputs:
- Research Structure Signature (RSS): {rss_json}
- Trend Analysis: {trend}
- Candidate inspirations: {inspiration_list}

Your task:
Select the best 1-3 inspirations (ranked) for final idea generation. Favor inspirations that enable a genuine STRUCTURAL or MECHANISTIC transfer addressing the RSS bottlenecks and that would yield a non-trivial but feasible idea.

Requirements:
1. Compare only the provided inspirations.
2. Select between 1 and 3 candidates (use fewer only if few are plausible).
3. Strongly PREFER at least one CROSS-DOMAIN inspiration (`domain` != "same_field") whenever any plausible cross-domain candidate exists: a cross-domain mechanism transfer is more valuable here than a same-field inspiration that would merely stack, route, or linearly combine primitives already standard in the field. Avoid a selection whose only outcome would be trivially combining existing same-field methods.
4. Prioritize structural / mechanism transfer over superficial topic similarity.
5. Output strict JSON only using this schema:
{{
  "selected_candidate_ids": [1],
  "justification": "brief explanation"
}}
6. Each id in `selected_candidate_ids` must be one of the candidate ids provided in `Candidate inspirations`, ordered best first.
"""

# --------------------------------------------------------------------------- #
# Step 9 — unified idea generation
# --------------------------------------------------------------------------- #
IDEA_GENERATION = """\
You are a scientific expert and research idea generator.

Inputs:
- Original research query (the user's query — anchor everything to this): {original_query}
- Serialized Citation Evolution Graph: {serialized_graph}
- Research Structure Signature (RSS): {rss_json}
- Trend Analysis: {trend}
- Cross-domain inspirations: {inspiration_list}
- Novelty feedback from the previous round: {novelty_feedback}
- Reference catalog (the ONLY works you may cite): {reference_catalog}

Relevance constraint (MUST):
Keep the PROBLEM on-topic: the idea must directly address the original research query above and resolve a real bottleneck in that domain, and the resulting artifact must live in and change the target domain. But "on-topic" refers to the PROBLEM, not the toolbox — you are expected to solve it by importing a concrete MECHANISM, principle, mathematical structure, or organizing scheme from a different field (the cross-domain inspirations above) and adapting it. A same-field idea that merely stacks, gates, routes, schedules, or linearly combines primitives already standard in the field (e.g. "dynamically route/select among convolution, attention, and state-space mixing", "a hybrid that schedules existing sub-quadratic primitives") is NOT acceptable — that is the trivial A+B outcome to avoid. The core novelty MUST be a non-obvious cross-domain mechanism transfer that resolves a specific RSS bottleneck; name the borrowed mechanism and its source field explicitly.

Substance constraint (MUST):
Produce a concrete scientific or technical advance in the target domain — a new method, mechanism, model/architecture, algorithm, or testable theory that someone could actually build and empirically evaluate, and that would change what practitioners in the field do. Do NOT produce a meta-level research tool: an idea whose deliverable is a way to map, reconstruct, complete, curate, type, score, benchmark, or analyze the literature / citation graph / research lineage itself is OFF-LIMITS unless the original query explicitly asks for such a tool. Outputs like "build a typed lineage/inheritance graph", "a framework to classify how field X evolved", "a controlled grammar for testing the lineage", or "a metric that separates architectural inheritance from citation/benchmark influence" are exactly the hollow, off-target results to avoid. When the query is about a scientific domain (e.g. how a family of methods works or what comes next), answer with a new artifact IN that domain — not with an instrument for studying its history. Prefer the most mechanistically specific idea the evidence supports over a broad descriptive framework.

Design-rigor constraint (MUST):
The idea must be a *controlled, falsifiable* research program, not a maximally ambitious combination of trendy concepts. Enforce all of the following:
1. State ONE primary, testable hypothesis that the idea is built to confirm or refute (e.g. "spatial proximity of A and B causally improves X"). Everything else is secondary.
2. Isolate that hypothesis. In the first phase, hold all other factors FIXED and vary only the single factor that tests the primary hypothesis (one factor at a time). Do NOT simultaneously vary many design parameters — if several parameters all change at once, an observed improvement cannot be attributed to the claimed mechanism.
3. Include strict controls that directly attribute any effect to the claimed mechanism: each active component alone, a physical/unstructured mixture of components, and a randomized vs. structured/organized arrangement, as applicable to the idea.
4. Explicitly identify and rule out trivial confounders that could explain the result without the claimed mechanism (e.g. surface area, adsorption, mass transport, support/matrix effects, component leaching, loss of the claimed structure under operating conditions). Build the controls and characterization needed to exclude each.
5. Phase the work: confirm the core mechanistic hypothesis under tightly controlled conditions BEFORE expanding to secondary variables, broader scope, or performance optimization. Secondary studies (e.g. electronic effects, substrate/condition scope) come only after the core effect is established.
6. Novelty must rest on a specific mechanistic or conceptual advance, NOT merely on combining popular materials/methods/concepts. If close prior work plausibly already combines the same ingredients, the idea must articulate what distinct, testable claim sets it apart.

Your task:
Generate exactly one novel, feasible, and well-grounded research idea by consolidating the RSS, the trend analysis, and the most useful mechanisms from the cross-domain inspirations. The idea must be motivated by a concrete research gap and must explain why the proposed mechanism can overcome specific bottlenecks in the current research context.

Generation procedure:
1. Understand the research context from the RSS.
2. Analyze the research trend and identify what remains unresolved.
3. Use reflection, analogy, and deep dive to consolidate one idea.
4. Select only inspiration mechanisms with a clear functional role.
5. Consolidate the final idea with motivation, methods, adaptation logic, and feasibility.
6. If novelty feedback is provided, directly address the cited overlap risks and improvement suggestions.
7. Apply the Design-rigor constraint: pick the single primary hypothesis, design the one-factor-at-a-time first phase, and enumerate the controls and confounders to exclude.

Citation requirements (MUST):
1. Ground the `motivation` and every method-component `description` in the literature by citing works from the Reference catalog above.
2. Cite ONLY works that appear in the Reference catalog. Never invent a citation, an author, a year, or a URL. If the catalog does not support a claim, state it without a citation.
3. Each inline citation MUST be a Markdown link in exactly this format: `[Author et al., Year](URL)`, copying the `citation` and `url` fields of a catalog entry verbatim (e.g. `[Karpov et al., 2019](https://doi.org/10.1007/978-3-030-30493-5_78)`).
4. Place citations inline, immediately after the specific claim, method, or prior result they support. Cite several distinct works across the motivation and the method components — do not lump them all at the end.
5. List every work you cited (and only those) in the `references` array.

Analogy / inspiration citation (MUST):
1. Catalog entries tagged `type: inspiration` are the source papers of the cross-domain inspirations above. Whenever the idea actually borrows, adapts, or builds on an inspiration's concept/mechanism (an analogy), you MUST cite that inspiration's source paper inline with its `[Author et al., Year](URL)` link at the point where the analogy is used.
2. For every analogy you use, add one entry to `analogy_mapping` recording: the inspiration's source paper title, the borrowed `concept`, how you `adaptation` it to the original query, and the inline `citation` you used. Do NOT list an inspiration in `analogy_mapping` unless its concept genuinely shapes the idea.
3. If a used inspiration's source paper is NOT in the Reference catalog, still record it in `analogy_mapping` (cite it by its title in the `citation` field), so the analogy is always traceable in the final idea.

Output requirements:
1. Output must be valid JSON with no commentary outside the JSON.
2. The `ideas` array must contain exactly one object.
3. Make the idea structured rather than a single prose blob: break the method into named components and give a concrete, ordered experiment plan. The `experiment_plan` MUST realize the Design-rigor constraint: an early phase that varies only the primary factor with confounders held fixed, the explicit control conditions (each component alone, physical mixture, randomized vs. structured), the characterization that rules out the named trivial confounders, and only then later phases for secondary variables and scope.
4. Before the components, write a `method_overview`: one short paragraph (2-4 sentences) that states what problem the method solves and how it is decomposed into its N parts. Begin it like "This method is intended to solve ...; it can be divided into <N> parts: ..." so the reader gets a top-down summary before the numbered components. Do NOT start the method straight from point 1.
5. Use this schema:
{{
  "ideas": [
    {{
      "title": "...",
      "motivation": "... grounded prose with inline [Author et al., Year](URL) citations ...",
      "method_overview": "This method is intended to solve ...; it can be divided into N parts: ...",
      "proposed_method": [
        {{"name": "Component / step name", "description": "... grounded prose with inline [Author et al., Year](URL) citations ..."}}
      ],
      "analogy_mapping": [
        {{"inspiration": "source paper title of the inspiration", "concept": "the borrowed concept/mechanism", "adaptation": "how it is adapted to the original query", "citation": "[Author et al., Year](URL)"}}
      ],
      "experiment_plan": ["Ordered step 1 ...", "Ordered step 2 ..."],
      "references": [
        {{"citation": "Author et al., Year", "url": "https://..."}}
      ]
    }}
  ]
}}
"""

# --------------------------------------------------------------------------- #
# Post-Step-9 — novelty check
# --------------------------------------------------------------------------- #
NOVELTY_CHECK = """\
You are a novelty analysis agent for scientific ideas.

Inputs:
- Proposed research idea: {idea_json}
- Key references: {reference_list}

Your task:
1. Assess how novel the proposed idea is relative to the provided key references.
2. Identify the CLOSEST existing combination: the reference (or small set of references) whose problem + ingredients + mechanism most overlap the idea, and state precisely what the idea adds beyond it. If the idea's core is plausibly already realized by such work, the novelty is at most `low`.
3. Flag "combination-only" novelty: if the idea's claim to novelty rests mainly on assembling already-popular materials/methods/concepts without a distinct, testable mechanistic or conceptual advance, say so explicitly and cap the level accordingly.
4. Categorize the novelty as one of: high, medium, low, none.
5. Provide a justification grounded in the comparison between the idea and the closest references.
6. If the level is medium or low, suggest at least one modification that raises *mechanistic or conceptual* novelty (a sharper distinct claim or mechanism), not merely adding more components or scope.

Requirements:
1. Base your assessment solely on the provided idea and references.
2. Output strict JSON with the following schema:
{{
  "novelty_level": "high | medium | low | none",
  "justification": "...",
  "improvement_suggestions": ["..."]
}}
3. If `novelty_level` is `high`, you may leave `improvement_suggestions` as an empty list.
4. Do not include any extraneous text outside of the JSON.
"""
