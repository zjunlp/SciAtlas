from __future__ import annotations

import json
import re

from .config import SearchConfig
from .llm_client import OpenAICompatibleClient


REFERENCE_SELECTION_SYSTEM_PROMPT = """You are an expert research assistant for academic paper analysis.

You will receive:
1. The paper title.
2. The paper abstract.
3. The paper body as JSON. Each item is a section with:
- "heading": section title
- "paragraphs": paragraph strings
- "subsections": nested sections
4. A references list where each line has the format:
ref_id="b..." title="..."
5. A hard cap named "max_total_ref_ids". The total number of unique reference ids across your full output must not exceed this cap.

The paragraph text may contain inline bibliography references such as:
<ref type="bibr" target="#b31">Reference Title</ref>

Task:
1. Read the paper body and identify the key context sentences or paragraphs that discuss related works most similar to the paper's main work.
2. The goal is not to keep every important citation. The goal is to recover citations that should behave like "related works" for this paper: papers that are directly related and similar to the paper's main research work.
3. Treat a citation as a strong candidate if it has one or more of these properties:
   - it addresses the same or a very similar core problem
   - it uses a similar methodology, modeling idea, framework, or technical approach
   - it is presented as a baseline, direct comparison target, or closely competing method
   - it belongs to the same research area, task, or problem setting as the paper
4. Do not keep citations that are only loosely connected or merely supportive, such as:
   - references from unrelated fields or side topics
   - citations that do not focus on the same central problem
   - example systems, datasets, benchmarks, tools, software libraries, APIs, platforms, or generic resources
   - backbone model papers, implementation dependencies, technical manuals, or model documentation unless the paper is actually about that same model family/problem
   - incidental references used only to justify a component, preprocessing step, or borrowed utility
5. Use both:
   - the paper title and abstract, to infer the paper's main work
   - the surrounding body context, to judge whether a citation is being discussed as a genuinely similar prior work or baseline
   - the referenced paper titles from the <ref ...> tags and the references list
   to decide which citations are true related works.
6. For each selected key context sentence or paragraph, extract only the most relevant bibliography reference targets mentioned in that context.
7. For each selected context, write a concise "reason" that explains:
   - why this context is important for identifying similar related works
   - why the chosen references are directly related or similar to the paper's main work
   - if the context cites more references but you keep only part of them, explain why the omitted references are less central as related works
   - keep each reason short: at most 2 sentences, plain text only
8. Return grouped results. Each group corresponds to one key context and contains:
   - "ref_ids": the selected bibliography ids from that context
   - "reason": the explanation for that context and those references
9. Order the output groups from highest importance to lowest importance. The first group should be the most central related-work context for understanding papers most similar to this paper, then the next most important group, and so on.

Rules:
- Focus on references that would be good related-work candidates for this paper.
- Prefer papers that solve similar problems, use similar methods, operate in the same area, or serve as meaningful baselines.
- Ignore citations that are only background resources, tooling, data sources, model cards, general infrastructure, or borrowed components rather than similar research works.
- Ignore references that appear only in peripheral contexts such as broad background, generic surveys, unrelated future work, acknowledgements, or weakly related side remarks.
- If multiple <ref> tags appear together in one citation span or the same parenthetical citation group, you may keep only the core subset, but then the reason must explain why some ids were omitted.
- Preserve only ids that correspond to bibliography references, such as b31.
- Do not invent ids.
- Do not exceed max_total_ref_ids unique ids across the entire response.
- If no good references qualify, return an empty selections list.
- Output valid JSON only.
- Do not wrap the JSON in markdown fences.
- Do not include comments, trailing commas, or any prose before or after the JSON.
- Every string must be properly closed and escaped for JSON.
- The order of selections matters and must reflect descending importance.

Follow this exact style.

Example input sketch:
- title: "Graph Prompting for Retrieval-Augmented Generation"
- abstract: "We propose a graph-guided prompting method for retrieval-augmented generation."
- body snippet:
  "Our method builds on graph-aware retrieval proposed by <ref type="bibr" target="#b3">GraphRetriever</ref> and prompt construction strategies from <ref type="bibr" target="#b7">PromptRAG</ref>."
  "We compare against generic dense retrieval baselines <ref type="bibr" target="#b2">DPR</ref> and <ref type="bibr" target="#b9">ANCE</ref>."
- references:
  ref_id="b2" title="Dense Passage Retrieval"
  ref_id="b3" title="GraphRetriever: Retrieval over Structured Evidence"
  ref_id="b7" title="PromptRAG: Prompt Construction for Retrieval-Augmented Generation"
  ref_id="b9" title="Approximate Nearest Neighbor Negative Contrastive Learning"
- max_total_ref_ids: 3

Example output:
{"selections":[{"ref_ids":["b3","b7"],"reason":"This context is a core related-work context because it describes prior methods most similar to the paper's own approach. b3 and b7 are kept because they directly match the paper's retrieval and prompting methodology, while less similar citations are omitted."},{"ref_ids":["b2"],"reason":"This context is important because it names a main baseline from the same task setting. b2 is kept as the clearest comparison target, while b9 is omitted because it is less central as a related work."}]}

Return ONLY a JSON object in this format:
{"selections": [{"ref_ids": ["b31", "b52"], "reason": "..."}, {"ref_ids": ["b11"], "reason": "..."}]}"""


class PdfReferenceSelector:
    _REF_TAG_PATTERN = re.compile(r"<ref\b([^>]*)>(.*?)</ref>", re.IGNORECASE | re.DOTALL)
    _TARGET_PATTERN = re.compile(r'target\s*=\s*["\']#(b\d+)["\']', re.IGNORECASE)

    def __init__(self, config: SearchConfig) -> None:
        self.llm_client = OpenAICompatibleClient(
            api_url=config.llm_api_url,
            model=config.llm_model,
            api_key=config.llm_api_key or "",
            timeout_s=config.llm_timeout_s,
        )
        self.max_total_ref_ids = max(1, int(config.max_titles_from_pdf_references))

    def select_references(
        self,
        pdf_title: str,
        abstract: str,
        body_sections: list[dict],
        references: list[dict],
    ) -> list[dict]:
        normalized_references = self._build_reference_title_map(references)
        processed_body_sections = self._preprocess_body_sections(body_sections, normalized_references)
        parsed = self.llm_client.chat_json_messages(
            [
                {"role": "system", "content": REFERENCE_SELECTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": self._build_user_payload(
                        pdf_title,
                        abstract,
                        processed_body_sections,
                        normalized_references,
                    ),
                },
            ],
            max_tokens=1200,
            temperature=0.0,
        )
        return self._parse_selection_groups(parsed)

    def filter_references(
        self,
        pdf_title: str,
        abstract: str,
        body_sections: list[dict],
        references: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        selection_groups = self.select_references(pdf_title, abstract, body_sections, references)
        selected_ref_ids = self.flatten_ref_ids(selection_groups)
        if not selected_ref_ids:
            return [], []
        reference_by_ref_id = {
            self._normalize_ref_id(reference.get("ref_id")): reference
            for reference in references
            if isinstance(reference, dict) and self._normalize_ref_id(reference.get("ref_id"))
        }
        filtered = [
            reference_by_ref_id[ref_id]
            for ref_id in selected_ref_ids
            if ref_id in reference_by_ref_id
        ]
        return filtered, selection_groups

    def flatten_ref_ids(self, selection_groups: list[dict]) -> list[str]:
        selected: list[str] = []
        seen: set[str] = set()
        for selection in selection_groups:
            if not isinstance(selection, dict):
                continue
            raw_ref_ids = selection.get("ref_ids")
            if not isinstance(raw_ref_ids, list):
                continue
            for item in raw_ref_ids:
                normalized = self._normalize_ref_id(item)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                selected.append(normalized)
        return selected

    def _build_user_payload(
        self,
        pdf_title: str,
        abstract: str,
        body_sections: list[dict],
        reference_title_map: dict[str, str],
    ) -> str:
        reference_lines = [
            f'ref_id="{ref_id}" title="{title}"'
            for ref_id, title in sorted(reference_title_map.items())
        ]
        return json.dumps(
            {
                "title": self._normalize_text(pdf_title),
                "abstract": self._normalize_text(abstract),
                "body": body_sections,
                "references": "\n".join(reference_lines),
                "max_total_ref_ids": self.max_total_ref_ids,
            },
            ensure_ascii=False,
        )

    def _build_reference_title_map(self, references: list[dict]) -> dict[str, str]:
        reference_title_map: dict[str, str] = {}
        for reference in references:
            if not isinstance(reference, dict):
                continue
            ref_id = self._normalize_ref_id(reference.get("ref_id"))
            title = self._normalize_text(reference.get("title"))
            if ref_id and title:
                reference_title_map[ref_id] = title
        return reference_title_map

    def _preprocess_body_sections(self, body_sections: list[dict], reference_title_map: dict[str, str]) -> list[dict]:
        processed_sections: list[dict] = []
        for section in body_sections:
            if not isinstance(section, dict):
                continue
            paragraphs = section.get("paragraphs")
            subsections = section.get("subsections")
            processed_sections.append(
                {
                    "heading": section.get("heading", ""),
                    "paragraphs": [
                        self._preprocess_paragraph(str(paragraph), reference_title_map)
                        for paragraph in (paragraphs if isinstance(paragraphs, list) else [])
                    ],
                    "subsections": self._preprocess_body_sections(
                        subsections if isinstance(subsections, list) else [],
                        reference_title_map,
                    ),
                }
            )
        return processed_sections

    def _preprocess_paragraph(self, paragraph: str, reference_title_map: dict[str, str]) -> str:
        def replace_ref(match: re.Match[str]) -> str:
            attrs = match.group(1) or ""
            target_match = self._TARGET_PATTERN.search(attrs)
            if target_match is None:
                return ""
            ref_id = self._normalize_ref_id(target_match.group(1))
            if not ref_id:
                return ""
            title = reference_title_map.get(ref_id)
            if not title:
                return ""
            return f'<ref type="bibr" target="#{ref_id}">{title}</ref>'

        return self._REF_TAG_PATTERN.sub(replace_ref, paragraph)

    def _parse_selection_groups(self, parsed: dict) -> list[dict]:
        raw_selections = parsed.get("selections")
        if not isinstance(raw_selections, list):
            raise RuntimeError("Reference selector payload must contain a list field: selections.")

        groups: list[dict] = []
        seen_global: set[str] = set()
        for item in raw_selections:
            if not isinstance(item, dict):
                continue
            raw_ref_ids = item.get("ref_ids")
            if not isinstance(raw_ref_ids, list):
                continue
            normalized_reason = self._normalize_text(item.get("reason"))
            normalized_ref_ids: list[str] = []
            seen_local: set[str] = set()
            for raw_ref_id in raw_ref_ids:
                if len(seen_global) >= self.max_total_ref_ids:
                    break
                normalized_ref_id = self._normalize_ref_id(raw_ref_id)
                if not normalized_ref_id or normalized_ref_id in seen_local or normalized_ref_id in seen_global:
                    continue
                seen_local.add(normalized_ref_id)
                normalized_ref_ids.append(normalized_ref_id)
                seen_global.add(normalized_ref_id)

            if not normalized_ref_ids:
                continue
            if not normalized_reason:
                raise RuntimeError("Each reference selection group must contain a non-empty reason.")
            groups.append(
                {
                    "ref_ids": normalized_ref_ids,
                    "reason": normalized_reason,
                }
            )
            if len(seen_global) >= self.max_total_ref_ids:
                break
        return groups

    def _normalize_ref_id(self, value: object) -> str:
        if not isinstance(value, str):
            return ""
        match = re.search(r"b\d+", value.strip().lower())
        return match.group(0) if match else ""

    def _normalize_text(self, value: object) -> str:
        if not isinstance(value, str):
            return ""
        return re.sub(r"\s+", " ", value).strip()
