from __future__ import annotations

import unittest

from review import report


class ReportFormattingTests(unittest.TestCase):
    def test_meta_inline_star_lists_are_normalized(self) -> None:
        text = (
            "Reviewers converge on issues: * **Motivation & Novelty:** The motivation is clear. "
            "* **Methodological Clarity:** The method needs formalization. "
            "These points of consensus are central."
        )

        rendered = report._clean_report_markdown_block(text)

        self.assertIn("Reviewers converge on issues:\n- **Motivation & Novelty:**", rendered)
        self.assertIn("\n- **Methodological Clarity:**", rendered)
        self.assertIn("\n\nThese points of consensus are central.", rendered)
        self.assertNotIn(": * **", rendered)

    def test_reviewer_report_includes_section_assessment_before_strengths(self) -> None:
        rendered = report._format_reviewer_reports(
            [
                {
                    "overall": {"confidence": "high", "summary": "Reviewed 1 section."},
                    "section_reviews": [
                        {
                            "section": "Motivation",
                            "assessment": "The motivation is strong but needs clearer novelty positioning.",
                            "strengths": [{"point": "Identifies a specific gap.", "citations": []}],
                            "weaknesses": [{"point": "Does not compare enough prior work.", "citations": []}],
                        }
                    ],
                }
            ]
        )

        section_start = rendered.index("#### Motivation")
        assessment_index = rendered.index("The motivation is strong", section_start)
        strengths_index = rendered.index("**Strengths**", section_start)
        self.assertLess(assessment_index, strengths_index)

    def test_rubric_required_evidence_stays_nested_under_dimension(self) -> None:
        rendered = report._format_rubric(
            {
                "sections": [
                    {
                        "section": "Motivation",
                        "standards": [
                            {
                                "dimension_name": "Gap in Multimodal SSL",
                                "core_philosophy": "Identify the aligned-data gap.",
                                "required_evidence": "Show why existing methods are suboptimal.",
                            }
                        ],
                    }
                ]
            }
        )

        self.assertIn("- **Gap in Multimodal SSL**\n", rendered)
        self.assertIn("  - **Evaluation focus**: Identify the aligned-data gap.", rendered)
        self.assertIn("  - **Required evidence**: Show why existing methods are suboptimal.", rendered)
        self.assertNotIn("\n\n  **Required evidence**", rendered)

    def test_bulleted_notes_strip_existing_list_markers(self) -> None:
        rendered = report._format_bulleted_notes(
            [
                "- **Existing bullet:** Keep the label.",
                "* Star marker should be normalized.",
                "1. Number marker should be normalized.",
            ]
        )

        self.assertIn("- **Existing bullet:** Keep the label.", rendered)
        self.assertIn("- Star marker should be normalized.", rendered)
        self.assertIn("- Number marker should be normalized.", rendered)
        self.assertNotIn("- - **Existing bullet", rendered)
        self.assertNotIn("- * Star marker", rendered)
        self.assertNotIn("- 1. Number marker", rendered)

    def test_short_report_includes_idea_overview_and_numbered_rubric_headings(self) -> None:
        short_meta_review = {
            "overall_assessment": "Overall assessment.",
            "reviewer_consensus": "Consensus.",
            "reviewer_disagreements": "No disagreement.",
            "shared_strengths": ["Strength."],
            "shared_weaknesses": ["Weakness."],
            "revision_advice": ["Revise."],
            "rubric_summaries": [
                {
                    "section": "Motivation",
                    "dimensions": [
                        {
                            "standard_id": "motivation_01",
                            "dimension_name": "Gap in Multimodal SSL",
                            "review_text": "This rubric is largely convincing, although reviewers note that the paper should position the gap more concretely.",
                        },
                        {
                            "standard_id": "motivation_02",
                            "dimension_name": "Mechanistic Hypothesis",
                            "review_text": "The mechanistic story is plausible, but the current presentation remains too implicit about why the joint objective should work.",
                        },
                    ],
                }
            ],
        }

        rendered = report.build_short_markdown_report(
            short_idea_overview=[
                {"label": "Basic Idea", "text": "Core idea."},
                {"label": "Motivation", "text": "Motivation point."},
            ],
            normalized_reviewers={"reviewers": []},
            evidence_bank={},
            reviewer_reviews=[],
            short_meta_review=short_meta_review,
        )

        self.assertIn("## Idea Overview", rendered)
        self.assertIn("**Basic Idea**", rendered)
        self.assertIn("## Section Review", rendered)
        self.assertIn("- **Gap in Multimodal SSL:**", rendered)
        self.assertIn("- **Mechanistic Hypothesis:**", rendered)
        self.assertNotIn("#### 1. Gap in Multimodal SSL", rendered)
        self.assertNotIn("**Reviewer Positions**", rendered)


if __name__ == "__main__":
    unittest.main()
