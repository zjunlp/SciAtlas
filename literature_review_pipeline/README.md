# Literature Review Pipeline

This folder contains the survey / literature-review generation pipeline centered on `downstream/lr_search/`.

Main entry points:
- `downstream/lr_search/literature_review_search.py`
- `downstream/lr_search/generate_literature_review_report.py`

Recommended repository entry point:

```bash
python run_sciatlas.py literature-review --query "retrieval augmented generation" --workflow flash
python run_sciatlas.py literature-review --query "retrieval augmented generation" --workflow full
```

`flash` is the default interactive path: it uses smaller KG/S2 budgets, fewer planning actions, and may stop after bibliography, outline, or evidence packs. `full` keeps broader multi-round retrieval and continues into formal review drafting/integration.

Main local modules:
- `downstream/lr_search/`

Shared retrieval dependencies:
- `kg_search/`
- `search/merge_search.py`
- `search/pdf_xml_pipeline.py`
- `s2api/`
- `downstream/repos/academic-research-skills/scripts/`
