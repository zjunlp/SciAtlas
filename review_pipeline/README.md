# Review Pipeline

This folder contains the automated review pipeline built around top-level `pipeline.py`.

Main entry point:
- `pipeline.py`

Recommended repository entry point:

```bash
python run_sciatlas.py idea-evaluate --idea "LLM-based idea evaluation" --workflow flash
python run_sciatlas.py idea-evaluate --idea "LLM-based idea evaluation" --workflow full
```

`flash` is the default interactive path: it uses smaller KG/S2/manifest budgets, fewer reviewer and evidence branches, and compact reporting. `full` keeps the broader reviewer, rubric, grounding, review, and report path.

Main local modules:
- `review/`
- `workers/`
- `reviewer-evaluate/`

Shared retrieval dependencies:
- `kg_search/`
- `search/merge_search.py`
- `search/pdf_xml_pipeline.py`
- `s2api/`
