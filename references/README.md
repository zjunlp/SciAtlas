# References

This directory contains supplemental reference code released together with the open demo.

- `search/` keeps a standalone copy of the search stack, including the KG, S2, merge, and reranking paths.
- It is separate from the runnable `run_sciatlas.py` flow in the demo root.
- It is mainly for reading and demonstrating the search logic. Running the full KG flow still requires Neo4j, the expected graph schema, and related indexes.
