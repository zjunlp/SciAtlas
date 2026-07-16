# Method-Timeline Case Study — Workflow

## What we are building

For a chosen research idea (e.g. "deep neural network architectures",
"reinforcement learning algorithms"), produce a **swimming-lane figure** that
shows the canonical anchor papers of each method family laid out on a time
axis, suitable for a case-study figure in a paper.

Goal of the figure: convey "evolution flow" — early methods cluster on the
left, recent methods on the right; each lane is one method family.

The pipeline output alone is not enough to make a credible figure: retrieval
misses many canonical anchors (e.g. AlexNet / ResNet / BERT / Backprop /
LSTM 1997 were all missing from the DNN run). The figure therefore mixes
three sources, distinguished visually:

- **filled dot ⬤** — the anchor exists in the figure's corpus (either the
  original `lr_search` retrieval, or fetched on-demand from Semantic Scholar
  by title via `migrate_method_spec.py`).
- **dashed hollow dot ◯** — the anchor is a canonical work we want on the
  figure but neither `lr_search` retrieval nor S2 title-match returned it;
  the metadata (label, title, year, first author, citation count, venue)
  comes from the human-written outline / spec yaml.

---

## Two phases

### Phase 1 · Pipeline (automated, ~10 min total)

Goal: from a single enriched idea string, produce a per-method cluster
membership table covering as many retrieved papers as possible.

```
idea string  ─►  literature_review_search.py  ─►  search_result.json
                                                       │
                                                       ▼
                              recluster_all_papers.py  (preview + map-reduce
                                                       across all 400-600 papers,
                                                       not just the top-80 the
                                                       original pipeline kept)
                                                       │
                                                       ▼
                                            clusters_recovered.json
```

Key fact: the **enriched idea string** is the real input. A bare 4-character
topic like "强化学习" is too underspecified — it gets a vague topic_profile
and the LLM cluster step has no structure to lean on. The format we used for
both DNN and RL:

```
Evolution of <topic>. Scope: trace the algorithmic development of <area>
from <classical era> to <modern era>. Method families to cover, in
chronological order:
  (1) <family 1 with key methods>
  (2) <family 2 with key methods>
  ...
Anchor works that must appear: <Author Year>, <Author Year>, ...
Time range: <min_year> to <max_year>.
Exclude: <list of off-narrative sub-domains>
```

Both DNN and RL used this format. Both produced enriched topic_profile.json
with the right method-family list, exclusion rules, etc.

After `recluster_all_papers.py`, every paper in the search is in exactly one
cluster (or in lexical-rescue fallback). The cluster names come from LLM
synthesis of all batches, so they reflect actual paper content, not just the
idea string.

### Phase 2 · Manual curation + visualization

Goal: turn the auto-clustered paper soup into a credible figure with the
canonical anchors of each method family.

```
clusters_recovered.json
       │
       │  (human review:  which clusters are core to the narrative?
       │   which canonical anchors does each cluster need?
       │   which ones did the pipeline retrieve vs. miss?)
       │
       ▼
method_timeline_spec.yaml          ◄── this is the source of truth
       │                                 (one file per case study)
       │
       │  render_method_timeline.py
       ▼
method_timeline.html  +  method_timeline.svg
```

The spec file is the only thing a human edits in Phase 2. It contains, per
cluster: a name, a color, and an ordered list of `(label, full_title, year,
first_author, citation_count, in_corpus)`. Editing is straightforward — add a
row, change a year, recolor a lane, etc.

Curation rules we used for DNN:
- Pick focus clusters that match the narrative (e.g. for DNN we kept C9 MLP /
  C1 CNN / C5 RNN / C2 Transformer / C3 Diffusion / C7 SSL; we dropped the
  NAS, Evolutionary, Multimodal, Surveys clusters that retrieval over-pulled).
- For each focus cluster, enumerate ~10-16 canonical anchors per textbook
  consensus, mark each `in_corpus: true/false`, and supply rough citation
  counts.
- We did NOT try to inject every paper retrieval missed — only the textbook
  anchors needed to tell the story.

---

## Current status — both cases have v0, v2, *and* v2 manual refinement

Both DNN and RL have been taken through Phase 1 + Phase 2 to a renderable
HTML/SVG figure: a v0 first pass (single-shot manual yaml editing), a v2
(declarative outline + `migrate_method_spec.py` — see "Phase 3" below), and a
final **manual refinement** round on top of the v2 specs (lane taxonomy +
anchor coverage curated by hand — this is the current source of truth).

### v0 (single-shot manual curation)

| Case | Phase 1 done? | Phase 2 done? | Run dir | Spec file |
|---|---|---|---|---|
| DNN architectures | ✅ search + recluster (435 papers, 11 clusters) | ✅ spec + HTML + SVG (6 lanes, 81 anchors, 15 retrieved / 66 injected) | `downstream/runs/case_dnn_arch/artifacts/lr_search/` | `downstream/lr_search/method_timeline_spec.yaml` |
| RL algorithms | ✅ search + recluster (562 papers, 9 clusters) | ✅ spec + HTML + SVG (6 lanes, 43 anchors, 17 retrieved / 26 injected) | `downstream/runs/case_rl_evolution/artifacts/lr_search/` | `downstream/lr_search/method_timeline_spec_rl.yaml` |

### v2 (outline-driven re-curation, then hand-refined) — CURRENT

The `*.v2.yaml` specs are the current source of truth. Counts below reflect
the contents *after* the manual refinement round. "filled" = solid dot
(retrieved by pipeline or S2-verified); "dashed" = hollow dot injected from
the outline/spec metadata.

| Case | Lanes | Anchors | filled | dashed | Spec |
|---|---:|---:|---:|---:|---|
| DNN architectures | **9** | **112** | 37 | 75 | `method_timeline_spec.v2.yaml` |
| RL algorithms     | **9** | **66**  | 24 | 42 | `method_timeline_spec_rl.v2.yaml` |

The manual refinement round added, on top of the auto-migrated v2:

- **DNN (+8 anchors)**: C9 MLP gained Xavier Init · ReLU · He Init · LayerNorm
  · AdamW (init / activation / norm / optimizer chain); C13 GAN gained
  CycleGAN · SNGAN; C14 GNN gained Graphormer.
- **RL (+17 anchors, +1 lane)**: M2 gained C51 (distributional RL); M3 gained
  IMPALA; M5a completed the AlphaGo → AlphaGo Zero → AlphaZero → MuZero line;
  M6 gained the post-GRPO reasoning-RL branch (**Dr. GRPO · DAPO · VAPO ·
  ARPO · GiGPO**) plus DeepSeek-R1; and a brand-new **M8 Multi-Agent RL**
  lane (MADDPG · COMA · QMIX · AlphaStar · OpenAI Five · MAPPO).

The refinement anchors were written straight into the yaml with
`in_corpus: false` (dashed). To turn them into solid S2-verified dots later,
re-run `migrate_method_spec.py --skip-stages corpus_rescan recommend_expansion`
and they will pick up a `corpus_paper_id`.

#### Per-lane contents (current)

DNN v2 — 9 lanes / 112 anchors / 37 filled · 75 dashed (**bold** = refinement add):

| Lane | n | filled | what's in it |
|---|---:|---:|---|
| C9 MLP / Foundational | 16 | 0 | Perceptron · Backprop · Universal Approx. · DBN · Reducing dim. · Denoising AE · Stacked DAE · **Xavier Init** · **ReLU** · Dropout (idea) · Dropout (JMLR) · **He Init** · BatchNorm · Adam · **LayerNorm** · **AdamW** |
| C1 CNN Architectures | 15 | 1 | LeNet-5 · AlexNet · OverFeat · VGG · GoogLeNet · Inception-v3 · ResNet · DenseNet · Xception · MobileNet · ResNeXt · SENet · EfficientNet · NFNet · ConvNeXt |
| C5 RNN / Sequence | 11 | 0 | Elman RNN · LSTM · Bi-RNN · LSTM Forget · Deep Speech RNN · GRU · Seq2Seq · Bahdanau Attn · Luong Attn · ConvS2S · ELMo |
| C2 Transformer Family | 15 | 4 | Transformer · GPT-1 · BERT · GPT-2 · XLNet · RoBERTa · T5 · GPT-3 · ViT · Swin · Switch · PaLM · Chinchilla · LLaMA · GPT-4 |
| C12 State-Space Models / Mamba | 7 | 7 | HiPPO · S4 · S5 · Mamba · Mamba-2 · RWKV · Hyena |
| C3 Diffusion / Score-Based | 13 | 6 | Sohl-Dickstein · NCSN · DDPM · DDIM · Improved DDPM · Score SDE · Classifier Guidance · CFG · Stable Diffusion · Imagen · EDM · DiT · Consistency |
| C13 Generative Adversarial Nets | 11 | 9 | GAN · DCGAN · WGAN · WGAN-GP · **CycleGAN** · ProgressiveGAN · **SNGAN** · StyleGAN · BigGAN · StyleGAN2 · StyleGAN3 |
| C7 Self-Supervised | 16 | 4 | Word2Vec · GloVe · Skip-Thought · CPC · MoCo · SimCLR · BYOL · SwAV · SimSiam · Wav2Vec 2.0 · Barlow Twins · DINO · HuBERT · BEiT · CLIP · MAE |
| C14 Graph Neural Networks | 8 | 6 | Spectral GCN · ChebNet · GCN · GraphSAGE · MPNN · GAT · GIN · **Graphormer** |

RL v2 — 9 lanes / 66 anchors / 24 filled · 42 dashed (**bold** = refinement add):

| Lane | n | filled | what's in it |
|---|---:|---:|---|
| M1 Classical Foundations | 8 | 2 | TD Learning · Q-Learning (thesis) · Q-Learning (proof) · REINFORCE · SARSA · TD-Gammon · Policy Gradient · Actor-Critic |
| M2 Deep Value-Based RL | 8 | 2 | DQN (workshop) · DQN (Nature) · Double DQN · Dueling DQN · PER · **C51** · Rainbow · IQN |
| M3 Actor-Critic / Continuous Control | 7 | 4 | DPG · A3C · DDPG · Soft Q-Learning · **IMPALA** · TD3 · SAC |
| M4 Trust Region / PPO | 4 | 2 | TRPO · GAE · PPO · ACKTR |
| M5a Model-Based RL | 8 | 2 | Dyna · **AlphaGo** · AlphaGo Zero · **AlphaZero** · World Models · MuZero · Dreamer · DreamerV3 |
| M5b Offline RL | 4 | 3 | BCQ · CQL · IQL · Decision Transformer |
| M6 RL for LLMs / Alignment | 15 | 3 | RLHF (Christiano) · Summarize-HF · InstructGPT · Constitutional AI · DPO · RLAIF · GRPO · RLOO · **Dr. GRPO** · **DAPO** · **VAPO** · **ARPO (Replay)** · **ARPO (Agentic)** · **GiGPO** · **DeepSeek-R1** |
| M7 Exploration / Intrinsic Motivation | 6 | 6 | Pseudo-counts · ICM · RND · NoisyNet · Go-Explore · Agent57 |
| **M8 Multi-Agent RL** *(new lane)* | 6 | 0 | MADDPG · COMA · QMIX · AlphaStar · OpenAI Five · MAPPO |

### Two figure flavors per case: interactive vs. demo

`render_method_timeline.py` emits two different HTML/SVG figures from the same
spec, selected by the `--demo` flag:

| Flavor | Flag | What it is | Use for |
|---|---|---|---|
| **Interactive** | *(default)* | Full figure + subtitle + legend + per-lane debug subline (`M2 · n=8 · 2/8 in corpus`) + hover tooltips + an anchor reference table below the SVG. Solid dots = in corpus, dashed hollow = injected. | inspecting / auditing the data |
| **Demo / clean** | `--demo` | Presentation-only: just a title `<h1>` and the swimming-lane figure. No subtitle, legend, debug subline, table, or filled/dashed distinction — **every dot is solid**. Compact rows, larger dots + labels, and the SVG scales to fit the viewport (no horizontal/vertical scroll). | dropping into slides / paper as a clean case-study figure |

Demo-figure layout requirements we settled on (all live in `render_svg`'s
`demo` branch, no per-spec editing needed):

- fit-to-viewport, no scrolling — SVG uses `viewBox` + `preserveAspectRatio`
  and the wrapper CSS caps it at `max-width/height: 100%`;
- wide/flat aspect (~2:1) so it fills a typical 16:9 screen without side gaps;
- compact lane height (`row_height` 86) but labels kept clear of the dots
  (labels start ≥ 22 px from the lane center, above/below alternating) so
  fat dots never cover text;
- larger dots (radius 7–14) and larger fonts (anchor 12.5 px, lane 16 px);
- title shown once (HTML `<h1>` only, not duplicated inside the SVG).

Both flavors are regenerated by the commands in "How to reproduce" below.

### Open follow-up work (human review)

The remaining work is purely curation / polish on the spec yamls. None of it
needs to re-run the pipeline.

1. **Method-cluster validation per case.** Do the lanes match how the field
   is actually taxonomized? Is anything missing (e.g. should DNN have a
   "GNN" lane? should RL have a "Meta-RL" or "Exploration" lane)? Should any
   lane be split or merged?
2. **Anchor coverage per lane.** For each lane, are the canonical anchors
   complete? Did we miss any obvious one (e.g. Word2Vec is in DNN/SSL but
   we could add ELECTRA)? Drop any anchor that doesn't belong?
3. **Anchor metadata correctness.** Are the year / first-author / citation
   numbers accurate? Several were S2 indexing artifacts we fixed manually
   (Transformer year=2025, PER year=2024 etc.). Spot-check the rest.
4. **`in_corpus` flags.** Some anchors marked retrieved may have been weak
   lexical-rescue matches. Cross-check against `clusters_recovered.json`
   `recluster_metadata.paper_assignment_log` to see if `source` is
   `llm_map_reduce` (confident) or `lexical_rescue` (weak).
5. **Figure aesthetics.** Lane colors, time-axis segment weights, label
   abbreviations, lane ordering top-to-bottom, figure width — all live in
   the spec and re-render in seconds.

Most of these can be done by editing the yaml and re-running the render
command (sub-second each iteration).

---

## File / script inventory

### Scripts under `downstream/lr_search/`

| Script | Phase | Purpose |
|---|---|---|
| `literature_review_search.py` | 1 | Idea → retrieval (KG + S2) → `search_result.json` |
| `recluster_all_papers.py` | 1 | All-corpus reclustering with preview + map-reduce; produces `clusters_recovered.json` |
| `organize_search_result.py` | (optional) | Lexical organization into evidence map; not needed for figures |
| `build_method_timeline.py` | (optional, diagnostic) | Per-method CSV/MD/JSON dumps with landmark scores from the retrieved corpus — useful for inspecting what retrieval found before doing Phase 2 |
| `render_method_timeline.py` | 2 | Reads a spec YAML, writes a self-contained HTML + standalone SVG. Two flavors: default (interactive, with legend + debug subline + reference table) and `--demo` (clean presentation figure — all solid dots, compact, fits viewport). |

### Phase-2 spec files (one per case study)

| File | Purpose |
|---|---|
| `downstream/lr_search/method_timeline_spec.yaml` | DNN v0 spec. 6 lanes, 81 anchors. |
| `downstream/lr_search/method_timeline_spec_rl.yaml` | RL v0 spec. 6 lanes, 43 anchors. |
| `downstream/lr_search/method_timeline_spec.v2.yaml` | **DNN current spec.** 9 lanes, 112 anchors. |
| `downstream/lr_search/method_timeline_spec_rl.v2.yaml` | **RL current spec.** 9 lanes, 60 anchors. |

### Rendered figure outputs (per case)

Each case's `timeline/` dir holds both figure flavors, interactive + demo:

| File | Flavor |
|---|---|
| `.../timeline/method_timeline.v2.html` · `.v2.svg` | interactive (default) |
| `.../timeline/method_timeline.v2.demo.html` · `.v2.demo.svg` | demo / clean (`--demo`) |

where `...` is
`downstream/runs/case_dnn_arch/artifacts/lr_search` (DNN) or
`downstream/runs/case_rl_evolution/artifacts/lr_search` (RL). The demo HTMLs
are also copied to `/tmp/dnn_v2_demo.html` and `/tmp/rl_v2_demo.html` for quick
opening.

### Phase-3 outline + migration files (one per case study)

| File | Purpose |
|---|---|
| `downstream/lr_search/migrate_method_spec.py` | Outline-driven migration tool — reads a target-outline yaml and emits a renderable spec yaml. See "Phase 3" below. |
| `downstream/lr_search/target_outline_dnn_v2.yaml` | Worked example: DNN outline that produces the v2 spec (adds SSM / GAN / GNN lanes). |
| `downstream/lr_search/target_outline_rl_v2.yaml` | Worked example: RL outline that produces the v2 spec (splits M5; adds Exploration). |
| `downstream/lr_search/_smoke_outlines/` | Three smaller smoke-test outlines used during development. |

### Run-dir artifacts (per case)

```
downstream/runs/<case_name>/artifacts/lr_search/
├── search_result.json            # raw retrieval result (Phase 1)
├── topic_profile.json            # the LLM-built topic profile
├── time_windows.json             # auto-derived 5-stage timeline
├── clusters_final.json           # original (top-80-truncated) clustering — DON'T USE
├── clusters_recovered.json       # ★ full-corpus reclustering — USE THIS
├── recluster/                    # preview / batches / reduce intermediate JSONs
└── timeline/                     # Phase-2 outputs (only for DNN today)
    ├── papers_assigned.csv       # all retrieved papers with landmark scores
    ├── method_timeline.html      # ★ self-contained figure to view in a browser
    └── method_timeline.svg       # standalone SVG for paper inclusion
```

---

## How to start a NEW case study

Example: a new topic "Graph Neural Networks for molecular property prediction".

1. **Write an enriched idea string** (~5-10 method families, anchor work
   names, time range, exclusion rules). Use the DNN/RL strings as templates.

2. **Phase 1 — pipeline**
   ```bash
   TOPIC="Evolution of graph neural networks for molecular property prediction. Scope: ... Method families: (1) ... Anchor works: ... Time range: 2014–2025. Exclude: ..."
   RUN=/data2/yunx/innoeval/downstream/runs/case_gnn_mol/artifacts/lr_search
   python3 downstream/lr_search/literature_review_search.py \
     --topic "$TOPIC" --output-dir "$RUN" \
     --min-year 2014 --max-year 2025 \
     --probe-kg-top-k 50 --probe-s2-top-k 50 \
     --round-kg-top-k 40 --round-s2-top-k 40 \
     --enable-round2 --enable-embedding-expansion \
     --enable-query-cleaning --enable-relevance-guard
   python3 downstream/lr_search/recluster_all_papers.py \
     --search-result "$RUN/search_result.json" \
     --output        "$RUN/clusters_recovered.json" \
     --work-dir      "$RUN/recluster" \
     --batch-size 50 --max-parallel-batches 64 \
     --target-cluster-count-min 6 --target-cluster-count-max 12 \
     --enable-lexical-rescue --resume
   ```

3. **Phase 2 — curation + figure**

   a. Inspect what came out: `head $RUN/clusters_recovered.json | jq`,
      check landmark CSV via `build_method_timeline.py` if useful.

   b. Decide focus clusters (typically 4–7).

   c. Copy `method_timeline_spec.yaml` to
      `method_timeline_spec_<case>.yaml`. Edit:
      - `topic.title`, `subtitle`, `time_range`, `time_segments`
      - `clusters`: replace each lane's `name`, `color`, and rewrite the
        `anchors` list (textbook canonical anchors + their citation counts).
      - For each anchor that the pipeline found, set
        `in_corpus: true` and ideally `corpus_paper_id`. Use the
        `papers_assigned.csv` to find which retrieved paper matches.

   d. Render:
      ```bash
      python3 downstream/lr_search/render_method_timeline.py \
        --spec downstream/lr_search/method_timeline_spec_<case>.yaml \
        --output-html "$RUN/timeline/method_timeline.html" \
        --output-svg  "$RUN/timeline/method_timeline.svg" \
        --width 1600
      ```

   e. Open the HTML, iterate the YAML, re-render. Each round is sub-second.

---

## How to ITERATE on an existing case

### Cheap iterations (Phase 2 only)
- Reword a label: edit YAML → re-render.
- Add/remove an anchor: edit YAML → re-render.
- Change time-axis emphasis: edit `topic.time_segments` weights → re-render.
- Change cluster color or lane order: edit YAML → re-render.

### Medium iteration (rerun Phase 1)
- Change the enriched idea string → re-run `literature_review_search.py`.
- Re-run `recluster_all_papers.py` on the new search_result.
- The Phase-2 spec is decoupled, so curated anchors survive — but you may
  want to flip some `in_corpus` flags based on the new retrieval.

### Expensive iteration
- Switch retrieval backend, change ranking weights — invalidates Phase 1.

---

## Phase 3 · Outline-driven migration (`migrate_method_spec.py`)

Phase 3 is an **optional** extra path through Phase 2 that scales to "I want
to reorganize / extend the lane taxonomy across many iterations". Direct
yaml editing (Phase 2 as described above) is still valid and remains the
cheapest path for small tweaks. Phase 3 makes sense when you want to:

- split one lane into several (e.g. RL's `M5 Offline & Model-Based` →
  `M5a Model-Based` + `M5b Offline`),
- add a brand-new family the original retrieval missed (e.g. add a
  Mamba / GAN / GNN lane to DNN),
- rename / reorder / merge / drop lanes,
- re-run the same migration whenever the corpus or your method-family
  taxonomy changes.

### What it does

Inputs an **outline yaml** (declares the target lane structure) and an
existing `clusters_recovered.json` corpus, and produces a renderable spec
yaml. Per-lane, you choose a `source.mode`:

| Mode | Behaviour | Dependencies |
|---|---|---|
| `inherit` | Copy anchors verbatim from a lane in the current spec. | none |
| `manual`  | Pass anchors through from the outline yaml. | none |
| `reclassify` | Pull paper_ids from listed `from_clusters` (cluster ids in `clusters_recovered.json`) and use an LLM to bucket each paper into one of the reclassify-mode target lanes. Used for cluster splits. | LLM |
| `discover` | A brand-new family. Up to three stages run independently: `corpus_rescan` (keyword prefilter + LLM family-fit), `named_fetch` (S2 paper/search/match for each seed title), `recommend_expansion` (S2 recommendations from each found seed → LLM family filter). | S2 always; LLM for `corpus_rescan` and `recommend_expansion` |

All LLM batches and S2 calls are **cached on disk** (`<work_dir>/llm_cache/`,
`<work_dir>/s2_cache/`) keyed by a hash of the prompt or request. `--resume`
makes re-runs near-instant. Failed calls are not cached, so a retry after a
quota fix re-issues the request.

Every newly-fetched S2 paper is also persisted to
`<work_dir>/added_papers.json` so that subsequent rounds (or other
downstream tools) can ingest those papers as part of the corpus.

### CLI

```bash
python3 downstream/lr_search/migrate_method_spec.py \
  --outline   downstream/lr_search/target_outline_rl_v2.yaml \
  --out-spec  downstream/lr_search/method_timeline_spec_rl.v2.yaml \
  [--work-dir <dir>] [--resume] [--dry-run] \
  [--skip-stages corpus_rescan named_fetch recommend_expansion]
```

After migrate, render is the same single-line command as in Phase 2.

### v2 outcome — without LLM (DMX quota dry, S2 only)

Both v2 outlines were resolved with `--skip-stages corpus_rescan recommend_expansion`
(i.e. only `inherit`, `named_fetch`, and the seed-binding parts of `reclassify`
and `discover` ran). Each new lane's seed anchors were given to S2 by title;
matching papers became filled dots with `corpus_paper_id` of the form
`F_<lane>_NNN`. Unmatched seeds remain dashed dots (metadata is fully present
in the outline yaml — no further editing required).

#### DNN v2 — 9 lanes / 104 anchors / 37 filled · 67 dashed

| Lane | retrieved (Pxxx) | S2-fetched (F\_\*) | dashed | total | what's in it |
|---|---:|---:|---:|---:|---|
| C9 MLP / Foundational | 0 | 0 | 11 | 11 | Perceptron · Backprop · Universal Approx. · DBN · Reducing dim. · Denoising AE · Stacked DAE · Dropout (idea/JMLR) · BatchNorm · Adam |
| C1 CNN Architectures | 1 | 0 | 14 | 15 | LeNet-5 · AlexNet · OverFeat · VGG · GoogLeNet · Inception-v3 · ResNet · DenseNet · Xception · MobileNet · ResNeXt · SENet · **EfficientNet** · NFNet · ConvNeXt |
| C5 RNN / Sequence | 0 | 0 | 11 | 11 | Elman RNN · LSTM · Bi-RNN · LSTM Forget · Deep Speech RNN · GRU · Seq2Seq · Bahdanau Attn · Luong Attn · ConvS2S · ELMo |
| C2 Transformer Family | 4 | 0 | 11 | 15 | **Transformer** · GPT-1 · BERT · GPT-2 · XLNet · RoBERTa · T5 · GPT-3 · **ViT** · **Swin** · **Switch** · PaLM · Chinchilla · LLaMA · GPT-4 |
| **C12 State-Space Models / Mamba** *(new)* | 0 | 7 | 0 | 7 | **HiPPO** · **S4** · **S5** · **Mamba** · **Mamba-2** · **RWKV** · **Hyena** |
| C3 Diffusion / Score-Based | 6 | 0 | 7 | 13 | Sohl-Dickstein · NCSN · **DDPM** · **DDIM** · Improved DDPM · Score SDE · **Classifier Guidance** · CFG · **Stable Diffusion** · Imagen · **EDM** · **DiT** · Consistency |
| **C13 Generative Adversarial Nets** *(new)* | 0 | 9 | 0 | 9 | **GAN** · **DCGAN** · **WGAN** · **WGAN-GP** · **ProgressiveGAN** · **StyleGAN** · **BigGAN** · **StyleGAN2** · **StyleGAN3** |
| C7 Self-Supervised | 4 | 0 | 12 | 16 | Word2Vec · GloVe · Skip-Thought · **CPC** · **MoCo** · SimCLR · BYOL · SwAV · **SimSiam** · Wav2Vec 2.0 · Barlow Twins · DINO · **HuBERT** · BEiT · CLIP · MAE |
| **C14 Graph Neural Networks** *(new)* | 0 | 6 | 1 | 7 | Spectral GCN *(missed by S2)* · **ChebNet** · **GCN** · **GraphSAGE** · **MPNN** · **GAT** · **GIN** |

**Bold = filled dot** (in corpus or S2-verified). The three new lanes contributed
22 new filled anchors; only one seed across all three (Bruna 2014 Spectral
Networks) failed S2 title-match and remained dashed.

#### RL v2 — 8 lanes / 49 anchors / 24 filled · 25 dashed

| Lane | retrieved (Pxxx) | S2-fetched (F\_\*) | dashed | total | what's in it |
|---|---:|---:|---:|---:|---|
| M1 Classical Foundations | 2 | 0 | 6 | 8 | TD Learning · Q-Learning (thesis) · Q-Learning (proof) · **REINFORCE** · SARSA · TD-Gammon · **Policy Gradient** · Actor-Critic |
| M2 Deep Value-Based RL | 2 | 0 | 5 | 7 | DQN (workshop) · **DQN (Nature)** · Double DQN · Dueling DQN · **PER** · Rainbow · IQN |
| M3 Actor-Critic / Continuous Control | 4 | 0 | 2 | 6 | **DPG** · **A3C** · DDPG · Soft Q-Learning · **TD3** · **SAC** |
| M4 Trust Region / PPO | 2 | 0 | 2 | 4 | **TRPO** · GAE · **PPO** · ACKTR |
| **M5a Model-Based RL** *(split from M5)* | 2 | 0 | 4 | 6 | Dyna · AlphaGo Zero · **World Models** *(pinned P032)* · MuZero · Dreamer · **DreamerV3** *(matched P137)* |
| **M5b Offline RL** *(split from M5)* | 3 | 0 | 1 | 4 | BCQ · **CQL** · **IQL** · **Decision Transformer** *(pinned P307)* |
| M6 RL for LLMs / Alignment | 3 | 0 | 5 | 8 | **RLHF (Christiano)** · Summarize-HF · InstructGPT · Constitutional AI · **DPO** · **RLAIF** · GRPO · RLOO |
| **M7 Exploration / Intrinsic Motivation** *(new)* | 1 | 5 | 0 | 6 | **Pseudo-counts** · **ICM** · **RND** · **NoisyNet** · **Go-Explore** · **Agent57** *(was already P049 in corpus)* |

**Bold = filled dot.** M7 is fully filled: 5 anchors came from S2 named_fetch
and Agent57 was already in the corpus (auto-discovered by the fuzzy title
matcher). M5a + M5b show only their seed anchors; once DMX is available the
`reclassify` LLM stage will additionally bucket the ~80 papers in clusters
C4 + C6 into either lane and likely add 2–3 extras per lane.

### What LLM would add (currently skipped)

Three stages need DMX:

1. **`reclassify`** — needed to bucket the ~80 papers in RL clusters C4 + C6
   into the M5a / M5b split. Without it, M5a / M5b only contain seed anchors.
   Estimated impact: +2–3 filled extras per lane.
2. **`corpus_rescan`** — keyword prefilter + LLM family-fit over the full
   corpus. Expected to be low-yield for the new families (the original
   DNN/RL retrievals didn't target them; only one DNN paper hits any SSM
   keyword), but cheap to re-run when DMX is back.
3. **`recommend_expansion` (LLM filter)** — S2 already returned 40 unique
   "similar to SSM seeds" papers (cached); the LLM gate that decides which
   of them are *actually* SSM-family is what's missing. Likely candidates:
   H3, RetNet, Griffin, etc. Cache is already populated, so resume cost is
   only the LLM filtering pass (a few seconds per lane).

When DMX is topped up, re-running with `--resume` will:
- skip all S2 calls (already cached);
- run only the LLM batches;
- promote any above-confidence extras into the v2 spec.

Expected wall-clock to fully resolve both cases end-to-end with LLM:
~5–8 minutes total.

### How to re-render the current figures

The two `*.v2.yaml` specs are now **hand-refined after migration**, so they are
the source of truth — do NOT blindly re-run `migrate_method_spec.py`, which
would regenerate the spec from the outline and drop the manual additions
(unless you first fold those additions into the outline yaml). For figure
tweaks just re-run the renderer; each flavor is sub-second.

```bash
# ---- DNN ----
# interactive (default): full figure + legend + debug subline + reference table
python3 downstream/lr_search/render_method_timeline.py \
  --spec        downstream/lr_search/method_timeline_spec.v2.yaml \
  --output-html downstream/runs/case_dnn_arch/artifacts/lr_search/timeline/method_timeline.v2.html \
  --output-svg  downstream/runs/case_dnn_arch/artifacts/lr_search/timeline/method_timeline.v2.svg \
  --width 1800

# demo / clean: title + figure only, all dots solid, fits viewport
python3 downstream/lr_search/render_method_timeline.py \
  --spec        downstream/lr_search/method_timeline_spec.v2.yaml \
  --output-html downstream/runs/case_dnn_arch/artifacts/lr_search/timeline/method_timeline.v2.demo.html \
  --output-svg  downstream/runs/case_dnn_arch/artifacts/lr_search/timeline/method_timeline.v2.demo.svg \
  --width 1800 --demo

# ---- RL ----
python3 downstream/lr_search/render_method_timeline.py \
  --spec        downstream/lr_search/method_timeline_spec_rl.v2.yaml \
  --output-html downstream/runs/case_rl_evolution/artifacts/lr_search/timeline/method_timeline.v2.html \
  --output-svg  downstream/runs/case_rl_evolution/artifacts/lr_search/timeline/method_timeline.v2.svg \
  --width 1600

python3 downstream/lr_search/render_method_timeline.py \
  --spec        downstream/lr_search/method_timeline_spec_rl.v2.yaml \
  --output-html downstream/runs/case_rl_evolution/artifacts/lr_search/timeline/method_timeline.v2.demo.html \
  --output-svg  downstream/runs/case_rl_evolution/artifacts/lr_search/timeline/method_timeline.v2.demo.svg \
  --width 1700 --demo
```

To regenerate the spec from scratch via the outline (only if you want to redo
migration — it will overwrite the hand-refined spec), see "Phase 3" and run
`migrate_method_spec.py --skip-stages corpus_rescan recommend_expansion
--resume` first, then the render commands above. When DMX comes back, drop the
`--skip-stages` flag and re-run with `--resume` to pick up the LLM extras.

---

## DNN-specific note: retrieval gaps

The Phase-1 retrieval for DNN missed ~85% of the textbook anchors (AlexNet,
ResNet, VGG, BERT, GPT, LSTM 1997, Backprop, ...). That's why the DNN figure
ended up 66/81 = 82% injected.

Two reasons:
1. `literature_review_search.py` query generation is LLM-driven and tends to
   produce broad descriptive queries, not "find this paper by exact title".
2. KG / S2 ranking puts recent high-citation papers ahead of pre-2015
   classics, so the top-k cut drops the classics.

Possible future improvement (not done yet): add a `named_work_query` step
that lifts every anchor name from the topic string and issues a direct
title-based query, then merges results back. This would shrink the
"injected" share dramatically. Out of scope for the demo.

---

## Quick re-render commands

For the **current** figures (v2 specs, both interactive + demo flavors), use
the command block in "How to re-render the current figures" above.

The v0 specs are kept only for reference; to re-render them:
```bash
# DNN v0
python3 downstream/lr_search/render_method_timeline.py \
  --spec        downstream/lr_search/method_timeline_spec.yaml \
  --output-html downstream/runs/case_dnn_arch/artifacts/lr_search/timeline/method_timeline.html \
  --output-svg  downstream/runs/case_dnn_arch/artifacts/lr_search/timeline/method_timeline.svg \
  --width 1600

# RL v0
python3 downstream/lr_search/render_method_timeline.py \
  --spec        downstream/lr_search/method_timeline_spec_rl.yaml \
  --output-html downstream/runs/case_rl_evolution/artifacts/lr_search/timeline/method_timeline.html \
  --output-svg  downstream/runs/case_rl_evolution/artifacts/lr_search/timeline/method_timeline.svg \
  --width 1600
```
