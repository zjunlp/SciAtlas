<div align="center">
  <h1>SciAtlas: 面向自动化科学研究的大规模知识图谱</h1>
</div>

<p align="center">
  🌐 <a href="README.md">English</a> · <strong>简体中文</strong>
</p>

<p align="center">
  <a href="http://sciatlas.openkg.cn/api/docs/?lang=zh">📚 SciAtlas 文档网站</a>
</p>

<p align="center">
  一个可通过 uv 一键下载/安装，也可通过 pip 安装的 SciAtlas 客户端与命令行工具，用于调用托管 SciAtlas API 完成文献驱动的科研工作流。
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.22878">📄 arXiv</a>
  ·
  <a href="http://sciatlas.openkg.cn/register">🔑 获取 API Token</a>
  ·
  <a href="http://sciatlas.openkg.cn/healthz">🩺 API 健康检查</a>
</p>

---

## ✨ 项目概览

你可以把 SciAtlas 理解成一张面向科研的“知识地图”。输入一个研究主题、一个 idea、一位作者，或一条论文线索，SciAtlas 会帮你检索相关文献、沿着知识图谱寻找证据，并把结果整理成易读报告和可复现的 JSON 产物。

这张图谱不只是论文列表。它把论文、作者、机构、期刊会议、关键词、引用关系，以及从 Domain 到 Topic 的四级学科体系连接起来。因此，SciAtlas 的检索不只是在匹配关键词，也可以顺着研究领域、概念、人物和论文之间的关系继续探索。

本仓库提供的是面向用户的轻量级 **SciAtlas 客户端包**。新用户可以通过 `uv` 一键下载仓库并安装 CLI，也可以通过 `pip` 安装包；注册 API Token 后，就可以在本地运行文献检索和科研工作流。你无需自己部署 Neo4j、维护图数据库，也不用关心后端基础设施。

<p align="center">
  <img src="imgs/field_distribution_pie.png" alt="SciAtlas 各学科领域分布" width="92%">
</p>

<p align="center">
  <em>SciAtlas 覆盖医学、社会科学、工程、计算机科学、材料科学、数学等多个学科领域，适合跨学科科研探索。</em>
</p>

<p align="center">
  <img src="imgs/schema.png" alt="SciAtlas 知识图谱结构" width="92%">
</p>

<p align="center">
  <em>图谱把论文与作者、机构、来源、关键词、引用、相关工作，以及 Domain-Field-Subfield-Topic 层级连接起来。</em>
</p>

通过这个客户端，SciAtlas 可以直接服务于这些场景：

- **图谱增强论文检索**：结合关键词、语义、标题锚点、参考文献和图传播，不局限于普通关键词匹配；
- **科研工作流自动化**：运行文献综述、idea grounding、idea evaluation、idea generation、趋势分析、相关作者检索和研究者画像；
- **Agent 友好的输出**：保留 `request.json`、`response.json` 等机器可读产物，同时生成面向用户的 `summary.txt` 和 `report.md`；
- **可编辑 CLI skills**：把常用下游任务保存为可查看、可复制、可修改、可复用的 JSON skill；
- **通用 Agent Skill 包**：通过 [`agent-skill/`](agent-skill/) 把基础检索和当前专用 workflow 包装成端到端下游任务技能；文献综述、自动评审和 idea 生成都提供 `flash/full` 两条路径，让 Codex、Claude Code 等工具型 Agent 从检索一路完成科研目标。

## 📑 目录

- [✨ 项目概览](#-项目概览)
- [🚀 快速开始](#-快速开始)
- [🔑 API Token](#-api-token)
- [🧩 支持任务](#-支持任务)
- [🛠️ CLI 优先工作流](#️-cli-优先工作流)
- [🧰 可编辑 Skills](#-可编辑-skills)
- [🖊Agent Skill](#agent-skill)
- [🐍 Python SDK](#-python-sdk)
- [📦 输出与运行产物](#-输出与运行产物)
- [📂 仓库结构](#-仓库结构)
- [🧯 常见问题](#-常见问题)
- [📝 TODO](#-todo)
- [✍️ Citation](#️-citation)
- [📄 License](#-license)

## 🚀 快速开始
### 1. 安装

推荐使用 `uv` 一键下载完整仓库并安装：

Linux / macOS：

```bash
curl -LsSf https://raw.githubusercontent.com/zjunlp/SciAtlas/main/scripts/install-sciatlas-uv.sh | sh
```

Windows PowerShell：

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/zjunlp/SciAtlas/main/scripts/install-sciatlas-uv.ps1 | iex"
```

安装脚本会把完整仓库下载到 `~/SciAtlas`，创建 `uv` 虚拟环境，安装 SciAtlas CLI，并通过 `uv tool install` 暴露全局 `sciatlas` 命令。

也可以只安装 Python 包：

从 GitHub 直接安装：
```bash
pip install "git+https://github.com/zjunlp/SciAtlas.git#subdirectory=sciatlas"
```

如果只希望隔离安装 CLI：
```bash
pipx install "git+https://github.com/zjunlp/SciAtlas.git#subdirectory=sciatlas"
```

> **专用 workflow 前提**：上面的 GitHub 子目录和 `pipx` 安装只提供核心 CLI（健康检查、配置、计划、论文和作者检索）。`literature-review`、`idea-evaluate` 与 `idea-generate` 必须使用完整仓库及其工作流依赖；一键 `uv` 安装会自动完成。手动安装时执行：
>
> ```bash
> git clone https://github.com/zjunlp/SciAtlas.git
> cd SciAtlas
> python -m pip install -e ./sciatlas
> python -m pip install -r requirements-workflows.txt
> ```

安装后检查：

```bash
sciatlas -h
```

### 2. 注册 API Token

访问：
```text
http://sciatlas.openkg.cn/register
```

完成邮箱验证码注册，并复制个人 Token。

快速链接：[🔑 API Token](#-api-token)。

### 3. 配置

最少需要配置托管 SciAtlas API 地址和你的个人 Token。

Linux / macOS：
```bash
export SCIATLAS_API_BASE_URL="http://sciatlas.openkg.cn"
export SCIATLAS_API_KEY="your-personal-sciatlas-token"
export SCIATLAS_TIMEOUT=900
export SCIATLAS_RUNS_DIR="./runs"
```

Windows CMD：
```bat
set SCIATLAS_API_BASE_URL=http://sciatlas.openkg.cn
set SCIATLAS_API_KEY=your-personal-sciatlas-token
set SCIATLAS_TIMEOUT=900
set SCIATLAS_RUNS_DIR=.\runs
```


📕 可选：用自己的 LLM 提升关键词抽取

```bash
export LLM_PROVIDER="chat_completions"
export LLM_API_KEY="your-provider-api-key"
export LLM_BASE_URL="https://your-provider-or-gateway.example/v1"
export LLM_MODEL="your-model-name"
# 如果服务商使用特殊接口地址或鉴权头，可按需打开：
# export LLM_CHAT_COMPLETIONS_URL="https://your-provider-or-gateway.example/v1/chat/completions"
# export LLM_AUTH_HEADER="x-api-key: your-provider-api-key"
export LLM_TIMEOUT=180
export LLM_TEMPERATURE=0.7
export LLM_MAX_TOKENS=512
```

这是可选项。只有当你希望 SciAtlas 在检索前，用你的 LLM API 从自然语言输入中提炼更好的关键词时，才需要配置。

`LLM_PROVIDER` 保持 `chat_completions`，然后把 `LLM_API_KEY`、`LLM_BASE_URL` 和 `LLM_MODEL` 换成你的服务商参数。若服务商直接给出完整 chat-completions 接口地址，填写 `LLM_CHAT_COMPLETIONS_URL`；若需要自定义鉴权头，填写 `LLM_AUTH_HEADER`。

不需要 LLM 时可以全部留空。核心检索 CLI 会自动使用内置关键词抽取。文献综述、Idea 评审和 Idea 生成这三条专用 workflow 有各自的凭证要求；运行前请查看上面的专用 workflow 前提和对应说明。

用户需要编辑的配置模板：[.env.example](.env.example#L19-L45)。只有需要 LLM 辅助关键词抽取时，才填写这些变量。

🖊 可选：OpenAlex 元数据支持

```bash
export OA_API_KEY=""
export OPENALEX_MAILTO=""
```

OpenAlex 主要用于补充元数据或辅助 PDF 相关流程。README 中的常用 CLI 示例不依赖它。即使不填写这些变量，普通 SciAtlas 检索也可以正常运行。

用户需要编辑的配置模板：[.env.example](.env.example#L50-L52)。只有需要 OpenAlex 辅助元数据支持时才填写。

🖌 可选：本地 PDF 工作流需要 GROBID

只有当你要处理本地 PDF 文件时，才需要配置 GROBID。它会从科研 PDF 中抽取标题、作者、摘要和参考文献。如果你只运行上面的文本检索和下游 CLI 命令，可以直接跳过这一段。

本地启动 GROBID：

```bash
docker pull lfoppiano/grobid:latest
docker run -d --rm --name grobid -p 8070:8070 lfoppiano/grobid:latest
curl http://127.0.0.1:8070/api/isalive
```

然后设置：

```bash
export GROBID_BASE_URL="http://127.0.0.1:8070"
```

Windows CMD：

```bat
set GROBID_BASE_URL=http://127.0.0.1:8070
```

用户需要编辑的配置模板：[.env.example](.env.example#L47-L48)。不处理本地 PDF 时可以留空。

运行时变量说明：

| 变量 | 所需场景 | 说明 |
|---|---|---|
| `SCIATLAS_API_BASE_URL` | 所有托管 SciAtlas 任务 | 托管 SciAtlas API 基础 URL。 |
| `SCIATLAS_API_KEY` | 所有托管 SciAtlas 任务 | 作为 `X-API-Key` 和 `Authorization: Bearer` 发送。 |
| `LLM_PROVIDER` | 可选前端增强 | 保持为 `chat_completions`。 |
| `LLM_API_KEY` | 可选前端增强 | 你的服务商密钥；本地或无鉴权服务可留空。 |
| `LLM_BASE_URL` | 可选前端增强 | 服务商基础 URL，通常以 `/v1` 结尾。 |
| `LLM_CHAT_COMPLETIONS_URL` | 可选前端增强 | 只有服务商给出完整接口地址时才填写。 |
| `LLM_MODEL` | 可选前端增强 | 服务商提供的模型名称。 |
| `LLM_AUTH_HEADER` | 可选前端增强 | 只有需要自定义鉴权时才填写，例如 `x-api-key: your-provider-api-key`。 |
| `LLM_HTTP_HEADERS` | 可选前端增强 | 可选的额外请求头，填写 JSON 对象。 |
| `GROBID_BASE_URL` | PDF 任务 | 使用 `--pdf-path` 工作流时需要。 |
| `OA_API_KEY` | 可选 | OpenAlex 元数据 / PDF 支持。 |
| `OPENALEX_MAILTO` | 可选 | OpenAlex 联系邮箱。 |

### 4. 测试

```bash
sciatlas health
sciatlas config
```

### 5. 运行论文检索
```bash
sciatlas search-papers \
  --query "open world agent" \
  --keyword "high:open world agent" \
  --top-k 10
```

---

## 🔑 API Token

SciAtlas 对公开用户使用个人 API Token。

### 浏览器注册

访问：
```text
http://sciatlas.openkg.cn/register
```

流程：

1. 输入姓名、邮箱、机构和使用目的；
2. 点击 **Send code**；
3. 查收邮箱验证码；
4. 输入验证码并创建 Token；
5. 复制返回的 `sciatlas_xxx` Token。

Token 只显示一次，请妥善保存。

### 查询 Token 状态
```bash
curl -H "Authorization: Bearer $SCIATLAS_API_KEY" \
  http://sciatlas.openkg.cn/v1/auth/token/status
```

### 查询用量

```bash
curl -H "Authorization: Bearer $SCIATLAS_API_KEY" \
  "http://sciatlas.openkg.cn/v1/auth/usage?days=7"
```

---

## 🧩 支持任务

| 命令 | 场景 | 主要输出 |
|---|---|---|
| `sciatlas search-papers` | 论文检索 | 相关论文和 Markdown 报告 |
| `sciatlas related-authors` | 相关作者发现 | 候选作者与分数 |
| `sciatlas author-papers` | 作者论文查询 | 指定作者论文 |
| `sciatlas support-papers` | 支撑论文检索 | 候选作者的相关证据论文 |
| `sciatlas paper-search` | 轻量底层论文检索 | 快速论文候选 |
| `sciatlas literature-review` | 文献综述 | `literature_review_pipeline` flash/full workflow 产物 |
| `sciatlas idea-grounding` | idea 定位 | 相似工作和差异化证据 |
| `sciatlas idea-evaluate` | idea 评估 | `review_pipeline` flash/full 自动评审产物 |
| `sciatlas idea-generate` | idea 生成 | `sciatlas_idea_gen` flash/full 多步 workflow 产物和 idea seeds |
| `sciatlas trend-report` | 趋势分析 | 发展脉络和代表工作 |
| `sciatlas researcher-review` | 研究者背景综述 | 研究轨迹与代表论文 |
| `sciatlas skill` | 可编辑 skill 注册表 | 可复用工作流预设 |

---

## 🛠️ CLI 优先工作流

SciAtlas 以 CLI 为优先界面。新用户可以先查看帮助，再跑一次基础检索，最后按任务选择下游工作流；每次运行都会保存完整结果，方便复现、调试和交给 AI Agent 继续处理。

文档网站：[📚 SciAtlas 文档网站](http://sciatlas.openkg.cn/api/docs/?lang=zh)。用于查看 API 配置、CLI 命令、参数含义和可运行示例。

### 帮助

```bash
sciatlas -h
sciatlas search-papers -h
sciatlas literature-review -h
sciatlas skill -h
```

### 输入方式

SciAtlas 支持两种输入方式：专家参数输入和自然语言输入。正式使用时，推荐优先使用专家参数，因为每个检索条件都写得清楚，结果也更容易复现；自然语言输入更适合快速试跑和探索。

#### 推荐：专家参数输入

```bash
sciatlas --timeout 900 search-papers \
  --retrieval-mode hybrid \
  --query "open world agent" \
  --domain "artificial intelligence" \
  --time-range 2020-2024 \
  --keyword "high:open world agent" \
  --keyword "middle:embodied agent" \
  --title "middle:Voyager: An Open-Ended Embodied Agent with Large Language Models" \
  --reference "low:JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models" \
  --top-k 5 \
  --top-keywords 0 \
  --max-titles 0 \
  --max-refs 0 \
  --bias-keyword high \
  --bias-related high \
  --bias-exploration low \
  --ranking-profile precision \
  --report-max-items 5
```

示例中使用的是通用英文参数名；`--查询`、`--检索领域`、`--时间范围` 也可以作为中文别名使用。

#### 兼容：自然语言输入

使用 `--text` 时，SciAtlas 会从一段说明中解析检索意图。你也可以在文本里加入 `关键词[high]：...` 这类结构化提示，让关键词更稳定。

```bash
sciatlas --timeout 900 search-papers \
  --retrieval-mode hybrid \
  --text "检索 open world agent 相关论文，领域是 artificial intelligence，从 2020 年以后，返回 3 篇。

关键词[high]：open world agent" \
  --top-k 3 \
  --top-keywords 1 \
  --max-titles 0 \
  --max-refs 0
```

### 基础检索

适合快速围绕一个主题获取有证据支撑的论文列表。

```bash
sciatlas search-papers \
  --query "open world agent" \
  --domain "artificial intelligence" \
  --time-range 2020-2024 \
  --keyword "high:open world agent" \
  --top-k 5 \
  --top-keywords 0 \
  --max-titles 0 \
  --max-refs 0
```

### 五种下游工作流

下面的命令可以直接作为起点。你只需要替换 `--query`、`--idea`、`--author`、`--domain`、`--time-range` 和 `--keyword` 等参数。

#### 如何选择 `flash` 和 `full`

建议先用 `flash`。它适合快速交互、smoke test、初步判断方向是否值得继续。需要正式写综述、做更完整评审或保留更丰富证据链时，再切到 `full`。CLI 默认是 `flash`；可编辑 skill 中的 `literature-review-full`、`idea-evaluate-full` 和 `idea-generate-full` 对应完整版。

| Workflow | `flash` | `full` |
|---|---|---|
| `literature-review` | 压缩 KG/S2 检索轮次、规划 action 和论文预算，通常停在 bibliography / outline / evidence packs 等轻量阶段。 | 保留更完整的多轮检索、论文池、outline、section packs，并继续生成/整合正式综述。 |
| `idea-evaluate` | 压缩 KG/S2/manifest、reviewer 和 evidence budget，适合快速得到新颖性、可行性和修改方向判断。 | 保留更完整的 reviewer、rubric、grounding、evidence、review 和 report 路径，适合深入自动评审。 |
| `idea-generate` | 压缩锚点检索、研究图、gate/selection 阶段，快速生成少量 idea seeds。 | 扩大 seed、研究图和灵感检索范围，并保留更完整的新颖性反馈路径。 |

#### 文献综述

运行当前 `literature_review_pipeline` workflow。`flash` 会压缩检索轮次和报告深度，适合快速形成阅读清单；`full` 会保留更完整的多轮检索、证据筛选和综述生成。

```bash
sciatlas literature-review \
  --workflow flash \
  --query "retrieval augmented generation" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:retrieval augmented generation" \
  --top-k 10
```

```bash
sciatlas literature-review \
  --workflow full \
  --query "retrieval augmented generation" \
  --domain "artificial intelligence" \
  --time-range 2020-2025
```

#### Idea Evaluation

运行当前 `review_pipeline` 自动评审 workflow。`flash` 会压缩 reviewer、manifest 和 evidence budget，适合快速判断；`full` 会保留更完整的 reviewer/rubric/evidence/report 路径。

```bash
sciatlas idea-evaluate \
  --workflow flash \
  --idea "LLM-based multi-perspective evaluation for scientific research ideas" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:idea evaluation" \
  --keyword "middle:LLM as a judge" \
  --top-k 10
```

```bash
sciatlas idea-evaluate \
  --workflow full \
  --idea "LLM-based multi-perspective evaluation for scientific research ideas" \
  --domain "artificial intelligence"
```

#### Idea Generation

运行当前 `sciatlas_idea_gen` 多步 workflow：锚点论文检索、研究图构建、趋势/RSS 提取、灵感检索、idea 生成和新颖性检查。`flash` 会压缩 gate / selection 阶段，适合更快的实时交互；`full` 适合更完整的证据构建。

```bash
sciatlas idea-generate \
  --query "knowledge editing for large language models" \
  --domain "artificial intelligence" \
  --workflow flash
```

```bash
sciatlas idea-generate \
  --query "knowledge editing for large language models" \
  --domain "artificial intelligence" \
  --workflow full \
  --top-k 5
```

#### Trend Report

用于梳理一个研究主题的发展脉络，找出代表性论文和阶段性变化。

```bash
sciatlas trend-report \
  --query "retrieval augmented generation" \
  --domain "artificial intelligence" \
  --time-range 2020-2025 \
  --keyword "high:retrieval augmented generation" \
  --keyword "middle:knowledge graph" \
  --top-k 10
```

#### Researcher Review

用于快速了解一位研究者的研究轨迹、代表论文和相关方向。

```bash
sciatlas researcher-review \
  --author "Yoshua Bengio" \
  --limit 10 \
  --no-abstract
```

### 检索模式

| 模式 | 含义 | 适用场景 |
|---|---|---|
| `keyword` | 关键词驱动 KG 检索 | 术语明确 |
| `semantic` | 语义检索 | 宽泛语义匹配 |
| `title` | 标题锚点检索 | 已知代表论文 |
| `hybrid` | 关键词 + 语义 + 标题 + 图游走 | 默认推荐 |

未指定 `--retrieval-mode` 时，默认使用 `hybrid`。

### 专家锚点

当你已经知道高质量关键词、代表论文标题或参考文献时，可以用锚点引导图检索从这些线索出发。

```bash
--keyword "high:open world agent"
--title "middle:Voyager: An Open-Ended Embodied Agent with Large Language Models"
--reference "low:JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models"
```

### 图检索偏好
| 参数 | 含义 |
|---|---|
| `--bias-keyword` | 关键词路径强度 |
| `--bias-non-seed-keyword` | 非种子关键词扩展 |
| `--bias-citation` | 引用边强度 |
| `--bias-related` | 论文相关边强度 |
| `--bias-authorship` | 作者-论文关系强度 |
| `--bias-coauthorship` | 合作者网络强度 |
| `--bias-cooccurrence` | 关键词共现强度 |
| `--bias-exploration` | 图探索程度 |
| `--ranking-profile` | 排序偏好：`precision`、`balanced`、`discovery`、`impact` |

推荐的稳妥起点：

```bash
--top-k 10
--top-keywords 0
--max-titles 0
--max-refs 0
--bias-exploration low
```

---

## 🧰 可编辑 Skills

SciAtlas skills 是下游科研工作流的 JSON 预设，方便用户查看、复用和自定义。
```bash
sciatlas skill list
sciatlas skill show literature-review
sciatlas skill run literature-review --query "open world agent" --keyword "high:open world agent"
sciatlas skill run literature-review-full --query "open world agent" --keyword "high:open world agent"
sciatlas skill run idea-evaluate --idea "LLM-based idea evaluation"
sciatlas skill run idea-evaluate-full --idea "LLM-based idea evaluation"
sciatlas skill run --dry-run literature-review --query "open world agent" --keyword "high:open world agent"
```

创建自定义 skill：
```bash
sciatlas skill init my-review --from literature-review
```

它会生成：
```text
./skills/my-review.json
```

用户可以直接修改 JSON，然后运行：

```bash
sciatlas skill run my-review --query "your topic"
```

---

## 🖊Agent Skill

> **想直接描述科研目标，并让 Agent 把流程跑完时，就使用 Agent Skill。** 如果你希望自己执行、调试和微调每条命令，请继续使用上面的 CLI 模板。

Agent Skill 是供 Codex、Claude Code 等能够读取 `SKILL.md` 的工具使用的任务说明。安装后，你只需告诉 Agent 想了解或产出什么；它会完成安装或定位 SciAtlas、注册引导、环境配置、执行、读取运行产物和最终综合，交付科研结果而不只是命令模板。

<p align="center">
  <img src="imgs/agent-skill-demo.gif" alt="SciAtlas Agent Skill 工作流演示" width="92%">
</p>

<p align="center">
  <em>Agent Skill 演示：把科研需求变成检索或 workflow 运行、产物读取和面向任务的答案。</em>
</p>

### 三步开始

1. 按下表选择最接近目标的 Skill。
2. 通过 [Agent Skill 包安装说明](agent-skill/README.md#use) 安装该一个 Skill 目录（推荐）；只有确定会使用全部任务时，才安装全部 Skill。
3. 开启新的 Agent 会话，用自然语言描述目标。例如：“梳理 2020 年以来 retrieval-augmented generation 的文献脉络”或“评估这个研究想法”。

### 按目标选择 Skill

| 你想做什么 | 使用哪个 Skill | 你会得到什么 |
|---|---|---|
| 在投入更多时间前先判断一个主题 | `sciatlas-quick-paper-search` | 小规模、可核查的论文证据和下一步建议 |
| 规划阅读路径、论文地图、related work 或综述 | `sciatlas-literature-review` | 综述大纲、证据包或正式综述 |
| 判断一个 idea 是否与已有工作重叠 | `sciatlas-idea-grounding` | 前序工作矩阵、差异化风险和下一轮检索 |
| 判断一个 idea 或论文是否值得继续做 | `sciatlas-idea-evaluate` | 自动评审、reviewer/rubric 证据和修改建议 |
| 从一个主题出发寻找研究方向 | `sciatlas-idea-generate` | 有文献依据的 idea seeds 和验证方向 |
| 解释一个领域如何随时间演化 | `sciatlas-trend-report` | 时间线、代表论文和新兴信号 |
| 根据检索到的论文整理研究者工作 | `sciatlas-researcher-review` | 有证据依据的研究者画像，不是权威履历 |

### Agent 会做什么，你需要提供什么

Agent 会安装或定位 SciAtlas、引导浏览器注册、配置环境变量、运行允许的检索或 workflow、读取 `runs/<run_id>/` 产物并撰写最终答案。你只需提供程序无法知道的信息：邮箱验证码、返回的 SciAtlas Token、缺失的 LLM/S2/KG 密钥、本地 PDF，或一次必要的任务范围澄清。不要把 Token 和运行产物放进 `agent-skill/`。

### 怎么选择 `flash` 和 `full`

文献综述、idea 评估和 idea 生成这三类 Skill 都先用 `flash`：速度更快，但保留主要证据链。只有你要求更广覆盖，或 `flash` 的产物对所需深度不足时，Agent 才切到 `full`。无论哪种模式都只安装同一个 Skill 目录；`literature-review-full`、`idea-evaluate-full` 和 `idea-generate-full` 是 `sciatlas skill run` 的 CLI JSON 预设名，不是 Agent Skill 目录。

底层执行时，快速检索、idea grounding、趋势分析和研究者画像只使用 `search-papers`；文献综述、idea 评估和 idea 生成分别调用专用 workflow。完整的执行边界和 artifact 说明见 [`agent-skill/README.md`](agent-skill/README.md)。

### 安装所选 Skill

[Agent Skill 包安装说明](agent-skill/README.md#use) 是唯一的安装参考，其中包含 Codex 与 Claude Code 在 Windows、macOS 和 Linux 上的命令，并明确区分“安装一个所选 Skill”和“安装全部 Skill”。安装后开启新的会话并直接描述科研目标；所选 `SKILL.md` 就是 Agent 进行安装、执行、读取产物和综合回答时遵循的唯一操作说明。

---

## 🐍 Python SDK

```python
from sciatlas import SciAtlasClient

client = SciAtlasClient()
print(client.health())

result = client.search_papers(
    query="open world agent",
    keywords=[{"text": "open world agent", "score": 10}],
    top_k=3,
)
print(result)
```

也可以直接传入配置：

```python
client = SciAtlasClient(
    base_url="http://sciatlas.openkg.cn",
    api_key="your-personal-sciatlas-token",
)
```

## 📦 输出与运行产物
终端默认输出简洁表格，完整结果保存在：

```text
runs/<run_id>/
```

常见文件：
| 文件 | 说明 |
|---|---|
| `plan.json` | 结构化检索计划 |
| `request.json` | 发送给 SciAtlas API 的完整请求 |
| `response.json` | 后端原始响应 |
| `summary.txt` | 简短摘要 |
| `report.md` | 面向用户的 Markdown 报告 |
| `metadata.json` | 运行元信息 |

---

## 📂 仓库结构

下面只保留仓库展示和新用户上手最需要的主入口，已省略生成产物、虚拟环境和缓存目录。

```text
SciAtlas/
  README.md / README_zh.md       # 项目文档
  .env.example                   # 根目录运行配置模板
  requirements.txt
  run_sciatlas.py                  # 轻量本地运行入口
  agent-skill/                   # 通用 Agent Skill 包
  docs/api/                      # 统一静态 API 与 CLI 文档网站
  imgs/                          # README 图片资源
  sciatlas/                        # 可 pip 安装的 SciAtlas 客户端包
    pyproject.toml
    src/sciatlas/                  # 打包发布的 CLI、client、config、skills
    core/ search/ tasks/         # 检索规划与科研工作流逻辑
    evidence/ llm/ renderers/    # PDF 证据、可选 LLM、报告渲染
    examples/ tests/
  references/search/             # KG 检索参考实现
  runs/                          # 生成的 CLI 运行产物
```

---

## 🧯 常见问题

### `sciatlas health` 成功，但 `search-papers` 返回 401

说明 Token 缺失或无效。
```bash
echo $SCIATLAS_API_KEY
export SCIATLAS_API_KEY="your-personal-sciatlas-token"
```

Windows CMD：
```bat
set SCIATLAS_API_KEY=your-personal-sciatlas-token
```

### 没有收到邮箱验证码

请检查邮箱地址、垃圾邮件和验证码重发间隔。

### 检索很慢或超时

使用轻量参数：
```bash
--top-k 3
--top-keywords 0
--max-titles 0
--max-refs 0
--bias-exploration low
```

### Windows 上找不到 `sciatlas` 命令

```bat
.venv\Scripts\sciatlas.exe -h
```

或重新安装：

```bat
.venv\Scripts\python.exe -m pip install -e .
```

---

## 📝 TODO

- [x] **命令行工具。** 增加更多面向用户的命令行功能，以便下游用户和 AI 代理无需接触数据库内部即可调用检索工作流。
- [x] **通用 Agent Skill 包。** 为常见的科学发现工作流打包可重用的代理技能，并将最佳实践作为更易于加载的组件提供。
- [ ] **更多知识。** 整合超越以论文为中心的实体之外的更多知识形式，例如数据集、代码、标准、定理和实验经验。
- [ ] **基准测试与评估。** 为 SciAtlas 支持的下游科学研究任务构建专用基准测试和评估协议。
- [ ] **动态更新。** 改进动态知识更新机制，使其更加系统化并提高刷新频率。

---

## ✍️ Citation

如果 SciAtlas 对你的研究有帮助，请引用：
```
@misc{qiao2026sciatlaslargescaleknowledgegraph,
      title={SciAtlas: A Large-Scale Knowledge Graph for Automated Scientific Research}, 
      author={Shuofei Qiao and Yunxiang Wei and Jiazheng Fan and Bin Wu and Busheng Zhang and Mengru Wang and Yuqi Zhu and Ningyu Zhang and Keyan Ding and Qiang Zhang and Huajun Chen},
      year={2026},
      eprint={2605.22878},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.22878}, 
}
```

---

## 📄 License

本项目采用 MIT License。详见 [LICENSE](LICENSE)。
