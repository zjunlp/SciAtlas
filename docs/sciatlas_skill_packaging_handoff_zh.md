# 从 API 到 CLI Skill 与 Agent Skill

本文面向其它有 API 接口、希望包装成 CLI 和 Agent Skill 的项目，总结 SciAtlas 的包装思路与可复用做法。文中保留了关键代码位置，便于阅读时对照实现。

一句话结论：先做稳定 CLI 原语和可复盘 artifacts，再做 CLI Skill 参数预设，最后做 Agent Skill 工作流说明书。


## 1. 总体分层

推荐任何 API 型项目按这条链路包装：

```text
后端 API
  -> SDK / API client
  -> CLI 原语命令
  -> 运行产物 artifacts
  -> CLI Skill 参数预设
  -> Agent Skill 工作流说明书
```

SciAtlas 的对应关系：

| 通用层 | SciAtlas 案例 | 作用 |
|---|---|---|
| 后端 API | Hosted SciAtlas API `/v1/search` | 提供核心检索能力 |
| CLI 原语 | `sciatlas search-papers` | 最稳定、最通用的可运行命令 |
| 运行产物 | `runs/<run_id>/` | 给用户复盘，也给 Agent 读取 |
| CLI Skill | `sciatlas skill run literature-review` | 保存常用参数组合 |
| Agent Skill | `agent-skill/*/SKILL.md` | 指导 Agent 安装、配置、运行、读证据、生成交付物 |

要点：

- CLI 是稳定执行层。
- artifacts 是证据和复盘层。
- CLI Skill 是参数复用层。
- Agent Skill 是任务方法层。

代码定位：

- 项目包装入口：[sciatlas/pyproject.toml](../sciatlas/pyproject.toml#L42-L49)
- CLI 命令注册和分发：[build_parser](../sciatlas/src/sciatlas/cli.py#L5594)、[main](../sciatlas/src/sciatlas/cli.py#L6012)

## 2. 先选一个稳定 CLI 原语

Agent Skill 不应直接依赖大量接口。应先选一个最稳定、覆盖面最大的 CLI 原语；如果下游任务已经有维护中的专用 workflow，则把 workflow 包装成清晰的 `flash/full` 路径，并让 Agent 只读取保存下来的 artifacts。

SciAtlas 选择：

```bash
sciatlas search-papers
```

现在的分层规则是：quick search、grounding、trend、researcher 等检索型 Agent Skill 继续以 `search-papers` 作为唯一稳定原语；`sciatlas-literature-review` 只调用 `literature-review`，`sciatlas-idea-evaluate` 只调用 `idea-evaluate`，`sciatlas-idea-generate` 只调用 `python -m sciatlas_idea_gen.main`。三条专用 Skill 从压缩非必要 stage 的 `flash` 路径开始；只有用户要求更广覆盖或 flash artifacts 不足时才切到 `full`。`*-full` 是 CLI JSON preset 的名称，不是额外的 Agent Skill 目录。Agent 通过读取 artifacts 完成综述、自动评审、趋势分析或 idea-generation 等下游任务。

选择原语时，重点关注 5 个判断标准：

- 是否稳定，长期不频繁改参数？
- 是否覆盖大部分下游任务？
- 是否能保存完整请求和响应？
- Agent 是否能基于输出继续综合分析？
- 普通用户是否也能直接运行？

代码定位：

- `search-papers` 命令实现：[cmd_search_papers](../sciatlas/src/sciatlas/cli.py#L4841)
- 下游命令复用同一检索链路：[_run_plan_search_channel](../sciatlas/src/sciatlas/cli.py#L4460)
- 专用 workflow Agent Skill 示例：[sciatlas-literature-review/SKILL.md](../agent-skill/sciatlas-literature-review/SKILL.md#L8)

## 3. CLI 包装流程

CLI 的目标不是把 API 参数原样暴露给用户，而是把“用户友好输入”转换成“后端稳定 schema”。

通用执行链路：

```text
用户输入
  -> 标准化任务计划 plan
  -> API 请求 options
  -> 后端响应 response
  -> 保存 artifacts
  -> 输出简短终端摘要
```

SciAtlas 的 `search-papers` 对应：

```text
read_text_input()
  -> build_plan_from_text()
  -> build_options_from_text()
  -> request_json(POST /v1/search)
  -> save_artifacts()
  -> render_user_output()
```

要点：

- `plan` 负责表达用户想查什么。
- `options` 负责表达怎么查，例如 `top_k`、时间范围、retrieval mode。
- CLI 可以支持自然语言输入，也可以支持专家参数。
- stdout 只做摘要，不能替代 artifacts。

代码定位：

- 包入口和包数据：[pyproject.toml](../sciatlas/pyproject.toml#L42-L49)
- plan 构造：[build_plan_from_text](../sciatlas/src/sciatlas/cli.py#L1981)
- options 构造：[build_options_from_text](../sciatlas/src/sciatlas/cli.py#L2470)
- API 请求：[request_json](../sciatlas/src/sciatlas/cli.py#L727)

## 4. Artifacts 是 Agent 的关键接口

这是最值得重点讲的部分：Agent 不应解析 stdout，而应读取稳定文件。

SciAtlas 每次运行保存：

```text
runs/<run_id>/
  plan.json
  request.json
  response.json
  summary.txt
  report.md
  metadata.json
```

文件职责：

| 文件 | 用途 |
|---|---|
| `plan.json` | 用户输入解析后的任务计划 |
| `request.json` | 发给 API 的完整请求 |
| `response.json` | 后端响应，必要时可压缩但不能丢核心证据 |
| `summary.txt` | 快速判断成功、失败、结果数量 |
| `report.md` | 面向用户和 Agent 的可读报告 |
| `metadata.json` | 命令、时间、状态码、endpoint、耗时 |

要点：

- artifacts 是“人类复盘”和“Agent 接续工作”的共同接口。
- 其它项目至少应保存 `request.json`、`response.json`、`summary.txt`、`report.md`、`metadata.json`。
- 响应可以压缩，但不能破坏核心证据。

代码定位：

- artifact 保存入口：[save_artifacts](../sciatlas/src/sciatlas/cli.py#L4100)
- 命令结束时统一保存：[finish_with_artifacts](../sciatlas/src/sciatlas/cli.py#L4220)
- artifact 压缩测试：[test_cli_artifacts.py](../sciatlas/tests/test_cli_artifacts.py)

## 5. CLI Skill：参数预设，不是新执行器

CLI Skill 解决的是“常用参数组合复用”的问题。

SciAtlas 的内置预设：

```text
sciatlas/src/sciatlas/builtin_skills.json
```

典型结构：

```json
{
  "name": "literature-review",
  "aliases": ["review", "lit-review"],
  "command": "literature-review",
  "defaults": {
    "retrieval_mode": "hybrid",
    "top_k": 5,
    "ranking_profile": "balanced"
  }
}
```

要点：

- CLI Skill 不是新的业务逻辑。
- 它只是把 preset 展开成普通 CLI argv。
- 展开后仍交回原 CLI parser 执行。
- `--dry-run` 很适合检查和调试。

快速验证：

```bash
python run_sciatlas.py skill run --dry-run literature-review --query "open world agent"
```

代码定位：

- 内置 preset 示例：[builtin_skills.json](../sciatlas/src/sciatlas/builtin_skills.json#L3)
- 加载内置 skill：[_builtin_skills](../sciatlas/src/sciatlas/skills.py#L11)
- 用户目录覆盖：[_dirs](../sciatlas/src/sciatlas/skills.py#L20)、[load_skills](../sciatlas/src/sciatlas/skills.py#L43)
- preset 展开：[expand_skill](../sciatlas/src/sciatlas/skills.py#L75)
- `skill run` 分发：[dispatch_skill_cli](../sciatlas/src/sciatlas/skills.py#L169)
- `main()` 接收展开后的 argv：[cli.py main](../sciatlas/src/sciatlas/cli.py#L6012)

## 6. Agent Skill：任务 playbook

Agent Skill 解决的是“多步任务怎么交给 Agent 稳定执行”的问题：安装、注册引导、配置、运行命令、读取 artifacts、综合证据、输出最终结果。SciAtlas 的目标范式是小白友好：Agent 尽量代办所有可自动化步骤，用户只提供邮箱、验证码、API Token、LLM/S2/KG 密钥或一次必要的任务澄清。

其中要区分核心 CLI 与专用 workflow：`search-papers` 等核心检索命令可以只安装 `sciatlas` 子包；`literature-review`、`idea-evaluate` 和 `idea-generate` 必须在完整仓库中运行，并先安装 `requirements-workflows.txt`。不要把 GitHub `#subdirectory=sciatlas` 安装方式用于这三条专用 workflow。

当前七个 Skill 的入口与交付边界如下：

| Skill 类别 | 固定入口 | 典型交付 |
|---|---|---|
| quick paper search / idea grounding / trend report | 仅 `sciatlas search-papers` | 小型证据表、前序工作矩阵或阶段化时间线 |
| researcher review | 仅 `sciatlas search-papers` | 文献证据驱动的研究者画像，不是完整 CV |
| literature review | 仅 `sciatlas literature-review` 或 `python run_sciatlas.py literature-review` | 综述大纲、论文地图、证据包或正式综述 |
| idea evaluate | 仅 `sciatlas idea-evaluate` 或 `python run_sciatlas.py idea-evaluate` | go/revise/no-go、rubric/reviewer/evidence 报告 |
| idea generate | 仅 `python -m sciatlas_idea_gen.main` | 研究图、灵感链路、idea seeds 与新颖性风险 |

SciAtlas 的结构：

```text
agent-skill/
  sciatlas-literature-review/
    SKILL.md
    agents/openai.yaml
```

一个 `SKILL.md` 重点包含：

- Operating Contract：允许做什么、禁止做什么。
- Zero-start bootstrap：没安装 CLI、没 token 时怎么处理。
- Search / Run Plan：只调用哪个稳定原语或专用 workflow。
- Reading Artifacts：读哪些文件，按什么顺序。
- Synthesis Method：如何把证据变成结果。
- Deliverable：最终答案必须包含什么。

要点：

- Agent Skill 是 playbook，不是代码。
- 它的强约束是“只调用稳定原语或明确指定的当前 workflow”。
- 它依赖 artifacts 做证据读取和综合。
- 它要写清 zero-start，因为 Agent 面向的是新手用户：不能只把命令扔给用户，而要主动安装/定位 CLI、引导注册、配置环境、运行检索或 workflow、读取 `runs/<run_id>/`，最后交付可直接阅读的结果。

代码定位：

- Agent Skill 总体定位：[agent-skill/README.md](../agent-skill/README.md#L3)
- CLI 层和 Agent 层互补：[agent-skill/README.md](../agent-skill/README.md#L11-L14)
- 不依赖下游 CLI 的设计说明：[agent-skill/README.md](../agent-skill/README.md#L48-L52)
- `literature-review` 操作契约：[SKILL.md Operating Contract](../agent-skill/sciatlas-literature-review/SKILL.md#L10)
- zero-start 流程：[SKILL.md Zero-Start Bootstrap](../agent-skill/sciatlas-literature-review/SKILL.md#L17)
- artifact 读取：[SKILL.md Reading Artifacts](../agent-skill/sciatlas-literature-review/SKILL.md#L57)
- 最终交付格式：[SKILL.md Deliverable](../agent-skill/sciatlas-literature-review/SKILL.md#L86)

## 7. 其它项目照做清单

把 SciAtlas 换成其它项目时，可以按这张清单落地。

### CLI 层

1. 定义 `yourtool.cli:main`。
2. 读取 API base URL、API key、timeout、runs dir。
3. 选一个稳定基础命令，例如 `search`、`analyze`、`query`。
4. 把用户输入转换成标准 request schema。
5. 调用 API。
6. 保存 `runs/<run_id>/`。
7. 输出简短摘要。

### CLI Skill 层

1. 新增 `builtin_skills.json`。
2. 为常见任务写 JSON preset。
3. 实现 `skill list/show/run/init`。
4. 允许用户目录覆盖内置 preset。
5. 提供 `--dry-run`。

### Agent Skill 层

1. 为每个任务建 `agent-skill/<name>/SKILL.md`。
2. 写清触发场景和操作边界。
3. 明确只允许调用哪个基础命令或专用 workflow。
4. 写安装、注册引导、配置和 token 获取流程，并强调只向用户索取邮箱、验证码、API Token、LLM/S2/KG 密钥等人类必须提供的信息。
5. 写命令模板。
6. 写 artifact 读取顺序。
7. 写最终交付格式，要求 Agent 输出下游结果而不是运行说明。
8. 可选添加 `agents/openai.yaml`。

### 最小测试

- 包可以 import。
- CLI help 可以运行。
- `skill list` 可以列出 preset。
- `skill run --dry-run` 展开正确。
- artifact 不丢核心字段。
- 缺 API key 时错误清楚，且不泄露密钥。

## 8. 常见问题

| 问题 | 建议回答 |
|---|---|
| 为什么不让 Agent 直接调 HTTP API？ | CLI 是稳定边界，能统一配置、参数、错误处理和 artifacts。 |
| 为什么 Agent Skill 要限制调用入口？ | 降低耦合，让 Agent 基于同一证据或 workflow artifacts 做不同任务综合。 |
| CLI Skill 和 Agent Skill 有什么区别？ | CLI Skill 是参数预设；Agent Skill 是端到端任务说明书。 |
| artifacts 为什么必要？ | 它让结果可复盘，也让 Agent 不必解析 stdout。 |
| 默认参数为什么重要？ | 默认参数体现任务策略，例如查准、查全、探索或影响力优先。 |

## 9. 代码索引

| 文件 | 关注点 |
|---|---|
| [sciatlas/pyproject.toml](../sciatlas/pyproject.toml#L42-L49) | Python 包入口、console script、package data |
| [sciatlas/src/sciatlas/cli.py](../sciatlas/src/sciatlas/cli.py#L5594) | CLI parser、命令注册 |
| [cmd_search_papers](../sciatlas/src/sciatlas/cli.py#L4841) | 基础原语实现 |
| [save_artifacts](../sciatlas/src/sciatlas/cli.py#L4100) | artifact 保存 |
| [sciatlas/src/sciatlas/skills.py](../sciatlas/src/sciatlas/skills.py#L169) | CLI Skill 加载、dry-run、argv 展开 |
| [builtin_skills.json](../sciatlas/src/sciatlas/builtin_skills.json#L3) | 内置 JSON preset |
| [agent-skill/README.md](../agent-skill/README.md#L3) | Agent Skill pack 定位 |
| [sciatlas-literature-review/SKILL.md](../agent-skill/sciatlas-literature-review/SKILL.md#L10) | Agent Skill 操作契约 |
| [install-sciatlas-uv.ps1](../scripts/install-sciatlas-uv.ps1#L112-L123) | zero-start 安装脚本 |
| [test_cli_artifacts.py](../sciatlas/tests/test_cli_artifacts.py) | artifact 核心字段保留测试 |

## 10. 验证记录

已运行无需 token 的验证：

```bash
python run_sciatlas.py skill list
python run_sciatlas.py skill run --dry-run literature-review --query "open world agent" --keyword "high:open world agent"
```

结果：

- 当前内置 CLI Skill 包括 `idea-evaluate`（flash 默认）、`idea-evaluate-full`、`idea-generate`（flash 默认）、`idea-generate-full`、`idea-grounding`、`literature-review`（flash 默认）、`literature-review-full`、`quick-paper-search`、`researcher-review`、`trend-report`。
- `literature-review` 与 `idea-evaluate` 能正确展开为当前 workflow 的 `flash/full` 底层 CLI 命令并带上默认参数。

真实检索和 workflow smoke 需要有效 `SCIATLAS_API_KEY` 以及可用的 KG/S2/LLM 配置；无凭据环境中优先运行 help、compile、dry-run 和 mock/smoke 测试。
