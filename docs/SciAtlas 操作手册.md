# SciAtlas 操作手册

SciAtlas 可以理解为一个科研文献检索助手。输入研究主题、研究想法或研究者姓名后，SciAtlas 会调用托管知识图谱服务，检索相关论文，并在本地生成可阅读的结果报告。

## 1. 使用前准备

| 准备项 | 用途 |
|---|---|
| 可联网电脑 | 访问 SciAtlas 托管服务 |
| 浏览器 | 打开注册页面并获取 API Token |
| 邮箱 | 接收注册验证码 |
| 终端窗口 | 执行 SciAtlas 命令 |
| SciAtlas API Token | 调用托管服务的身份凭证 |

注意事项：

- API Token 类似密码，不应发到聊天群、截图、共享文档或公开仓库。
- 首次运行建议只返回 3 篇论文，确认流程正常后再增加数量。

## 2. 打开终端

### Windows

1. 打开开始菜单。
2. 搜索 `PowerShell`。
3. 打开 `Windows PowerShell`。

### macOS

1. 打开启动台。
2. 搜索 `Terminal` 或 `终端`。
3. 打开终端应用。

### Linux

打开系统自带的 `Terminal`。

## 3. 安装 SciAtlas

推荐使用一键安装脚本。安装完成后，系统中会出现 `sciatlas` 命令。

### Windows PowerShell

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/zjunlp/SciAtlas/main/scripts/install-sciatlas-uv.ps1 | iex"
```

### macOS / Linux

```bash
curl -LsSf https://raw.githubusercontent.com/zjunlp/SciAtlas/main/scripts/install-sciatlas-uv.sh | sh
```

安装完成后，执行以下命令检查是否安装成功：

```bash
sciatlas -h
```

若终端显示帮助说明，表示安装成功。

## 4. 获取 API Token

浏览器打开：

```text
http://sciatlas.openkg.cn/register
```

注册流程：

1. 填写姓名、邮箱、机构和使用目的。
2. 点击发送验证码。
3. 前往邮箱查收验证码。
4. 回到注册页面输入验证码。
5. 复制页面返回的 `sciatlas_xxx` Token。

Token 通常只显示一次，建议保存到安全位置。

## 5. 配置 API Token

SciAtlas 命令需要读取两个环境变量：

| 变量 | 含义 |
|---|---|
| `SCIATLAS_API_BASE_URL` | SciAtlas 托管服务地址 |
| `SCIATLAS_API_KEY` | 注册得到的个人 Token |

### Windows PowerShell，当前窗口临时生效

将 `把Token粘贴在这里` 替换为注册得到的完整 Token：

```powershell
$env:SCIATLAS_API_BASE_URL = "http://sciatlas.openkg.cn"
$env:SCIATLAS_API_KEY = "把Token粘贴在这里"
```

该方式只在当前 PowerShell 窗口有效。

### Windows PowerShell，长期保存

```powershell
setx SCIATLAS_API_BASE_URL "http://sciatlas.openkg.cn"
setx SCIATLAS_API_KEY "把Token粘贴在这里"
```

执行 `setx` 后，需要关闭当前 PowerShell，并重新打开一个新的 PowerShell 窗口。

### macOS / Linux，当前窗口临时生效

```bash
export SCIATLAS_API_BASE_URL="http://sciatlas.openkg.cn"
export SCIATLAS_API_KEY="把Token粘贴在这里"
```

## 6. 检查配置

依次执行：

```bash
sciatlas health
sciatlas config
```

检查要点：

- `sciatlas health` 用于确认托管服务是否可访问。
- `sciatlas config` 用于确认服务地址和 Token 是否已经配置。

## 7. 首次检索示例

以下命令用于检索 `open world agent` 相关论文，只返回 3 篇结果，适合作为首次测试。

```bash
sciatlas --timeout 900 search-papers --query "open world agent" --domain "artificial intelligence" --time-range 2020-2024 --keyword "high:open world agent" --top-k 3 --top-keywords 0 --max-titles 0 --max-refs 0 --report-max-items 3
```

命令执行完成且没有报错，表示 SciAtlas 已可正常使用。

## 8. 查看运行结果

每次运行后，结果会保存到 `runs` 目录下的新文件夹中，例如：

```text
runs/20260505_013440_search_papers_xxxxxxxx/
```

常见结果文件：

| 文件 | 说明 |
|---|---|
| `report.md` | 面向阅读的 Markdown 报告，优先查看 |
| `summary.txt` | 简短结果摘要 |
| `request.json` | 本次请求参数，便于复现 |
| `response.json` | 后端返回的完整结果 |
| `metadata.json` | 运行时间、命令等元信息 |

普通阅读场景通常只需要打开 `report.md` 和 `summary.txt`。

## 9. 常用参数说明

| 参数 | 含义 | 示例 |
|---|---|---|
| `--query` | 检索主题 | `"retrieval augmented generation"` |
| `--domain` | 所属领域 | `"artificial intelligence"` |
| `--time-range` | 时间范围 | `2020-2025` |
| `--keyword "high:..."` | 核心关键词 | `"high:open world agent"` |
| `--top-k` | 返回结果数量 | `3`、`5`、`10` |
| `--report-max-items` | 报告展示数量 | `3`、`5`、`10` |

改写命令时，优先调整三处：

1. `--query`：替换为检索主题。
2. `--keyword "high:..."`：替换为最核心的关键词。
3. `--time-range`：替换为关注的年份范围。

## 10. 常用任务模板

以下模板可直接复制使用。引号中的主题、领域、关键词或作者姓名按实际需求替换。

### 10.1 检索某个主题的相关论文

```bash
sciatlas --timeout 900 search-papers --query "研究主题" --domain "领域名称" --time-range 2020-2025 --keyword "high:核心关键词" --top-k 5 --top-keywords 0 --max-titles 0 --max-refs 0 --report-max-items 5
```

适用场景：快速获得某个主题的相关论文列表。

### 10.2 生成文献综述起步材料

```bash
sciatlas --timeout 900 literature-review --workflow flash --query "研究主题" --domain "领域名称" --time-range 2020-2025 --keyword "high:核心关键词" --top-k 10 --report-max-items 8
```

适用场景：准备 related work、开题报告、阅读清单或综述初稿。默认建议使用 `flash`；需要更完整的多轮检索、证据包和正式综述草稿时改为 `--workflow full`。

### 10.3 评估一个研究想法

```bash
sciatlas --timeout 900 idea-evaluate --workflow flash --idea "一句话研究想法" --domain "领域名称" --time-range 2020-2025 --keyword "high:核心关键词" --top-k 8 --report-max-items 8
```

适用场景：检查研究想法的新颖性、可行性、相关工作和差异化空间。默认建议使用 `flash`；需要更完整的 reviewer、rubric、grounding、evidence 和 report 路径时改为 `--workflow full`。

### 10.4 为研究想法寻找相关工作

```bash
sciatlas --timeout 900 idea-grounding --idea "一句话研究想法" --domain "领域名称" --keyword "high:核心关键词" --top-k 8 --report-max-items 8
```

适用场景：为已有想法寻找 prior work、动机支撑和差异化证据。

### 10.5 生成研究想法线索

```bash
sciatlas --timeout 900 idea-generate --query "研究方向" --domain "领域名称" --workflow flash
```

```bash
sciatlas --timeout 900 idea-generate --query "研究方向" --domain "领域名称" --workflow full --top-k 5
```

适用场景：运行当前 `sciatlas_idea_gen` 多步 workflow，围绕一个方向构建研究图、检索灵感、生成新想法并检查新颖性。`flash` 会压缩 gate / selection 阶段，适合快速交互；`full` 适合更完整的证据构建。

### 10.5.1 `flash` 与 `full` 的选择

| Workflow | `flash` | `full` |
|---|---|---|
| `literature-review` | 快速形成阅读清单、outline 或 evidence packs。 | 更完整的多轮检索、section packs 和正式综述生成。 |
| `idea-evaluate` | 快速得到新颖性、可行性、主要风险和修改方向。 | 更完整的 reviewer/rubric/grounding/evidence/report 自动评审链路。 |
| `idea-generate` | 快速生成少量有文献依据的 idea seeds。 | 更大的研究图、更广的灵感检索和更完整的新颖性反馈。 |

### 10.6 分析主题发展趋势

```bash
sciatlas --timeout 900 trend-report --query "研究主题" --domain "领域名称" --time-range 2018-2025 --keyword "high:核心关键词" --top-k 10 --report-max-items 10
```

适用场景：梳理一个方向的时间线、代表论文和发展趋势。

### 10.7 整理研究者背景

```bash
sciatlas --timeout 900 researcher-review --author "研究者姓名" --limit 10 --report-max-items 10
```

适用场景：了解某位研究者的代表工作和研究方向。

## 11. 可选：使用 Agent Skill

SciAtlas 还提供了 `agent-skill/` 目录，用于让支持 `SKILL.md` 的 Agent 工具理解 SciAtlas 的使用流程。常见 Agent 工具包括 Codex、Claude Code 等。

通俗理解：

- CLI 是实际执行检索的工具，负责调用 SciAtlas 后端并生成 `runs/` 结果文件。
- Agent Skill 是给 Agent 阅读的操作说明，目标是让新手用户从零开始得到最终下游结果：Agent 负责安装或定位 CLI、引导注册、配置环境变量、运行稳定检索命令或当前专用 workflow、读取 `runs/` 中的报告和 JSON 文件，并把结果整理成更自然的回答。
- 用户只需要提供 Agent 无法代办的人类输入，例如邮箱、验证码、SciAtlas API Token、LLM/S2/KG 密钥，或一次必要的任务范围澄清。不要把这些密钥写进 `agent-skill/` 目录。

### 11.1 Agent Skill 的安装

一键安装脚本会把完整 SciAtlas 仓库下载到本地，因此仓库中会包含 `agent-skill/` 目录。但它不会自动复制到 Codex 或其它 Agent 工具的技能目录中。



### 11.2 什么时候适合使用 Agent Skill

Agent Skill 适合这些情况：

| 场景 | 推荐 skill |
|---|---|
| 快速查几篇相关论文 | `sciatlas-quick-paper-search` |
| 生成文献综述材料 | `sciatlas-literature-review` |
| 判断研究想法是否已有类似工作 | `sciatlas-idea-grounding` |
| 评估研究想法的新颖性和可行性 | `sciatlas-idea-evaluate` |
| 从一个方向运行多步 idea-generation workflow | `sciatlas-idea-generate` |
| 梳理研究趋势 | `sciatlas-trend-report` |
| 整理研究者画像 | `sciatlas-researcher-review` |

#### Agent Skill 的固定执行边界

下表描述的是 Agent Skill 的约束，不是普通 CLI 的命令清单。安装 Skill 后，Agent 必须按对应边界执行并读取保存的 artifacts；不能因为某个 CLI 子命令可用，就替换成别的检索或工作流。

| Agent Skill | 允许的执行入口 | 交付重点 |
|---|---|---|
| `sciatlas-quick-paper-search` | 仅 `sciatlas search-papers` | 3 篇左右的证据表，并推荐下一步 Skill |
| `sciatlas-literature-review` | 仅 `sciatlas literature-review` 或 `python run_sciatlas.py literature-review` | 综述大纲、论文地图、证据包或正式综述 |
| `sciatlas-idea-grounding` | 仅 `sciatlas search-papers` | 前序工作矩阵、差异化风险和下一轮检索 |
| `sciatlas-idea-evaluate` | 仅 `sciatlas idea-evaluate` 或 `python run_sciatlas.py idea-evaluate` | go/revise/no-go 建议、rubric 与 reviewer 证据 |
| `sciatlas-idea-generate` | 仅 `python -m sciatlas_idea_gen.main` | 有文献依据的 idea seeds、验证实验与新颖性风险 |
| `sciatlas-trend-report` | 仅 `sciatlas search-papers` | 按阶段组织的时间线、代表论文与新兴信号 |
| `sciatlas-researcher-review` | 仅 `sciatlas search-papers` | 基于检索证据的研究者画像；不应当当作完整履历或权威 CV |

三条专用 workflow Skill 使用同一个 Skill 目录，默认从 `flash` 开始。只有用户要求更广覆盖，或 `flash` 的 artifacts 对所需深度不足时，Agent 才传入 `--workflow full`。`literature-review-full`、`idea-evaluate-full` 和 `idea-generate-full` 是 `sciatlas skill run` 的 CLI JSON 预设名，不是要复制安装的 Agent Skill 目录。

如果只是想自己复制命令跑一次，直接使用第 10 节的 CLI 模板即可。如果希望 Agent 根据自然语言需求自动选择命令、运行检索、读取结果并总结，就适合使用 Agent Skill。

### 11.3 安装到 Codex 的常见方式

以下示例以 Codex 为例。其它 Agent 工具的技能目录可能不同，按对应工具文档调整即可。

Windows PowerShell：

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.codex\skills" | Out-Null
Copy-Item -Recurse ".\agent-skill\sciatlas-literature-review" "$env:USERPROFILE\.codex\skills\"
```

macOS / Linux：

```bash
mkdir -p ~/.codex/skills
cp -R ./agent-skill/sciatlas-literature-review ~/.codex/skills/
```

复制完成后，重启或刷新 Agent 工具，使其重新读取技能目录。

### 11.4 多个 Agent Skill 的安装方式

如果需要同时安装多个 skill，可以按需复制多个目录。

Windows PowerShell：

```powershell
Copy-Item -Recurse ".\agent-skill\sciatlas-quick-paper-search" "$env:USERPROFILE\.codex\skills\"
Copy-Item -Recurse ".\agent-skill\sciatlas-literature-review" "$env:USERPROFILE\.codex\skills\"
Copy-Item -Recurse ".\agent-skill\sciatlas-idea-evaluate" "$env:USERPROFILE\.codex\skills\"
```

macOS / Linux：

```bash
cp -R ./agent-skill/sciatlas-quick-paper-search ~/.codex/skills/
cp -R ./agent-skill/sciatlas-literature-review ~/.codex/skills/
cp -R ./agent-skill/sciatlas-idea-evaluate ~/.codex/skills/
```

### 11.5 使用时怎么表达需求

Agent Skill 安装完成后，在 Agent 对话中直接描述科研任务即可。示例：

```text
请使用 SciAtlas 帮我查找 open world agent 方向的相关论文，返回 5 篇代表性工作，并说明这些论文分别解决了什么问题。
```

```text
请使用 SciAtlas 帮我评估这个研究想法是否值得继续做：用大语言模型进行科研 idea 的多角度评估。
```

```text
请使用 SciAtlas 帮我整理 Yoshua Bengio 的研究背景和代表论文。
```

Agent 正常工作时，会先检查是否已安装 `sciatlas`；如果缺少 CLI、Token 或 workflow 所需的 LLM/S2/KG 配置，Agent 应尽量完成可自动化的安装与配置，并只向用户索取邮箱、验证码、Token、密钥等必要信息。随后 Agent 运行 SciAtlas 检索命令或当前专用 workflow，读取 `runs/` 目录下的 `summary.txt`、`report.md`、`request.json`、`response.json` 或 workflow artifacts，最后给出面向阅读的下游结果，而不是只给命令模板。

### 11.6 使用 Agent Skill 的注意事项

- 不要把 API Token 写进 `agent-skill/` 目录，也不要提交到公开仓库。
- 快速检索、grounding、趋势和研究者画像类 Agent Skill 只允许使用 `search-papers`；文献综述、自动评审和 idea 生成类 Agent Skill 只调用各自当前 workflow。专用 workflow 先用 `flash`，仅在用户要求更深覆盖或 flash 证据不足时切换到 `full`；不要寻找或安装不存在的 `*-full` Agent Skill 目录。
- 如果 Agent 提示找不到 `sciatlas` 命令，应优先让 Agent 按第 3 节安装或定位 CLI。
- 如果 Agent 提示 Token 缺失，应让 Agent 引导注册并配置 `SCIATLAS_API_KEY`；用户只需要提供邮箱、验证码和返回的 Token。
- 如果结果文件不存在，检查 `runs/` 目录是否生成了新的运行文件夹。

## 12. 结果调优

### 返回结果太少

增加返回数量：

```bash
--top-k 10 --report-max-items 10
```

### 返回结果太散

增加更明确的强关键词：

```bash
--keyword "high:核心关键词"
```

补充领域和年份范围：

```bash
--domain "artificial intelligence" --time-range 2020-2025
```

### 只想快速试跑

降低返回数量：

```bash
--top-k 3 --report-max-items 3
```

减少自动扩展：

```bash
--top-keywords 0 --max-titles 0 --max-refs 0
```

## 13. 常见问题

### 13.1 终端提示 `sciatlas` 不是内部或外部命令

常见原因是安装后终端环境尚未刷新。

处理方式：

1. 关闭当前终端。
2. 重新打开 PowerShell 或 Terminal。
3. 重新执行：

```bash
sciatlas -h
```

若仍无法识别，可重新运行安装命令。

### 13.2 提示 401、Missing token 或 Unauthorized

通常表示 Token 未配置或配置不正确。重新执行第 5 节配置命令，并确认 `SCIATLAS_API_KEY` 是完整的 `sciatlas_xxx` Token。

### 13.3 注册后未收到验证码

先检查垃圾邮件。若仍未收到，可更换邮箱或稍后重试。

### 13.4 命令运行较慢

可先降低返回数量：

```bash
--top-k 3 --report-max-items 3
```

复杂检索建议保留：

```bash
--timeout 900
```

### 13.5 找不到报告文件

查看 `runs` 目录下最新生成的文件夹，优先打开其中的 `report.md`。

### 13.6 中文主题是否可用

中文主题可以使用，但英文关键词通常更适合学术论文检索。可采用“中文主题 + 英文核心关键词”的方式。

示例：

```bash
sciatlas --timeout 900 search-papers --query "大语言模型用于科研想法评估" --domain "artificial intelligence" --keyword "high:LLM idea evaluation" --top-k 5
```

## 14. 更多文档

完整 API 与 CLI 文档：

```text
http://sciatlas.openkg.cn/api/docs/?lang=zh
```

面向开发者和工具包装的说明：

```text
docs/sciatlas_skill_packaging_handoff_zh.md
```

## 15. 最简流程

1. 安装 SciAtlas。
2. 访问 `http://sciatlas.openkg.cn/register` 获取 Token。
3. 设置 `SCIATLAS_API_KEY`。
4. 运行 `sciatlas search-papers ...`。
5. 打开 `runs` 目录中的 `report.md` 查看结果。
6. 可选：将 `agent-skill/` 中需要的目录复制到 Agent 工具的技能目录，并重启 Agent 工具。
