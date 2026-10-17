# 父子块证据修复与第二轮实验

第一轮结果保留在 `outputs/printer_retrieval_comparison/manual_run_v1`，平均引文覆盖为：旧版无过滤 59.67%、旧版型号过滤 82.78%、新版无过滤 85.22%、新版型号过滤 87.22%。这不是问答准确率。本轮先修复已确认的脚注遗漏与功能前提核验；不接入上传界面、不修改线上知识库、不调用生成模型。

## 表格脚注

原 `*.tables.json` 中的 `footnotes` 现在按表格 ID 绑定到父块的 `table_notes`。所有实际引用该表格的子块，包括小表格、混合正文表格和按行切片，都继承这些脚注并附加到 embedding 文本。没有该表格的普通子块不携带它的脚注。

父块 `content` 和既有 `source_spans` 保持原样。附加脚注记录表格 ID、原文、来源文件 SHA-256、JSON Pointer、文档身份，不伪造原始正文块中的偏移或精确页码。回取结果中的 `evidence_content` 是正文加脚注；消费者必须使用这个字段或 `table_notes.evidence_text(parent)` 构造证据。只读取 `content` 仍只能看到原始正文。

新草稿读取时验证产物哈希、表格来源文件哈希，并从来源重新构造预期脚注，拒绝脚注丢失、篡改或错误归属。脚注不按数字星号猜测关联；保守保留该表的全部脚注，避免漏掉不带编号的 `*`。

这修复 C 手册错误代码表的寿命说明遗漏。它没有凭空补充原始解析产物中也不存在的内容。

## 功能适用性核验

父块型号范围规则保持不变：文件名默认，只有整段明确限制才缩小。新 `CapabilityIndex` 独立读取类似原文：

```text
以下功能仅适用于 L5298。
O 传真
O ADF（自动文稿送纸器）
O 以太网连接
```

程序只识别当前已实现的“以下功能仅/只适用于型号”声明和其后的连续列表，功能词表覆盖传真、ADF、以太网及明确别名。声明型号必须属于文件名所列型号。每条结果保留原文、父块范围、原始结构块范围与位置。没有相应声明或不能识别的写法返回 `unknown`，不能从功能词或缺少说明推断不支持；冲突声明返回 `conflict`。

`ParentChildIndex.resolve(..., required_features=['adf'])` 对目标型号核验该功能。明确不支持或声明冲突时不返回操作父块，但在 `capability_evidence` 中保留原文，供后续解释前提。没有传 `required_features` 的旧接口仍按原有型号/平台/领域规则工作。未知状态有 `requires_review`，不代表功能已经确认可用。

第二轮对照的新版过滤组根据问题中明确出现的型号、功能词触发核验，将声明原文优先纳入证据预算。限制声明作为解释证据，不会因为它描述 L5298 就从 L3268 问题中全部过滤掉。候选操作仍需要通过型号检查。该版本是“向量召回 + 型号过滤 + 明确功能核验 + 父块/脚注回取”，并非仅比较切块。

这是有限文字规则的核验，不是通用自然语言能力理解；不包含软件版本、全部平台条件、否定意图分析或完整故障诊断。Q19 的跨章节步骤补齐及 Q30 的多型号分别召回仍是后续工作。

## 本地产物与检查

新草稿位于 `outputs/printer_parent_child/20261017_v3/{A,B,C}`，仍为草稿。历史 v1/v2 与第一轮实验保留。子块数仍为 A 598、B 764、C 619，共 1981；内容偏移和父块范围不因补充元数据改变，子块版本/ID 更新。

可仅运行不联网的来源与功能核验检查（输出需使用新文件名）：

```powershell
.\.venv\Scripts\python.exe -m lightrag.evaluation.check_parent_child_evidence `
  --bundle outputs/printer_parent_child/20261017_v3/A `
  --bundle outputs/printer_parent_child/20261017_v3/B `
  --bundle outputs/printer_parent_child/20261017_v3/C `
  --questions examples/parent_blocks/printer_retrieval_cases.jsonl `
  --output outputs/printer_parent_child/20261017_v3/evidence_audit_user.json
```

它只检查语料是否保留引文和确定性功能拦截，不计算向量、不证明实际检索准确率；题库引文仅用于离线核对，不进入索引。

## 用户自行运行第二轮

预检：

```powershell
Set-Location D:\GitHubCode\LightRAG
.\examples\parent_blocks\run_printer_evidence_repair.ps1 -Resume
```

默认输出新目录 `outputs/printer_retrieval_comparison/manual_run_v2`，使用 v3 草稿，并读取第一轮 `manual_run_v1` 的向量缓存。预检只读旧缓存，不请求 API，打印 `reusable_inputs` 和 `remaining_embedding_inputs`。同配置、同文本的向量可以复用，改变的文本需要新增计算。复用同时检查 provider URL、模型、维度、dimensions 发送策略和向量有效性，配置不一致时拒绝。

实际实验由用户执行：

```powershell
.\examples\parent_blocks\run_printer_evidence_repair.ps1 -RunExperiment -Resume
```

仅为缺失输入请求 embedding，不调用生成模型。旧向量库和旧实验不写入；新目录中的 `embedding_reuse.json` 记录缓存来源、哈希与导入键，`embedding_requests.jsonl` 只记录本目录发生的新 API 请求。第一次导入后中断，恢复时直接复用新目录已落盘向量。更换任何计划输入或配置需要新输出目录，不能在第一轮目录上恢复新版本。

结果仍输出 `report.md`、`results.jsonl`、`summary.json`。逐题记录增加 `capability_checks`；证据中明确标识 `capability_declaration`。不能通过补进限制声明就宣称问题答案已正确生成，也不能据此证明全部条件冲突已解决。

当前本地预检：2411 条唯一输入中，2408 条可复用第一轮缓存，只有 3 条文本变化需新增 embedding。这是当前文件和配置下的预检值，实际运行以脚本打印为准。现有 `manual_run_v2` 仅有预检计划，没有第二轮检索结果；正式运行需带 `-Resume`。

本地确定性核查：表格来源脚注 B/C 各 1 条，Q16 的整机寿命说明从正文缺失变为附加证据可用；功能声明提取为 3 条，L3266 传真与 L3268 ADF 均判为不支持，操作候选被拒绝且限制原文保留。该检查不等于实际向量召回验收。

## 回归检查

```powershell
.\.venv\Scripts\python.exe -m pytest tests/document_structure tests/evaluation -q
```

测试使用本地假分词器、假向量和 mock API，检查脚注的来源与回取、排他声明、冲突/未知、功能步骤拦截、跨实验缓存复用及变更文本新增请求。真实检索实验仍由用户运行。

2026-10-17 验证：`tests/document_structure` 与 `tests/evaluation` 共 201 项通过，Ruff 检查通过；B 手册原有 7 项真实父子映射检查全部通过。新 PowerShell 入口已执行无 API 的真实预检。
