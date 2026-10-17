# 父子块对照实验运行说明

实验由用户自行运行。第一轮已完成，结果在 `outputs/printer_retrieval_comparison/manual_run_v1`；本页命令和输入描述对应第一轮。第二轮已准备脚注补全、功能核验及缓存复用，见 [第二轮运行说明](ParentChildEvidenceRepair-zh.md)。旧知识库、基线和首轮结果保留。

## 当前输入

- 旧版 407 个块：`rag_storage/learning_paratera/kv_store_text_chunks.json`。
- 新版 1981 个子块及父块：`outputs/printer_parent_child/20261017_v2/A`、`B`、`C`。
- 36 道冻结测试题：`examples/parent_blocks/printer_retrieval_cases.jsonl`，完整保留原基线题库，包含 30 道检索题、6 道边界题；旁边的 manifest 记录来源与 SHA-256。只将 query 字符串作为查询，参考要点和评分引文不进入文档索引或 embedding 输入。不要把这份测试文件作为知识文档上传。
- 本地分词规则：`tmp/parent_blocks_tools/cl100k_base.tiktoken`。文件已经准备在当前工作区，不随 Git 分发；迁移实验时需一并复制。

5 个超长表格已按完整行拆成 11 个片段，每片保留经过验证的表头。正文均不超过 500 个 cl100k token；索引文本附加章节、型号及必要脚注，因此另记 retrieval_tokens。本文的 500 不是实际 embedding 服务的硬上限。

## 先预检

在仓库根目录的 PowerShell 执行：

```powershell
.\examples\parent_blocks\run_printer_retrieval_comparison.ps1
```

默认仅检查文件、配置、哈希和本地 token 数，写入 `experiment.json`，不会调用 embedding API。默认目录为 `outputs/printer_retrieval_comparison/manual_run_v1`；此前由代理创建的 `20261017_v1` 是旧预检记录，不要拿它恢复新版实验。

脚本优先使用项目 `.venv\Scripts\python.exe`；如需使用当前激活环境，可加 `-PythonExecutable python`。使用项目现有依赖，不自动安装或重建环境。

配置读取 `.env`，进程环境变量优先，使用 `EMBEDDING_BINDING=openai` 对应的兼容服务，以及 `EMBEDDING_MODEL`、`EMBEDDING_BINDING_HOST`、`EMBEDDING_BINDING_API_KEY`、`EMBEDDING_DIM`。不输出或保存 API 密钥。`EMBEDDING_SEND_DIM` 与服务端默认一致，为 false 时不发送 dimensions 字段；收到的向量维度始终校验。

当前对照脚本只实现对称 embedding 输入；如果显式启用了 `EMBEDDING_ASYMMETRIC_ENABLED`，预检会拒绝运行，避免混用文档/查询角色的缓存。

## 正式运行

预检后，对同一目录执行：

```powershell
.\examples\parent_blocks\run_printer_retrieval_comparison.ps1 -RunExperiment -Resume
```

这一条才会调用现有 embedding API；不需要启动 LightRAG 或 Docling 服务，不调用生成模型。旧块、新块和问题全部使用同一 embedding 配置，旧块重新计算到独立实验缓存，不改写旧向量库。

当前预估去重后约 2411 条输入，本地 cl100k 统计约 83 万 token；实际 API 计费以服务商记录为准。默认 batch size 为 16。所有配置、输入哈希、模型和 token 数以你执行后的预检计划为准。

如果预检和正式运行期间改了模型、来源文件、top-k 或上下文预算，恢复会被拒绝，应使用新的目录。例如：

```powershell
.\examples\parent_blocks\run_printer_retrieval_comparison.ps1 `
  -OutputDirectory 'outputs/printer_retrieval_comparison/manual_run_v2'
```

网络中断或 Ctrl+C 后，使用相同正式运行命令恢复，完整缓存的向量不会重复请求。没有完成的请求可能需要重试。不要同时运行同一实验目录；异常强杀遗留 `embedding.lock` 时，应先确认原进程停止，再移除该锁。缓存损坏会报错，不会悄悄丢弃并重新付费计算。

## 对照设置与输出

固定同一问题、模型、top-k（默认 5）和上下文预算（默认 8000 cl100k token），比较四个方案：

本实验直接使用归一化向量的余弦相似度排序，不调用 LightRAG 问答 API，不运行 reranker，也不复制线上查询流程中的所有阈值与调度逻辑。

| 方案 | 内容 |
| --- | --- |
| legacy_raw | 旧版块直接向量召回 |
| legacy_model_filter | 旧版增加文件名型号过滤 |
| child_raw | 新子块向量召回后回取父块与前置说明 |
| child_model_filter | 新版增加当前父块型号范围过滤，再回取父块 |

型号过滤仅使用问题明确写出的型号，不利用评分标签选择候选。没有自动实现领域路由或所有操作系统条件识别。多型号问题采用候选适用型号与问题型号相交的规则，逐题结果保留来源，便于检查是否漏掉某个型号的证据。

完整父块与必要前置说明作为一组纳入预算；放不下就跳过整组，不截断警告或条件。原始命中片段和父块回取后的引文覆盖分别报告。

| 文件 | 含义 |
| --- | --- |
| `experiment.json` | 冻结的配置、输入哈希、向量任务身份 |
| `embedding_cache.jsonl` | 可恢复的向量缓存 |
| `embedding_requests.jsonl` | 成功请求的输入数和服务端 usage；不含密钥 |
| `results.jsonl` | 36 题 × 4 方案的命中、相似度、证据与预算记录 |
| `report.md` | 可读的汇总及逐题对照 |
| `summary.json` | 最后写入的结果摘要及累计请求记录 |

自动指标是“正确文档来源中的规范化原文引文覆盖”，不是答案正确率；格式差异、同义改写或只召回部分依据可能造成漏记，需复核关键题。6 道边界题不纳入普通召回汇总，因为本轮不生成答案或测试追问。

新旧方案同时改变解析、切分和元数据，因此实验不能把所有差异都归因于父块回取。没有重新评估图谱模式或证明图谱歧义已解决。尤其应人工查看 Q25/Q26：当前文件名默认策略不会仅因 ADF/传真功能词就排除某型号，是否取到了明确的限制说明需要从结果核实。

## 自动化测试

以下命令只运行使用假向量和 mock HTTP 的测试，不会调用真实 API：

```powershell
.\.venv\Scripts\python.exe -m pytest tests/document_structure tests/evaluation -q
```

表格 header override 及证据位于 `examples/parent_blocks/printer_table_headers.json`；原四张表直接使用 sidecar 的表头，第五张表的首行已经对照原 PDF 第 202 页核实。表格行、重复表头、脚注和源字符区间均有回归检查。

2026-10-17 本轮本地验证：`tests/document_structure` 与 `tests/evaluation` 共 174 项通过，Ruff 格式和静态检查通过。PowerShell 入口已使用假 Python 程序检查默认参数及显式运行/恢复参数，没有执行真实实验。
