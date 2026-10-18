# 已有完整子块的小样本真实入库

入口读取 `outputs/printer_parent_child/20261017_v3/{A,B,C}/children.jsonl` 与父块包。默认各选 4 个完整子块，共 12 个。优先覆盖网络/显示屏代码、W-11 和明确型号限制，候选不足时补充诊断/规格类子块。实际选择及完整原始记录保存在 `selected_children.json`，可以先看再决定执行。

不使用旧六段提示词实验材料，不裁剪表格行，不重新切分 PDF 或子块，不调用 Docling。子块必要表头与脚注沿用已有保存内容；模型看到的是完整子块及其必要说明。

## 三份文档全量入库

全量入口复用本文的同一入库程序，不创建另一个知识库。读取三份保存包的全部 **1981 个子块**（A：598，B：764，C：619），保留全部 **710 个父块**（197／291／222）、原始文档 ID、文件名、适用范围和父子来源映射。使用 `printer_v2` 业务抽取、现有图谱与向量存储，不重新解析、切分或调用 Docling。

在项目根目录预检（无 API 调用或知识库写入）：

```powershell
.\examples\parent_blocks\run_printer_full_ingestion.ps1
```

确认输出 `children=1981`、`workspace=learning_paratera`。停止本地 LightRAG 后端和其他同工作区写入脚本，然后执行：

```powershell
.\examples\parent_blocks\run_printer_full_ingestion.ps1 -Run
```

此命令会真实调用现有 EXTRACT 和 embedding API。全量 PowerShell 入口默认最多并发 4 个抽取请求，每个请求只处理一个完整子块；文档写入仍为单进程、逐文档处理。一轮最多为 1981 个子块发起新抽取，每块遇到临时请求故障最多尝试 3 次，另有子块、实体、关系的 embedding 请求；不生成问答，不调用额外通用摘要模型。不修改 `.env`。保存每次请求、原始响应、向量用量和 `progress.json`；终端也会显示每份文档的子块抽取及图谱合并进度。

通过 `-ExtractConcurrency` 选择 1–16 个并发请求，建议先用 4。例如恢复已有任务：

```powershell
.\examples\parent_blocks\run_printer_full_ingestion.ps1 -Run -Resume -ExtractConcurrency 4
```

并发数是执行参数，允许在同一计划的恢复中调整，不改变输入、提示、文档身份或缓存键；实际参数单独写入 `execution.json`／`resume_..._execution.json` 和最终报告。并发请求可以乱序返回，每个请求在发送前分配固定编号，原始响应与对应请求分别保存，不混合来源。向量调用仍为单并发、每批 10 条。不要同时启动多个入库脚本；HTTP 请求并发不等于多个知识库写入进程。若服务返回 429，可将并发降为 2 后恢复；实际提速受服务端限流和生成速度影响。

已经运行的进程不会动态修改并发。需要切换时按一次 Ctrl+C，等待当前脚本完全退出、释放目录锁后再用上面的恢复命令，不要另开第二个写入进程。已有正常缓存继续复用。小样本脚本与直接 Python 入口的默认并发仍为 1。

逐块进度示例：`Extraction progress: document B 10/764 chunks; overall 608/1981 chunks (30.7%); skipped 2 chunks`。这里 A 的 598 块、B 的 10 块已审查，审查数包含跳过的块和缓存结果，不把 API 请求数当作块数。单份文档抽取结束后仍需图谱合并和向量索引；最终日志区分 `Ingestion completed` 与 `Ingestion completed_with_skips`。`progress.json` 保存 `chunks_reviewed`、`chunks_total`、`skipped_chunks`、各文档进度和 `stage`，调用用量继续放在 `calls` 中。已经运行的旧进程不会自动加载新日志代码，不必仅为日志而中断任务。

进度快照使用每次唯一的临时文件，关闭句柄后原子替换。Windows 文件被短暂占用（错误码 5、32、33）时，在 0.05／0.1／0.2／0.4／0.8 秒后有限重试，不发起额外模型请求。持续失败仍报错，保留上一版快照及唯一的 `.tmp` 文件供排查；不能因报告失败将文档误报成功。进度报告不是恢复检查点，恢复依据仍是原生文档状态与可用抽取缓存，不读取遗留临时文件。无需删除原报告或知识库。

全量三份文档都达到 `processed`、子块完整性校验通过、父块数量一致且没有整块跳过时，才按原生删除流程清理对应小样本运行的三份 `child_smoke_...` 文档，避免重复参与检索。原始 PDF、本地父子块包、小样本响应和实验输出文件均保留。文档失败或存在跳过块时保留小样本，避免移除已有有效图谱；报告用 `sample_cleanup_deferred` 说明。需要保留小样本时，预检、运行和恢复都添加 `-KeepSampleDocuments`。

结果保存在 `outputs/printer_full_ingestion/saved_children_v1/`；这只是运行报告目录，实际数据仍写入 `.env` 指定的 `rag_storage/learning_paratera`。查看 `summary.json`：A／B／C 的 `stored_children` 应为 598／764／619，`stored_parents` 应为 197／291／222，`intact_saved_children` 均为 `true`。运行完成并退出后，按平时方式重新启动后端，即可检索完整三份文档。

失败或中断后保留原目录，先排查报错，停止其他写入，再用同一计划恢复：

```powershell
.\examples\parent_blocks\run_printer_full_ingestion.ps1 -Run -Resume
```

恢复会跳过已经处理成功的文档，复用可用的原生抽取缓存；未完成的本次文档也可由原生管线恢复。其他文档的排队或失败记录会使脚本拒绝启动。缺失或未持久化的响应、重建时的向量仍可能产生 API 用量；不保证断电前的每次调用都能复用。恢复报告使用 `resume_..._summary.json` 等新文件名，不覆盖原始响应。

抽取请求显式传递 `response_format={"type":"json_object"}`。JSON 损坏、必要字段缺失或输出截断时，跳过该子块的实体关系入图，继续处理其他子块；不补空字段来冒充正确输出，也不在本轮自动重试。原文和父子映射保留，原文仍可向量化并参与证据召回。原始回答和失败原因保存在 `business_extraction_attempt`、`business_extraction.review` 中；恢复时保留可用及格式错误的缓存，按同一规则审查并跳过，无需为已有的损坏回答再次付费。

临时请求故障（连接/超时、HTTP 408、429、5xx）先等待 2 秒、4 秒重试，共最多 3 次。仍失败时，只跳过该子块的图谱抽取并继续后续子块。错误保存在 `request_编号_attempt_编号_error.json`，源块的 `business_extraction_attempt` 明确保存 `raw_response=null` 和 `request_failure`，不会用空 JSON 冒充成功或把网络失败写入抽取缓存。`calls.extraction_requests` 统计未命中缓存的子块调用数，`calls.extraction_api_attempts` 统计含重试的实际 API 尝试数。重试策略记入执行报告，不改变原冻结计划、提示和缓存身份。请求超时后服务端仍可能已经计费，重试不保证免费。

`skipped_chunks.json`（恢复时为 `resume_..._skipped_chunks.json`）逐条记录文档、子块、父块、标题和错误。`summary.json` 记录各文档及总计 `skipped_extraction_chunks`；有跳过时 `completed_with_skips=true`。这表示剩余合格数据已入库，不代表每个子块都产生了图谱结果。单条实体/关系错误仍按记录隔离；鉴权、请求配置、TLS 证书、来源归属、合同冲突、存储及未分类的程序错误仍导致失败，不伪装为正常跳过。错误收据或来源证据保存失败也必须报错。

已成功入库但带跳过块的文档不会由 `-Resume` 自动再跑；保留缺口记录供后续明确补提取。若整篇文档因其他故障重试，临时请求失败没有成功缓存，相关块可以重新请求，已有正常响应仍复用。持续的服务故障可能产生大量跳过块，运行结束必须检查 `skipped_chunks.json`，不能只看文档 `processed` 状态。

向量调用同样对连接超时、响应中途断开（`IncompleteRead`）、SSL 提前关闭以及 HTTP 408／429／5xx 做有限重试。保持原批次文本和顺序，在 1／2／4 秒后重试，总共最多 4 次，单次 HTTP 超时为 60 秒；入库队列的执行超时留足整组重试时间。鉴权、证书、向量数量/维度/数值错误不重试，向量持续失败也不跳过或写零向量，而是保留原生失败/排队状态，等明确恢复。执行报告保存此策略。`embedding_requests` 统计程序调用的批次数，内部 HTTP 重试未单独计数，费用以平台账单为准。

索引失败后，其他已经排队的向量批次可能继续打印成功日志，最终是否完成须看文档 `processed` 和整轮汇总。恢复跳过已完成文档，失败文档沿用抽取缓存重建索引；未提交或未保留的向量可能再次生成并收费，当前入库入口没有逐条向量响应的持久缓存。无需重新解析 PDF，也不要删除知识库或原报告。

本次沿用当前已验证的抽取合同：超限、端点或引用不合格的记录仍会被隔离并保存原因。全量指所有保存子块都进入处理流程，`processed` 不代表每一条事实均已提取或语义正确。

## 运行

先在项目根目录执行预检，无 API 调用，不连接或写入知识库：

```powershell
Set-Location D:\GitHubCode\LightRAG
.\examples\parent_blocks\run_printer_child_ingestion.ps1
```

输出应显示当前 `working_dir`、`workspace=learning_paratera`、`children=12` 和 `parsing_requests=0`。配置读取现有 `.env`，本入口显式使用 `printer_v2`，不需要再修改默认抽取方案。

确认选中的子块后，停止本地 LightRAG 后端及同工作区的其他写入脚本，再执行：

```powershell
.\examples\parent_blocks\run_printer_child_ingestion.ps1 -Run
```

当前使用 NetworkX，同库测试需要单写入，见 [单写入契约](design/NetworkXSingleWriterContract.md)。脚本会检查常用后端端口和已排队/未完成文档；发现已有队列就拒绝启动，不为测试顺带处理原三份完整文档。

`-Run` 会写入当前知识库，并使用已有 EXTRACT 和 embedding 配置。新抽取最多覆盖选中子块数量，每块临时请求故障最多尝试 3 次，另有子块、实体、关系的向量调用；不执行问题关键词抽取或答案生成。图谱合并阶段不允许额外摘要模型请求。需要更小批次时，预检和执行都使用 `-PerDocument 2 -OutputDirectory outputs/printer_child_ingestion/saved_children_small_v1`，这会选 6 个完整子块，与过去的六段投影材料无关。

## 如何进入现有系统

通过既有 C 自定义块入口提交已经保存的块；该适配器不执行 C 或 H 切分算法。后续仍走正常排队、业务抽取、记录校验、身份转换、恢复锚点、图和向量入库。保存父块及来源映射，使 H 的父块回查可用。

测试使用三个可识别的 `child_smoke_...` 文档名，在当前工作区写入少量数据，不新建知识库目录或工作区，不覆盖原三份手册。测试记录拥有新的文档 ID，原文档 ID 另存为 `source_doc_id`；原始 child_id、parent_id、范围、坐标和投影信息均保留。这意味着本次可检查测试文档内部的场景消歧及跨文档隔离，但不修改原文档的已有图谱。

## 只恢复失败测试文档

若已有运行因模型输出超过条数上限失败，保留原目录，不重新运行整批。原生入库现在对超出的记录逐条隔离：实体仍只接纳原顺序前 12 条，关系只接纳前 16 条，且须通过全部既有类型和引用检查。关系引用被隔离实体时也会被拒绝。不根据后续记录补齐或伪造端点，不改写原始响应。实验入口的整份严格检查保持不变。

停止同工作区写入进程后，先预检恢复计划，再实际执行：

```powershell
.\examples\parent_blocks\run_printer_child_ingestion.ps1 -ResumeFailed
.\examples\parent_blocks\run_printer_child_ingestion.ps1 -Run -ResumeFailed
```

恢复沿用原 plan_hash 和文档 ID，校验样本、模型、范围和存储目标未变化；只允许带审计的实现修复。成功文档不重新排队，失败文档通过原生人工重试信箱恢复，复用可用的抽取缓存。若工作区还有其他失败或未完成文档，则拒绝批量重试，避免影响无关文档。没有可确认归属的响应不会按编号猜测复用，缺失缓存的块仍需要新抽取。

原始计划、summary、响应和导出文件不覆盖。恢复结果写成 `resume_<时间>_<标识>_summary.json` 等文件，同时保存本次实现变更审计；新的请求另存提示指纹和输入，方便定位响应归属。超限隔离让流程可以完成，但不表示事实覆盖完整，仍需人工核对缺失内容。

为了复现，计划冻结选中子块、源指纹、模型、存储目标和实现指纹。计划变化时要求新输出目录。已开始的同一测试不自动重跑或升级失败文档；中途失败先检查保存结果，不要通过换目录反复付费运行。

## 看哪些结果

默认输出目录 `outputs/printer_child_ingestion/saved_children_v1`：

- `plan.json`：选中块、模型、当前存储位置及合同指纹，不含密钥。
- `selected_children.json`：实际选中的完整子块及原始元数据。
- `summary.json`：三个测试文档的状态、块数、完整性检查和隔离数量。
- `stored_chunks_A/B/C.json`：从真实 text_chunks 读回的块、引用、范围、限定条件和隔离原因。
- `graph_nodes_A/B/C.json`、`graph_edges_A/B/C.json`：从真实图存储读回的节点和边。
- `response_*.json`：本次模型请求的原始响应。

先确认三份测试记录为 `processed` 且 `intact_saved_children=true`，再核对：同名故障码是否按场景/文档隔离；关系是否连接到正确内部 ID；局部型号限制是否只作用于对应事实；按钮、前提、否定和脚注是否完整；来源是否能回到选中的完整子块及父块。

`processed` 表示系统流程完成，不等于所有事实语义正确。隔离记录和整块跳过通过 WARNING、审核字段和跳过清单呈现。本轮未接入查询时型号/场景硬过滤，不能用检索结果证明该功能已经完成。测试后可重启后端，在文档列表查看这三个 `child_smoke_...` 记录；通过普通按文档删除入口清理它们，不删除原手册。
