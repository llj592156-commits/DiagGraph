# 离线父块整理原型

本工具把原 PDF 书签与已生成的 LightRAG Docling sidecar 对齐，输出可审阅的父块候选。它是实验准备工具，不是入库功能；不会发送解析、LLM 或 embedding 请求，不会修改原 PDF、sidecar、旧知识库。

核心整理逻辑已提取至 `lightrag/document_structure/parents.py`，原 CLI 调用共享实现。共享 API、离线子块与父块回取见 [离线父子块说明](ParentChildOfflinePreparation-zh.md)。

2026-10-17 策略更新：用户确定文件名型号为父块默认范围，只有父块明确限制时才缩小。以下描述以当前 v3 为准；历史 v1/v2 实验仍保留其当时策略，不会自动改写。[一键上传接入设计](ParentChildIngestionIntegration-zh.md)说明系统集成的接入位置及尚未完成部分。

## 输入与输出

输入为同一份文档的原 PDF、包含一个 `.blocks.jsonl` 的 `.parsed` 目录，以及可选的范围规则和结构规则。PDF 必须有书签。`pypdf` 已属于项目 API 依赖。sidecar 有正文指纹时验证该指纹；邻近 `.docling_raw/_manifest.json` 存在时，还校验 PDF 字节是否与原解析输入一致。缺少 raw manifest 会明确记录来源身份未验证。

```powershell
python -m lightrag.evaluation.prepare_parent_blocks `
  --pdf 'D:\桌面\L3266_L3267_L3268_L3269_L5298_UG.pdf' `
  --parsed-dir '.\outputs\printer_docling_validation\B_no_ocr\L3266_L3267_L3268_L3269_L5298_UG.pdf.parsed' `
  --scope-rules '.\examples\parent_blocks\printer_scope_rules.json' `
  --structure-rules '.\examples\parent_blocks\printer_structure_rules.json' `
  --output '.\outputs\printer_parent_blocks\B_new_experiment'
```

每次使用新的输出目录。目录存在时拒绝覆盖，防止把一次实验混入另一次。

| 文件 | 内容 |
| --- | --- |
| `sections.jsonl` | 原始书签树、页码/坐标、与 sidecar 的匹配方式 |
| `parents.jsonl` | 父块候选原文、来源块、页码、范围、待复核标记 |
| `parents_preview.md` | 人工可读的候选预览 |
| `issues.jsonl` | 对齐、范围和长度方面的待复核事项 |
| `summary.json` | 输入指纹、数量、覆盖率和限制；最后写入，缺少它代表输出不完整 |

## 对齐和分组

1. 从 PDF 书签读取层级与目标位置，不相信 Docling 已压平的 `level`。
2. 优先匹配同页且规范化标题一致的块；同页有重名标题时，按垂直位置选择。
3. 标题被识别为图片时，核对 raw JSON 中图片子文字、书签标题、同页坐标、drawing ID 和原块归属。证据唯一且一致时，按原文 drawing 标签的准确偏移拆开跨页源块，记录 `picture_text_and_page`。原始 JSON 的哈希必须与 manifest 一致。没有充分证据时保留附近位置回退 `position_only` 或 `unmatched`。
4. 保持原始块顺序，在每个书签区间内组织原文；独立故障码和明确型号限定标题建立额外边界。
5. 型号限定只用于当前局部单元；进入下一个普通局部标题时停止继承，避免因标题层级缺失而扩大范围。
6. 全文目录、带点线页码的章节目录和前置内容保留用于核对，分别标为 `table_of_contents`、`section_contents`、`front_matter`，不作为正式正文索引。章节目录识别是启发式规则，需抽查，不代表已解决所有目录识别问题。
7. 未解决的跨书签源块仍标记 `source_block_crosses_bookmark`。比较坐标时使用 12 点容差，避免将标题自身与书签目的位置之间的轻微差异误判为跨节。

例如，原 PDF 书签保存“故障排除 → 本产品未能正常工作 → 无法打印”的父子关系和目标页。程序用这些已存储的数据对齐同页 Docling 标题，再把该标题之后、下一书签之前的结构块放到对应章节。它不是让模型猜目录。原项目的 PDF 文本提取器也可以扩展读取书签；Docling 在这里另外提供表格、图片引用和位置。

这不是最终的通用章节恢复算法。书签以下的子标题可能仍被组合在一起。候选不能自动写入知识库，需先抽查流程完整性、表格表头、脚注、型号边界和跨页位置。

## 已复核的超长章节细分

可选的 `printer_structure_rules.json` 只允许这三份已核对手册、三个完整书签路径。默认 10000 字符是审阅阈值，不是 token 上限；仅细分超过阈值的单元。其他文档、其他章节和缺少位置的标题不会套用这组规则。

- Windows 打印大节：在约 13.99 点高的标题框处识别独立操作，例如“双面打印”“打印小册子”；较小的海报拼接标题继续归于海报操作。
- 无法打印、无法收发传真：结合约 13.99 / 13.20 点的标题框恢复两层局部路径，使 Windows、Mac、手机端和其连接情境分开。原因、解决方案、注释继续附着在所属情境下。
- 高度指侧车记录的标题框高度，不是字号，也不是适用于所有 PDF 的通用标准。它来自这三份手册的版式核对，容差为 0.15 点；输出 `structural_evidence` 记录规则、标题来源块、高度和局部深度。
- `context_parent_ids` 指向同章节内路径上的前置父块。例如 Windows 网络问题保留对 Windows 连接检查说明的引用，不能在后续检索中丢掉这些前置条件。这只是离线元数据，尚未接入查询回取。

警告、注意和编号步骤不作为这种细分边界。没有可靠结构边界的超长单元继续保留并标记，禁止按字符数盲切。

## 范围规则

历史示例规则放在 `examples/parent_blocks/printer_scope_rules.json`，记录 B 手册第 6 页的功能说明。指定它们时仍验证原文引文，匹配项仅记为 `reference_rule_ids`，不再据此自动缩小某个父块范围。新实验可省略 `--scope-rules`。

- `document_models` 从原文件名提取，普通父块默认适用这些型号，状态为 `filename`，记录文件名证据。
- 父块标题、章节路径或正文开头明确限定整个单元时取交集；明确排除时取差集。结果为空标记冲突，不扩大范围。
- 仅限制按钮、表格项目或步骤的行不用于缩小整个父块，保留原文并标记复核。
- 没有文件名型号和明确范围时才保留 `unknown`，不能解释为通用。
- 当前只自动识别可验证的标题和开头声明格式；其他限定句保留复核标记，不声称已解决所有自然语言条件。

## 长度与来源完整性

当前草稿 schema 为 `parent-block-draft-v3`。未启用结构规则或细分后仍超长的单元标记 `oversized_parent_requires_structural_split`。

每个原块的内容按原顺序完整分配。`source_block_ids` 是原 sidecar 的块编号；`source_fragment_ids` 标识分配片段；`source_spans` 保存原块编号和左闭右开的 Python 字符偏移。未拆块的范围覆盖整个原块，图片章节边界处拆开的原块可由多个父块分担。这里的偏移由原块字符串直接计算，**不是 raw JSON 中按文本项计数的 `charspan`**。

校验要求所有原块范围连续覆盖，无遗漏、重叠、重排，且父块原文等于其来源切片按双换行连接的结果。拆分边界处会产生父块展示分隔；要还原原文应按 `source_spans` 拼回同一原块，而不是简单在所有父块之间插入双换行。原图片、表格标签和内容不改写。父块 ID 基于版本、文档 ID 和片段编号，重名标题不会覆盖。

源位置保留；预览中的图片路径仍相对原 `.parsed` 目录，预览文件不是可以独立搬走的完整副本。`summary.json` 记录规则哈希及图片边界恢复所用的 raw JSON、drawings 哈希。

所有候选固定 `ready_for_indexing=false`。后续还需处理超长单元、疑难边界和范围缺口，再构建子块，验证父块回取及图谱归属。不要把本原型描述成已支持父子块检索。
