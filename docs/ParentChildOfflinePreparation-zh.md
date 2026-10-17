# 共享整理模块与离线父子块

2026-10-17 更新：最新草稿为 `outputs/printer_parent_child/20261017_v3`，补充表格 sidecar 脚注与独立功能适用性核验，仍为 1981 个子块。下文 v2 计数为上一版记录；新字段、修复边界和第二轮命令见 [证据修复说明](ParentChildEvidenceRepair-zh.md)。

当前阶段已实现共享整理 API、子块草稿及按已选子块 ID 回取父块的离线验证。独立向量对照脚本已准备，实验由用户自行执行，见 [检索对照运行说明](PrinterRetrievalComparison-zh.md)。尚未接入上传 UI、图谱或在线查询，也未修改旧知识库。

## 共享 API

父块核心逻辑现位于 `lightrag/document_structure/parents.py`；原 `python -m lightrag.evaluation.prepare_parent_blocks` 命令保持可用，调用同一份实现。

```python
from lightrag.document_structure import organize

result = organize(pdf_path, parsed_dir, structure_rules=rules_path)
# result.parents / sections / issues / summary remain in memory.
```

`organize` 只读取本地 PDF 和解析产物，不写实验目录。`write_preparation` 是显式输出接口，拒绝覆盖已有目录。后续上传适配器可直接使用内存结果。

## 子块

`lightrag/document_structure/children.py` 提供 `build_children(parents, tokenizer, config)`，返回附加条件的父块副本及子块，原输入不变。默认目标 500 token、普通文本重叠目标 60 token；这是一组实验参数，不代表最优值。

- 只切正文父块，目录、章节目录和前置内容不产生子块。
- 一个父块内部的普通文本复用现有 R 切分器，按段落、换行及中文标点等递归分割；不同父块之间没有重叠。
- v2 支持带明确表头的 JSON 表格按完整行切片，每片重复原表头。表头必须与原表格的行前缀一致，禁止猜测或截断单元格；无法拆分的单行和其他超长资产仍保留复核标记。图片与公式标签保持完整。
- 普通子块是父块的准确子串；表格行片段记录 `transformation=json_table_rows`，是经验证的来源投影，不冒充连续子串。`parent_span` 是原整张表格的范围，`source_spans` 映射被选中的原行、重复表头和标签位置。重复表头不会被误计为原数据行重复。
- `required_notes` 保留相关脚注原文、父块内范围和原始来源区间；检索文本附加这些脚注，父块回取仍取得完整说明。所有范围是左闭右开的 Python 字符偏移，不使用 Docling 每项局部 charspan 代替原块偏移。
- `source_positions` 只表示相交来源片段的 bbox，可能包含跨页的额外区域；`position_precision=source_fragment` 明示它不是子块的精确矩形。
- 型号范围、章节路径、功能域和前置父块引用继承自父块。平台条件仅由标题路径提取，未宣称覆盖正文中所有版本和条件。
- `retrieval_text` 是供后续实验索引使用的上下文文本，包含文件名、章节、型号、平台和原始内容。500 token 是内容目标，不包含这个前缀；`retrieval_tokens` 另计，正式索引必须采用实际 embedding 模型的 tokenizer 检查总输入上限。
- 子块 ID 包含父块内容、适用元数据、来源区间和切分参数；改变文件名型号范围会改变子块身份。

校验每个正文父块的非空白内容都被覆盖，允许普通文本的受控重叠；检查子串或表格行投影、顺序、继承元数据、来源区间、重复表头、脚注及原始数据行的完整覆盖。所有结果保持 `ready_for_indexing=false`。

## 分词与命令

离线 CLI 使用显式指定的本地 `cl100k_base.tiktoken` 文件，按固定 SHA-256 校验并直接读取，不在运行时联网下载。分词规则与实际部署的 embedding 模型可能不同，当前计数用于可复现的草稿实验。

```powershell
python -m lightrag.evaluation.prepare_child_blocks `
  --parent-dir '.\outputs\printer_parent_blocks\20261017_v6\B' `
  --tokenizer-file '.\tmp\parent_blocks_tools\cl100k_base.tiktoken' `
  --target-tokens 500 --overlap-tokens 60 `
  --table-header-overrides '.\examples\parent_blocks\printer_table_headers.json' `
  --output '.\outputs\printer_parent_child\B_new_experiment'
```

首次准备分词文件只下载公开规则，无文档上传或模型调用；上述命令本身只访问本地文件。共享 Python API 可以注入系统实际使用的 tokenizer，上传接入时不必绑定这份实验词表。

输出 `parents.jsonl`、`children.jsonl` 和最后写入的 `summary.json`；记录输入和输出哈希、计数、配置、资产根目录及复核事项。读取已有父块实验时检查完成标记、文档身份、sidecar 哈希和原文完整性。

最新工作数据位于 `outputs/printer_parent_child/20261017_v2`：A 598、B 764、C 619 个子块，共 1981 个。此前的 5 张超长表格已拆为 11 个完整行片段，正文最多 500 token；索引文本连同前缀和必要脚注最多 1010 token。表头 override 仅用于 C 手册第 202 页已复核的表，其余表头来自原 sidecar。历史 v1 产物保留。

## 离线回取

```python
from lightrag.document_structure.retrieval import ParentChildIndex

index = ParentChildIndex.from_directory(bundle_dir)
result = index.resolve(child_ids, model="L5298", platform="Windows")
```

该接口消费调用方已经选出的子块 ID，按型号、可选平台和功能域过滤，去重父块并补回必要的前置父块。前置说明不兼容时拒绝该证据组，不能返回丢失前提的操作。返回的局部复核标记仍需要下游处理。

这里没有向量搜索、LLM、相似度评分或生产上下文预算；用手工选定的典型子块验证映射，不是检索质量评测。相同 E-12 在网络和显示屏错误两个领域分别回取，不能据此宣称现有图谱歧义已经解决。

下一阶段按 `docs/ParentChildIngestionIntegration-zh.md` 接入持久化、上传配置、恢复/删除流程和实际查询，并在独立 workspace 验证。
