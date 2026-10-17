# 打印机故障诊断业务抽取规范 v1

本阶段定义业务实体、关系及输出约定，提供可运行的六样本真实抽取入口，以及兼容原生 LightRAG 的提示词配置。真实 API 由用户执行；开发验证使用 mock。没有注册上传策略，没有将三个文档全量抽取或写入知识库，也没有接入正式实体身份合并、范围过滤、父块展开、Adaptive RAG 或 Self-RAG。

## 实体和关系

唯一规范源是 `lightrag/extraction/printer.py`，版本 `printer-business-extraction-v1`。

实体类型：

| 类型 | 含义与例子 |
| --- | --- |
| DeviceModel | 原文明确出现的设备型号，例如 L5298 |
| FaultCode | 故障/状态码，完整保留 E-12、W-11、002 等格式 |
| Symptom | 可观察症状或明确的故障、维护状态 |
| Cause | 原文明确陈述的原因，不能从建议检查项或解决方法反推 |
| Component | 部件与耗材，例如无边距打印废墨收集垫 |
| Procedure | 具体处理动作，例如联系服务机构更换；机构本身不是操作 |
| Feature | 功能，例如传真、ADF、无边距打印 |
| Organization | 厂商或认证服务机构 |
| Condition | 原文明确的操作前提 |
| Setting | 配置项或配置值 |
| Specification | 参数及其数值、单位、上下限 |
| Other | 必要但不能归入以上类型的对象，谨慎使用 |

关系是连接两个实体的边，不是节点。允许孤立实体，不允许缺少任何端点的关系。处理动作可以作为实体。关系及端点类型受约束：

| 关系 | 起点 → 终点 | 语义 |
| --- | --- | --- |
| INDICATES | FaultCode → Symptom | 代码表示某种状态，不凭处理方法推断原因 |
| CAUSED_BY | FaultCode/Symptom → Cause | 仅限原文明确的因果陈述 |
| INVOLVES | FaultCode/Symptom/Procedure → Component/Feature/Setting | 涉及对象，不自动解释为因果 |
| HANDLED_BY | FaultCode/Symptom → Procedure | 处理方式，保留临时恢复与修复的区别 |
| PERFORMED_BY | Procedure → Organization | 执行机构 |
| REQUIRES | Procedure/Feature → Condition | 明确前提，不是任意依赖或步骤顺序 |
| SUPPORTS | DeviceModel → Feature | 原文明确支持，文件名不构成支持证据 |
| LIMITED_TO | Feature/Procedure → DeviceModel | 明确型号限制，必须有带引用的限制条件 |
| HAS_SPEC | DeviceModel/Component/Feature → Specification | 参数、单位与边界 |
| CONFIGURED_BY | DeviceModel/Feature → Setting | 明确配置关联 |

方向属于本次抽取语义约定。原生 NetworkX 后端仍是无向图，本阶段没有改变存储契约；不能据此声称已支持有向依赖、多个平行关系或操作执行计划。

## 增强输出及验证边界

实体字段：`name/type/description/evidence_quotes`。

关系字段：`source/target/relation/description/evidence_quotes/qualifiers`。

`qualifiers` 是条件数组。每项为 `kind/text/evidence_quotes`，kind 只能是：

- `model_restriction`：明确的局部适用型号或排除限制。
- `prerequisite`：操作前提。
- `prohibition`：禁止事项、不可由用户执行等。
- `button`：按钮及原始符号。
- `footnote`：限定本条事实的脚注。

没有相关限制时数组为空；描述也须保留限定，不能只藏在条件数组中。多段出处分别引用，不能拼成一条连续引文；不额外包引号、不用省略号替代原文。示意：

```json
{
  "source": "E-12",
  "target": "联系认证服务机构更换废墨收集垫",
  "relation": "HANDLED_BY",
  "description": "联系认证服务机构更换，不能由用户更换",
  "evidence_quotes": ["请联系爱普生认证服务机构更换无边距打印废墨收集 垫 *2 。它不能由用户更换。"],
  "qualifiers": [
    {
      "kind": "prohibition",
      "text": "不能由用户更换",
      "evidence_quotes": ["它不能由用户更换。"]
    }
  ]
}
```

这里只展示字段；正式关系须有引用支持整个事实，并且两个端点都存在于实体数组。引用存在不代表关系成立。

校验检查字段、数量、类型、关系端点类型、完整型号/代码以及引文连续存在；仅忽略 PDF 空白差异，不改标点、否定或按钮符号。逐条隔离不合格记录，依赖被拒实体的关系一并隔离；截断或无法解析的整份响应不拼接恢复。未知关系不自动映射，原始响应保持完整。

校验不能自动判断语义正确、条件是否漏抽、因果是否合理、某个脚注是否限定整条事实，或者 model_restriction 是否完整。输出始终标记 `semantic_review_required=true`，需要审阅；不能把通过计数当成诊断正确率。

来源由程序附加到 `evidence_records.json` 的每条记录：文档、父块、子块 ID，章节、页码、原文跨度、表格与脚注位置、原始范围及输入摘要。模型只读取文档/章节/场景背景及原文，不生成 ID，也不参与全局合并。

父块范围继续采用文件名默认型号，只在已有整理程序发现明确的整个父块限制时收窄。模型抽出的局部型号限制单独保存，不自动修改父块范围；后续正式入库与查询必须消费已审阅的限制，不能把原始父块范围直接当成每条事实的最终适用性。

## 六样本运行

复用原先 B 网络 E-12、B 显示屏 E-12、B/C W-11、B 明确功能限制、B ADF 规格这六份真实片段，保留表头和相关脚注。输入仍是从父子块中选出的实验片段，不等于已经对全部 1981 个子块运行抽取。新入口不构建实验图谱，也不改变原来的图谱实验；先检查新业务输出，再接入正式图谱。

本地 PowerShell 先预检：

```powershell
Set-Location D:\GitHubCode\LightRAG
.\examples\parent_blocks\run_printer_business_extraction.ps1 -Resume
```

默认新目录 `outputs/printer_business_extraction/manual_run_v1`。查看 `prompts.json` 和 `samples.json`；`experiment.json` 冻结提示、源样本、配置和依赖代码摘要，不保存密钥。预检不请求任何模型。

用户决定执行真实抽取时：

```powershell
.\examples\parent_blocks\run_printer_business_extraction.ps1 -RunExperiment -Resume
```

最多六次新抽取请求，使用现有 EXTRACT 配置；没有 embedding、关键词、回答或评分模型调用。默认输出上限 6144，每份最多 12 实体、16 关系。无自动重试；成功响应保存后恢复复用，包括有隔离记录的响应。更改提示、模型、参数、输入或依赖代码后使用新目录，不能复用旧计划。

`summary.json` 为累计 usage/计数；`validation.json` 保存被隔离记录及原因；`extractions.json` 为保留输出；`evidence_records.json` 为带来源输出；`report.md` 列出逐项语义验收要点。存在解析错误、隔离记录或空样本时退出码 2，PowerShell 会提示检查文件；不是六次请求都失败，也不会自动重试补全。

## 原生 LightRAG 兼容配置

`prompts/samples/printer_diagnosis.v1.yml` 从同一规范导出，包含业务类型、规则、关系词表和 text/JSON 两种原生输出模板。原生模板只使用其支持的字段：实体 `name/type/description`，关系 `source/target/keywords/description`。原生关系词表是提示约束；本阶段没有给原生入库增加强制 schema 校验。

额外的引用和条件数组由增强实验入口保存。仅启用 YAML 不会让原生存储自动保存这些字段，也不会实现消歧或硬型号过滤。

本轮不改 `.env`。以后在独立知识库试用原生配置时，先复制配置：

```powershell
New-Item -ItemType Directory -Force .\prompts\entity_type | Out-Null
Copy-Item .\prompts\samples\printer_diagnosis.v1.yml .\prompts\entity_type\printer_diagnosis.v1.yml
$env:ENTITY_TYPE_PROMPT_FILE = 'printer_diagnosis.v1.yml'
$env:ENTITY_EXTRACTION_USE_JSON = 'true'
```

使用自定义 `PROMPT_DIR` 时复制到对应目录。该配置由重启后的服务或新建 LightRAG 实例加载，现有实例可按核心文档通过 `addon_params` 更新。上面的环境变量只作用于原生流程；六样本入口从规范模块直接构造增强提示，不读取这个 YAML 开关。不要据此重建现有知识库，本轮只准备配置与验证。

修改规范后，可导出到一个不存在的新路径：

```powershell
.\.venv\Scripts\python.exe -m lightrag.extraction.printer --export-profile tmp/printer_diagnosis.updated.yml
```

导出拒绝覆盖已有文件，避免覆盖手动修改的提示词。配置与规范的一致性、原生两种模式可加载性、逐条隔离、代码前导零、来源范围不变、预检/恢复不重复请求均有离线软件测试；业务质量仍由新的真实输出验收。
