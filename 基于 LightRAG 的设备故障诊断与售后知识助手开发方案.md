# 基于 LightRAG 的设备故障诊断与售后知识助手开发方案

**技术方案：Adaptive RAG + Self-RAG**

**用途：** 提供给 Codex 阅读、分析现有仓库，并分阶段完成二次开发。

**编写日期：** 2026-10-16

**实施原则：** 优先通过业务编排层实现新能力，不直接侵入 LightRAG 底层知识图谱和检索算法。所有接口与源码位置均应以本地项目版本为准。

## 1. 项目目标

基于开源项目 [LightRAG](https://github.com/HKUDS/LightRAG)，开发一个设备故障诊断与售后知识助手。

主要提供以下能力：

- **设备故障诊断：** 根据设备型号、故障码、异常症状等信息，检索维修手册与历史案例，辅助分析可能原因。
- **售后知识问答：** 回答保修政策、维修流程、使用说明、维护规范等问题。
- **自适应检索：** 根据问题复杂度，动态选择检索策略，避免所有问题使用同一种检索流程。
- **检索质量评价：** 判断检索文档是否与问题相关、是否适用于当前设备、能否支持诊断。
- **答案自我验证：** 判断生成内容是否有可靠证据支持，减少幻觉。
- **多轮交互：** 当缺少设备型号、故障码等必要信息时，主动向用户追问。
- **安全与人工升级：** 对高风险维修操作设置独立安全规则，必要时转人工售后。

### 1.1 核心技术改进

本项目重点增加两种 RAG 机制。

**Adaptive RAG**

在检索前判断问题意图和复杂度，选择：

- 不需要检索：直接回答。
- 简单事实问题：单次检索。
- 复杂诊断问题：多轮检索与问题拆解。
- 信息不足：优先追问。
- 高风险问题：触发安全处理流程。

**Self-RAG**

在检索和生成过程中增加质量评价：

- 判断文档是否相关。
- 判断设备型号和版本是否匹配。
- 判断证据是否充分。
- 根据缺失信息补充检索。
- 判断生成答案是否受到证据支持。
- 必要时修改答案、追问或转人工。

注意：本项目第一版采用基于 LLM 和规则的 Self-RAG 式评价流程，并不等于原论文通过 Reflection Tokens 训练模型的完整实现。

## 2. LightRAG 项目结构与关键接口

### 2.1 LightRAG 核心机制

LightRAG 结合知识图谱和向量检索，实现基于文档的知识增强问答。

知识入库的基本流程：

```
设备文档
   ↓
文档解析与切块
   ↓
实体和关系抽取
   ↓
知识图谱构建
   ↓
向量索引构建
   ↓
提供知识检索能力
```

查询的基本流程：

```
用户问题
   ↓
QueryParam
   ↓
检索策略选择
   ↓
图谱检索 / 向量检索
   ↓
上下文构建
   ↓
LLM生成答案
```

### 2.2 查询模式

LightRAG 的主要查询模式包括：

| mode     | 主要能力               | 业务使用场景     |
| -------- | ---------------------- | ---------------- |
| `bypass` | 跳过知识检索           | 允许的通用交流   |
| `naive`  | 文本向量检索           | 售后政策、说明书 |
| `local`  | 具体实体及其关联信息   | 故障码、设备部件 |
| `global` | 较广泛的关系信息       | 多设备故障关系   |
| `hybrid` | local + global         | 多实体关联分析   |
| `mix`    | 知识图谱与文本检索结合 | 综合故障诊断     |

不同模式不代表固定的准确率高低，需要通过测试选择。

### 2.3 需要研究的源码

**查询入口**

```
lightrag/api/routers/query_routes.py
```

研究 HTTP 查询请求、查询参数构建与核心接口调用。

**查询参数**

```
lightrag/base.py
```

研究 `QueryParam` 中的检索模式、Top-K、Token 上限等配置。

**核心查询接口**

```
lightrag/lightrag.py
```

研究 LightRAG 对外暴露的查询方法，例如：

```
rag.aquery_llm(...)
rag.aquery_data(...)
```

具体接口是否存在、参数和返回结构必须在实际版本中核实。

**检索执行逻辑**

```
lightrag/operate.py
```

研究知识图谱检索、文本块检索、上下文构建和答案生成相关流程。

**提示词**

```
lightrag/prompt.py
```

研究知识抽取和问答提示词，以及支持的自定义配置方式。

### 2.4 开发前核实事项

Codex 开始修改前必须：

1. 检查实际 LightRAG 仓库版本和 git commit。
2. 确认当前部署使用 Python SDK 还是独立 LightRAG Server。
3. 检查 `QueryParam` 的真实字段。
4. 确认是否支持 `aquery_data()` 和 `aquery_llm()`。
5. 检查结构化检索结果的数据结构。
6. 判断能否将检索与答案生成分离。
7. 检查自定义实体抽取的配置方式。
8. 记录现有查询 API 的行为，避免破坏兼容性。

不要直接假定最新 GitHub `main` 分支的接口与本地版本完全相同。

## 3. 总体技术架构

推荐采用：

```
             用户输入
                |
                v
        设备信息提取和规范化
                |
                v
          安全与信息检查
                |
                v
       Adaptive RAG Router
                |
      +---------+---------+
      |         |         |
      v         v         v
   直接回答  单次检索   多轮检索
                |         |
                |     Query Planner
                |         |
                +----+----+
                     |
                     v
             LightRAG Retriever
                     |
                     v
              Document Grader
                     |
            +--------+--------+
            |                 |
         证据充分           证据不足
            |                 |
            |           Query Rewriter
            |                 |
            |             补充检索
            |                 |
            +<----------------+
            |
            v
       Diagnosis Generator
            |
            v
         Answer Grader
            |
      +-----+------+
      |            |
    通过          未通过
      |            |
      |         修正/追问
      |            |
      v            |
    安全规则校验 <---+
      |
      v
带引用的答案 / 追问 / 转人工
```

主要分为四层：

1. **交互层：** Web、聊天接口、售后工单入口。
2. **智能编排层：** Adaptive Router、Query Planner、Document Grader、Answer Grader、Diagnosis Agent。
3. **检索与工具层：** LightRAG、订单查询、设备状态查询、工单查询。
4. **数据层：** 设备手册、维修案例、故障码、保修政策及相关业务数据库。

推荐以 LightRAG 为知识引擎，在外部增加自己的业务编排层。

## 4. Adaptive RAG 设计与接入位置

### 4.1 接入位置

Adaptive RAG 应位于 LightRAG 查询执行之前。

参考调用链：

```
用户请求
   ↓
query_routes.py
   ↓
Adaptive Router（新增）
   ↓
生成 RouteDecision
   ↓
构建 QueryParam
   ↓
LightRAG 查询接口
```

推荐在业务模块中新增：

```
app/adaptive_rag/
├── intent_classifier.py
├── complexity_router.py
└── query_planner.py
```

不要将分类逻辑直接写入 LightRAG 的底层图谱检索函数。

### 4.2 路由分为两个阶段

**第一阶段：用户意图识别**

建议的意图类型：

```
general_chat
after_sales_policy
fault_code_lookup
basic_troubleshooting
complex_diagnosis
warranty_or_order_status
unknown
```

**第二阶段：处理复杂度判断**

```
no_retrieval
single_step
multi_step
clarify
escalate
```

其中：

- `no_retrieval`：无需外部知识。
- `single_step`：执行一次检索。
- `multi_step`：拆分问题并进行多次检索。
- `clarify`：追问用户必要信息。
- `escalate`：转人工或触发安全升级。

### 4.3 路由示例

| 用户问题                                    | 意图                     | 策略               |
| ------------------------------------------- | ------------------------ | ------------------ |
| 什么是设备故障码？                          | general_chat             | no_retrieval       |
| X100 的保修期限是多少？                     | after_sales_policy       | single_step        |
| X100 的 E03 故障码是什么意思？              | fault_code_lookup        | single_step        |
| X100 运行十分钟后自动关机，可能是什么原因？ | complex_diagnosis        | multi_step         |
| 我的设备还在保修期内吗？                    | warranty_or_order_status | 业务工具           |
| 设备出现危险异常或需要拆解高风险部件        | safety-related           | clarify / escalate |

售后政策属于企业事实，即使问题简单，也应检索可靠资料，不应仅依靠模型自身知识。

### 4.4 路由数据结构

建议业务层定义：

```
from typing import Literal
from pydantic import BaseModel, Field

class RouteDecision(BaseModel):
    intent: str

    strategy: Literal[
        "no_retrieval",
        "single_step",
        "multi_step",
        "clarify",
        "escalate"
    ]

    mode: Literal[
        "bypass",
        "naive",
        "local",
        "global",
        "hybrid",
        "mix"
    ]

    reasoning_summary: str = ""

    extracted_entities: dict = Field(
        default_factory=dict
    )

    missing_fields: list[str] = Field(
        default_factory=list
    )
```

设备相关实体建议提取：

```
device_model
fault_code
symptoms
firmware_version
hardware_revision
operating_context
region
```

其中 `reasoning_summary` 仅保存简短的可解释路由理由。

### 4.5 路由调用示意

```
# 设计伪代码，需适配真实接口

async def route_and_retrieve(query, rag):
    decision = await classify_query(query)

    if decision.strategy == "clarify":
        return ask_for_missing_information(
            decision.missing_fields
        )

    if decision.strategy == "escalate":
        return make_safe_escalation_response()

    if decision.strategy == "no_retrieval":
        return await respond_without_kb(query)

    param = make_query_param(decision)

    if decision.strategy == "single_step":
        return await retrieve_evidence(
            rag, query, param
        )

    subqueries = await plan_subqueries(
        query, decision
    )

    return await retrieve_and_merge(
        subqueries, rag, param
    )
```

### 4.6 Adaptive RAG 的关键要求

不能将 Adaptive RAG 简单理解为动态修改 `mode`。

一个完整的 Adaptive Router 至少需要：

- 意图分类。
- 复杂度判断。
- 信息充分性判断。
- 检索模式选择。
- 单轮与多轮策略选择。
- 必要时的问题拆解。
- 安全规则优先级控制。

第一版可以使用规则和 LLM 结构化输出实现，后续再考虑训练专门的复杂度分类器。

## 5. Self-RAG 设计与接入位置

### 5.1 核心目标

在 LightRAG 检索之后、生成答案之前和之后增加评价闭环。

推荐工作流：

```
LightRAG 检索
     ↓
Document Grader
     ↓
判断证据是否充分
     ↓
必要时 Query Rewriter
     ↓
补充检索
     ↓
Generator
     ↓
Answer Grader
     ↓
通过验证后输出
```

### 5.2 接口选择

优先研究 LightRAG 的结构化检索接口：

```
result = await rag.aquery_data(
    query,
    param=QueryParam(mode="mix")
)
```

其目的是获取检索证据，而非直接生成答案。

应由 Codex 检查当前版本是否提供类似数据：

```
entities
relationships
chunks
references
```

不要直接假定这些字段一定存在。

如果实际结构不同，应由 `LightRAGAdapter` 统一转换。

特别注意：

不应在调用 `aquery_data()` 完成证据评价之后，又无条件调用会重新检索的完整问答接口，否则会产生重复检索，也无法保证答案基于已验证证据。

### 5.3 统一证据数据结构

建议新增：

```
from pydantic import BaseModel, Field

class EvidenceItem(BaseModel):
    evidence_id: str
    source_id: str
    content: str
    retrieval_mode: str

    source_title: str | None = None
    source_version: str | None = None
    device_model: str | None = None
    relevance_score: float | None = None

    metadata: dict = Field(
        default_factory=dict
    )
```

这是业务层自定义结构，不是 LightRAG 官方返回的数据结构。

### 5.4 Document Grader

新增：

```
app/self_rag/document_grader.py
```

Document Grader 接收：

- 用户问题。
- 当前设备信息。
- 检索证据。
- 用户当前诊断上下文。

主要评价内容：

**相关性**

文档是否与故障问题有关？

**设备适用性**

文档型号、硬件版本、固件版本、地区和生效日期是否适用？

**证据充足性**

是否覆盖故障原因、排查流程和限制条件？

**来源可靠性**

文档是否来自可接受的资料来源，是否存在过期或冲突情况？

建议先通过确定性规则过滤错误型号和过期政策，再交给 LLM 评价语义相关性。

模型生成的相关性分数不能被当作真实概率或最终安全保证。

### 5.5 Query Rewriter

新增：

```
app/self_rag/query_rewriter.py
```

当证据不足时，根据缺失信息生成新查询。

例如：

原始问题：

> X100 运行十分钟后自动关机。

可以改写为多个检索任务：

```
1. X100 自动关机相关故障说明
2. X100 运行中关机的官方排查流程
3. X100 运行时间与异常关机相关案例
4. X100 对应型号的安全限制和维修要求
```

将不同查询得到的证据合并并去重。

需要设置严格的检索预算。

例如：

```
MAX_RETRIEVAL_ROUNDS = 2
MAX_SUBQUERIES = 4
MAX_ANSWER_REVISIONS = 2
```

这些数值只是初始建议，应根据性能评测调整。

如果检索预算耗尽，系统应该追问、表达证据不足或转人工，而不是无限搜索。

### 5.6 Generator

新增：

```
app/diagnosis/answer_generator.py
```

Generator 只消费通过评价的证据。

建议答案包含：

1. 已确认的设备信息。
2. 当前故障现象。
3. 可能的故障原因。
4. 原因对应的证据来源。
5. 允许用户执行的安全检查。
6. 需要补充的信息。
7. 后续维修或售后处理建议。

应明确区分：

- 文档中直接支持的事实。
- 基于证据的合理推断。
- 暂时无法确认的内容。

不能把猜测写成确定诊断。

### 5.7 Answer Grader

新增：

```
app/self_rag/answer_grader.py
```

检查生成答案是否存在：

- 错误设备型号。
- 错误故障码解释。
- 无来源的维修操作。
- 缺少依据的确定性结论。
- 过期保修政策。
- 忽视安全警示。
- 没有回答用户真正的问题。

推荐输出：

```
pass
revise
retrieve_more
clarify
escalate
```

含义分别是：

- `pass`：通过评价。
- `revise`：需要修正答案。
- `retrieve_more`：需要补充检索。
- `clarify`：需要追问用户。
- `escalate`：需要人工处理。

注意：最终风险控制不能仅依赖 LLM 自我评分，必须设置独立安全规则。

### 5.8 Self-RAG 控制循环

```
# 设计伪代码

async def answer_with_evidence(query, route):
    state = init_state(query, route)

    for _ in range(MAX_RETRIEVAL_ROUNDS):
        evidence = await retriever.fetch(
            state.search_queries,
            route.mode
        )

        graded = await document_grader.evaluate(
            query,
            evidence,
            state.device_info
        )

        state.add_evidence(graded.accepted)

        if graded.sufficient:
            break

        state.search_queries = (
            await query_rewriter.rewrite(
                query,
                graded.missing_information,
                state
            )
        )

    if not state.has_minimum_evidence:
        return clarify_or_escalate(state)

    for _ in range(MAX_ANSWER_REVISIONS):
        answer = await generator.generate(
            query,
            state.accepted_evidence
        )

        verdict = await answer_grader.evaluate(
            answer,
            state.accepted_evidence
        )

        if (
            verdict.status == "pass"
            and safety_rules_allow(answer, state)
        ):
            return answer

        if verdict.status in {"clarify", "escalate"}:
            return clarify_or_escalate(state)

        state.add_feedback(verdict)

    return answer_with_uncertainty_or_escalation(
        state
    )
```

代码仅用于说明模块关系，需要 Codex 根据实际项目封装接口并实现错误处理。

### 5.9 是否修改 `operate.py`

首版不建议直接修改。

优先使用 LightRAG 公开的检索接口，由外部 Document Grader 评价证据。

如果后续发现检索结果在返回前已经被大量截断，导致无法对所有候选文档评价，可以研究在检索完成之后、Token 截断之前增加评价钩子。

需要先核实当前版本 `operate.py` 中的内部函数调用链，不能假设所有版本具有相同实现。

## 6. 设备故障诊断工作流

### 6.1 示例

用户输入：

> 我的 X100 设备运行十分钟后自动关机，应该怎么处理？

**步骤一：设备信息抽取**

```
{
  "device_model": "X100",
  "symptoms": [
    "运行约十分钟后自动关机"
  ],
  "fault_code": null,
  "firmware_version": null
}
```

**步骤二：Adaptive Router**

将问题分类为：

```
{
  "intent": "complex_diagnosis",
  "strategy": "multi_step",
  "mode": "mix"
}
```

**步骤三：Query Planner**

拆分检索任务：

```
X100 自动关机故障说明
X100 运行中关机排查流程
X100 与当前症状相关的维修案例
X100 安全操作和维修限制
```

**步骤四：LightRAG 检索**

获取知识图谱关系与相关文档片段。

**步骤五：Document Grader**

检查资料：

- 是否适用于 X100。
- 是否属于有效版本。
- 是否真正描述了类似症状。
- 是否提供了可信的排查依据。

**步骤六：Diagnosis Agent**

根据证据分析可能原因。

如果缺少关键条件，可以向用户提问，例如：

> 设备关机前是否显示故障代码或报警信息？

**步骤七：Answer Grader**

检查诊断依据、维修建议和安全条件。

**步骤八：最终回答**

输出有依据的可能原因、必要的下一步检查和售后建议。

示例中的设备和故障均为虚构，不代表真实设备诊断结果。

### 6.2 售后问答与故障诊断的区别

售后问答通常偏向事实查询：

```
用户问题
   ↓
资料检索
   ↓
证据验证
   ↓
回答
```

故障诊断则需要根据证据逐步缩小可能原因：

```
用户问题
   ↓
提取症状
   ↓
判断信息是否充分
   ↓
多路径检索
   ↓
分析可能原因
   ↓
必要时追问
   ↓
诊断建议与验证
```

两类请求应该由 Router 分流，不应该全部使用复杂诊断流程。

## 7. 设备领域知识库改造

除了 Adaptive RAG 与 Self-RAG，项目还应该定制设备领域实体及关系。

### 7.1 实体类型

| 实体            | 示例         |
| --------------- | ------------ |
| Device          | X100         |
| Component       | 电源模块     |
| FaultCode       | E03          |
| Symptom         | 自动关机     |
| FailureCause    | 散热异常     |
| RepairProcedure | 故障排查 SOP |
| Policy          | 保修条款     |

### 7.2 关系类型

建议使用：

```
HAS_COMPONENT
HAS_SYMPTOM
POSSIBLE_CAUSE
CHECK_BY
COVERED_BY
APPLIES_TO_VERSION
```

例如：

```
Device
   |
   | HAS_SYMPTOM
   v
Symptom
   |
   | POSSIBLE_CAUSE
   v
FailureCause
   |
   | CHECK_BY
   v
RepairProcedure
```

这些是建议的领域知识模型，而不是 LightRAG 原生保证的强类型图谱约束。

### 7.3 元数据

对每份文档记录：

```
source_id
document_type
device_model
hardware_revision
firmware_version
effective_date
region
source_title
chunk_id
ingested_at
approval_status
```

并确保：

- 型号别名能够统一。
- 故障码与设备版本对应。
- 过期售后政策不会被当作有效承诺。
- 文档引用可以追溯到原始资料。
- 不同客户或企业的数据具有权限隔离。
- 敏感工单信息在入库前完成脱敏。

对于自动抽取的设备故障关系，应通过人工抽样和测试进行验收，不能仅因为关系存在于知识图谱中，就认为它是确定的因果关系。

## 8. 建议代码目录

```
device-service-assistant/
│
├── app/
│   │
│   ├── api/
│   │   └── diagnosis_routes.py
│   │
│   ├── adaptive_rag/
│   │   ├── intent_classifier.py
│   │   ├── complexity_router.py
│   │   └── query_planner.py
│   │
│   ├── self_rag/
│   │   ├── document_grader.py
│   │   ├── answer_grader.py
│   │   └── query_rewriter.py
│   │
│   ├── diagnosis/
│   │   ├── workflow.py
│   │   ├── safety_rules.py
│   │   └── answer_generator.py
│   │
│   ├── retrieval/
│   │   ├── lightrag_client.py
│   │   └── evidence_schema.py
│   │
│   └── models/
│       ├── requests.py
│       └── responses.py
│
├── prompts/
│   ├── device_extraction.yaml
│   ├── routing.yaml
│   └── grading.yaml
│
├── tests/
│   ├── unit/
│   └── integration/
│
├── evaluations/
│   ├── test_queries.jsonl
│   └── evaluate_rag.py
│
└── README.md
```

这是逻辑目录结构，不要求完全照搬。

如果用户已经在 LightRAG 项目仓库中开展二次开发，应优先适配现有目录，避免创建不必要的嵌套项目。

## 9. API 设计

推荐新增业务 API：

```
POST /assistant/query
POST /assistant/diagnose
```

前者负责普通问答，后者负责多轮故障诊断。

如现有系统具备会话持久化能力，可以增加：

```
GET /assistant/diagnose/{session_id}
```

### 9.1 推荐响应结构

```
{
  "status": "answered",
  "intent": "complex_diagnosis",
  "strategy": "multi_step",
  "retrieval_mode": "mix",
  "device_model": "X100",
  "answer": "需要确认故障码和设备运行条件。",
  "possible_causes": [],
  "citations": [
    {
      "source_id": "manual-x100-v2",
      "chunk_id": "chunk-123"
    }
  ],
  "missing_information": [
    "故障码",
    "固件版本"
  ],
  "confidence_label": "insufficient_evidence",
  "escalation_required": false
}
```

这是自定义业务层结构，不是 LightRAG 默认 API。

还应支持：

```
answered
needs_clarification
escalated
failed
```

等响应状态。

## 10. 分阶段开发计划

### P0：LightRAG 基线验证

任务：

- 检查本地 LightRAG 版本。
- 确认查询 API。
- 导入一批设备手册和测试案例。
- 测试不同检索模式。
- 记录原始 RAG 的准确率、延迟和 Token 消耗。

验收：

能够查询设备知识，并将证据关联至原始文档。

### P1：Adaptive Router

任务：

- 实现意图识别。
- 实现复杂度分类。
- 实现 `QueryParam` 动态配置。
- 实现必要信息追问。
- 加入安全规则优先级。

验收：

不同类型问题能够进入正确的检索或处理流程。

### P2：Document Grader

任务：

- 封装 LightRAG 检索接口。
- 统一证据格式。
- 实现型号和版本规则过滤。
- 实现 LLM 相关性评价。
- 保留来源引用。

验收：

能够过滤型号不匹配、过期或不相关的资料。

### P3：复杂故障多轮检索

任务：

- Query Planner。
- Query Rewriter。
- 子查询执行。
- 多轮检索状态管理。
- 证据合并与去重。
- 检索次数限制。

验收：

复杂故障能够执行有限次补充检索，并在证据不足时追问。

### P4：Answer Grader 与安全输出

任务：

- 构建基于证据的 Generator。
- 实现答案支持度评价。
- 实现答案修改机制。
- 添加独立安全规则。
- 实现转人工处理。

验收：

能够识别无依据的维修建议、错误型号引用和高风险回答。

### P5：工程优化

任务：

- Rerank 与检索参数优化。
- Router 分类器优化。
- 缓存。
- 日志与监控。
- 性能评测。
- 测试覆盖率提升。
- 必要时考虑专用模型微调。

验收：

能够展示改造前后的客观评测结果。

## 11. 评测体系

建议准备包含多种故障和售后问题的测试集。

每条测试数据包含：

- 用户问题。
- 设备型号。
- 设备版本。
- 参考文档。
- 正确处理策略。
- 期望答案或期望追问。
- 可接受的来源证据。
- 不允许出现的错误。

### 11.1 评价指标

| 指标                    | 评估目标             |
| ----------------------- | -------------------- |
| Intent Accuracy         | 意图分类是否正确     |
| Strategy Accuracy       | 检索策略是否正确     |
| Recall@k                | 检索证据召回能力     |
| Precision@k             | 检索结果准确性       |
| Model Matching Accuracy | 设备型号适用性       |
| Citation Support Rate   | 答案是否受到引用支持 |
| Diagnosis Accuracy      | 诊断正确率           |
| Safety Violation Rate   | 不安全建议出现比例   |
| P50/P95 Latency         | 系统延迟             |
| Token Cost              | 模型调用成本         |
| Retrieval Rounds        | 平均检索次数         |

### 11.2 消融实验

建议对比：

**实验 A：原始 LightRAG**

固定使用 `mix`，进行单次检索和生成。

**实验 B：LightRAG + Adaptive RAG**

增加智能路由，比较检索效率与回答质量。

**实验 C：LightRAG + Self-RAG 式评价**

增加证据和答案评价，比较答案支持率。

**实验 D：LightRAG + Adaptive RAG + Self-RAG**

构建完整工作流，评估综合表现。

所有实验使用相同设备资料、问题集，并尽量保持 LLM 配置一致。

## 12. 关键工程注意事项

### 12.1 优先避免侵入底层源码

尽量使用公开 API。

除非确有必要，不直接修改图谱构建、存储层和底层检索算法。

### 12.2 避免重复检索

如果通过结构化检索接口获取证据并完成评分，应将这些证据交给自己的 Generator。

不要再无条件调用会重新检索的完整问答接口。

### 12.3 保障诊断安全

对高压电、高温、机械运动、危险拆卸等场景设置独立规则。

不应允许模型仅通过自我评分决定高风险操作是否安全。

### 12.4 保证引用可靠

引用应关联真实的 `source_id` 和 `chunk_id`。

不允许模型编造文档名称、页码、维修案例和售后政策。

### 12.5 控制流程预算

所有循环都应该受到限制：

```
最大检索轮数
最大子查询数量
最大答案修改次数
最大 Token 数量
超时限制
并发限制
```

### 12.6 保证数据权限

需要考虑：

- 企业知识库访问权限。
- 客户工单隐私。
- 设备数据保护。
- Prompt Injection 防护。
- 日志中的敏感信息脱敏。

### 12.7 可观测性

建议记录：

```
route_decision
selected_mode
retrieval_rounds
retrieval_latency
document_grade
answer_grade
token_usage
escalation_reason
```

从而分析系统为什么选择某种检索策略，以及在哪一步出现问题。

## 13. 交给 Codex 的开发任务

请按照以下顺序实施，而不是一次修改所有文件。

### Task 1：分析现有仓库

- 检查当前 LightRAG 版本。
- 追踪查询执行链路。
- 找出公开查询 API。
- 研究检索与生成的解耦可能性。
- 给出实际需要新增或修改的文件清单。

第一阶段只分析，不修改核心代码。

### Task 2：实现 LightRAG Adapter

封装统一接口，例如：

```
async def retrieve_structured_evidence(
    query,
    mode,
    params
):
    ...
```

使后续模块不依赖 LightRAG 的内部返回结构。

### Task 3：实现 Adaptive Router

包括：

- 业务意图分类。
- 问题复杂度判断。
- 检索模式选择。
- 多轮策略选择。
- 必要信息识别。

### Task 4：实现 Document Grader

包括：

- 型号和版本规则过滤。
- 文档相关性判断。
- 证据覆盖情况检查。
- 引用信息保留。

### Task 5：实现诊断工作流

包括：

- Query Planner。
- Query Rewriter。
- 多轮检索。
- 状态管理。
- 追问。
- 转人工。

### Task 6：实现 Generator 与 Answer Grader

包括：

- 带来源的答案生成。
- 证据支持度验证。
- 错误回答修正。
- 安全规则审核。

### Task 7：测试与评估

包括：

- 单元测试。
- 集成测试。
- 评测数据集。
- 消融实验。
- 运行说明。
- 与原始 LightRAG 的性能对比。

**要求：** 每项任务在修改前先说明准备修改的文件及原因，修改后运行对应测试。

## 14. 参考资料

**LightRAG 官方仓库**

https://github.com/HKUDS/LightRAG

**LightRAG 中文说明**

https://github.com/HKUDS/LightRAG/blob/main/README-zh.md

**LightRAG 核心参数**

https://github.com/HKUDS/LightRAG/blob/main/lightrag/base.py

**LightRAG 查询路由**

https://github.com/HKUDS/LightRAG/blob/main/lightrag/api/routers/query_routes.py

**LightRAG 查询实现**

https://github.com/HKUDS/LightRAG/blob/main/lightrag/lightrag.py

**LightRAG 检索操作**

https://github.com/HKUDS/LightRAG/blob/main/lightrag/operate.py

**Adaptive-RAG 论文**

https://aclanthology.org/2024.naacl-long.389/

**Adaptive-RAG GitHub**

https://github.com/starsuzi/Adaptive-RAG

**Self-RAG 论文**

https://arxiv.org/abs/2310.11511

**Self-RAG GitHub**

https://github.com/AkariAsai/self-rag

## 15. 最终开发原则

本项目的核心工作流为：

```
Adaptive Router
      ↓
LightRAG Retriever
      ↓
Document Grader
      ↓
Diagnosis Generator
      ↓
Answer Grader
      ↓
安全校验
      ↓
最终答案 / 追问 / 转人工
```

**Adaptive RAG 解决如何选择检索策略的问题。**

**Self-RAG 式评价解决检索证据是否可靠，以及生成答案是否受到证据支持的问题。**

LightRAG 负责知识检索，新增智能编排层负责业务决策、评价、诊断和流程控制。

首版实现重点是构建可运行、可测试、可评估的完整闭环，而不是立即修改 LightRAG 最底层的知识图谱算法。