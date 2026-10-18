# 并行智算云重排与父块证据选择

本轮使用现有并行智算云的 `GLM-Rerank`，接口为 `https://llmapi.paratera.com/v1/p002/rerank`，走项目已有 `cohere` 兼容适配器。官方说明见 https://ai.paratera.com/document/llm/modalAbility/reRanking 。模型列出的上下文为 4K，因此先送子块和简短来源信息，不直接将长父块作为重排输入。复用本地已有同平台密钥，不在文档或报告中保存密钥。

## 当前查询配置

本地 `.env` 的目标配置如下；密钥由现有同平台配置复用，不需要换厂商或重新购买另一套服务。

```dotenv
RERANK_BINDING=cohere
RERANK_MODEL=GLM-Rerank
RERANK_BINDING_HOST=https://llmapi.paratera.com/v1/p002/rerank
RERANK_BINDING_API_KEY=<existing local Paratera key>
RERANK_BY_DEFAULT=true
RERANK_CANDIDATE_K=20
CHUNK_TOP_K=5
MIN_RERANK_SCORE=0
MAX_ASYNC_RERANK=1
RERANK_TIMEOUT=60
RERANK_ENABLE_CHUNKING=true
RERANK_MAX_TOKENS_PER_DOC=3000
```

保存后需要用户重启后端，已运行的服务不会自动加载这些修改。前端若保存了显式关闭重排或其他文本块数的设置，需将重排打开、文本块数设为 5；显式请求参数优先于默认值。

`RERANK_CANDIDATE_K` 映射到 `addon_params['rerank_candidate_k']`。未配置时默认为 0，保持普通切分查询原有的候选数量逻辑；本项目配置为 20。向量分支先取候选，实体和关系分支关联各自原文；各路按原顺序轮流合并、按子块 ID 去重后，进入重排的池最多 20 条。`chunk_top_k=5` 表示重排最多保留 5 个子块。候选不足时不补空结果；调用方显式请求更多最终块时，候选池至少扩到该数量。

## 证据处理顺序

1. 从现有知识库召回候选，保持各模式原来的召回来源；不重新入库，不增加故障码精确检索。
2. 合并去重，取有界候选池。重排输入为原子块正文、文件名、标题和已保存的适用型号/场景信息，后两者仅作排序输入，不执行硬过滤。
3. `GLM-Rerank` 返回候选索引和分数，保留最多 5 个子块。排序输入中的元数据不会冒充原文加入最终证据。
4. 对 H 子块回取同一文档的父块与必要前置说明；同一父块的多个命中合并成一个证据组。普通切分块仍直接作为原文。
5. 在总预算内优先保留完整原文组。图谱描述过多时缩减其末尾记录，为至少一个完整证据组留出空间。父块过长则回退到完整命中子块及其必要前提、表头和脚注；同父块的多个选中子块在回退中都保留。
6. 某组仍放不下时继续检查后面的候选，不因第一组过长丢弃全部后续来源；不任意截断步骤、前提或警告。若所有完整证据组都无法容纳，结果会记录预算遗漏，不能保证任意小预算都能保留原文。

普通切分方式兼容现有重排和证据选择。没有候选数覆盖、也不涉及 H 的查询保持旧流程。不会改已有图谱实体、关系、向量或父子块包。

`metadata.retrieval` 记录候选数、候选 ID、所选子块 ID、分数、`rerank_requested`、`rerank_applied`、回退块、预算遗漏块与最终证据数。`metadata.parent_child.parents` 只列最终保留的证据来源，标明完整父块或子块回退，不将预算裁掉的父块冒充已提供的原文。

重排异常时底层保留有日志的原顺序降级，`rerank_applied=false`；批量验证将这种请求记为 `unverified`，避免把 HTTP 成功当成重排成功。配置了新候选池但缺重排函数、请求却要求重排时直接报错。

## 用户运行验证

先启动已重启的本地后端。预检不会请求后端或模型：

```powershell
.\examples\parent_blocks\run_printer_system_retrieval.ps1 -DryRun
```

开启重排运行 Q01–Q30、naive/hybrid/mix 三种模式，共 90 次查询：

```powershell
.\examples\parent_blocks\run_printer_system_retrieval.ps1
```

结果默认存入 `outputs/printer_system_retrieval/glm_rerank_v1`，与原始基线隔离。只检索，不生成最终答案，但会调用问题 embedding、图谱关键词抽取和重排服务。不能承诺总测试免费。

在同一知识库上重跑关闭重排的对照，单独存入 `no_rerank_v2`：

```powershell
.\examples\parent_blocks\run_printer_system_retrieval.ps1 -NoRerank
```

旧小样本已清理，不能把此前保留小样本的第一轮差异全部归因于重排。查看 `summary.json` 中的 `confirmed_rerank_applied`，以及逐题 `metadata.retrieval`；运行成功率不是证据或答案正确率。预算、候选或代码再改变时换新的输出目录。故障恢复用同一命令和目录，未开始任务自动续跑；排查服务问题后可添加 `-RetryErrors` 重试未验证/错误任务。

## 明确保留到后续的工作

- 用户要求：故障码精确匹配通道仅记录设想，暂不加入，待后续明确授权。
- 用户要求：型号/场景硬过滤暂不加入；本轮只让已保存的范围可见于重排输入。
- 10 个跳过抽取块未补提取；不重新建图，不运行付费实验，不自动重启用户服务。
