# 维知助手 · DiagGraph

基于 [LightRAG](https://github.com/HKUDS/LightRAG) 开展二次开发的设备故障诊断与售后知识项目，计划通过自适应检索、证据检查与多轮排查，提供有来源依据的售后建议。

本仓库保留 LightRAG 的实现、原有文档和 MIT 许可证。设备诊断、自适应检索与答案验证属于分阶段建设内容，具体完成情况见下表。

## 当前状态

| 内容 | 状态 |
| --- | --- |
| 文档入库、知识图谱和多种检索模式 | 复用 LightRAG 现有能力 |
| 打印机手册开发题库与人工评分依据 | 已整理，包含 30 道模式对比题和 6 道边界题 |
| 批量评估工具 | 已提供，支持答案或检索材料采集、续跑、错误记录和结果汇总 |
| 批量自动语义评分 | 尚未完成；助手流程已有 LLM 证据/答案核验，运行成功不代表答案正确 |
| Adaptive RAG 检索计划与迭代控制 | 已有 LLM 五策略路由和有界补检索后端首版，待真实模型验收 |
| 答案证据验证、多轮诊断状态和工单 | 已有同证据回答与核验；历史由客户端发送，持久会话和工单尚未实现 |

助手后端入口为 `POST /assistant/query`，沿用现有鉴权与 query 角色模型。管理端和用户端聊天现已默认使用“助手对话”，支持连续追问、来源展开和业务失败展示；原查询方式保留在“检索问答”。使用方式、验证结果与限制见 [助手首版说明](docs/AssistantV1-zh.md)。独立训练路由小模型留待后续接入。

## 项目文档

- [设备故障诊断与售后知识助手开发方案](<基于 LightRAG 的设备故障诊断与售后知识助手开发方案.md>)
- [测试题与操作说明](outputs/printer_rag_baseline/01_测试题与操作说明.md)
- [参考依据与人工评分要点](outputs/printer_rag_baseline/02_参考依据与评分要点.md)
- [结构化题库](outputs/printer_rag_baseline/03_题库.jsonl)
- [单次结果记录模板](outputs/printer_rag_baseline/04_单次结果记录模板.json)
- [自动批量测试说明](outputs/printer_rag_baseline/05_自动批量测试说明.md)
- [LightRAG 原有安装和使用说明](README.md)

产品手册请从题库来源登记中的官方入口取得，再导入自己的知识库。仓库不附带原始手册、个人环境配置或运行中的知识库存储。题库和参考答案用于评价，不能重新导入被测知识库。

## 获取项目

```bash
git clone https://github.com/llj592156-commits/DiagGraph.git
cd DiagGraph
```

环境安装和服务器启动沿用 LightRAG 原有说明。开始评估前，应确认所需手册完成处理，并记录知识范围、模型、提示词和检索配置。

## 批量评估

工具入口为 [batch_baseline.py](lightrag/evaluation/batch_baseline.py)。直接运行该脚本仅需 Python 标准库，实际采集需要已经运行的 LightRAG API。

先检查执行计划，不发送模型请求：

```bash
python lightrag/evaluation/batch_baseline.py --dataset outputs/printer_rag_baseline/03_题库.jsonl --output outputs/printer_rag_baseline/answer_smoke --ids Q01 Q29 Q30 --modes naive mix --dry-run
```

确认 API 地址、认证和知识库范围后，按[批量测试说明](outputs/printer_rag_baseline/05_自动批量测试说明.md)执行实际采集。只发送问题和查询参数，不把参考答案发送给被测模型；每次采集使用的模型会产生正常调用费用。

生成的答案、原始响应和本地复核材料保存在本地。仓库只提交可复用题库、模板和工具文档，避免把临时记录混入项目源码。

## 开发检查

安装项目测试依赖后运行：

```bash
python -m pytest tests/evaluation -q
python -m ruff check lightrag/evaluation/batch_baseline.py tests/evaluation/test_batch_baseline.py
```

下一阶段将新增检索控制器，完成“分析问题 → 制定计划 → 检索 → 检查证据 → 继续检索、追问或回答”的闭环，再与固定检索模式进行对照。

## 许可证与来源

基础项目来自 [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG)，代码按仓库中的 [MIT License](LICENSE) 发布。产品名称、手册及其内容属于相应权利人，手册来源单独登记；它们不因本仓库的代码许可证而自动改为 MIT 许可。
