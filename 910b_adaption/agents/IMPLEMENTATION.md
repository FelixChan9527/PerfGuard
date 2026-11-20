# 原流程对照与复现范围

本实现依据原 `sample_test_benchmark.py` 的推理流程和项目内 `paper.pdf`（PerfGuard，25页）整理，使用显式 Python 调度替换 MetaGPT。新工程中的 `docs/paper.txt` 是论文文本，原工程保持原样。

原 `run_slim.py` 将规划、工具调用和评估集中在 Planner 中；`sample_test_benchmark.py` 包含 Analyst、Planner、Worker，视觉评估放在 `Role_AIGC.unit_test` 中。这里选择后者为主流程，并将原 `unit_test` 原样的评估请求与评分公式提取成独立 Evaluator，形成论文第3、4节的四角色结构。

2026-11-09 按新的两工具要求调整默认 Worker：根据 Planner 的子任务返回生成/编辑类别，然后为 FLUX 或 Step1X 构造官方函数参数并直接执行，省略 PASM 七维权重和排名。`WORKER_ROUTE_PROMPT` 从原 Worker 提示的类别选择部分派生，增加编辑必须有输入图像的约束。Analyst、Planner、Evaluator 提示、评分公式、依赖图和终止条件保持原实现。下面描述的原 PASM 仍可用 `--worker-mode pasm --tool-config ToolSets/config.yaml` 运行。

同日后续按用户要求优化并精简活跃提示，默认模式为 `precise`。各阶段固定提示为40–79个英文词，原始需求仅传一份，重点保留数量、属性、空间、尺寸及编辑保留要求。Planner 使用精简任务图而不再嵌套先前请求；Worker 编辑时查看源图；原有输出读图调用移至评分前，提取不含目标提示的对象观察，Evaluator 结合输出图和编辑源图评估。原评分聚合与五轮/0.9规则保持；精简评分提示采用共享0–100标尺，不能视为与原prompt逐字等价。`--prompt-profile original` 保留原提示、完整历史传递和先评分后读图顺序，原始档案没有更改。以下原提示说明适用于 original 模式。

本地后端后续切换为用户指定的Qwen3.6-35B-A3B-AWQ（昇腾使用同一量化参数的FP16展开）。真实轨迹曾出现Planner将 `operation` 写成 `None`，因此明确只有 `depend` 允许该占位值，操作必须为具体动作，并按同一schema进行一次语义修复。观察提示简化为对象类别、颜色材质、数量、位置和一句整体描述；评分解释要求20词以内。这些变化不改原始prompt档案。工具改为独立常驻服务，流程退出不卸载，详见 [工具服务](../deployment/tools/README.md)。

后续数量样例中，Planner两次只返回 `generate`／`create_scene` 动作标签，语义检查在调用工具前拒绝。精简提示补充一句完整操作的通用示例后，独立真实API复查首次返回包含数量、颜色、布局和排除项的可执行指令，见 [Planner复查](../outputs/validation/qwen36_concrete_planner_probe.json)。没有将标签自动替换为用户需求来掩盖规划失败。

| 角色/功能 | 原实现 | 新实现 |
| --- | --- | --- |
| 图像语义分析 | `MLLM_read_img` | 同名类，图像通过 API 发送 |
| 任务总结、目标语义和评估维度 | `Analyst` 第573–701行、`prompt.txt` | `Analyst`、`ANALYST_EVAL_CONTEXT` |
| 根据历史结果规划下一操作、选择依赖轮次 | `Planner` 第703–1133行 | `Planner` |
| PASM维度权重、工具排名及条件构造 | `Worker` 第1135–1541行 | `Worker` |
| 各维度图像评估与最终分数 | `Role_AIGC.unit_test` 第325–550行 | `Evaluator.evaluate` |
| MetaGPT消息调度与共享内存 | `Environment`、`Message`、类级共享字典 | `Workflow`、轻量 `Message`、每次任务独立字典 |
| 角色通信 | 手工XML标签与字符串拆分 | 官方Structured Outputs、JSON Schema、宿主严格验证 |
| 工具调用 | 同进程加载所有工具、模型回复XML条件 | 官方function calling、宿主参数绑定、独立conda子进程 |
| 工具运行结果 | `<save_path>…</save_path>` | PIVOT边界：`<result><image>…</image></result>` |

## 保留的推理行为

1. Analyst先读输入图像，再总结任务和目标图像语义，并抽取对象数量、位置、属性、风格、背景等目标。
2. Planner每轮生成一个操作。首轮从用户输入开始，后续可选择任务图中任意已成功轮次作为依赖，也可从根节点重新开始。历史图包含各轮结果、分数、评估反馈和图像路径。
3. Worker使用原任务偏好定义确定工具类别和七维任务偏好，将权重归一化，再与原工具偏好向量相乘并排序。四个保留工具的偏好数据使用原 `prompt/preference_logs/preference_lastest.json`，与原 `Basic_Tool` 的实际运行读取源一致。
4. 原 sample 推理代码虽然计算 `top2 + random1`，随后明确执行 `tools_list = [tools_list[0]]`。本实现保留“仅执行排名第一工具”的实际行为。不会宣称已经执行论文中的在线多候选APU训练。
5. Evaluator保留原逐维0–100评分请求、各局部维度等权平均、局部平均与全局semantic各占一半的公式。原脚本未实现identity维度评分，本实现也未新加该评分prompt。
6. 默认最多5轮、达到0.9分停止；达到轮数上限时从整个任务图选最高分的有效图像，而非固定返回最后一轮。命令行可调整轮数与阈值。
7. 原 `planner_use_exp=False` 默认值保留。经验文件读取格式保留，原脚本中注释掉的经验写回、偏好更新以及训练代码不自动启用。

## 运行时Prompt与通信格式

`prompts.py` 只保留当前流程使用的原始 prompt 文本，并以有意义的常量名提供给各 agent。没有运行时引用的源码行号索引和字符串已移除。

按照后续要求，当前发送给模型的提示在 `active_prompts.py` 从原文派生：保留角色职责、任务要求、七维偏好解释、操作约束以及完整评分rubric，删除XML输出标签、展示输出格式的示例、重复格式提醒和要求模型输出think标签的后缀。用户任务和图像信息在静态模板处理之后才插入，不会被格式清理误删。工具参数说明由官方function参数schema提供，提示中不再重复整个schema或XML使用示例。

`schemas.py` 定义各角色的官方JSON Schema。Analyst返回任务/目标语义与评估目标，Planner返回操作与依赖，Worker返回类别/偏好并通过官方function call提供工具参数，Evaluator返回各维分数/说明及总评。目标对象数量通过`[{category,number}]`数组传输，以满足strict schema禁止开放对象属性的要求；宿主随后恢复原`{对象名:数量}`格式。工具图像路径、1024尺寸、随机种子和步数由宿主绑定，模型不能覆盖。

当前流程使用官方schema约束各角色输出。离线检查不代表真实API或图像模型推理通过。

## 解析与错误处理

参考PIVOT通信边界，API适配器严格解析JSON并验证完整schema：重复字段、未知字段、NaN/Infinity、缺失维度、错误类型、未提供的工具或不匹配的强制工具名均失败，不使用宽松XML拆分或模拟回复兜底。Planner依赖只允许None或已有有效结果的轮次；Worker还校验类别对应的七维集合、权重有限且在0–1；Analyst拒绝重复对象类别。

格式或语义协议错误最多进行一次受约束修复，与网络重试分开。每次API尝试和协议错误写入轨迹的`attempts`。角色边界错误分类为`protocol`，连接/超时为`network`，服务端API状态错误为`api`，模型明确拒绝为`refusal`（不作为格式错误修复）；真实工具执行错误单独记录为`tool_execution`，继续按原算法交Planner规划，所有工具失败时明确终止。工具参数检查在实际执行前完成，非法参数不会触发工具调用。必需任务、语义、操作与评估说明均须包含非空白字符。

后续参考PIVOT的 `utils/llm_stream.py` 和超时分类，增加官方流式接收。空闲时限与按token预算扩展的总时限分开，正文、思考或函数参数增量均计为进度。要求服务明确给出结束原因，才把完整内容交给原schema解析；断流和半截函数参数直接失败，不自动重复提交。连接建立失败、429及临时5xx仍可退避重试，本地超时默认不重放。JSON Schema与强制函数选择完整保留，不复制PIVOT的XML字段解析或专用grammar。每次传输保留进度统计与终止原因，正常、失败和取消路径均关闭连接。

## 为可运行性修复的细节

- 所有角色和视觉读取器统一使用所选配置的 OpenAI 兼容 API；当前默认 `config/llm.local.yaml` 连接独立部署的本地 Qwen3.6-35B-A3B-AWQ 服务，也可指定远端 `config/llm.yaml`。调度进程不加载 LLM、CLIP 或 MetaGPT。
- 原Analyst设置了description但没有设置system_prompt，这里使用它已有的原description构造已有格式的system prompt。
- 原Planner保存了历史却默认没有传入模型；这里将原样历史传入API，使历史观测真正参与下一轮决策。Planner返回值顺序同时修复为决策使用去除think标签后的内容。
- 经验检索以确定性的词频余弦替换原本地CLIP，避免额外本地模型。经验库为空且默认关闭规划经验时不影响原默认路径；如果提供已有经验库，召回排序与原CLIP并不等价，不能据此声称经验训练完全复现。
- 每次运行隔离共享内存，过滤 `image:[None]`，检查输入路径和依赖轮次。编辑工具的输入路径由主进程从真实依赖覆盖，避免模型改写路径。
- 原目标问卷的 `<goal>…<goal>` 笔误保留在档案，当前官方schema通信不再依赖该标签。
- API返回的评分必须为有限的0–100数值；缺失或非法评分明确失败，不把解析失败当成图像低分。
- 保留原semantic-only评分行为：如果不存在可评估局部维度，最终分数等于semantic分数的一半。这是原公式的边界行为；没有悄悄更改算法。
- 全部工具失败时明确报错并记录实际错误。存在图像但所有评分为0时仍返回有效的最高分图像，修复原代码 `os.path.exists(None)` 的崩溃。
- 输出目录保存完整调用请求、回复、任务图、工具条件、维度评分和失败原因；不记录API密钥或编码后的图像内容。

## 训练与论文数值

论文第4.2节APU需要多工具输出比较及偏好矩阵更新，第4.3节CAPO需要收集候选规划并训练Planner。原 sample 推理脚本中前者被注释，后者不属于该文件；用户要求所有后端切换为同一API模型后，也不再使用论文训练的本地Planner。因此这里复现的是原推理框架、原角色prompt、PASM选择与闭环反馈，不能将其结果当作论文原模型配置的完整指标复现。

## 回归检查

```bash
python -m unittest agents.test_workflow -v
python -m unittest discover -s tests -v
```

这些检查使用明确的假API和假工具，覆盖任意历史依赖、全图最优输出、阈值停止、零分图像、工具失败、非法评分以及原prompt一致性。真实生成/编辑测试请使用根目录 `test_tools.py`。
