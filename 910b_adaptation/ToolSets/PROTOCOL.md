推理默认工具配置位于 `ToolSets/config.flux_step1x.yaml`，包含 FLUX（`npu:5`）与 Step1X（`npu:7`）；完整四工具配置仍保存在 `ToolSets/config.yaml`。Agent 与 API 的工具通信使用官方 OpenAI function calling：`tools[].function.parameters` 从同一份 `conds_info` 生成 JSON Schema。每个工具保留原 Agent4AIGC 名称、类别、说明和运行时偏好，执行时仍使用 PIVOT 的独立环境方式和模型生命周期：

```python
server_load_model()
server_inference(**conditions)
server_unload_model()
```

注册表在主环境中读取配置，模型只在对应conda环境加载。默认两工具使用独立常驻HTTP服务：FLUX在18105、Step1X在18106，仅绑定本机。注册表关闭或推理进程退出后，模型继续驻留；新进程复用同一PID。显式管理模型启停使用 `python deployment/tools/serve.py start|status|stop`。传输采用原 `action/request_id/args` 信封；API工具参数和PIVOT结果XML不变。`server_unload_model` 在常驻客户端中不终止服务，实际卸载仅由显式stop触发。

设备优先级为显式device、非auto的 `AGENT4AIGC_DEVICE`、工具自身device、配置默认设备。已运行常驻服务的配置指纹必须与客户端匹配；修改权重或设备前需显式停止该服务。`AGENT4AIGC_MODEL_ROOT`、`AGENT4AIGC_CONDA_ENVS_ROOT` 可覆盖模型与环境根目录。NPU服务缺少 `ASCEND_HOME_PATH` 时自动加载 `/usr/local/Ascend/ascend-toolkit/set_env.sh`，可用 `AGENT4AIGC_CANN_SETUP` 覆盖。

`ToolsetBase.function_spec(name, host_inputs=...)` 返回可以直接放进 OpenAI `tools` 的单项定义。其 `strict` 为 `true`，对象禁止额外属性，所有暴露属性均列入 `required`；可选字段通过 `null` 表达，执行前恢复宿主默认值或省略。`parameter_schema(name)` 则返回包含全部本地参数的普通 JSON Schema。

图片路径由宿主根据当前依赖绑定，编辑工具请求 schema 前必须提供实际存在的输入文件。已绑定路径及 `host_only` 参数不会暴露给模型；四个已部署工具因此只要求模型填写 `text`、`prompt` 或 `instruction`。步数、种子和 1024 尺寸继续使用已验证的宿主默认配置。模型返回的参数必须是原生 JSON 类型，宿主拒绝未知参数、宿主字段覆盖、缺失字段和类型错误。

```python
name = "Step1X_Edit_ImageEditTool"
host_inputs = {"original_img_path": "/absolute/path/to/source.png"}
spec = tools.function_spec(name, host_inputs=host_inputs)
# 将 [spec] 作为 OpenAI tools 参数，选择该 function 后获得 call：
call = {"name": name, "arguments": {"instruction": "Change the teapot color to blue."}}
result = await tools.invoke_function(call, host_inputs=host_inputs)
```

`invoke_function` 接受 API 客户端规范化后的 `{name, arguments}`，将严格验证后的参数适配到已完成真实推理测试的 `ToolsetBase.inference`。`bind_function_arguments` 可在执行前返回完整条件，便于记录轨迹。`function_specs(names, host_inputs=...)` 可为使用相同宿主输入字段的一组工具生成定义；编辑工具应绑定各自的源图字段。

输入条件由 `conds_info` 定义，保留四工具原来的内部字段：

| 名称 | 原输入字段 | 默认步数 |
| --- | --- | --- |
| `FLUX` | `text` | 28 |
| `SD3_Tool` | `prompt` | 28 |
| `Step1X_Edit_ImageEditTool` | `original_img_path`, `instruction` | 28 |
| `AnySD_EditTool` | `instruction`, `image_path` | 100 |

独立测试和 PIVOT 内部仍兼容以下 XML 条件格式；API function calling 直接传 JSON 参数，无需模型生成 XML：

```xml
<conditions>
  <original_img_path>/absolute/path/to/source.png</original_img_path>
  <instruction>Change the teapot color to blue.</instruction>
</conditions>
```

解析器拒绝未知字段、重复字段、缺失必需条件和非 1024 的尺寸。XML 文本中的 `&`、`<` 和 `>` 应转义。编辑工具可接受 `image`/`img_path` 作为源图字段别名；流程已知的依赖图像路径可由宿主覆盖 LLM 输出。生成工具不消费此路径，保留 Planner 在有输入图时选择重新生成的能力。

工具返回 PIVOT 通用结果容器中的图像路径：

```xml
<result><image>/absolute/path/to/generated.png</image></result>
```

图像输出必须由实际推理产生且尺寸为 1024×1024，适配器不会放大工具输出。后端和适配器共用 `tool_backends/` 的加载与推理函数。失败向调用者抛出，子进程详细错误保存在 `outputs/logs/`，不会把失败包装为成功图像。

```python
from ToolSets.toolset_utils.toolset import ToolsetBase

with ToolsetBase(config_path="ToolSets/config.flux_step1x.yaml") as tools:
    result = tools.inference_sync("FLUX", {"text": "A red teapot"})
```

完整流程入口为根目录的 `inference.py`。LLM 的 OpenAI 兼容地址和模型名在 `config/llm.yaml` 中设置；工具权重默认从 `external/Step1X-Edit/models/` 下的各模型目录读取，Step1X 上游源码位于 `external/Step1X-Edit/`。

