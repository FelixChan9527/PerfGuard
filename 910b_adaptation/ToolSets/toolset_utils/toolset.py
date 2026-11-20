from __future__ import annotations

import asyncio
import atexit
import importlib
import os
import textwrap
import threading
from pathlib import Path

from .condition_protocol import build_condition_prompt, format_conditions, parse_conditions
from .function_schema import bind_function_arguments, function_spec, schema_from_conditions
from .runtime import SubprocessTool
from .tool_base import read_config


class _AgentTool:
    def __init__(self, owner, name, info):
        self.owner, self.name, self.tool_info = owner, name, info
        self.category = info["category"]
        self.preference = dict(info.get("preference", {}))
        self.description = textwrap.dedent(info["description"]).replace("\n", " ").strip()
        self.query = build_condition_prompt(name, info)
        example = {key: meta.get("example", "") for key, meta in info["conds_info"].items() if not meta.get("host_only")}
        self.example = "<example>\n" + self.query + "\n" + format_conditions(example, info["conds_info"]) + "\n</example>\n"

    async def run(self, reply, image_override=None):
        image_key = next((key for key, info in self.tool_info["conds_info"].items() if info.get("context") == "image"), None)
        schema = self.tool_info["conds_info"]
        if image_override and image_key:


            reduced = {key: dict(info) for key, info in schema.items()}
            reduced[image_key]["required"] = False
            conditions = parse_conditions(reply, reduced)
            image_override = str(Path(image_override).resolve())
            if not Path(image_override).is_file():
                raise FileNotFoundError(image_override)
            conditions[image_key] = image_override
            conditions = parse_conditions(conditions, schema)
        else:


            conditions = self.owner.parse_conditions(self.name, reply)
        clean = format_conditions(conditions, self.tool_info["conds_info"])
        result = await self.owner.inference(self.name, conditions)
        return result, clean


class ToolsetBase:
    def __init__(self, toolset_name=None, device="auto", *, config_path=None, execution_mode=None, enabled_tools=None, max_loaded_tools=None):
        self.config = read_config(config_path)
        execution_mode = execution_mode or self.config.get('execution_mode', 'subprocess')
        if execution_mode not in {"subprocess", "in_process", "resident"}:
            raise ValueError("execution_mode must be subprocess, in_process or resident")
        self.toolset_name = toolset_name or self.config.get("tools_set_name", "AIGC")
        selected = os.getenv("AGENT4AIGC_DEVICE") if device == "auto" else device
        self._device_override = selected if selected not in {None, '', 'auto'} else None
        self.device = self._device_override or self.config.get("default_device", "npu:6")
        self.execution_mode = execution_mode
        self.max_loaded_tools = max(1, int(max_loaded_tools if max_loaded_tools is not None
                                           else self.config.get("max_loaded_tools", 1)))
        self.tool_configs = self.config["tools_set_info"]
        if enabled_tools is not None:
            self.tool_configs = {name: self.tool_configs[name] for name in enabled_tools}
        self._instances = {}
        self._lock = threading.RLock()
        self.AIGC_Tools = {name: _AgentTool(self, name, info) for name, info in self.tool_configs.items()}
        self.AIGC_Tools_description = self.describe()
        self.Auxiliary_Tools = {}
        self.Auxiliary_Tools_description = "The following are the **Auxiliary Tools** descriptions:\n"
        atexit.register(self.close)

    def get_info(self, name):
        return dict(self.tool_configs[name])

    def device_for(self, name):
        return self._device_override or self.tool_configs[name].get("device", self.device)

    def describe(self):
        return "The following are the **AIGC Tools** descriptions:\n" + "".join(
            f"**{name}**: {tool.description}\n" for name, tool in self.AIGC_Tools.items()
        )

    def condition_prompt(self, name, task=""):
        return build_condition_prompt(name, self.tool_configs[name], task)

    def parse_conditions(self, name, response):
        return parse_conditions(response, self.tool_configs[name]["conds_info"])

    def parameter_schema(self, name):
        return schema_from_conditions(self.tool_configs[name]["conds_info"])

    def function_spec(self, name, *, host_inputs=None):
        return function_spec(name, self.tool_configs[name], host_inputs=host_inputs)

    def function_specs(self, names=None, *, host_inputs=None):
        return [self.function_spec(name, host_inputs=host_inputs) for name in
                (self.tool_configs if names is None else names)]

    def bind_function_arguments(self, name, arguments, *, host_inputs=None):
        return bind_function_arguments(self.tool_configs[name], arguments, host_inputs=host_inputs)

    async def invoke_function(self, call, *, host_inputs=None):
        if not isinstance(call, dict) or set(call) != {"name", "arguments"}:
            raise ValueError("Function call requires exactly name and arguments")
        conditions = self.bind_function_arguments(call["name"], call["arguments"], host_inputs=host_inputs)
        return await self.inference(call["name"], conditions)

    def _load(self, name):
        if name in self._instances:
            return self._instances[name]
        while len(self._instances) >= self.max_loaded_tools:
            old = next(iter(self._instances))
            self._instances.pop(old).server_unload_model()
        info = self.tool_configs[name]
        device = self.device_for(name)
        if self.execution_mode == "resident":
            from .resident import ResidentTool
            instance = ResidentTool(name, info, device)
        elif self.execution_mode == "subprocess":
            instance = SubprocessTool(name, info, device)
        else:
            cls = getattr(importlib.import_module(info["module"]), info["class_name"])
            instance = cls(device=device, model_path=info.get("model_path"), tool_info=info)
        try:
            instance.server_load_model()
        except BaseException:
            instance.server_unload_model()
            raise
        self._instances[name] = instance
        return instance

    def inference_sync(self, tool_name, conditions):
        conditions = self.parse_conditions(tool_name, conditions)
        for key, info in self.tool_configs[tool_name]["conds_info"].items():
            if info.get("context") == "image":
                path = Path(conditions[key]).expanduser()
                if not path.is_file():
                    raise FileNotFoundError(f"Input image does not exist: {path}")
                conditions[key] = str(path.resolve())
        with self._lock:
            try:
                return self._load(tool_name).server_inference(**conditions)
            except Exception:


                instance = self._instances.pop(tool_name, None)
                if instance is not None:
                    instance.server_unload_model()
                raise

    async def inference(self, tool_name, conditions):
        return await asyncio.to_thread(self.inference_sync, tool_name, conditions)

    async def execute(self, tool_name, condition_xml):
        return await self.inference(tool_name, condition_xml)

    def close(self):
        with self._lock:
            while self._instances:
                _, instance = self._instances.popitem()
                instance.server_unload_model()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

