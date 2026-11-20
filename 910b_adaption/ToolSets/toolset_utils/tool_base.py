from __future__ import annotations

import importlib
import json
import os
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from .condition_protocol import build_condition_prompt, parse_conditions

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "ToolSets" / "config.flux_step1x.yaml"
EMPTY_XML = "<result></result>"


def read_config(path=None):
    path = Path(path or os.getenv("AGENT4AIGC_TOOL_CONFIG", DEFAULT_CONFIG))
    raw = path.read_text(encoding="utf-8")
    try:
        config = json.loads(raw)
    except ValueError:
        import yaml
        config = yaml.safe_load(raw)
    model_root = Path(os.getenv("AGENT4AIGC_MODEL_ROOT", PROJECT_ROOT / "external" / "Step1X-Edit" / "models"))
    for info in config["tools_set_info"].values():
        for key, value in list(info.items()):
            if not key.endswith("_path") or not isinstance(value, str):
                continue
            value = value.replace("${AGENT4AIGC_MODEL_ROOT}", str(model_root))
            value = value.replace("${AGENT4AIGC_PROJECT_ROOT}", str(PROJECT_ROOT))
            info[key] = os.path.expandvars(os.path.expanduser(value))
    return config


class ToolBase:
    EMPTY_XML = EMPTY_XML

    def __init__(self, name, model_path=None, device="npu:6", *, tool_info=None):
        self.name = name
        self.tool_info = dict(tool_info if tool_info is not None else read_config()["tools_set_info"][name])
        self.description = self.tool_info.get("description", "")
        self.conds_info = self.tool_info.get("conds_info", {})
        self.conda_env = self.tool_info.get("conda_env")
        self.model_path = model_path or self.tool_info.get("model_path")
        selected = os.getenv("AGENT4AIGC_DEVICE") if device == "auto" else device
        self.device = selected if selected not in {None, '', 'auto'} else self.tool_info.get("device", "npu:6")
        self.model_dict = None

    @staticmethod
    def build_condition_prompt(tool_name, tool_info, task):
        return build_condition_prompt(tool_name, tool_info, task)

    @staticmethod
    def parse_condition_response(conds_info, llm_response):
        try:
            return parse_conditions(llm_response, conds_info), []
        except (TypeError, ValueError) as exc:
            return {}, [str(exc)]

    def _activate_device(self):
        import torch
        if self.device.startswith("npu"):
            import torch_npu
            if not torch.npu.is_available():
                raise RuntimeError("NPU was requested but torch_npu cannot access an NPU")
            torch.npu.set_device(self.device)
        elif self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            torch.cuda.set_device(self.device)
        elif self.device != "cpu":
            raise ValueError(f"Unsupported inference device: {self.device}")

    def output_path(self, suffix=".png"):
        root = Path(os.getenv("AGENT4AIGC_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "tools"))
        root.mkdir(parents=True, exist_ok=True)
        return str((root / f"{self.name}_{uuid.uuid4().hex}{suffix}").resolve())

    @staticmethod
    def lines_to_xml(value):
        if value is None:
            return EMPTY_XML
        if isinstance(value, str) and value.lstrip().startswith(("<result>", "<results>")):
            root = ET.fromstring(value)
            root.tag = "result"
        else:
            root = ET.Element("result")
            ET.SubElement(root, "image").text = os.fspath(value)
        return ET.tostring(root, encoding="unicode", short_empty_elements=False)

    def server_load_model(self):
        raise NotImplementedError

    def server_inference(self, **kwargs):
        raise NotImplementedError

    def server_unload_model(self):
        self.model_dict = None
        import gc
        gc.collect()


class StandaloneBackendTool(ToolBase):
    def server_load_model(self):
        self._activate_device()
        self._backend = importlib.import_module(self.tool_info["backend"])
        config = dict(self.tool_info, model_path=self.model_path)
        self.model_dict = self._backend.load_model(config, self.device)

    def server_inference(self, **kwargs):
        from PIL import Image
        conditions = parse_conditions(kwargs, self.conds_info)
        if self.model_dict is None:
            raise RuntimeError("server_load_model() must complete before inference")
        output = self.output_path()
        config = dict(self.tool_info, model_path=self.model_path,
                      _progress_callback=getattr(self, "_progress_callback", None))
        result = self._backend.infer(self.model_dict, conditions, output, config)
        if isinstance(result, Image.Image):
            result.save(output)
        elif isinstance(result, (str, os.PathLike)):
            output = os.fspath(result)
        if not Path(output).is_file():
            raise RuntimeError(f"Backend did not produce an output image: {output}")
        with Image.open(output) as image:
            image.load()
            if image.size != (1024, 1024):
                raise RuntimeError(f"Expected native 1024x1024 output; backend returned {image.size}")
        return self.lines_to_xml(str(Path(output).resolve()))

    def server_unload_model(self):
        unload = getattr(getattr(self, "_backend", None), "unload_model", None)
        if unload and self.model_dict is not None:
            unload(self.model_dict)
        super().server_unload_model()

