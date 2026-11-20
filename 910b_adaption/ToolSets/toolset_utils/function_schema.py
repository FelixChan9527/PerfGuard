from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping
from pathlib import Path

from .condition_protocol import parse_conditions

SCALAR_TYPES = {"string", "integer", "number", "boolean"}
SCHEMA_FIELDS = {"type", "description", "enum", "const", "minimum", "maximum", "minLength", "maxLength", "pattern", "default"}


def schema_from_conditions(conditions):
    if not isinstance(conditions, Mapping):
        raise TypeError("conds_info must be a mapping")
    properties, required = {}, []
    for name, info in conditions.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][\w.-]*", name):
            raise ValueError(f"Invalid condition name: {name!r}")
        if not isinstance(info, Mapping) or info.get("type") not in SCALAR_TYPES:
            raise ValueError(f"Condition {name} must declare a supported scalar type")
        properties[name] = {key: copy.deepcopy(value) for key, value in info.items() if key in SCHEMA_FIELDS}
        if info.get("required", True) and not info.get("optional", False) and "default" not in info:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def _host_bindings(conditions, host_inputs):
    if host_inputs is None:
        host_inputs = {}
    if not isinstance(host_inputs, Mapping):
        raise TypeError("host_inputs must be a mapping")
    aliases = {}
    for name, info in conditions.items():
        for alias in [name, info.get("tag", name), *info.get("aliases", [])]:
            if alias in aliases and aliases[alias] != name:
                raise ValueError(f"Ambiguous host condition alias: {alias}")
            aliases[alias] = name
    bindings = {}
    for alias, value in host_inputs.items():
        if alias not in aliases:
            raise ValueError(f"Unknown host input: {alias}")
        name = aliases[alias]
        if name in bindings:
            raise ValueError(f"Duplicate host input: {name}")
        bindings[name] = value


    reduced = {name: dict(info, required=False) for name, info in conditions.items()}
    return parse_conditions(bindings, reduced, defaults=False)


def model_conditions(conditions, *, host_inputs=None):
    bindings = _host_bindings(conditions, host_inputs)
    for name, info in conditions.items():
        if info.get("context") == "image":
            if name not in bindings or not bindings[name]:
                raise ValueError(f"The host must bind the actual input image before requesting {name}")
            if not Path(bindings[name]).expanduser().is_file():
                raise FileNotFoundError(f"Host input image does not exist: {bindings[name]}")
    exposed = {name: info for name, info in conditions.items() if name not in bindings and not info.get("host_only")}
    return exposed, bindings


def strict_parameters(conditions):
    ordinary = schema_from_conditions(conditions)
    properties = {}
    for name, raw in ordinary["properties"].items():
        info = conditions[name]
        prop = copy.deepcopy(raw)
        prop.pop("default", None)
        optional = not info.get("required", True) or info.get("optional", False) or "default" in info
        if "const" in prop:
            prop["enum"] = [prop.pop("const")]
        if optional:
            prop["type"] = [prop["type"], "null"]
            if "enum" in prop and None not in prop["enum"]:
                prop["enum"].append(None)
            description = prop.get("description", "")
            suffix = "Use null to apply the host default." if "default" in info else "Use null when this optional field is unused."
            prop["description"] = (description + " " + suffix).strip()
        properties[name] = prop
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def function_spec(tool_name, tool_info, *, host_inputs=None):
    if not isinstance(tool_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", tool_name):
        raise ValueError(f"Invalid OpenAI function name: {tool_name!r}")
    exposed, _ = model_conditions(tool_info["conds_info"], host_inputs=host_inputs)
    return {"type": "function", "function": {
        "name": tool_name, "description": str(tool_info.get("description", "")).strip(),
        "strict": True, "parameters": strict_parameters(exposed),
    }}


def _strict_value(name, value, info):
    if value is None:
        optional = not info.get("required", True) or info.get("optional", False) or "default" in info
        if not optional:
            raise ValueError(f"Function argument {name} cannot be null")
        return
    kind = info["type"]
    valid = (isinstance(value, str) if kind == "string" else
             isinstance(value, bool) if kind == "boolean" else
             isinstance(value, int) and not isinstance(value, bool) if kind == "integer" else
             isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))
    if not valid:
        raise ValueError(f"Function argument {name} requires a native JSON {kind}")
    if isinstance(value, str):
        if "minLength" in info and len(value) < info["minLength"]:
            raise ValueError(f"Function argument {name} is too short")
        if "maxLength" in info and len(value) > info["maxLength"]:
            raise ValueError(f"Function argument {name} is too long")
        if "pattern" in info and not re.search(info["pattern"], value):
            raise ValueError(f"Function argument {name} does not match its pattern")


def bind_function_arguments(tool_info, arguments, *, host_inputs=None):
    if not isinstance(arguments, Mapping):
        raise TypeError("Function arguments must be a JSON object")
    conditions = tool_info["conds_info"]
    exposed, bindings = model_conditions(conditions, host_inputs=host_inputs)
    unknown = set(arguments) - set(exposed)
    if unknown:
        raise ValueError(f"Unknown or host-owned function arguments: {sorted(unknown)}")
    missing = set(exposed) - set(arguments)
    if missing:
        raise ValueError(f"Strict function call omitted required properties: {sorted(missing)}")
    normalized = {}
    for name, value in arguments.items():
        _strict_value(name, value, exposed[name])
        if value is not None:
            normalized[name] = value
    return parse_conditions({**normalized, **bindings}, conditions)

