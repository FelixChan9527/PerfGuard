from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping


def schema_from_conditions(conditions):
    from .function_schema import schema_from_conditions as convert
    return convert(conditions)


def condition_tag(name, info):
    tag = info.get("tag", name)
    if not re.fullmatch(r"[A-Za-z_][\w.-]*", tag):
        raise ValueError(f"Invalid condition tag: {tag!r}")
    return tag


def format_conditions(values, conditions=None):
    root = ET.Element("conditions")
    for name, value in values.items():
        info = (conditions or {}).get(name, {})
        node = ET.SubElement(root, condition_tag(name, info))
        node.text = value if isinstance(value, str) else json.dumps(value, allow_nan=False)
    return ET.tostring(root, encoding="unicode", short_empty_elements=False)


def _decode(value, info, name):
    kind = info.get("type", "string")
    if kind == "string":
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        value = value.strip()
    elif kind in {"integer", "number"}:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise ValueError(f"{name} must be a JSON {kind}") from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite {kind}")
        if kind == "integer" and not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    elif kind == "boolean":
        if isinstance(value, str) and value in {"true", "false"}:
            value = value == "true"
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a boolean")
    else:
        raise ValueError(f"Unsupported AIGC scalar condition type: {kind}")
    if "const" in info and value != info["const"]:
        raise ValueError(f"{name} must be {info['const']}")
    if "enum" in info and value not in info["enum"]:
        raise ValueError(f"{name} must be one of {info['enum']}")
    for bound, operation in (("minimum", lambda x, y: x < y), ("maximum", lambda x, y: x > y)):
        if bound in info and operation(value, info[bound]):
            raise ValueError(f"{name} violates {bound} {info[bound]}")
    return value


def parse_conditions(text_or_values, conditions, *, defaults=True):
    if isinstance(text_or_values, Mapping):
        supplied = list(text_or_values.items())
    elif isinstance(text_or_values, str):
        text = text_or_values.strip()
        text = re.sub(r"^```(?:xml)?\s*|\s*```$", "", text)
        match = re.search(r"<conditions(?:\s[^>]*)?>.*?</conditions>", text, re.S)
        if match:
            text = match.group()
        elif not text.startswith("<conditions"):


            text = "<conditions>" + text + "</conditions>"
        try:
            root = ET.fromstring(text)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid condition XML: {exc}") from exc
        if root.tag != "conditions" or root.attrib or (root.text or "").strip():
            raise ValueError("Expected a plain <conditions> container")
        supplied = []
        for node in root:
            if node.attrib or len(node) or (node.tail or "").strip():
                raise ValueError(f"Condition {node.tag} must contain plain scalar text")
            supplied.append((node.tag, node.text or ""))
    else:
        raise TypeError("Conditions must be XML text or a mapping")
    aliases = {}
    for name, info in conditions.items():
        for alias in [name, condition_tag(name, info), *info.get("aliases", [])]:
            if alias in aliases and aliases[alias] != name:
                raise ValueError(f"Ambiguous condition alias: {alias}")
            aliases[alias] = name
    result = {}
    for alias, value in supplied:
        if alias not in aliases:
            raise ValueError(f"Unknown condition: {alias}")
        name = aliases[alias]
        if name in result:
            raise ValueError(f"Duplicate condition: {name}")
        result[name] = _decode(value, conditions[name], name)
    for name, info in conditions.items():
        if name not in result and "default" in info and defaults:
            result[name] = _decode(info["default"], info, name)
        if info.get("required", True) and not info.get("optional", False) and "default" not in info:
            if result.get(name) is None or result.get(name) == "":
                raise ValueError(f"Missing required condition: {name}")
    return result


def build_condition_prompt(tool_name, tool_info, task=""):
    lines = [
        f"Tool: {tool_name} — {tool_info.get('description', '').strip()}",
        "Extract only the listed condition fields inside <conditions>. Omit unused optional fields. Follow each field's XML form below; use item children for lists and keep scalar text free of invented markup.",
        "Use the step for intent and inputs for actual file/data values.",
        "For file tags, copy the exact matching path from Step information; never invent paths or substitute a mask for an RGB image.",
        "Use actual values only. Verify that every required path, label, box, index, or number is present in the step or inputs.",
        "XML-escape ampersands and angle brackets that are part of prompt text.",
        "<conditions>",
    ]
    for name, info in tool_info.get("conds_info", {}).items():
        if info.get("host_only"):
            continue
        tag = condition_tag(name, info)
        lines.append(f"<{tag}>{info.get('description', info.get('type', 'value'))}</{tag}>")
    lines.append("</conditions>")
    if task:
        lines.extend(["Step information:", task])
    return "\n".join(lines)

