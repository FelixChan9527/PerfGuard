from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import time
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def local_path(value: str | os.PathLike, *, directory: bool = True) -> str:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not (path.is_dir() if directory else path.is_file()):
        raise FileNotFoundError(f"Required {'directory' if directory else 'file'} does not exist: {path}")
    return str(path)


def activate_device(device: str) -> str:
    import torch

    selected = str(device or os.getenv("AGENT4AIGC_DEVICE", "cpu"))
    if selected == "auto":
        selected = os.getenv("AGENT4AIGC_DEVICE", "cpu")
        if selected == "auto":
            selected = "cpu"
    if selected.startswith("npu"):
        import torch_npu

        if not torch.npu.is_available():
            raise RuntimeError("NPU requested but torch_npu reports no available device")
        torch.npu.set_device(selected)
    elif selected.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available")
        torch.cuda.set_device(selected)
    elif selected != "cpu":
        raise ValueError(f"Unsupported device: {selected}")
    return selected


def dtype_for(device: str, config: dict):
    import torch

    name = config.get("dtype") or ("float32" if device == "cpu" else "bfloat16")
    if name not in ("float32", "float16", "bfloat16"):
        raise ValueError(f"Unsupported dtype: {name}")
    return getattr(torch, name)


def seed_generator(seed: int):
    import torch



    return torch.Generator(device="cpu").manual_seed(int(seed))


def required_text(conditions: dict, key: str) -> str:
    value = conditions.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing nonempty text condition: {key}")
    return value.strip()


def inference_size(config: dict, conditions: dict) -> tuple[int, int]:
    width = int(conditions.get("width", config.get("width", 1024)))
    height = int(conditions.get("height", config.get("height", 1024)))
    if (width, height) != (1024, 1024):
        raise ValueError("This reproduction deploys native 1024x1024 inference only")
    return width, height


def prepare_edit_image(path: str, config: dict, conditions: dict):
    from PIL import Image, ImageOps

    size = inference_size(config, conditions)
    with Image.open(local_path(path, directory=False)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")

    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return image


def validate_image(path: str, expected_size: tuple[int, int] = (1024, 1024)) -> dict:
    import hashlib
    from PIL import Image, ImageStat

    artifact = Path(path).resolve()
    with Image.open(artifact) as image:
        image.load()
        if image.size != expected_size:
            raise RuntimeError(f"Expected native {expected_size}, received {image.size}: {artifact}")
        extrema = image.convert("RGB").getextrema()
        if all(low == high for low, high in extrema):
            raise RuntimeError(f"Model produced a spatially constant image: {artifact}")
        stddev = ImageStat.Stat(image.convert("RGB")).stddev
    return {"path": str(artifact), "size": list(expected_size),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "bytes": artifact.stat().st_size, "stddev": stddev}


def save_result(image, output_path: str, *, expected_size: tuple[int, int] = (1024, 1024)) -> str:
    if image.size != expected_size:
        raise RuntimeError(f"Model generated {image.size}; native {expected_size} was requested")
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG")
    validate_image(str(target), expected_size)
    return str(target)


def unload_model(model_dict: dict[str, Any]) -> None:
    import gc
    import torch

    device = str(model_dict.get("device", "cpu"))
    model_dict.clear()
    gc.collect()
    if device.startswith("npu") and hasattr(torch, "npu"):
        with torch.npu.device(device):
            torch.npu.empty_cache()
    elif device.startswith("cuda") and torch.cuda.is_available():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a real model directly before XML tool integration")
    parser.add_argument("--tool", choices=("flux", "sd3", "step1x_edit", "anysd"), required=True)
    parser.add_argument("--config", help="JSON dict overriding backend configuration")
    parser.add_argument("--model-path")
    parser.add_argument("--base-model-path")
    parser.add_argument("--qwen-model-path")
    parser.add_argument("--source-path")
    parser.add_argument("--device", default=os.getenv("AGENT4AIGC_DEVICE", "npu:6"))
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    if args.device == "auto":
        args.device = config.get("device", "npu:6")
    model_root = Path(os.getenv("AGENT4AIGC_MODEL_ROOT", PROJECT_ROOT / "external" / "Step1X-Edit" / "models"))
    config.setdefault("model_path", str(model_root / args.tool))
    config.setdefault("base_model_path", str(model_root / "sd15"))
    config.setdefault("qwen_model_path", str(model_root / "qwen25vl"))
    config.setdefault("source_path", str(PROJECT_ROOT / "external" / "Step1X-Edit"))
    for key in ("model_path", "base_model_path", "qwen_model_path", "source_path"):
        if getattr(args, key):
            config[key] = getattr(args, key)
    conditions = {"text": args.prompt, "prompt": args.prompt, "instruction": args.prompt,
                  "original_img_path": args.image, "image_path": args.image, "seed": args.seed}
    if args.steps is not None:
        conditions["num_inference_steps"] = args.steps
    backend = importlib.import_module(f"tool_backends.{args.tool}")
    started = time.monotonic()
    model_dict = backend.load_model(config, args.device)
    try:
        path = backend.infer(model_dict, conditions, args.output, config)
        report = {"status": "passed", "mode": "direct_real_inference", "tool": args.tool,
                  "elapsed_seconds": time.monotonic() - started,
                  "num_inference_steps": conditions.get("num_inference_steps", config.get("num_inference_steps", 28 if args.tool != "anysd" else 100)),
                  "seed": args.seed, "device": model_dict["device"],
                  "model_path": str(config["model_path"]), "prompt": args.prompt,
                  **validate_image(path)}
        if args.image:
            import numpy as np
            from PIL import Image

            reference = prepare_edit_image(args.image, config, conditions)
            with Image.open(path) as edited:
                difference = np.abs(np.asarray(reference, dtype=np.float32)
                                    - np.asarray(edited.convert("RGB"), dtype=np.float32))
            report["input_image"] = local_path(args.image, directory=False)
            report["input_output_mean_absolute_difference"] = float(difference.mean())
            if not difference.any():
                raise RuntimeError("The editor returned the input image unchanged")
        Path(str(path) + ".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        unload_model(model_dict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
