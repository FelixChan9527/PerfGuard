from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from .common import (activate_device, local_path, prepare_edit_image,
                     required_text, save_result, unload_model)


def load_model(config: dict, device: str) -> dict:
    from .step1x_compat import prepare_source

    device = activate_device(device)
    if config.get("version", "v1.1") != "v1.1":
        raise ValueError("This reproduction pins the original Agent4AIGC Step1X-Edit version v1.1")
    model_path = Path(local_path(config["model_path"]))
    checkpoint = local_path(model_path / "step1x-edit-v1p1-official.safetensors", directory=False)
    vae = local_path(model_path / "vae.safetensors", directory=False)
    source_path = prepare_source(local_path(config["source_path"]), device)
    if source_path not in sys.path:
        sys.path.insert(0, str(source_path))
    module_name = "_agent4aigc_step1x_v11_inference"
    specification = importlib.util.spec_from_file_location(module_name, Path(source_path) / "inference.py")
    if specification is None or specification.loader is None:
        raise ImportError("Could not load prepared Step1X inference implementation")
    implementation = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = implementation
    specification.loader.exec_module(implementation)
    generator_class = implementation.ImageGenerator
    generator = generator_class(
        ae_path=vae, dit_path=checkpoint,
        qwen2vl_model_path=local_path(config["qwen_model_path"]),
        device=device, max_length=int(config.get("max_length", 640)),
        quantized=False, offload=bool(config.get("offload", True)),
        lora=None, mode="torch", version="v1.1",
    )
    return {"pipeline": generator, "device": device, "source_path": str(source_path),
            "implementation": implementation}


def infer(model_dict: dict, conditions: dict, output_path: str, config: dict) -> str:
    image_path = conditions.get("original_img_path") or conditions.get("image")
    if not image_path:
        raise ValueError("Step1X-Edit requires original_img_path")
    image = prepare_edit_image(image_path, config, conditions)
    num_steps = int(conditions.get("num_inference_steps", config.get("num_inference_steps", 28)))
    progress = config.get("_progress_callback")
    implementation = model_dict["implementation"]
    original_tqdm = implementation.tqdm
    progress_total = max(num_steps, 1)

    if progress:
        progress("denoising", 0, progress_total)

        def tracked_tqdm(iterable, *args, **kwargs):
            total = kwargs.get("total") or progress_total
            for current, item in enumerate(iterable, start=1):
                yield item
                progress("denoising", current, total)

        implementation.tqdm = tracked_tqdm
    try:
        result = model_dict["pipeline"].generate_image(
            required_text(conditions, "instruction"), negative_prompt="", ref_images=image,
            num_samples=1, num_steps=num_steps,
            cfg_guidance=float(conditions.get("guidance_scale", config.get("guidance_scale", 6.0))),
            seed=int(conditions.get("seed", config.get("seed", 42))), show_progress=True,
            size_level=1024, height=1024, width=1024,
        )[0]
    finally:
        implementation.tqdm = original_tqdm
    return save_result(result, output_path)
