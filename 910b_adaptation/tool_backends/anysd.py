from .common import (activate_device, dtype_for, local_path, prepare_edit_image,
                     required_text, save_result, seed_generator, unload_model)


def load_model(config: dict, device: str) -> dict:
    from diffusers import StableDiffusionInstructPix2PixPipeline, UNet2DConditionModel

    device = activate_device(device)
    dtype = dtype_for(device, config)
    unet = UNet2DConditionModel.from_pretrained(local_path(config["model_path"]), subfolder="unet",
                                              torch_dtype=dtype, local_files_only=True)
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        local_path(config["base_model_path"]), unet=unet, torch_dtype=dtype,
        safety_checker=None, feature_extractor=None, requires_safety_checker=False,
        local_files_only=True,
    ).to(device)
    pipe.enable_vae_slicing()
    return {"pipeline": pipe, "device": device, "adapter_name": "general"}


def infer(model_dict: dict, conditions: dict, output_path: str, config: dict) -> str:
    import torch

    if conditions.get("adapter_name", "general") != "general":
        raise ValueError("The original Agent4AIGC AnySD tool exposes only adapter_name='general'")
    image_path = conditions.get("image_path") or conditions.get("image")
    if not image_path:
        raise ValueError("AnySD requires image_path")
    image = prepare_edit_image(image_path, config, conditions)
    with torch.inference_mode():
        result = model_dict["pipeline"](
            prompt=required_text(conditions, "instruction"), image=image,
            guidance_scale=float(conditions.get("guidance_scale", config.get("guidance_scale", 3.0))),
            image_guidance_scale=float(conditions.get("image_guidance_scale", config.get("image_guidance_scale", 3.0))),
            num_inference_steps=int(conditions.get("num_inference_steps", config.get("num_inference_steps", 100))),
            generator=seed_generator(conditions.get("seed", config.get("seed", 42))),
        ).images[0]
    return save_result(result, output_path)
