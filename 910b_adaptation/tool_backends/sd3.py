from .common import (activate_device, dtype_for, inference_size, local_path,
                     required_text, save_result, seed_generator, unload_model)


def load_model(config: dict, device: str) -> dict:
    from diffusers import StableDiffusion3Pipeline

    device = activate_device(device)
    pipe = StableDiffusion3Pipeline.from_pretrained(local_path(config["model_path"]),
                                                   torch_dtype=dtype_for(device, config), local_files_only=True)
    pipe.to(device)
    pipe.vae.enable_slicing()
    return {"pipeline": pipe, "device": device}


def infer(model_dict: dict, conditions: dict, output_path: str, config: dict) -> str:
    import torch

    width, height = inference_size(config, conditions)
    with torch.inference_mode():
        image = model_dict["pipeline"](
            prompt=required_text(conditions, "prompt"), width=width, height=height,
            negative_prompt=config.get("negative_prompt", "monochrome, lowres, bad anatomy, worst quality, low quality"),
            guidance_scale=float(conditions.get("guidance_scale", config.get("guidance_scale", 7.0))),
            num_inference_steps=int(conditions.get("num_inference_steps", config.get("num_inference_steps", 28))),
            generator=seed_generator(conditions.get("seed", config.get("seed", 42))),
        ).images[0]
    return save_result(image, output_path)
