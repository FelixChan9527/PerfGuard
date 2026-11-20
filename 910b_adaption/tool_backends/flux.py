from .common import (activate_device, dtype_for, inference_size, local_path,
                     required_text, save_result, seed_generator, unload_model)


def load_model(config: dict, device: str) -> dict:
    from diffusers import FluxPipeline

    device = activate_device(device)
    pipe = FluxPipeline.from_pretrained(local_path(config["model_path"]),
                                       torch_dtype=dtype_for(device, config), local_files_only=True)
    pipe.to(device)
    pipe.enable_vae_slicing()
    return {"pipeline": pipe, "device": device}


def infer(model_dict: dict, conditions: dict, output_path: str, config: dict) -> str:
    import torch

    width, height = inference_size(config, conditions)
    num_steps = int(conditions.get("num_inference_steps", config.get("num_inference_steps", 28)))
    progress = config.get("_progress_callback")
    pipeline_args = {
        "prompt": required_text(conditions, "text"), "width": width, "height": height,
        "negative_prompt": config.get("negative_prompt", "monochrome, lowres, bad anatomy, worst quality, low quality"),
        "guidance_scale": float(conditions.get("guidance_scale", config.get("guidance_scale", 3.5))),
        "num_inference_steps": num_steps,
        "generator": seed_generator(conditions.get("seed", config.get("seed", 42))),
        "max_sequence_length": int(config.get("max_sequence_length", 512)),
    }
    if progress:
        progress("denoising", 0, num_steps)

        def on_step_end(_pipeline, step_index, _timestep, callback_kwargs):
            progress("denoising", step_index + 1, num_steps)
            return callback_kwargs

        pipeline_args["callback_on_step_end"] = on_step_end
        pipeline_args["callback_on_step_end_tensor_inputs"] = []
    with torch.inference_mode():
        image = model_dict["pipeline"](**pipeline_args).images[0]
    return save_result(image, output_path)
