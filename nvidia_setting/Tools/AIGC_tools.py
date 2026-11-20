import sys, gc, re
import os, random
sys.path.append("EasyControl")
sys.path.append("DreamO")
sys.path.append("ThreeDIS")
sys.path.append("Inpaint_Anything")
sys.path.append("DragonDiffusion")

sys.path.insert(0, "./UltraEdit/diffusers/src")
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3_instructpix2pix import StableDiffusion3InstructPix2PixPipeline
sys.path.pop(0) 
for mod in list(sys.modules):
    if mod == "diffusers" or mod.startswith("diffusers."):
        del sys.modules[mod]
#from diffusers import StableDiffusion3InstructPix2PixPipeline
from diffusers.utils import load_image

from Inpaint_Anything.lama_inpaint import inpaint_img_with_lama
from Inpaint_Anything.utils import load_img_to_array, save_array_to_img, dilate_mask
import json
import torch
from modelscope import FluxPipeline
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from .basic_tool import Basic_Tool
from modelscope import snapshot_download
from diffusers import FluxFillPipeline
from OminiControl.omini.pipeline.flux_omini import Condition, generate, seed_everything
from ThreeDIS.threeDIS.utils import seed_everything, load_migc
from ThreeDIS.threeDIS.pipeline_stable_diffusion_layout2depth import StableDiffusionL2DPipeline, MIGCProcessor
from ThreeDIS.threeDIS.pipeline_flux_rendering import FluxRenderingPipeline
from ThreeDIS.threeDIS.detail_renderer.detail_renderer_flux import DetailRendererFLUX
from ThreeDIS.threeDIS.utils import get_all_processor_keys
from DreamO.dreamo_generator import Generator
from EasyControl.src.pipeline import FluxPipeline as EasyControlFluxPipeline
from EasyControl.src.transformer_flux import FluxTransformer2DModel
from EasyControl.src.lora_helper import set_single_lora
from copy import deepcopy
from controlnet_aux import OpenposeDetector, HEDdetector, MidasDetector
import cv2
from torchvision.transforms import PILToTensor
from pytorch_lightning import seed_everything
from CreatiLayout.src.models.transformer_sd3_SiamLayout import SiamLayoutSD3Transformer2DModel
from CreatiLayout.src.pipeline.pipeline_sd3_CreatiLayout import CreatiLayoutSD3Pipeline

# 导入你的 DragonModels 类和其他必要函数
from DragonDiffusion.src.demo.model import DragonModels
from Step1X_Edit.inference import ImageGenerator

from diffusers import DiffusionPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from diffusers import StableDiffusionXLPipeline
from IP_Adapter.ip_adapter import IPAdapterXL
from IP_Adapter.ip_adapter import IPAdapterPlusXL

from AnySD.anysd.src.model import AnySDPipeline, choose_expert
from AnySD.anysd.src.utils import choose_book, get_experts_dir
from AnySD.anysd.train.valid_log import download_image

from diffusers import StableDiffusion3Pipeline

batch_num = 1

class FLUX(Basic_Tool):
    """
    文生图工具FLUX
    请按description文件输入的条件顺序进行提取输入，以及按对应顺序进行输出结果
    """
    async def run_function(self, conditions) -> list:
        """
        该代码是调用FLUX实现高质量文生图效果
        """
        
        prompt = conditions[0]      # 只有1个条件
        # save_path = conditions[1]
        save_path = self.get_save_output_dirs(num=batch_num)
        
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16
        ).to("cuda")
        pipe.safety_checker = None
        
        generator = torch.Generator("cuda").manual_seed(44)
        images = pipe(
            prompt,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            guidance_scale=3.5,
            num_inference_steps=28,  # 降低步数，加快推理
            max_sequence_length=512,
            num_images_per_prompt=batch_num,
            generator=generator,
            ).images
        # image = image.resize((1024, 1024) )
        # image.save(save_path)
    
        for idx, img in enumerate(images):
            img = img.resize((1024, 1024))  # 缩放每张图片为1024x1024
            img.save(save_path[idx])
        
        del pipe, img, images
        torch.cuda.empty_cache()
        gc.collect()
        
        return save_path
    
class Pose2Image(Basic_Tool):
    async def run_function(self, conditions) -> list:
        """
        conditions: [text_prompt, pose_image_path]
        """
        prompt, pose_image_path = conditions
        save_path = self.get_save_output_dirs(num=1)[0]
        pose_image = Image.open(pose_image_path).convert("RGB")
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-sd21-openpose-diffusers", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        result_image = pipe(prompt=prompt, image=pose_image, num_inference_steps=30).images[0]
        result_image = result_image.resize((1024, 1024) )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result_image.save(save_path)

        del result_image, pose_image, pipe, controlnet
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]


class Sketch2Image(Basic_Tool):
    async def run_function(self, conditions: list) -> list:
        """
        conditions: [text_prompt, sketch_image_path]
        """
        prompt, sketch_path = conditions
        save_path = self.get_save_output_dirs(num=1)[0]
        image = Image.open(sketch_path).convert("RGB")
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-sd21-hed-diffusers", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")

        result = pipe(prompt=prompt, image=image, num_inference_steps=30).images[0]
        result = result.resize((1024, 1024) )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.save(save_path)

        del result, image, pipe, controlnet
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]


class Depth2Image(Basic_Tool):
    async def run_function(self, conditions: list) -> list:
        """
        conditions: [text_prompt, depth_image_path]
        """
        prompt, depth_image_path = conditions
        save_path = self.get_save_output_dirs(num=1)[0]
        depth_image = Image.open(depth_image_path).convert("RGB")

        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-sd21-depth-diffusers", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.to("cuda")

        result = pipe(prompt=prompt, image=depth_image, num_inference_steps=30).images[0]
        result = result.resize((1024, 1024) )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result.save(save_path)

        del result, depth_image, pipe, controlnet
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]

class ICEdit(Basic_Tool):
    """
    图像编辑工具ICEdit
    请按description文件输入的条件顺序进行提取输入，以及按对应顺序进行输出结果
    """
    def load_and_resize_image(self, image_path: str, target_width: int = 512) -> Image.Image:
        """
        加载图像并调整宽度为 target_width，保持纵横比，并保证高度为8的倍数。

        Args:
            image_path (str): 图像路径
            target_width (int): 目标宽度（默认512）

        Returns:
            PIL.Image.Image: 调整尺寸后的图像
        """
        image = Image.open(image_path).convert("RGB")
        if image.size[0] != target_width:
            scale = target_width / image.size[0]
            new_height = (int(image.size[1] * scale) // 8) * 8
            image = image.resize((target_width, new_height))
        return image

    def create_diptych_and_mask(self, image: Image.Image) -> tuple[Image.Image, Image.Image]:
        """
        构造左右拼接图像和对应的mask（用于右半部分编辑）

        Args:
            image (PIL.Image.Image): 原始图像

        Returns:
            Tuple[PIL.Image.Image, PIL.Image.Image]: 拼接图像 和 mask 图像
        """
        width, height = image.size
        combined_image = Image.new("RGB", (width * 2, height))
        combined_image.paste(image, (0, 0))
        combined_image.paste(image, (width, 0))

        mask_array = np.zeros((height, width * 2), dtype=np.uint8)
        mask_array[:, width:] = 255
        mask = Image.fromarray(mask_array)

        return combined_image, mask

    def icedit_edit_image(self, 
        image_path: str,
        instruction: str,
        seed: int = 42,
        flux_path: str = "black-forest-labs/flux.1-fill-dev",   # "black-forest-labs/flux.1-fill-dev"
        lora_path: str = "RiverZ/normal-lora",
        guidance_scale: float = 50,
        num_inference_steps: int = 28,
    ) -> Image.Image:
        """
        使用ICEdit进行图像编辑（风格迁移、物体替换、人物修改等）

        参数:
            image_path (str): 输入图像路径（宽度建议512px）
            instruction (str): 自然语言指令
            seed (int): 随机种子，默认42
            flux_path (str): Flux模型路径
            lora_path (str): LoRA权重路径
            guidance_scale (float): 控制编辑强度，默认50
            num_inference_steps (int): 采样步数，默认28
            enable_cpu_offload (bool): 是否使用CPU卸载
            output_dir (str): 输出保存目录，默认 ./outputs

        Returns:
            PIL.Image.Image: 编辑结果图像
        """
        image = self.load_and_resize_image(image_path)
        combined_image, mask = self.create_diptych_and_mask(image)

        flux_path = snapshot_download(flux_path, max_workers=4)
        pipe = FluxFillPipeline.from_pretrained(flux_path, torch_dtype=torch.bfloat16)
        pipe.load_lora_weights(lora_path)

        pipe = pipe.to("cuda")

        full_instruction = (
            f"A diptych with two side-by-side images of the same scene. "
            f"On the right, the scene is exactly the same as on the left but {instruction}"
        )

        result_image = pipe(
            prompt=full_instruction,
            image=combined_image,
            mask_image=mask,
            height=combined_image.size[1],
            width=combined_image.size[0],
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]
        
        # 裁剪右半部分生成图像
        width = image.size[0]
        result_image = result_image.crop((width, 0, width * 2, image.size[1]))

        return result_image

    async def run_function(self, conditions: list) -> list:
        """
        该代码是调用ICEdit实insctrion图像编辑效果
        """
        instruction = conditions[0]
        img_path = conditions[1]
        save_dir = self.get_save_output_dirs(num=1)[0]
        
        result_image = self.icedit_edit_image(
            image_path=img_path,
            instruction=instruction,
            seed=123
        )
        
        # ori_img = Image.open(img_path)
        # ori_w, ori_h =  ori_img.size
        
        # result_image = result_image.resize((ori_w, ori_h))
        result_image = result_image.resize((1024, 1024) )
        result_image.save(save_dir)
        
        del result_image
        torch.cuda.empty_cache()
        gc.collect()
        
        return [save_dir]

class OminiControlInpainting(Basic_Tool):
    """
    图像编辑工具ICEdit
    请按description文件输入的条件顺序进行提取输入，以及按对应顺序进行输出结果
    """
    async def run_function(self, conditions: list) -> list:
        """
        该代码是调用OminiControl实现inpainting图像编辑效果
        """
        instruction = conditions[0]
        img_path = conditions[1]
        boxes = conditions[2]
        save_dir = self.get_save_output_dirs(num=1)[0]
        
        torch.cuda.empty_cache()
        
        pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
        pipe = pipe.to("cuda")
        pipe.load_lora_weights(
            "Yuanshi/OminiControl",
            weight_name=f"experimental/fill.safetensors",
            adapter_name="fill")
        
        image = Image.open(img_path).convert("RGB")
        
        original_width, original_height = image.size

        # resize 图像
        image = image.resize((512, 512))
        
        masked_image = image.copy()
        boxes = np.array(boxes)
        
        if boxes.ndim == 1:  # 一维，自动补成二维
            boxes = np.expand_dims(boxes, axis=0)
        # 计算缩放比例
        scale_x = 512 / original_width
        scale_y = 512 / original_height

        # 缩放 boxes
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y  # y1, y2
        boxes = boxes.astype(int)

        # 遮罩
        if len(boxes.shape) == 2:  # 多个 box
            for box in boxes:
                masked_image.paste((0, 0, 0), tuple(box))  # PIL 要 tuple
        elif len(boxes.shape) == 1:
            masked_image.paste((0, 0, 0), tuple(boxes))
        
        condition = Condition(masked_image, "fill")
        
        seed_everything(42)
        result_img = generate(
            pipe,
            instruction,
            conditions=[condition],
        ).images[0]
        
        result_img=result_img.resize((1024, 1024))
        result_img.save(save_dir)
        
        del result_img, pipe
        torch.cuda.empty_cache()
        gc.collect()
        
        return [save_dir]

class Pose2Image(Basic_Tool):
    async def run_function(self, conditions) -> list:
        """
        conditions: [text_prompt,input_image_path]
        return: [result_image_path]
        """
        prompt, input_image_path = conditions
        save_path = self.get_save_output_dirs(num=1)[0]

        # Step 1: 提取姿态图
        image = Image.open(input_image_path).convert("RGB")
        image = image.resize((1024, 1024) )
        pose_extractor = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        pose_image = pose_extractor(image)

        # Step 2: ControlNet 图像生成
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-sd21-openpose-diffusers",
            torch_dtype=torch.float16
        )

        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            controlnet=controlnet,
            torch_dtype=torch.float16
        ).to("cuda")

        result_image = pipe(
            prompt=prompt,
            image=pose_image,
            num_inference_steps=30
        ).images[0]
        result_image = result_image.resize((1024, 1024) )
        result_image.save(save_path)

        # 清理
        del image, pose_image, result_image, pipe, controlnet, pose_extractor
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]
    
class Sketch2Image(Basic_Tool):
    async def run_function(self, conditions: list) -> list:
        """
        conditions: [text_prompt, input_color_image_path]
        Returns: [save_path]
        """
        prompt, input_image_path = conditions
        save_path = self.get_save_output_dirs(num=1)[0]

        # === 1. 读取图像并提取草图 ===
        image = Image.open(input_image_path).convert("RGB")
        image = image.resize((1024, 1024) )
        sketch_extractor = HEDdetector.from_pretrained("lllyasviel/Annotators")
        sketch_image = sketch_extractor(image)

        # === 2. 加载 ControlNet 管道进行生成 ===
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-sd21-hed-diffusers", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")

        # === 3. 根据 sketch + prompt 生成图像 ===
        result = pipe(prompt=prompt, image=sketch_image, num_inference_steps=30).images[0]

        # === 4. 保存图像 ===
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result = result.resize((1024, 1024) )
        result.save(save_path)

        # === 5. 清理内存 ===
        del result, sketch_image, image, pipe, controlnet, sketch_extractor
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]

class Depth2Image(Basic_Tool):
    async def run_function(self, conditions: list) -> list:
        """
        conditions: [text_prompt, input_color_image_path]
        Returns: [save_path]
        """
        prompt, input_image_path = conditions
        save_path = self.get_save_output_dirs(num=1)[0]

        # === 1. 提取深度图 ===
        image = Image.open(input_image_path).convert("RGB")
        image = image.resize((1024, 1024) )
        depth_extractor = MidasDetector.from_pretrained("lllyasviel/Annotators")
        depth_image = depth_extractor(image)

        # === 2. 初始化 ControlNet 生成管道 ===
        controlnet = ControlNetModel.from_pretrained(
            "thibaud/controlnet-sd21-depth-diffusers", torch_dtype=torch.float16
        )
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            controlnet=controlnet,
            torch_dtype=torch.float16
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        pipe.to("cuda")

        # === 3. 图像生成 ===
        result = pipe(prompt=prompt, image=depth_image, num_inference_steps=30).images[0]

        # === 4. 保存结果 ===
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        result = result.resize((1024, 1024) )
        result.save(save_path)

        # === 5. 显存清理 ===
        del result, image, depth_image, pipe, controlnet, depth_extractor
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]


class EasyControl(Basic_Tool):
    async def run_function(self, conditions):
        """
        conditions: [prompt, image_path]
        """
        prompt, input_image_path = conditions
        save_path = self.get_save_output_dirs(num=1)[0]

        # 加载图像
        subject_image = Image.open(input_image_path).convert("RGB")
        subject_image = subject_image.resize((1024, 1024) )
        # 设置路径
        lora_path = "./EasyControl/models"
        subject_lora_path = f"{lora_path}/subject.safetensors"
        local_dir = "~/.cache/modelscope/FLUX.1-dev"
        modelscope_id = "black-forest-labs/FLUX.1-dev"

        if not os.path.isdir(local_dir):
            local_dir = snapshot_download(modelscope_id)

        base_path = local_dir
        device = "cuda"

        # 初始化 pipe（可根据需要改为外部初始化避免重复）
        pipe = EasyControlFluxPipeline.from_pretrained(base_path, torch_dtype=torch.bfloat16, device=device)
        transformer = FluxTransformer2DModel.from_pretrained(
            base_path,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device=device
        )
        pipe.transformer = transformer
        pipe.to(device)

        # 加载 LoRA 控制模块
        set_single_lora(pipe.transformer, subject_lora_path, lora_weights=[1], cond_size=512)

        # 推理生成
        image = pipe(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=15,
            max_sequence_length=256,
            generator=torch.Generator("cpu").manual_seed(5),
            subject_images=[subject_image],
            cond_size=512
        ).images[0]

        # 保存生成图像
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        image = image.resize((1024, 1024) )
        image.save(save_path)

        # 清除缓存
        for _, attn_processor in pipe.transformer.attn_processors.items():
            attn_processor.bank_kv.clear()

        del image, attn_processor, pipe
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]


class DreamO(Basic_Tool):
    async def run_function(self, conditions):
        """
        conditions: [prompt(str), image_path(str)]
        return: [save_path(str)]
        """
        prompt, image_path = conditions
        save_path = self.get_save_output_dirs(num=1)[0]

        # 初始化生成器（建议外部持久化，这里简化）
        generator_model = Generator(
            version='v1.1',
            quant='int8',
            no_turbo=False,
            offload=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        # 加载参考图像
        ref_image = np.array(Image.open(image_path).convert("RGB"))
        #ref_image = ref_image.resize((1024, 1024) )
        # 构造参考条件
        ref_conds, _, _ = generator_model.pre_condition(
            ref_images=[ref_image, None],
            ref_tasks=['style', 'style'],
            ref_res=512,
            seed=123456789,
        )

        # 推理生成图像
        image = generator_model.dreamo_pipeline(
            prompt=prompt,
            width=1024,
            height=1024,
            num_inference_steps=12,
            guidance_scale=4.5,
            ref_conds=ref_conds,
            generator=torch.Generator("cuda").manual_seed(42),
            true_cfg_scale=1.0,
            true_cfg_start_step=0,
            true_cfg_end_step=0,
            negative_prompt="",
            neg_guidance_scale=3.5,
            first_step_guidance_scale=4.5,
        ).images[0]
        image = image.resize((1024, 1024))
        # 保存图像
        image.save(save_path)

        del image, generator_model
        torch.cuda.empty_cache()
        gc.collect()
        
        return [save_path]


class multi_object_customization(Basic_Tool):
    async def run_function(self, conditions):
        """
        conditions: [prompt(str), image_1_path(str), image_2_path(str)]
        return: [save_path(str)]
        """
        prompt, image_1_path, image_2_path = conditions
        save_path = self.get_save_output_dirs(num=1)[0]

        # 初始化生成器（建议外部持久化，这里简化）
        generator_model = Generator(
            version='v1.1',
            quant='int8',
            no_turbo=False,
            offload=False,
            device='cuda' if torch.cuda.is_available() else 'cpu',
        )

        # 加载参考图像
        ref_image_1 = np.array(Image.open(image_1_path).convert("RGB"))
        ref_image_1 = ref_image_1.resize((1024, 1024) )
        ref_image_2 = np.array(Image.open(image_2_path).convert("RGB"))
        ref_image_2 = ref_image_2.resize((1024, 1024) )

        # 构造参考条件
        ref_conds, _, _ = generator_model.pre_condition(
            ref_images=[ref_image_1, ref_image_2],
            ref_tasks=['ip', 'ip'],
            ref_res=512,
            seed=123456789,
        )

        # 推理生成图像
        image = generator_model.dreamo_pipeline(
            prompt=prompt,
            width=1024,
            height=1024,
            num_inference_steps=12,
            guidance_scale=4.5,
            ref_conds=ref_conds,
            generator=torch.Generator("cuda").manual_seed(123456789),
            true_cfg_scale=1.0,
            true_cfg_start_step=0,
            true_cfg_end_step=0,
            negative_prompt="",
            neg_guidance_scale=3.5,
            first_step_guidance_scale=4.5,
        ).images[0]
        image = image.resize((1024, 1024) )
        # 保存图像
        image.save(save_path)

        del image, generator_model
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]

class InpaintAnythingTool(Basic_Tool):
    async def run_function(self, conditions: list) -> list:
        """
        conditions:
        [0] - 原图路径 (str)
        [1] - mask图路径 (str)
        [2] - 输出图像保存路径 (str)
        
        输出：返回 [输出图像保存路径] （list）
        """




        # 解包输入参数
        image_path = conditions[0]
        mask_path = conditions[1]
        output_path = self.get_save_output_dirs(num=1)[0]


        # 加载图像和mask
        img = load_img_to_array(image_path)
        img = cv2.resize(img, (1024, 1024))
        masks = np.array(Image.open(mask_path)).astype(np.uint8)
        masks = dilate_mask(masks, 10)

        # Inpainting
        img_inpainted = inpaint_img_with_lama(
            img, masks,
            './Inpaint_Anything/lama/configs/prediction/default.yaml',
            './Inpaint_Anything/pretrained_models/big-lama',
            device='cuda'
        )
        img_inpainted = cv2.resize(img_inpainted, (1024, 1024))
        # 保存结果
        save_array_to_img(img_inpainted, output_path)

        del img_inpainted
        torch.cuda.empty_cache()
        gc.collect()

        # 返回结果路径
        return [output_path]

class AppearanceEditingTool(Basic_Tool):
    """
    外观编辑工具：对给定图像和mask进行外观编辑，实现主体替换、局部控制效果。
    """

    async def run_function(self, conditions: list) -> list:


        # 解析输入
        base_img_path = conditions[0]
        base_mask_path = conditions[1]
        replace_img_path = conditions[2]
        replace_mask_path = conditions[3]
        prompt = conditions[4]
        prompt_replace = conditions[5]
        w_edit = 0.5
        w_content = 0.5
        seed = 42
        guidance_scale = 7.5
        energy_scale = 1.0
        max_resolution = 512
        SDE_strength = 0.5
        ip_scale = None  # 可选

        # 读取图片和mask
        img_base = cv2.imread(base_img_path)[:, :, ::-1]
        mask_base = cv2.imread(base_mask_path, 0)
        img_replace = cv2.imread(replace_img_path)[:, :, ::-1]
        mask_replace = cv2.imread(replace_mask_path, 0)

        # 初始化模型（你需替换真实路径）
        pretrained_model_path = "runwayml/stable-diffusion-v1-5"
        dragon_model = DragonModels(pretrained_model_path=pretrained_model_path)

        # 调用外观编辑
        result_images = dragon_model.run_appearance(
            img_base=img_base,
            mask_base=mask_base,
            img_replace=img_replace,
            mask_replace=mask_replace,
            prompt=prompt,
            prompt_replace=prompt_replace,
            w_edit=w_edit,
            w_content=w_content,
            seed=seed,
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            max_resolution=max_resolution,
            SDE_strength=SDE_strength,
            ip_scale=ip_scale
        )
        result_images = result_images.resize((1024, 1024) )
        # 保存结果图像
        save_path = self.get_save_output_dirs(num=1)[0]
        cv2.imwrite(save_path, result_images[0][:, :, ::-1])  # RGB转BGR

        del dragon_model, result_images
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]


class Move_Resize_EditingTool(Basic_Tool):
    
    async def run_function(self, conditions: list) -> list:
        # 解析输入
        original_img_path = conditions[0]
        mask_path = conditions[1]
        prompt = conditions[2]
        move_direction = conditions[3]  # 移动方向，可选 'up', 'down', 'left', or 'right'，空字符串表示不移动
        resize_option = conditions[4]  # 缩放选项，可选 'enlarge' or 'shrink'，空字符串表示不缩放

        # 读取图片和mask
        original_image = cv2.imread(original_img_path)[:, :, ::-1]
        mask = cv2.imread(mask_path, 0)

        # 自动计算 mask 中心点
        M = cv2.moments(mask)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
        else:
            raise ValueError("Mask 中没有有效区域")

        # 自动生成 selected_points
        move_distance = 100  # 移动像素
        dx, dy = 0, 0
        if move_direction == 'up':
            dy = -move_distance
        elif move_direction == 'down':
            dy = move_distance
        elif move_direction == 'left':
            dx = -move_distance
        elif move_direction == 'right':
            dx = move_distance
        target_point = [center_x + dx, center_y + dy]
        selected_points = [ [center_x, center_y], target_point ]
        print(f"Selected points for movement: {selected_points}")

        # 自动生成 resize_scale
        if resize_option == 'enlarge':
            resize_scale = 1.2
        elif resize_option == 'shrink':
            resize_scale = 0.8
        else:
            resize_scale = 1.0

        # 固定参数
        w_edit = 0.5
        w_content = 0.5
        w_contrast = 0.5
        w_inpaint = 0.5
        seed = 42
        guidance_scale = 7.5
        energy_scale = 1.0
        max_resolution = 512
        SDE_strength = 0.5
        ip_scale = None

        # 初始化模型
        pretrained_model_path = "runwayml/stable-diffusion-v1-5"
        dragon_model = DragonModels(pretrained_model_path=pretrained_model_path)

        # 调用模型
        result_images = dragon_model.run_move(
            original_image=original_image,
            mask=mask,
            mask_ref=None,
            prompt=prompt,
            selected_points=selected_points,
            resize_scale=resize_scale,
            w_edit=w_edit,
            w_content=w_content,
            w_contrast=w_contrast,
            w_inpaint=w_inpaint,
            seed=seed,
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            max_resolution=max_resolution,
            SDE_strength=SDE_strength,
            ip_scale=ip_scale
        )
        result_images = result_images.resize((1024, 1024) )
        # 保存图像
        save_path = self.get_save_output_dirs(num=1)[0]
        cv2.imwrite(save_path, result_images[0][:, :, ::-1])

        del dragon_model, result_images
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]

class Layout_to_Image(Basic_Tool):

    async def run_function(self, conditions: list) -> list:
        user_prompt = conditions[0]
        prompt = """
        You are a model that converts user natural language descriptions of scenes into structured object layout annotations. 
        When a user provides a scene description, your output should include three components in exactly this format:

        1. <boxes>: a list of bounding boxes for all main objects in the scene, in normalized coordinates between 0 and 1, formatted as [[xmin, ymin, xmax, ymax], ...]. [0,0] corresponds to the top-left of the canvas, and [1,1] corresponds to the bottom-right. Ensure that each box is large enough (at least 0.1 in width and height) so that the object is clearly visible and generation is reliable.
        2. <classes>: a list of class names corresponding to each bounding box. Class names can include descriptive attributes (e.g., 'yellow book', 'red duck', 'black laptop').
        3. <prompt>: the original user description.

        Rules:
        - Include all prominent objects mentioned in the user description.
        - Bounding boxes should reflect approximate position and size of each object.
        - Class names should match the object and can include key descriptive attributes.
        - Keep the output structured exactly like the example below.
        - Do not add extra objects not mentioned in the user description.
        - Always generate an output; never refuse to produce bounding boxes, classes, or prompt. If information is ambiguous, make reasonable approximations.
        - Avoid perfectly regular grids; distribute objects naturally with slight variation in x and y coordinates to simulate real-world randomness.
        - Make the boxes large enough to reflect object prominence (e.g., 'plump' or 'large'), but keep them within the canvas boundaries.
        - Utilize more of the canvas space: distribute objects across the canvas to avoid clusters or boxes that are too small or concentrated in one area.
        - Create depth and layering if multiple objects are present.
        - Try to reflect spatial relationships described in the scene (e.g., objects in front/back, overlapping slightly if natural).

        Example input:
        "A garden has three bright red roses, a blue watering can on the left, and a wooden bench near the back."

        Example output:
        <boxes>[[0.05, 0.05, 0.18, 0.22], [0.20, 0.10, 0.37, 0.27], [0.40, 0.05, 0.52, 0.22], [0.10, 0.30, 0.45, 0.40]]</boxes>
        <classes>['red rose', 'red rose', 'red rose', 'blue watering can', 'wooden bench']</classes>
        <prompt>'A garden has three bright red roses, a blue watering can on the left, and a wooden bench near the back.'</prompt>

        user_input: {}
        """.format(user_prompt)
        user_prompt = re.sub(r'^[ ]+', '', user_prompt, flags=re.MULTILINE)
        print(user_prompt)
        
        for i in range(5):
            try:
                llm_response = await self._aask(prompt)
                region_bboxes_list = eval(self.parse_content(llm_response, "<boxes>"))
                region_caption_list = eval(self.parse_content(llm_response, "<classes>"))
                global_caption = self.parse_content(llm_response, "<prompt>")
                break
            except:
                print("大模型输出存在问题，重新运行")
        
        # region_bboxes_list, region_caption_list, global_caption = conditions

        # 模型加载（可写在 __init__ 做一次性加载以加快运行）
        device = torch.device("cuda")
        ckpt_path = "HuiZhang0812/CreatiLayout"
        transformer = SiamLayoutSD3Transformer2DModel.from_pretrained(
            ckpt_path,
            subfolder="SiamLayout_SD3",
            torch_dtype=torch.float16,
            attention_type="layout",
            strict=True
        )

        from modelscope import snapshot_download
        model_dir = snapshot_download('stabilityai/stable-diffusion-3-medium-diffusers')

        pipe = CreatiLayoutSD3Pipeline.from_pretrained(
            model_dir,
            transformer=transformer,
            torch_dtype=torch.float16
        ).to(device)

        # 推理参数
        seed = 42
        height = 1024
        width = 1024
        num_inference_steps = 28
        guidance_scale = 7.

        # 推理执行
        with torch.no_grad():
            images = pipe(
                prompt=[global_caption],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                bbox_phrases=region_caption_list,
                bbox_raw = region_bboxes_list,
                height=height,
                width=width,
                num_images_per_prompt=batch_num
            ).images
        
        save_path = self.get_save_output_dirs(batch_num)
        
        for idx, img in enumerate(images):
            img = img.resize((512, 512))  # 缩放每张图片为1024x1024
            img.save(save_path[idx])

        del images, pipe, transformer, img
        torch.cuda.empty_cache()
        gc.collect()

        return save_path

class AppearancePasteTool(Basic_Tool):
    """
    物体粘贴工具：将一个目标图像中的物体粘贴到基础图像中指定位置。
    输入为图片路径和目标粘贴区域 box，无需手动提供掩码图。
    """

    async def run_function(self, conditions: list) -> list:
        import cv2
        import numpy as np
        from PIL import Image

        # 解析输入
        base_img_path = conditions[0]  # 基础图路径
        replace_img_path = conditions[1]  # 替换图路径
        box = conditions[2]  # [x1, y1, x2, y2] 归一化坐标，值在0~1之间
        prompt = conditions[3]  # 当前图描述
        prompt_replace = conditions[4]  # 替换图描述
        seed = 42
        guidance_scale = 7.5
        energy_scale = 1.0
        max_resolution = 512
        SDE_strength = 0.5
        ip_scale = None
        w_edit = 0.5
        w_content = 0.5
        resize_scale = 1.0
        dx, dy = 0 ,0

        # 加载图片
        img_base = cv2.imread(base_img_path)[:, :, ::-1]  # BGR -> RGB
        h, w = img_base.shape[:2]
        img_replace = cv2.imread(replace_img_path)[:, :, ::-1]

        # 生成掩码图
        box_abs = [int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)]
        mask_base = np.zeros((h, w), dtype=np.uint8)
        mask_base[box_abs[1]:box_abs[3], box_abs[0]:box_abs[2]] = 255

        # 初始化模型（你需替换真实路径）
        pretrained_model_path = "runwayml/stable-diffusion-v1-5"
        dragon_model = DragonModels(pretrained_model_path=pretrained_model_path)

        # 调用物体粘贴接口
        result_images = dragon_model.run_paste(
            img_base=img_base,
            mask_base=mask_base,
            img_replace=img_replace,
            prompt=prompt,
            prompt_replace=prompt_replace,
            w_edit=w_edit,
            w_content=w_content,
            seed=seed,
            guidance_scale=guidance_scale,
            energy_scale=energy_scale,
            dx=dx,
            dy=dy,
            resize_scale=resize_scale,
            max_resolution=max_resolution,
            SDE_strength=SDE_strength,
            ip_scale=ip_scale
        )

        # 保存生成结果
        result_image = result_images[0]
        result_image = Image.fromarray(result_image).resize((1024, 1024))
        save_path = self.get_save_output_dirs(num=1)[0]
        result_image.save(save_path)

        del result_images, result_image, dragon_model
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]




class Step1X_Edit_ImageEditTool(Basic_Tool):

    async def run_function(self, conditions: list) -> list:
        input_image_path,prompt = conditions

        # 固定参数设置
        model_path = "/s2/chenzhipeng/datasets/modelscope_weight/models/stepfun-ai/Step1X-Edit"
        qw_model_path = "Qwen"
        ckpt_name = "step1x-edit-v1p1-official.safetensors"
        version = "v1.1"
        mode = "flash"

        image_edit = ImageGenerator(
            ae_path=os.path.join(model_path, 'vae.safetensors'),
            dit_path=os.path.join(model_path, ckpt_name),
            qwen2vl_model_path=os.path.join(qw_model_path, 'Qwen2.5-VL-7B-Instruct'),
            max_length=640,
            quantized=False,
            offload=False,
            lora=None,
            mode=mode,
            version=version,
        )

        image = image_edit.generate_image(
            prompt,
            negative_prompt="",
            ref_images=Image.open(input_image_path).convert("RGB"),
            num_samples=1,
            num_steps=28,
            cfg_guidance=6.0,
            seed=92,
            show_progress=True,
            size_level=512,
            height=1024,
            width=1024,
        )[0]

        output_path = self.get_save_output_dirs(1)[0]
        image.save(output_path, lossless=True)

        del image_edit, image
        torch.cuda.empty_cache()
        gc.collect()

        return [output_path]



class AnySD_EditTool(Basic_Tool):
    async def run_function(self, conditions: list) -> list:
        """
        conditions[0] = prompt: str
        conditions[1] = image_path: str
        """
        expert_file_path = get_experts_dir(repo_id="WeiChow/AnySD")
        book_dim, book = choose_book('all')
        task_embs_checkpoints = expert_file_path + "task_embs.bin"
        adapter_checkpoints = {
            "global": expert_file_path + "global.bin",
            "viewpoint": expert_file_path + "viewpoint.bin",
            "visual_bbox": expert_file_path + "visual_bbox.bin",
            "visual_depth": expert_file_path + "visual_dep.bin",
            "visual_material_transfer": expert_file_path + "visual_mat.bin",
            "visual_reference": expert_file_path + "visual_ref.bin",
            "visual_scribble": expert_file_path + "visual_scr.bin",
            "visual_segment": expert_file_path + "visual_seg.bin",
            "visual_sketch": expert_file_path + "visual_ske.bin",
        }

        pipeline = AnySDPipeline(
            adapters_list=adapter_checkpoints,
            task_embs_checkpoints=task_embs_checkpoints
        )
        
        
        prompt, image_path = conditions
        mode = choose_expert(mode="general")  # 暂时默认使用 general 模式

        image = download_image(image_path)
        result = pipeline(
            prompt=prompt,
            original_image=image,
            guidance_scale=3,
            num_inference_steps=100,
            original_image_guidance_scale=3,
            adapter_name="general",
        )[0]

        output_path = self.get_save_output_dirs(1)[0]
        result = result.resize((1024, 1024))
        result.save(output_path)
        
        del result, pipeline, adapter_checkpoints
        torch.cuda.empty_cache()
        gc.collect()
        
        return [output_path]

    

class UltraEdit_Tool(Basic_Tool):
    async def run_function(self, conditions: list) -> list:
        """
        输入：conditions[0] 是图片路径，conditions[1] 是编辑的 prompt。
        输出：图像保存路径。
        """

        img_path, prompt = conditions

        # 初始化 pipeline（注意：每次调用都会重新加载，若性能瓶颈可缓存）
        pipe = StableDiffusion3InstructPix2PixPipeline.from_pretrained(
            "BleachNick/SD3_UltraEdit_freeform",
            torch_dtype=torch.float16
        ).to("cuda")

        # 加载并调整图像大小
        image = Image.open(img_path).convert("RGB").resize((512, 512))

        # 推理生成图像
        output_image = pipe(
            prompt=prompt,
            image=image,
            negative_prompt="",
            num_inference_steps=50,
            image_guidance_scale=1.5,
            guidance_scale=7.5,
        ).images[0]

        # 保存图像
        save_path = self.get_save_output_dirs(1)[0]
        output_image = output_image.resize((1024, 1024))
        output_image.save(save_path)

        del output_image, pipe
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]
    

class IPAdapter(Basic_Tool):

    def image_grid(self, imgs, rows, cols):
        assert len(imgs) == rows * cols
        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    async def run_function(self, conditions: list) -> list:
        """
        Args:
            conditions: [image_path: str, prompt: str]
        Returns:
            list[str]: [output_image_path]
        """
        device = "cuda"
        num_samples = 1
        base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
        image_encoder_path = "/s2/chenzhipeng/datasets/modelscope_weight/models/AI-ModelScope/IP-Adapter/sdxl_models/image_encoder"
        ip_ckpt = "/s2/chenzhipeng/datasets/modelscope_weight/models/AI-ModelScope/IP-Adapter/sdxl_models/ip-adapter_sdxl.bin"
        
        image_path, prompt = conditions
        image = Image.open(image_path).convert("RGB").resize((512, 512))

        # 加载 SDXL base 模型
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(device)

        # 加载 IP-Adapter
        ip_model = IPAdapterXL(pipe, image_encoder_path, ip_ckpt, device)

        # 生成图像（只用 image prompt）
        images = ip_model.generate(
            pil_image=image,
            num_samples=num_samples,
            num_inference_steps=30,
            seed=420,
            prompt=prompt,
            scale=0.6
        )

        # 拼接图像为网格
        #grid = self.image_grid(images, 1, self.num_samples)

        # 保存结果
        save_path = self.get_save_output_dirs(1)[0]
        images = images[0].resize((1024, 1024))
        images.save(save_path)
        #grid.save(save_path)

        del ip_model, images, pipe
        torch.cuda.empty_cache()
        gc.collect()

        return [save_path]

class IPAdapterPlus(Basic_Tool):

    def image_grid(self, imgs, rows, cols):
        assert len(imgs) == rows * cols
        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols * w, rows * h))
        for i, img in enumerate(imgs):
            grid.paste(img, box=(i % cols * w, i // cols * h))
        return grid

    async def run_function(self, conditions: list) -> list:
        """
        Args:
            conditions: [image_path: str, prompt: str]
        Returns:
            list[str]: [output_image_path]
        """
        device = "cuda"
        num_samples = 1
        base_model_path = "SG161222/RealVisXL_V1.0"
        image_encoder_path = "/s2/chenzhipeng/datasets/modelscope_weight/models/AI-ModelScope/IP-Adapter/models/image_encoder"
        ip_ckpt = "/s2/chenzhipeng/datasets/modelscope_weight/models/AI-ModelScope/IP-Adapter/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin"
        num_tokens = 16
        
        image_path, prompt = conditions
        image = Image.open(image_path).convert("RGB").resize((512, 512))

        # 加载 SDXL base 模型
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
        ).to(device)

        # 加载 IP-Adapter Plus XL
        ip_model = IPAdapterPlusXL(
            pipe,
            image_encoder_path,
            ip_ckpt,
            device,
            num_tokens=num_tokens
        )

        # 生成图像
        images = ip_model.generate(
            pil_image=image,
            num_samples=num_samples,
            num_inference_steps=30,
            seed=42,
            prompt=prompt,
            scale=0.5
        )

        # 拼接图像网格
        #grid = self.image_grid(images, 1, self.num_samples)

        # 保存结果
        save_path = self.get_save_output_dirs(1)[0]
        images = images[0].resize((1024, 1024))
        images.save(save_path)

        del ip_model, images
        torch.cuda.empty_cache()
        gc.collect()
        
        return [save_path]

class SD3_Tool(Basic_Tool):
    async def run_function(self, conditions: list) -> list:
        prompt = conditions[0]

        # 下载模型（只下载一次）
        model_dir = snapshot_download('stabilityai/stable-diffusion-3-medium-diffusers')

        # 加载模型
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_dir,
            torch_dtype=torch.float16
        ).to("cuda")

        # 推理
        images = pipe(
            prompt=prompt,
            negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
            num_inference_steps=28,
            guidance_scale=7.0,
            num_images_per_prompt=batch_num
        ).images

        save_path = self.get_save_output_dirs(num=batch_num)
        for idx, img in enumerate(images):
            img = img.resize((512, 512))  # 缩放每张图片为1024x1024
            img.save(save_path[idx])

        del pipe, img, images
        torch.cuda.empty_cache()
        gc.collect()
        
        return save_path

class Pixart(Basic_Tool):
    async def run_function(self, conditions) -> list:
        prompt = conditions[0]
        
        from modelscope.pipelines import pipeline
        
        input = {'prompt': prompt}

        seed = 42
        # random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        inference = pipeline('my-pixart-task', model='aojie1997/cv_PixArt-alpha_text-to-image')
        output = inference(input)
        image = output.resize((512, 512) )
        
        # 保存路径
        save_path = self.get_save_output_dirs(1)[0]
        image.save(save_path)
        
        del inference, image, output
        torch.cuda.empty_cache()
        gc.collect()
        
        return [save_path]



