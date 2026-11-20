from metagpt.actions import Action, UserRequirement
from metagpt.roles import Role
from metagpt.roles.role import RoleReactMode
from metagpt.schema import Message
from metagpt.context import Context
from metagpt.environment import Environment
from loguru import logger as _logger
from pathlib import Path
import asyncio
import json, os, sys, textwrap, shutil, re
from typing import ClassVar, Dict
from Tools.tools import Tools
from datetime import datetime
import base64
import openai
from transformers import CLIPTokenizer, CLIPModel
import torch
from tqdm import tqdm
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from modelscope import snapshot_download
from metagpt.config2 import Config
from pathlib import Path
import time, requests, random
import numpy as np
from prompt.normalize import normalize_tool_preferences


def define_log_level(project_root=".", name: str = None):
    """只输出 INFO 等级日志，带行号，仅 message 内容"""
    global _print_level
    _print_level = "INFO"

    current_date = datetime.now()
    formatted_date = current_date.strftime("%Y%m%d")
    current_time = current_date.strftime("%H%M%S")
    log_name = f"{name}_{formatted_date}" if name else formatted_date

    project_root = Path(project_root)
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    only_info = lambda record: record["level"].name == "INFO"

    # 带行号的输出格式
    log_format = "---------------- <line {line}> --------------------\n"+"{message}\n"

    _logger.remove()

    # 控制台输出：仅 INFO，带行号
    _logger.add(sys.stderr, filter=only_info, format=log_format)

    # 文件输出（纯文本），也仅 INFO，带行号
    _logger.add(log_dir / f"{log_name}-{current_time}.log", level="INFO", filter=only_info, format=log_format)

    return _logger

logger = define_log_level("./")

class MLLM:
    def __init__(self, config_file, max_tokens=1024):
        # 从 YAML 配置文件加载配置
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        llm_config = config.get('llm', {})
        
        # 设置 OpenAI API 配置
        openai.api_key = llm_config.get('api_key')
        openai.api_base = llm_config.get('base_url', 'https://api.openai.com/v1')  # 默认为 OpenAI 默认 API URL
        openai.api_version = llm_config.get('api_version', '2024-08-01')  # 设置 API 版本，默认使用最新版本
        self.model = llm_config.get('model', 'gpt-4')  # 默认模型为 gpt-4
        self.temperature = llm_config.get('temperature', 0.5)
        self.max_tokens = max_tokens

    def _encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def chat(self, prompt, history=None):
        """
        prompt 示例：
            [{"type": "image", "image": "path/to/image.png"},
             {"type": "text", "text": "Describe this image."}]
        """
        if history is None:
            history = []

        messages = []
        for item in prompt:
            if item["type"] == "image":
                image_base64 = self._encode_image(item["image"])
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                })
            elif item["type"] == "text":
                if messages and isinstance(messages[-1]["content"], list):
                    messages[-1]["content"].append({"type": "text", "text": item["text"]})
                else:
                    messages.append({
                        "role": "user",
                        "content": item["text"]
                    })

        for msg in history:
            if isinstance(msg["content"], str):
                messages.insert(0, msg)

        attempt = 0
        max_attempts = 10
        wait_seconds = 15
        while attempt < max_attempts:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                reply = response['choices'][0]['message']['content']
                break 
            except Exception as e:
                attempt += 1
                if attempt >= max_attempts:
                    raise e
                print(f"请求失败，第 {attempt}/{max_attempts} 次重试，{wait_seconds} 秒后再试...")
                time.sleep(wait_seconds)

        reply = response['choices'][0]['message']['content']
        history.append({"role": "user", "content": prompt})
        history.append({"role": "assistant", "content": reply})
        return reply, history


mllm = MLLM(config_file='./config/gpt4o.yaml')


class MLLM_read_img(Action):
    role_prompt: str = "You are an excellent Analyst.\n"
    description: str = """
    You are highly skilled at analyzing image information provided by users. 
    Please analyze the image content as thoroughly as possible and provide a closely matching description.\n
    """
    name: str = "ReadImage"

    def remove_think_tags(self, text: str) -> str:
        if "<think>" in text and "</think>" in text:
            # 去掉 <think> 和 </think> 之间的内容
            start = text.find("<think>")
            end = text.find("</think>") + len("</think>")
            return text[:start] + text[end:]
        return text

    async def run(self, image_path: list[str]):
        
        prompt = [{"role": "system", "content": f"{self.role_prompt}{self.description}"}]
        
        # input_text = "Please analyze the images information as thoroughly as possible."
        input_text = "Please describe the image in no more than 50 tokens, clearly stating the key entities, spatial relationships, "+\
                     "background, style, and other relevant details."
        usr_input = [{"type": "text", "text": input_text}]
        
        for img_dir in image_path:
            usr_input.append({"type": "image", "image": img_dir})
        
        reply, history = mllm.chat(usr_input, prompt)
        
        reply = self.remove_think_tags(reply)
        
        return reply

class Role_AIGC(Role):
    shared_memory: ClassVar[Dict] = {}
    tools: ClassVar[Tools] = Tools()
    read_img: ClassVar[MLLM_read_img] =  MLLM_read_img()
    experience_path: str = "./Experience"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.history = []
        self.description = ""
        self.system_prompt = ""
        
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    
    def find_top_n_indices(self, query: str, candidates: list[str], top_n: int = 1):
        """
        基于clip用于检索最匹配的n个经验样例
        """
        
        texts = [query] + candidates
        inputs = self.clip_tokenizer(texts, padding=True, return_tensors="pt", truncation=True)

        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        query_feature = text_features[0]
        candidate_features = text_features[1:]
        similarities = (candidate_features @ query_feature).squeeze()

        # 获取 top-n 索引（注意返回的是原 candidates 中的索引）
        top_n = min(top_n, len(candidates))
        top_scores, top_indices = torch.topk(similarities, top_n)

        top_indices_list = top_indices.tolist()
        # 如果只有一个元素，保证返回列表形式，而不是单个int
        if isinstance(top_indices_list, int):
            top_indices_list = [top_indices_list]

        return top_indices_list, top_scores.tolist()
    
    def get_system_prompt(self):
        self.description = textwrap.dedent(self.description)
        system_prompt = "You are an excellent {}. {}\n".format(self.name, self.description)
        
        return system_prompt
    
    def remove_think_tags(self, text: str) -> str:
        if "<think>" in text and "</think>" in text:
            # 去掉 <think> 和 </think> 之间的内容
            start = text.find("<think>")
            end = text.find("</think>") + len("</think>")
            return text[:start] + text[end:]
        return text
    
    def get_closing_tag(self, tag: str) -> str:
        if tag.startswith("<") and tag.endswith(">"):
            return "</" + tag[1:]
        raise ValueError("Invalid tag format")
    
    def parse_content(self, reply:str, tag:str):
        close_tag = self.get_closing_tag(tag)
        content = (reply.split(tag)[1]).split(close_tag)[0].strip()
        
        return content
    
    def write_json(self, json_dir, trajectory):
        # 如果文件不存在，创建一个空的 JSON 文件
        if not os.path.exists(json_dir):
            with open(json_dir, "w", encoding="utf-8") as f:
                json.dump([], f, ensure_ascii=False, indent=4)
        
        # 读取 JSON 文件，处理空文件的情况
        try:
            with open(json_dir, "r", encoding="utf-8") as f:
                data_list = json.load(f)  # 尝试加载 JSON 数据
        except json.decoder.JSONDecodeError:  # 如果 JSON 解析失败（文件为空或无效）
            data_list = []  # 如果解析失败，初始化为空列表

        # 添加新的轨迹数据
        data_list.append(trajectory)

        # 写回文件
        with open(json_dir, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)
    
    async def compair_test(self, task: str, output_dir_list: list, input_dir=None) -> list:
        if input_dir != None:
            labels = [f"Image {chr(66 + i)}" for i in range(len(output_dir_list))]
            labels_str = ", ".join(labels)
            prompt = """
            task: {}
            
            The first image (Image A) is the original input image. After executing the task, multiple output images are generated 
            ({} i.e., all images except the first one). Please compare these output images and analyze to provide their ranking from best to worst.

            """.format(task, labels_str)
        
        else:
            labels = [f"Image {chr(65 + i)}" for i in range(len(output_dir_list))]
            labels_str = ", ".join(labels)
            prompt = """
            task: {}
            
            Multiple output images are generated ({}) after executing the task. Please compare these output images and analyze to provide their ranking from **best to worst**.
            
            """.format(task, labels_str)
        
        prompt = prompt + "Please provide a detailed explanation first, and then give your selection in XML format, for example: <rank>Image B, Image E, Image D,...</rank>"
        prompt = textwrap.dedent(prompt)
        prompt = re.sub(r'^[ ]+', '', prompt, flags=re.MULTILINE)
        
        imgs_list = output_dir_list.copy()
        if input_dir != None:
            imgs_list.insert(0, input_dir)
            
        usr_input = [{"type": "text", "text": prompt}]
        for img_dir in imgs_list:
            usr_input.append({"type": "image", "image": img_dir})
        logger.info(prompt)
        
        max_retries = 10
        for i in range(max_retries):
            try:
                reply, _ = mllm.chat(usr_input)
                break
            except:
                logger.warning(f"Rate limit hit, retrying in {10} seconds...")
                time.sleep(10)
        logger.info(reply)
        
        rank_str = self.parse_content(reply, "<rank>")
        rank_list = [item.strip() for item in rank_str.split(',')]
        
        rank = [labels.index(x) for x in rank_list]
        
        return rank
        
    
    async def unit_test(self, task: str, output_dir: str, input_dir=None, tool_name=None):
        goal_eval = self.shared_memory.get("goal_eval", {})
        tar_semantic = goal_eval.get("semantic", "")
        original_semantic = self.shared_memory.get("original_semantic", "")

        # 1. 构建 task 和目标语义
        if original_semantic:
            evaluation_reasoning = f"""
        task: {task}
        original/reference image Semantic: {original_semantic}
        target image semantic: {tar_semantic}
        """
        else:
            evaluation_reasoning = f"""
        task: {task}
        target image semantic: {tar_semantic}
        """

        # 2. 通用评价标准（统一出现一次）
        evaluation_reasoning += """
        Evaluation criteria for all dimensions:
        - L1: Fully matches target requirements.
        - L2: Mostly matches with minor inconsistencies.
        - L3: Significant mismatches affecting meaning, realism, or naturalness.
        Explicitly indicate any missing, extra, or unnatural objects, attributes, relationships, styles, backgrounds, or semantics.
        In final evaluation, explicitly list all specific issues identified across dimensions.
        """

        # 3. 各维度评估 prompt（根据 goal_eval 中存在的维度生成）
        if "category_number" in goal_eval:
            evaluation_reasoning += f"""
        [Category Number]
        Analyze whether objects exist and counts match target: {goal_eval['category_number']}.
        Objects with quantity **-1** are unspecified; presence matters. Quantity **0** means object should not exist.
        Explicitly note missing, extra, or misplaced objects.

        Output format:
        <level_category_number>L1/L2/L3</level_category_number>
        <evaluation_category_number>within 20 words, explicitly noting issues</evaluation_category_number>
        """

        if "position" in goal_eval:
            pos_list = "\n".join(f"- {o_p}" for o_p in goal_eval["position"])
            evaluation_reasoning += f"""
        [Position]
        Analyze if positional relationships follow target criteria:
        {pos_list}
        Explicitly note incorrect placements, overlaps, or unnatural arrangements.

        Output format:
        <level_position>L1/L2/L3</level_position>
        <evaluation_position>within 20 words, explicitly noting position errors</evaluation_position>
        """

        if "attribute-binding" in goal_eval:
            attr_list = "\n".join(f"- {o_a}" for o_a in goal_eval["attribute-binding"])
            evaluation_reasoning += f"""
        [Attribute Binding]
        Analyze if attribute bindings conform to requirements:
        {attr_list}
        Explicitly note wrong, missing, or inconsistent attributes.

        Output format:
        <level_attribute_binding>L1/L2/L3</level_attribute_binding>
        <evaluation_attribute_binding>within 20 words, noting incorrect or missing attributes</evaluation_attribute_binding>
        """

        if "style" in goal_eval:
            evaluation_reasoning += f"""
        [Style]
        Analyze whether image style matches: {goal_eval['style']}.
        Explicitly note deviations or inconsistencies.

        Output format:
        <level_style>L1/L2/L3</level_style>
        <evaluation_style>within 20 words, mentioning style inconsistencies</evaluation_style>
        """

        if "background" in goal_eval:
            evaluation_reasoning += f"""
        [Background]
        Analyze whether background matches: {goal_eval['background']}.
        Explicitly note errors or unnatural areas.

        Output format:
        <level_background>L1/L2/L3</level_background>
        <evaluation_background>within 20 words, noting background issues</evaluation_background>
        """

        if "semantic" in goal_eval:
            evaluation_reasoning += f"""
        [Semantic & Naturalness]
        Analyze whether overall semantics align with target: {goal_eval['semantic']}.
        Explicitly note semantic inconsistencies or unrealistic elements.

        Output format:
        <level_semantic>L1/L2/L3</level_semantic>
        <evaluation_semantic>within 20 words, noting semantic misalignments</evaluation_semantic>
        """

        # 4. 最终总评价说明
        evaluation_reasoning += """
        ## Note:
        1. Please provide your reasoning process first within <think></think>.
        2. Carefully analyze the image, noting any missing, incorrect, misplaced, or unusual elements.
        3. Compare each dimension (objects, positions, attributes, style, background, semantics) with the target requirements.
        4. Explicitly justify each level and precisely indicate the exact elements or locations where errors, mismatches, or inconsistencies occur.
        5. The final output is enclosed using the '<result></result>' XML format.

        # Example (only for reference, follow the same format and text style for actual evaluation; do not copy content values):
        <result>
        <level_category_number>L1</level_category_number>
        <evaluation_category_number>All objects present in correct quantities</evaluation_category_number>
        <level_position>L2</level_position>
        <evaluation_position>Guitar slightly misaligned to the right of amplifier; microphone stand slightly farther left than target</evaluation_position>
        <level_attribute_binding>L3</level_attribute_binding>
        <evaluation_attribute_binding>Amplifier is dark gray instead of black; guitar green slightly desaturated</evaluation_attribute_binding>
        <level_background>L2</level_background>
        <evaluation_background>Wooden floor reflects golden light unevenly; stage curtains darker than target</evaluation_background>
        <level_semantic>L3</level_semantic>
        <evaluation_semantic>Guitar position off relative to amplifier; amplifier color wrong; stage setup differs from target</evaluation_semantic>
        </result>
        """
        
        evaluation_reasoning = textwrap.dedent(evaluation_reasoning)
        evaluation_reasoning = re.sub(r'^[ ]+', '', evaluation_reasoning, flags=re.MULTILINE)
        usr_input = [
            {"type": "image", "image": output_dir},
            {"type": "text", "text": evaluation_reasoning}
        ]
        logger.info(evaluation_reasoning)
        max_retries = 10
        for i in range(max_retries):
            try:
                reply, _ = mllm.chat(usr_input)
                reply = self.parse_content(reply, "<result>")
                if reply == "" or reply == None:
                    raise ValueError("GLM评估有误")
                break
            except:
                logger.warning(f"Rate limit hit, retrying in {5} seconds...")
                time.sleep(5)
        logger.info(reply)

        # 4. 解析结果
        test_results = {}
        def get_score(tag):
            """
            将 L1/L2/L3 等级映射为浮点分数：
                L1 -> 1.0
                L2 -> 0.66
                L3 -> 0.33
            """
            try:
                level = self.parse_content(reply, f"<level_{tag}>").strip()
                mapping = {"L1": 1.0, "L2": 0.66, "L3": 0.33}
                return mapping.get(level, 0.0)
            except:
                return 0.0

        def get_eval(tag):
            try:
                return self.parse_content(reply, f"<evaluation_{tag}>")
            except:
                return ""

        if "category_number" in goal_eval:
            test_results["object category & number"] = {
                "score": get_score("category_number"),
                "evaluation": get_eval("category_number"),
                "weight": 1.0
            }
        if "position" in goal_eval:
            test_results["positional relationships"] = {
                "score": get_score("position"),
                "evaluation": get_eval("position"),
                "weight": 1.0
            }
        if "attribute-binding" in goal_eval:
            test_results["attribute binding"] = {
                "score": get_score("attribute_binding"),
                "evaluation": get_eval("attribute_binding"),
                "weight": 1.0
            }
        if "style" in goal_eval:
            test_results["style consistency"] = {
                "score": get_score("style"),
                "evaluation": get_eval("style"),
                "weight": 1.0
            }
        if "background" in goal_eval:
            test_results["background match"] = {
                "score": get_score("background"),
                "evaluation": get_eval("background"),
                "weight": 1.0
            }

        if "semantic" in goal_eval:
            test_results["semantic"] = {
                "score": get_score("semantic"),
                "evaluation": get_eval("semantic"),
                "weight": 1.0
            }

        # 计算 final_score
        other_scores = [r["score"] for k, r in test_results.items() if k != "semantic"]
        other_weights = [r["weight"] for k, r in test_results.items() if k != "semantic"]
        other_weight_sum = sum(other_weights)
        normalized_weights = [w / other_weight_sum for w in other_weights] if other_weight_sum else [0] * len(other_weights)
        other_scores = sum(s * w for s, w in zip(other_scores, normalized_weights))
        final_score = round((other_scores + test_results["semantic"]["score"]) / 2, 4)

        # 拼接各维度 evaluation 生成 final_evaluation
        final_evaluation = ""
        for key, value in test_results.items():
            final_evaluation += f"{key}: {value['evaluation']}\n"

        logger.info(f"工具 {tool_name} 使用分数：\ngpt: {test_results}\nfinal_score: {final_score}")
        logger.info(f"最终评价: {final_evaluation}")

        return final_evaluation, final_score

    async def chat(self, prompt: str, use_history=False):
        prompt = textwrap.dedent(prompt)
        prompt = re.sub(r'^[ ]+', '', prompt, flags=re.MULTILINE)
        
        # sys_prompt = " The reasoning process must not contain any repetition or redundancy. The reasoning process must be logical and as concise as possible."  # 对qwen3
        sys_prompt = " Please provide a step-by-step and logical reasoning process first (Thought process provided through '<think></think>'), and then give the output that meets the required format. Output the result in the given format (e.g. <answer></answer>)."     # 对gpt-4o
        
        if use_history:
            reply = await self.llm.aask(prompt+sys_prompt, format_msgs=self.history, system_msgs=[self.system_prompt])
            reply = self.remove_think_tags(reply)
            self.history.append({'role': 'user', 'content': prompt})
            self.history.append({'role': 'assistant', 'content': reply})
        else:
            reply = await self.llm.aask(prompt+sys_prompt, system_msgs=[self.system_prompt])
            reply = self.remove_think_tags(reply)
        return reply

    def clean_history(self):
        self.history = []

class Analyst(Role_AIGC):
    name: str = "Analyst"
    profile: str = "Analyst"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._watch([UserRequirement])
        self.rc.react_mode = RoleReactMode.PLAN_AND_ACT
        self.description = "You are highly skilled at summarizing users' requirements for "+\
                            "image generation or image editing based on the image information "+\
                            "and text descriptions they provide.\n"

        with open('prompt.txt', 'r', encoding='utf-8') as f:
            self.eval_context = f.read()
        self.eval_query = "Please strictly follow the JSON format provided in the example and directly output your analysis results."
    
        self.eval_context = self.eval_context + \
        "Please strictly follow the JSON format provided in the example and directly output your analysis results.\n" +\
        "Output format: <goal>Your goal in JSON format<goal>"

    async def get_goal(self, task:dict, semantic:str, img_info:str):
        """
        基于task和目标图像语义，以获取评估目标
        """
        
        user_info = "task: " + task + "\n"
        if img_info != "":
            user_info = user_info + img_info
        goal_massage = self.eval_context + "prompt: " + user_info + "Target image semantics: " + semantic + "\n"
        request_eval = await self.chat(goal_massage)
        
        goal_str = self.parse_content(request_eval, "<goal>")
        goal = json.loads(goal_str)
        
        goal_eval = {k: v for k, v in goal.items() if v is not None}
        goal_logit = {k: False for k, v in goal.items() if v is not None}
        
        goal_eval["semantic"] = semantic
        goal_logit["semantic"] = False
        
        return goal, goal_eval, goal_logit

    async def analysis_task_semantic(self, user_instruction, img_info):

        if img_info != "":
            user_info = user_instruction + img_info
        else:
            user_info = user_instruction
        
        prompt = "user_info: " + user_info + "\n" +\
        """
        1. Please analyze the user's needs based on the provided content and summarize their requirements.
        If a specific image is referenced, the path to the reference image must be specified.
        **No assumptions are allowed about the user-provided information; the output must closely align with the user’s given information.**
        **The output must be derived through precise and correct reasoning, rather than copying the user's input.**
        Transform the user input into concrete visual elements for the final image, avoiding overly simple or abstract terms.
        The output task must be **precise and concise, within 20 tokens**.
        Output the task in the format: <task>Your summary task</task>.
        
        2. Please provide the semantics of the final output image (i.e., what the final rendered image looks like) in textual form.
        The output semantic should be described in terms of key objects in the image, their attributes (numeracy, categories, color, texture, etc.), spatial relationships, background, and image style, etc..
        The output semantic must be **precise and concise, within 20 tokens**.
        Output the semantic in the format: <semantic>Textual semantic information of the target image.</semantic>.
        
        ================= The following is a reference example ================= 
        Example 1:
        <task>Generate an image of a cyberpunk city at night, with neon lights, flying cars, and a futuristic atmosphere.</task>
        <semantic>A cyberpunk city at night with neon lights and flying cars.</semantic>
        
        Example 2:
        <task>Generate an image with a pig and a dog in a forest. The appearance of the pig is based on reference image './temp/pig.png'.</task>
        <semantic>A cyberpunk city at night with neon lights and flying cars.</semantic>
        
        Example 3:
        <task>Replace the dog in the image with a pig, and replace the apple on the table with an orange.  Image to be edited: './outputs/tool_out.png'</task>
        <semantic>A pig and a cow are in a pasture, with a table nearby. There is an orange on the table.</semantic>
        
        **Please pay attention to distinguish between editing tasks and generation tasks. If it's not a generation task, please avoid using the word "generate."**
        **The target image semantics should describe the output image after executing the task and must not conflict with the task.**
        **Tasks and semantics should be as concise as possible while retaining key elements, avoiding excessive redundant or irrelevant content.**
        """
        
        prompt = re.sub(r'^[ ]+', '', prompt, flags=re.MULTILINE)
        
        logger.info("user: "+prompt)
        reply = await self.chat(prompt)
        logger.info("Analyst: "+reply)

        task = self.parse_content(reply, "<task>")
        semantic = self.parse_content(reply, "<semantic>")
        logger.info("Analyst总结任务："+task)
        logger.info("Analyst目标图像语义："+semantic)
        
        return task, semantic, user_info

    async def _plan_and_act(self) -> Message:
        logger.info("分析用户给定图像信息")

        context_msg = (self.get_memories(k=1)[0]).content 
        context_msg = json.loads(context_msg)
        user_instruction = context_msg['text'][0]

        if "image" in context_msg:   # 存在图像
            image_path = context_msg['image']
            img_description = await self.read_img.run(image_path)
            img_info = "Image path: " + str(image_path) + "\n" +\
                       "Image description: " + img_description + "\n"
            # img_info = "Image description: " + img_description + "\n"
            self.shared_memory["original_semantic"] = img_description
            logger.info("图像信息: "+img_info)
            self.shared_memory["user img path"] = str(image_path[0])
        else:
            img_info = ""
            logger.info("img_info: "+"无图像输入")
        
        task, semantic, user_info = await self.analysis_task_semantic(user_instruction, img_info)
        goal, goal_eval, goal_logit = await self.get_goal(task, semantic, img_info)
        logger.info("Analyst: "+str(goal))
        
        self.shared_memory["task"] = task
        self.shared_memory["target_semantic"] = semantic
        self.shared_memory["img_info"] = img_info
        self.shared_memory["goal_eval"] = goal_eval
        self.shared_memory["goal_logit"] = goal_logit
        
        msg_send2planner = Message(content=img_info, role=self.profile)
        msg_send2planner.send_to = "Planner"
        
        return msg_send2planner

class Planner(Role_AIGC):
    name: str = "Planner"
    profile: str = "Planner"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rc.react_mode = RoleReactMode.PLAN_AND_ACT
        self.description = """
        You are highly skilled at analyzing the given task and the information provided by the user.
        By leveraging the available tool functionalities, you can decompose the user's task into logical and reasonable processing steps (excluding evaluation of results),
        enabling step-by-step handling to produce the desired output.
        """
        self.system_prompt = self.get_system_prompt()
        self.trajectory_info = {}
        self.task_graph = {}
        self.shared_memory["dependance score"] = 0.
        self.round = 0
        
        self.n_planner_experience = 3
        self.planner_experience = self.read_experience()
        
        self.try_count = 0
        self.try_num = 1
        self.round_info_buff = []
        
        self.use_experience = False
        self.experience_prompt = ""
    
    def extract_operations(self, operations_str):
        operations_str = self.parse_content(operations_str, "<operation>")
        lines = operations_str.strip().split('\n')
        result = []

        for line in lines:
            line = line.strip()
            if line.startswith('<operation>') or line.startswith('</operation>') or not line:
                continue
            if re.match(r'^\d+\.', line):
                content = line.split('.', 1)[1].strip()
                result.append(content)
        
        return result
    
    def extract_subtask_path(self, task_graph: dict) -> list:
        if not task_graph:
            return []

        all_rounds = set(task_graph.keys())
        depended_rounds = set(
            val["depends_on"] for val in task_graph.values()
            if val.get("depends_on") is not None
        )

        final_rounds = list(all_rounds - depended_rounds)

        if not final_rounds:
            raise ValueError("未能确定最终的子任务节点。请检查 depends_on 结构。")

        # 如果有多个最终节点，选得分最高的一个
        if len(final_rounds) > 1:
            final_rounds.sort(
                key=lambda k: task_graph[k].get("result", {}).get("final goal score", 0),
                reverse=True
            )

        # 回溯路径
        path = []
        current_key = final_rounds[0]
        while current_key:
            current_task = task_graph[current_key]
            path.append(current_task["subtask"])
            current_key = current_task["depends_on"]

        return list(reversed(path))
    
    def read_experience(self):
        plan_file = os.path.join(self.experience_path, "task_graph.json")
        if os.path.exists(plan_file) and os.path.getsize(plan_file) > 0:
            try:
                with open(plan_file, "r", encoding="utf-8") as f:
                    planner_experience = json.load(f)
            except json.JSONDecodeError:
                planner_experience = []
        else:
            planner_experience = []
        
        planner_experience_task_list = [item["task"] for item in planner_experience]
        planner_experience_taskgraph_list = [item["task_graph"] for item in planner_experience]
        
        planner_experience_sorted = {
            "task_list": planner_experience_task_list,
            "taskgraph_list": planner_experience_taskgraph_list
        }
        
        return planner_experience_sorted
    
    def refresh_experience(self):   # 更新经验
        self.planner_experience = self.read_experience()
    
    def select_experience(self, task, top_n=3):
        task_idx_list, scores = self.find_top_n_indices(task, self.planner_experience["task_list"], top_n)
        selected_tasks = [self.planner_experience["task_list"][i] for i in task_idx_list]
        selected_taskgraphs = [self.planner_experience["taskgraph_list"][i] for i in task_idx_list]
        taskgraph_list = []
        for task_data, taskgraph_data in zip(selected_tasks, selected_taskgraphs):
            taskgraph_path = self.extract_subtask_path(taskgraph_data)
            taskgraph = {
                "task": task_data,
                "operations": taskgraph_path
            }
            
            taskgraph_list.append(taskgraph)
        
        return taskgraph_list
        
    async def chat(self, prompt: str, use_history=False):
        prompt = textwrap.dedent(prompt)
        prompt = re.sub(r'^[ ]+', '', prompt, flags=re.MULTILINE)
        
        # 这一句非常重要，可避免QWEN发生重复冗余的推理。
        # sys_prompt = " The reasoning process must not contain any repetition or redundancy. The reasoning process must be logical and as concise as possible."  # 对qwen3
        sys_prompt = " Please provide a step-by-step and logical reasoning process first (Thought process provided through '<think></think>'), and then give the output that meets the required format. Output the result in the given format (e.g. <answer></answer>)."     # 对gpt-4o
        
        if use_history:
            reply_ori = await self.llm.aask(prompt+sys_prompt, format_msgs=self.history, system_msgs=[self.system_prompt])
            reply = self.remove_think_tags(reply_ori)
        else:
            reply_ori = await self.llm.aask(prompt+sys_prompt, system_msgs=[self.system_prompt])
            reply = self.remove_think_tags(reply_ori)
        return reply, reply_ori
    
    async def _plan_and_act(self) -> Message:
        
        self.try_count += 1
        
        task = self.shared_memory["task"]
        semantic = self.shared_memory["target_semantic"]
        img_info = self.shared_memory["img_info"]
        
        if img_info == "":
            img_info = "None"
        
        self.trajectory_info["task"] = task
        self.trajectory_info["user_img_info"] = img_info
        self.trajectory_info["target_semantic"] = semantic
        
        if len(self.round_info_buff) < self.try_num:
            if "current result" in self.shared_memory:
                subtask_result = self.shared_memory["current result"]
                current_operation = self.shared_memory["current operation"]
                dependance = self.shared_memory["current_dependance"]
                current_prompt = self.shared_memory["current_prompt"]
                current_reply = self.shared_memory["current_reply"]
                if self.round == 0:
                    dependance = None
                self.round_info_buff.append({
                    "subtask": current_operation,
                    "depends_on": dependance,
                    "result": subtask_result,
                    "prompt": current_prompt,
                    "reply": current_reply
                })
        
        if len(self.round_info_buff) == self.try_num:
            scores_cadidate = [round_info["result"]["final goal score"] for round_info in self.round_info_buff]
            max_score_idx = scores_cadidate.index(max(scores_cadidate))
            max_score_round_info = self.round_info_buff[max_score_idx]
            self.task_graph["round {}".format(self.round+1)] = {
                "subtask": max_score_round_info["subtask"],
                "depends_on": max_score_round_info["depends_on"],
                "result": max_score_round_info["result"]
            }
            
            self.history.append({'role': 'user', 'content': max_score_round_info["prompt"]})
            self.history.append({'role': 'assistant', 'content': max_score_round_info["reply"]})
            
            self.round_info_buff = []
            self.try_count = 1
            self.round += 1

        logger.info("** planner基于当前观测进行规划 round: {} try: {} **".format(self.round+1, self.try_count))

        if self.round == 0:
            task_ref = self.select_experience(task, top_n=self.n_planner_experience)
            self.experience_prompt = """
                The following are the operations experiences that best match the given task, presented in the form of a task graph:
                {}
                The provided experiences are for reference only. 
                """.format(str(task_ref))
            
            prompt = """
            Task: {}
            Current image semantics: {}
            Target image semantics: {}
            =======================================================================
            Available features:
            1. Image generation: Create an image strictly matching the target semantics. Specify only required dimensions: quantity (use "exactly" if needed), position, attributes, material, color, style, lighting, or semantic relationships.
            2. Image editing: Modify an existing image to gradually match target semantics. Adjust only the necessary dimensions; do not add unrelated objects. Regeneration of the whole image is not allowed.

            Instructions:
            - Analyze the **target image semantics**, **task requirements**, and **historical operation information** (if available). 
            - **Provide the next processing step to gradually meet the final task requirements through subsequent multi-round interactions.**
            - Each operation should be concise (≤30 words) while retaining essential elements.
            - Use precise instructions (e.g., "remove the apple on the far right"), avoiding vague expressions.
            - Preferably output a single most effective operation per round; if the task is complex and model capability allows, multiple operations can be included in one round.
            - For generation tasks, output images should be natural and harmonious.
            - For editing tasks, do not regenerate images arbitrarily; only modify necessary parts.
            - If multiple operations can achieve the task, select the one with the highest success rate.
            - Specify dependencies clearly: <depend>None</depend> if independent, or <depend>round X</depend> if based on a previous round (not necessarily the immediately preceding round).

            Output format:
            <operation>Describe the operation using only required dimensions. Use "exactly" for strict quantities.</operation>
            <depend>None or round X</depend>

            Examples:
            <operation>Generate exactly one red apple and one green pear on a wooden table under soft morning light.</operation>
            <depend>None</depend>

            <operation>Add a yellow banana to the right of the dog in the image.</operation>
            <depend>round 2</depend>

            Provide only the current round's operation(s); do not output further steps.
            """.format(task, img_info, semantic)
            
            prompt_ori = prompt
            if self.use_experience:      # 随机性和经验参考并存
                prompt = prompt + self.experience_prompt
                
            prompt_ori = re.sub(r'^[ ]+', '', prompt_ori, flags=re.MULTILINE)
            prompt = re.sub(r'^[ ]+', '', prompt, flags=re.MULTILINE)
            logger.info("user: "+prompt)
            reply_, reply = await self.chat(prompt)
            logger.info("planner: "+reply)
            
            self.shared_memory["current_prompt"] = prompt_ori
            self.shared_memory["current_reply"] = reply
            
            # operations = self.extract_operations(reply)
            operation = self.parse_content(reply, "<operation>")
            
            msg_send2worker = img_info + "\n" +\
                            "task: " + operation
            
            self.shared_memory["current operation"] = operation
            
            if "user img path" in self.shared_memory:
                msg_send2worker = msg_send2worker +\
                            "<input>"+self.shared_memory["user img path"]+"</input>"
            
            self.shared_memory["current_dependance"] = None
            
            self.shared_memory["woker_task_info"] = msg_send2worker
            msg_send2worker = Message(content=msg_send2worker, role=self.profile)
            msg_send2worker.send_to = "Worker"
            
            return msg_send2worker
        
        else:
            if self.round < 5 and self.task_graph["round {}".format(self.round)]["result"]["final goal score"] < 0.8:    # 如果不是全部正确，则需要做出修改
                semantic = self.shared_memory["target_semantic"]
                prompt = """
                Task: {}
                Current image semantics: {}
                Target image semantics: {}
                Historical operations (task graph):
                {}

                Score range: 0 to 1
                =======================================================================
                Available features:
                1. Image generation: Create an image strictly matching the target semantics. Specify only required dimensions: quantity (use "exactly" if needed), position, attributes, material, color, style, lighting, or semantic relationships.
                2. Image editing: Modify an existing image to gradually match target semantics. Adjust only the necessary dimensions; do not add unrelated objects. Regeneration of the whole image is not allowed.

                Instructions:
                - Analyze the **target image semantics**, **task requirements**, and **historical operation information** (if available). 
                - **Provide the next processing step to gradually meet the final task requirements through subsequent multi-round interactions.**
                - Each operation should be concise (≤30 words) while retaining essential elements.
                - Use precise instructions (e.g., "remove the apple on the far right"), avoiding vague expressions.
                - Preferably output a single most effective operation per round; if the task is complex and model capability allows, multiple operations can be included in one round.
                - For generation tasks, output images should be natural and harmonious.
                - For editing tasks, do not regenerate images arbitrarily; only modify necessary parts.
                - If multiple operations can achieve the task, select the one with the highest success rate.
                - Specify dependencies clearly: <depend>None</depend> if independent, or <depend>round X</depend> if based on a previous round (not necessarily the immediately preceding round).

                Output format:
                <operation>Describe the operation using only required dimensions. Use "exactly" for strict quantities.</operation>
                <depend>None or round X</depend>

                Examples:
                <operation>Generate exactly one red apple and one green pear on a wooden table under soft morning light.</operation>
                <depend>None</depend>

                <operation>Add a yellow banana to the right of the dog in the image.</operation>
                <depend>round 2</depend>

                Provide only the current round's operation(s); do not output further steps.
                """.format(task, img_info, semantic, str(self.task_graph))

                prompt_ori = prompt

                prompt_ori = re.sub(r'^[ ]+', '', prompt_ori, flags=re.MULTILINE)
                prompt = re.sub(r'^[ ]+', '', prompt, flags=re.MULTILINE)
                logger.info("user: "+prompt)
                reply_, reply = await self.chat(prompt)
                logger.info("planner: "+reply)

                self.shared_memory["current_prompt"] = prompt_ori
                self.shared_memory["current_reply"] = reply

                operation = self.parse_content(reply, "<operation>")
                
                dependance = self.parse_content(reply, "<depend>")
                if dependance != "None":
                    input_img_dir = self.task_graph[dependance]["result"]["output_img_dir"]
                    
                    self.shared_memory["current_dependance"] = dependance
                    
                    msg_send2worker = "image path: {}\n".format(input_img_dir) +\
                                    "task: {}\n".format(operation) +\
                                    "<input>"+input_img_dir+"</input>"
                    
                    self.shared_memory["dependance score"] = self.task_graph[dependance]["result"]["final goal score"]
                    
                else:
                    img_info = self.shared_memory["img_info"]
                    if img_info == "":
                        img_info = "None"
                    msg_send2worker = img_info + "\n" +\
                                    "task: {}\n".format(operation)
                    
                    if "user img path" in self.shared_memory:
                        msg_send2worker = msg_send2worker +\
                                    "<input>"+self.shared_memory["user img path"]+"</input>"
                
                    self.shared_memory["current_dependance"] = None
                
                    self.shared_memory["dependance score"] = 0.
                
                self.shared_memory["current operation"] = operation
                self.shared_memory["woker_task_info"] = msg_send2worker
                msg_send2worker = Message(content=msg_send2worker, role=self.profile)
                msg_send2worker.send_to = "Worker"
                
                return msg_send2worker
        
            else:       # 直接输出最高分的图像
                max_score = 0.
                final_dir = None

                for round_key, round_val in self.task_graph.items():
                    score = round_val["result"].get("final goal score", 0)
                    if score > max_score:
                        max_score = score
                        final_dir = round_val["result"].get("output_img_dir")

                if os.path.exists(final_dir):
                    logger.info("==================== 图像生成已完成 ====================")
                    now_time = datetime.now()
                    save_dir = now_time.strftime("./outputs/final_output_%Y%m%d%H%M.png")
                    shutil.copy(final_dir, save_dir)
                    logger.info("输出图像路径：\n{}".format(save_dir))
                    self.trajectory_info["task_graph"] = self.task_graph
                    self.trajectory_info["final_output"] = save_dir
                    self.trajectory_info["max_score"] = max_score
                    self.trajectory_info["history"] = self.history
                    # self.write_json(os.path.join(self.experience_path, "task_graph.json"), self.trajectory_info)
                else:
                    logger.info("==================== 生成失败 ====================")
        
class Worker(Role_AIGC):
    name: str = "Worker"
    profile: str = "Worker"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rc.react_mode = RoleReactMode.PLAN_AND_ACT
        self.description = """
        You manage all the tools used for image generation or editing. You are very skilled at selecting and using the appropriate tools based on the given task.
        """
        self.system_prompt = self.get_system_prompt()
        
        self.n_candidate_tools = 2      # 每个子任务尝试的工具数目
        self.n_random = 1
        self.n_experience_tools = 5     # 每个子任务经验选取的工具数目（需要小于尝试工具数目）
        self.n_usage_experience = 3     # 每个工具使用的参考经验数
        self.k = 0.5
        self.lr = 0.1
        
        self.subtask_experience_sorted, self.tool_usage_experience_sorted = self.read_experience()

    def read_experience(self):
        # 初始化经验数据字典
        subtask_experience_sorted = {}

        # ---------- 处理 subtask_tool_select.json ----------
        subtask_file = os.path.join(self.experience_path, "subtask_tool_select.json")
        if os.path.exists(subtask_file) and os.path.getsize(subtask_file) > 0:
            try:
                with open(subtask_file, "r", encoding="utf-8") as f:
                    subtask_experience = json.load(f)
            except json.JSONDecodeError:
                subtask_experience = []
        else:
            subtask_experience = []

        subtask_experience_task_list = [item["task"] for item in subtask_experience]
        subtask_experience_tool_list = [item["tool_name"] for item in subtask_experience]
        subtask_experience_sorted["task_list"] = subtask_experience_task_list
        subtask_experience_sorted["tool_list"] = subtask_experience_tool_list

        # ---------- 处理 tool_usage_experience.json ----------
        usage_file = os.path.join(self.experience_path, "tool_usage_experience.json")
        if os.path.exists(usage_file) and os.path.getsize(usage_file) > 0:
            try:
                with open(usage_file, "r", encoding="utf-8") as f:
                    tool_use_experience = json.load(f)
            except json.JSONDecodeError:
                tool_use_experience = []
        else:
            tool_use_experience = []

        tool_usage_experience_sorted = {}
        for item in tool_use_experience:
            tool_name = item["tool_name"]
            usage_data = {
                "task": item["subtask"],
                "pre_tool": item["pre_tool"],
                "conditions": item["conditions"]
            }
            if tool_name in tool_usage_experience_sorted:
                tool_usage_experience_sorted[tool_name]["subtask_list"].append(item["subtask"])
                tool_usage_experience_sorted[tool_name]["usage"].append(usage_data)
            else:
                tool_usage_experience_sorted[tool_name] = {
                    "subtask_list": [item["subtask"]],
                    "usage": [usage_data]
                }
        
        return subtask_experience_sorted, tool_usage_experience_sorted

    def select_tools_from_experience(self, subtask: str, top_n: int):
        """
        按照经验，选取前n个不同的工具作为参考
        """
        if not self.subtask_experience_sorted.get("task_list") or not self.subtask_experience_sorted.get("tool_list"):
            return []

        tool_idx_list, scores = self.find_top_n_indices(subtask, self.subtask_experience_sorted["task_list"], 1000)

        selected_tools = []
        selected_tasks = []
        seen_tools = set()

        for i in tool_idx_list:
            tool = self.subtask_experience_sorted["tool_list"][i]
            task = self.subtask_experience_sorted["task_list"][i]
            if tool not in seen_tools:
                selected_tools.append(tool)
                selected_tasks.append(task)
                seen_tools.add(tool)
            if len(selected_tools) == top_n:
                break
        
        select_experience = []
        for task, tool in zip(selected_tasks, selected_tools):
            select_experience.append({
                "similar task": task,
                "experienced tool": tool
            })

        return select_experience
    
    def select_toolusage_from_experience(self, tool_name: str, subtask: str, top_n: int):
        # 如果该工具没有使用经验，直接返回空
        if tool_name not in self.tool_usage_experience_sorted:
            return []

        tool_usage_experience = self.tool_usage_experience_sorted[tool_name]
        
        if not tool_usage_experience.get("subtask_list") or not tool_usage_experience.get("usage"):
            return []

        usage_idx_list, scores = self.find_top_n_indices(subtask, tool_usage_experience["subtask_list"], top_n)
        selected_usage = [tool_usage_experience["usage"][i] for i in usage_idx_list]

        return selected_usage

    def refresh_experience(self):   # 更新经验
        self.subtask_experience_sorted, self.tool_usage_experience_sorted = self.read_experience()

    def get_preference_weight(self, reply: str, tool_category: str):
        if tool_category == 'text2image-generation tool':
            preference_list = ["color", "shape", "texture", "2D-spatial", "3D-spatial", "numeracy", "non-spatial"]
        
        elif tool_category == 'image-editing tool':
            preference_list = ["addition", "removement", "replacement", "attribute-alter", "motion-change", "style-transfer", "background-change"]
        
        preference_weight = {}
        preference_content = self.parse_content(reply, "<preference>")
        for preference_type in preference_list:
            if preference_type in preference_content:
                preference_weight[preference_type] = float(self.parse_content(preference_content, "<{}>".format(preference_type)))
        
        total = sum(preference_weight.values())
        preference_weight = {k: v / total for k, v in preference_weight.items()}
        
        return preference_weight

    async def _plan_and_act(self) -> Message:
        task = self.shared_memory["task"]
        success_flag = False
        dependance_score = self.shared_memory["dependance score"]
        
        current_operation = self.shared_memory["current operation"]
        task_info = self.shared_memory["woker_task_info"]
        input_dir = None
        if "<input>" in task_info:
            input_dir = self.parse_content(task_info, "<input>")
            task_info = task_info.split("<input>")[0]
        
        select_experience = self.select_tools_from_experience(current_operation, top_n=self.n_experience_tools)   # 选取经验中最佳的n个工具
        # candidate_num = self.n_candidate_tools - len(selected_tools)
        
        AIGC_Tools_description = self.tools.AIGC_Tools_description
        # 工具选择
        prompt = """
            Task : {}
            
            1. You have two types of tools to choose from: **text2image-generation tool**, **image-editing tool**.
            Please choose the appropriate tool-type based on the task requirements.
            Please output in the xml format: <category>your select tools-type</category>
            
            2. If you choose the 'text2image-generation tool' category, please analyze the task requirements and assign weights for the following preferences:
            **'color', 'shape', 'texture', '2D-spatial', '3D-spatial', 'numeracy', 'non-spatial'**
            'color' indicates a requirement for the object’s color in the generated image.
            'shape' indicates a requirement for the object’s shape in the generated image.
            'texture' indicates a requirement for the object’s material or surface quality in the generated image, such as 'wooden', 'metallic', etc.
            '2D-spatial' indicates a requirement for the 2D spatial relationships between objects in the generated image, such as 'on the side of', 'on the left', 'on the top of', 'next to', etc.
            '3D-spatial' indicates a requirement for the 3D spatial relationships between objects in the generated image, such as 'behind', 'hidden by', 'in front of', etc.
            'numeracy' indicates a requirement for the number of objects in the generated image.
            'non-spatial' indicates a requirement for non-spatial relationships between objects in the generated image, such as 'A is holding B', 'C is looking at D', 'E is sitting on F', etc.
            Please output in the xml format: 
            <preference>
            <color>your 'color' weight</color>
            <shape>your 'shape' weight</shape>
            <texture>your 'texture' weight</texture>
            <2D-spatial>your '2D-spatial' weight</2D-spatial>
            <3D-spatial>your '3D-spatial' weight</3D-spatial>
            <numeracy>your 'numeracy' weight</numeracy>
            <non-spatial>your 'non-spatial' weight</non-spatial>
            </preference>
            
            3. If you choose the 'image-editing tool' category, please analyze the task requirements and assign weights for the following preferences:
            **'addition', 'removement', 'replacement', 'attribute-alter', 'motion-change', 'style-transfer', 'background-change'**
            'addition' indicates that the task involves adding objects to the image.
            'removement' indicates that the task involves removing objects from the image.
            'replacement' indicates that the task involves replacing objects in the image.
            'attribute-alter' indicates that the task involves modifying the attributes of objects in the image.
            'motion-change' indicates that the task involves modifying the actions, movements, or spatial positions of objects in the image.
            'style-transfer' indicates that the task involves modifying the overall style of the image.
            'background-change' indicates that the task involves modifying the background of the image.
            Please output in the xml format: 
            <preference>
            <addition>your 'addition' weight</addition>
            <removement>your 'removement' weight</removement>
            <replacement>your 'replacement' weight</replacement>
            <attribute-alter>your 'attribute-alter' weight</attribute-alter>
            <motion-change>your 'motion-change' weight</motion-change>
            <style-transfer>your 'style-transfer' weight</style-transfer>
            <background-change>your 'background-change' weight</background-change>
            </preference>
            
            Notes:  
            - Weights range from 0 to 1; higher values indicate greater importance.  
            - The sum of the weights must be 1.
            - Unimportant dimensions should be assigned very low values, even 0.  
            - Ensure all weights strictly align with the user’s input and task requirements, emphasizing key dimensions.
            - Do not confuse the preferences of the 'text2image-generation tool' and the 'image-editing tool'.
        """.format(task_info)    # 其余未靠经验选取的，由大模型自行选择
        prompt = re.sub(r'^[ ]+', '', prompt, flags=re.MULTILINE)
        logger.info("user: "+prompt)
        reply = await self.chat(prompt)
        logger.info("Worker: "+reply)
        
        tool_category = self.parse_content(reply, "<category>")
        preference_weight = self.get_preference_weight(reply, tool_category)
        logger.info("子任务归一化权重: "+str(preference_weight))
        candidate_tools = []
        tools_scores = []
        for tool_name in self.tools.AIGC_Tools.keys():
            if self.tools.AIGC_Tools[tool_name].category == tool_category:
                preference_score = round(sum(self.tools.AIGC_Tools[tool_name].preference[key] * preference_weight[key] for key in preference_weight.keys()), 4)
                candidate_tools.append(tool_name)
                tools_scores.append(preference_score)
        
        sorted_candidate_tools = [name for _, name in sorted(zip(tools_scores, candidate_tools), key=lambda x: x[0], reverse=True)]
        logger.info("任务分数计算: "+str(candidate_tools)+" "+str(tools_scores))
        
        tools_list = sorted_candidate_tools[:self.n_candidate_tools]    # 选取前top m个
        remaining = sorted_candidate_tools[self.n_candidate_tools:]     # 剩下的工具，随机选取n个
        if len(remaining) > 0:
            if len(remaining) > self.n_random:
                tools_list.extend(random.sample(remaining, self.n_random))
            else:
                tools_list.extend(remaining)
        logger.info("agent选择: "+str(tools_list))

        tool_result = {}
        tool_name_list = []
        status_list = []
        output_list = []
        
        tools_list = [tools_list[0]]
        for tool_name in tools_list:    # 执行所有候选工具
            logger.info("工具选择："+str(tool_name))
            
            try:
                example = self.tools.AIGC_Tools[tool_name].example
                select_aigc_tool = self.tools.AIGC_Tools[tool_name]
            except:
                continue
            
            tool_usage_experence = self.select_toolusage_from_experience(tool_name, current_operation, top_n=self.n_usage_experience)
            
            prompt = """
            Task : {}
            
            You have selected {} to complete the task.
            Tool destription: {}
            Usage example: 
            {}
            
            Please provide the properly formatted conditions for {} according to the given example in order to solve the task.
            
            **The given tool conditions should strictly follow the style and meaning of the sample, using concise text and avoiding excessive complexity.**
            
            """.format(task_info, tool_name, self.tools.AIGC_Tools[tool_name].description, example, tool_name)
            prompt = re.sub(r'^[ ]+', '', prompt, flags=re.MULTILINE)
            logger.info("user: "+prompt)
            reply = await self.chat(prompt)
            logger.info("Worker: "+reply)
            
            is_tool_success = False
            try:
                result, condition_str_clean = await select_aigc_tool.run(reply)
                logger.info(result)
                is_tool_success = True
            except:
                logger.info("条件提供错误")
            
            if is_tool_success:
                img_dir = self.parse_content(result, "<save_path>")
                tool_name_list.append(tool_name)
                output_list.append(img_dir)
        
        if len(output_list) > 1:
            best_output_dir = output_list[0]
            best_tool_name = tool_name_list[0]
            logger.info("最佳工具 {}".format(best_tool_name))
            success_flag = True


        elif len(output_list) == 1:
            best_output_dir = output_list[0]
            best_tool_name = tool_name_list[0]
            success_flag = True
        
        else:
            success_flag = False
        
        if success_flag:
            evaluation, score = await self.unit_test(task=current_operation, output_dir=best_output_dir, input_dir=input_dir, tool_name=best_tool_name)
                    
            tool_result[best_tool_name] = {
                "final goal score": score,
                "final goal evaluation": evaluation,
                "output_img_dir": best_output_dir
            }
            
            self.clean_history()
        
            if dependance_score < score:
                subtask_tool_select = {
                                            "task": current_operation,
                                            "tool_name": best_tool_name,
                                        }
            
            img_semantic = await self.read_img.run([tool_result[best_tool_name]["output_img_dir"]])
            tool_result[best_tool_name]["img_semantic"] = img_semantic
            
            self.shared_memory["current result"] = tool_result[best_tool_name]

        else:   # 所选用的工具全部失败.
            failure_result = {
                "final goal score": 0,
                "final goal evaluation": "The result of this operation is unsatisfactory. Please reconsider the operation at this node or analyze whether the dependent pipeline is correct.",
                "suggestions": "None",
                "output_img_dir": "None"
            }
            self.shared_memory["current result"] = failure_result

        msg_send2Planner = Message(content=str(self.shared_memory["current result"]), role=self.profile)
        msg_send2Planner.send_to = "Planner"

        return msg_send2Planner

async def main(input_info, planner_use_exp=False):
    other_config_file = Path("/s2/chenzhipeng/chenzhipeng_folder/project/Agent4AIGC/config/others.yaml")
    other_config = Config.from_yaml_file(other_config_file)
    
    planner_config_file = Path("/s2/chenzhipeng/chenzhipeng_folder/project/Agent4AIGC/config/planner.yaml")
    planner_config = Config.from_yaml_file(planner_config_file)
    
    context = Context()     # 配置上下文环境信息
    context.cost_manager.max_budget = 100.      # 配置最多100$上限
    env = Environment(context=context)
    
    Role_AIGC.shared_memory = {}
    worker = Worker(config=other_config)
    worker.refresh_experience()  # 每次刷新经验
    worker.clean_history()
    planner = Planner(config=other_config)
    planner.refresh_experience()  # 每次刷新经验
    planner.clean_history()

    analyst = Analyst(config=other_config)
    
    if planner_use_exp:
        planner.use_experience = True
    else:
        planner.use_experience = False
    
    env.add_roles([analyst, planner, worker])
    input_info = json.dumps(input_info)
    
    env.publish_message(Message(content=input_info, send_to="Analyst"))
    while not env.is_idle:
        await env.run()

if __name__ == "__main__":
    
    input_info = {
                    'text': ["A black laptop sits on a wooden table beside a red chair in the library, with a yellow book lying on the floor to the left."]
                    }
    
    # input_info = {
    #                 'image': ['./assets/image.png'],
    #                 'text': ["Place the dish on a marble countertop, decorate it with a mint leaf garnish next to the cherries, position a silver spoon beside it."]
    #                 }
    
    asyncio.run(main(input_info, planner_use_exp=False))


