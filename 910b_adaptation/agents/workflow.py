from __future__ import annotations
import xml.etree.ElementTree as ET
import asyncio
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import random
import re
import shutil
import textwrap
import time
from typing import Dict
from .prompts import (
    IMAGE_ANALYST_DESCRIPTION,
    IMAGE_DESCRIPTION_PROMPT,
    PLANNER_DESCRIPTION,
    PLANNER_EXPERIENCE_PROMPT,
    TOOL_ERROR_EVALUATION,
    WORKER_DESCRIPTION,
)
from . import active_prompts as active, schemas
logger = logging.getLogger(__name__)

def error_category(error):
    if isinstance(error, WorkflowError):
        return 'tool_execution'
    if type(error).__name__ == 'LLMRefusalError':
        return 'refusal'
    if isinstance(error, (schemas.AgentProtocolError, ValueError, KeyError, TypeError)):
        return 'protocol'
    if type(error).__name__ in {'APIConnectionError', 'APITimeoutError', 'ConnectError', 'ReadTimeout', 'ConnectTimeout', 'TimeoutException'} or isinstance(error, (ConnectionError, TimeoutError)):
        return 'network'
    if type(error).__name__ in {'APIStatusError', 'RateLimitError', 'AuthenticationError', 'PermissionDeniedError', 'NotFoundError', 'BadRequestError', 'InternalServerError'}:
        return 'api'
    return 'runtime'

@dataclass
class Message:
    content: str
    role: str = ''
    send_to: str = ''

class MLLM_read_img:

    def __init__(self, llm, prompt_profile='original'):
        self.llm = llm
        self.prompt_profile = prompt_profile
        self.last_observation = {}
    role_prompt: str = 'You are an excellent Analyst.\n'
    description: str = IMAGE_ANALYST_DESCRIPTION
    name: str = 'ReadImage'

    async def run(self, image_path):
        detailed = self.prompt_profile == 'precise'
        text = active.BLIND_IMAGE_PROMPT if detailed else IMAGE_DESCRIPTION_PROMPT + 'background, style, and other relevant details.'
        prompt = [{'type': 'text', 'text': text}]
        prompt.extend(({'type': 'image', 'image': path} for path in image_path))
        options = {'max_tokens': active.TOKEN_BUDGETS['read_image']} if detailed else {}
        result = await self.llm.arequest_json(prompt, schema=schemas.IMAGE_OBSERVATION if detailed else schemas.READ_IMAGE, name='read_image', system_msgs=[self.role_prompt + self.description], **options)
        self.last_observation = result
        return result['semantic']

class Role_AIGC:

    def __init__(self, *, llm, tools, shared_memory, experience_path, evaluator=None, prompt_profile='original'):
        self.llm = llm
        self.tools = tools
        self.shared_memory = shared_memory
        self.experience_path = str(experience_path)
        self.evaluator = evaluator
        self.prompt_profile = prompt_profile
        self.history = []
        self.description = ''
        self.system_prompt = ''
        self.read_img = MLLM_read_img(llm, prompt_profile)
        self._inbox = []

    def precision(self, stage, prompt, observation=None, **context):
        if self.prompt_profile == 'original':
            return prompt
        return active.precise_prompt(stage, self.shared_memory.get('user_request', ''), observation, **context)

    def get_memories(self, k=1):
        return self._inbox[-k:]

    def find_top_n_indices(self, query, candidates, top_n=1):
        if not candidates or top_n <= 0:
            return ([], [])

        def counts(text):
            return Counter(re.findall('\\w+', str(text).casefold()))
        query_vec = counts(query)
        qnorm = math.sqrt(sum((v * v for v in query_vec.values())))
        scores = []
        for (i, candidate) in enumerate(candidates):
            vec = counts(candidate)
            norm = math.sqrt(sum((v * v for v in vec.values())))
            score = sum((v * vec.get(k, 0) for (k, v) in query_vec.items())) / (qnorm * norm) if qnorm * norm else 0.0
            scores.append((i, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        selected = scores[:min(top_n, len(scores))]
        return ([i for (i, _) in selected], [v for (_, v) in selected])

    def get_system_prompt(self):
        self.description = textwrap.dedent(self.description)
        system_prompt = 'You are an excellent {}. {}\n'.format(self.name, self.description)
        return system_prompt

    def clean_history(self):
        self.history = []

    async def request(self, prompt, schema, name, use_history=False, validate=None):
        options = {'max_tokens': active.TOKEN_BUDGETS[name]} if self.prompt_profile == 'precise' else {}
        result = await self.llm.arequest_json(prompt, schema=schema, name=name, system_msgs=[self.system_prompt], history=self.history if use_history else None, validate=validate, **options)
        if use_history:
            self.history.extend([{'role': 'user', 'content': prompt}, {'role': 'assistant', 'content': json.dumps(result, ensure_ascii=False)}])
        return result

class Evaluator(Role_AIGC):

    async def evaluate(self, task, output_dir, input_dir=None, tool_name=None):
        goal_eval = self.shared_memory.get('goal_eval', {})
        prompt = active.evaluation_prompt(task, goal_eval, self.shared_memory.get('original_semantic', ''))
        prompt = self.precision('evaluator', prompt, self.shared_memory.get('last_observation'),
                                goals={key: value for key, value in goal_eval.items() if key != 'semantic'})
        content = [{'type': 'image', 'image': output_dir}, {'type': 'text', 'text': prompt}]
        if self.prompt_profile == 'precise':
            content = []
            if input_dir:
                content += [{'type': 'text', 'text': 'BEFORE image: reference for preservation only.'},
                            {'type': 'image', 'image': input_dir}]
            content += [{'type': 'text', 'text': 'AFTER image: evaluate this output.'},
                        {'type': 'image', 'image': output_dir}, {'type': 'text', 'text': prompt}]
        options = {'max_tokens': active.TOKEN_BUDGETS['evaluator_scores']} if self.prompt_profile == 'precise' else {}
        result = await self.llm.arequest_json(content, schema=schemas.evaluator_scores(goal_eval), name='evaluator_scores', **options)
        test_results = {}
        for (key, label) in schemas.DIMENSIONS.items():
            if key in goal_eval:
                dimension = result['dimensions'][key]
                score = float(dimension['score'])
                if not math.isfinite(score) or not 0 <= score <= 100:
                    raise schemas.AgentProtocolError(f'Evaluator returned invalid {key} score: {score}')
                test_results[label] = {'score': score / 100, 'evaluation': dimension['explanation'], 'weight': 1.0}
        other = [item['score'] for (key, item) in test_results.items() if key != 'semantic']
        local_mean = sum(other) / len(other) if other else 0.0
        final_score = round((local_mean + test_results['semantic']['score']) / 2, 4)
        evaluation = result['final_evaluation']
        self.shared_memory['last_evaluation'] = {'dimensions': test_results, 'final_score': final_score, 'final_evaluation': evaluation}
        dimensions = {key: round(value['score'], 3) for key, value in test_results.items()}
        logger.info('[Evaluator] tool=%s score=%.4f dimensions=%s evaluation=%s',
                    tool_name, final_score, dimensions, evaluation)
        return (evaluation, final_score)

class Analyst(Role_AIGC):
    name: str = 'Analyst'
    profile: str = 'Analyst'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = "You are highly skilled at summarizing users' requirements for " + 'image generation or image editing based on the image information ' + 'and text descriptions they provide.\n'
        self.system_prompt = self.get_system_prompt()

    async def get_goal(self, task, semantic, img_info):
        prompt = active.ANALYST_GOALS_PROMPT + '\n\nprompt: task: ' + task + '\n' + img_info + 'Target image semantics: ' + semantic
        prompt = self.precision('analyst_goals', prompt, self.shared_memory.get('original_observation'))
        response = await self.request(prompt, schemas.ANALYST_GOALS, 'analyst_goals', validate=schemas.original_goals)
        goal = schemas.original_goals(response)
        goal_eval = {key: value for (key, value) in goal.items() if value is not None}
        goal_eval['semantic'] = semantic
        return (goal, goal_eval, {key: False for key in goal_eval})

    async def analysis_task_semantic(self, user_instruction, img_info):
        user_info = user_instruction + img_info
        prompt = 'user_info: ' + user_info + '\n' + active.ANALYST_TASK_PROMPT
        prompt = self.precision('analyst_task', prompt, self.shared_memory.get('original_observation'))
        result = await self.request(prompt, schemas.ANALYST_TASK, 'analyst_task')
        return (result['task'], result['semantic'], user_info)

    async def _plan_and_act(self) -> Message:
        logger.info('[Analyst] 开始分析用户任务和输入图像')
        context_msg = self.get_memories(k=1)[0].content
        context_msg = json.loads(context_msg)
        user_instruction = context_msg['text'][0]
        self.shared_memory['user_request'] = user_instruction
        if 'image' in context_msg:
            image_path = context_msg['image']
            img_description = await self.read_img.run(image_path)
            img_info = 'Image path: ' + str(image_path) + '\n' + 'Image description: ' + img_description + '\n'
            self.shared_memory['original_semantic'] = img_description
            self.shared_memory['original_observation'] = self.read_img.last_observation
            logger.info('[Analyst] input_image=%s semantic=%s', image_path, img_description)
            self.shared_memory['user img path'] = str(image_path[0])
        else:
            img_info = ''
            logger.info('[Analyst] input_image=None')
        (task, semantic, user_info) = await self.analysis_task_semantic(user_instruction, img_info)
        (goal, goal_eval, goal_logit) = await self.get_goal(task, semantic, img_info)
        logger.info('[Analyst] task=%s semantic=%s goals=%s', task, semantic, goal)
        self.shared_memory['task'] = task
        self.shared_memory['target_semantic'] = semantic
        self.shared_memory['img_info'] = img_info
        self.shared_memory['goal_eval'] = goal_eval
        self.shared_memory['goal_logit'] = goal_logit
        msg_send2planner = Message(content=img_info, role=self.profile)
        msg_send2planner.send_to = 'Planner'
        return msg_send2planner

class Planner(Role_AIGC):
    name: str = 'Planner'
    profile: str = 'Planner'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.description = PLANNER_DESCRIPTION
        self.system_prompt = self.get_system_prompt()
        self.trajectory_info = {}
        self.task_graph = {}
        self.shared_memory['dependance score'] = 0.0
        self.round = 0
        self.n_planner_experience = 3
        self.planner_experience = self.read_experience()
        self.try_count = 0
        self.try_num = 1
        self.round_info_buff = []
        self.use_experience = False
        self.experience_prompt = ''

    def extract_subtask_path(self, task_graph: dict) -> list:
        if not task_graph:
            return []
        all_rounds = set(task_graph.keys())
        depended_rounds = set((val['depends_on'] for val in task_graph.values() if val.get('depends_on') is not None))
        final_rounds = list(all_rounds - depended_rounds)
        if not final_rounds:
            raise ValueError('未能确定最终的子任务节点。请检查 depends_on 结构。')
        if len(final_rounds) > 1:
            final_rounds.sort(key=lambda k: task_graph[k].get('result', {}).get('final goal score', 0), reverse=True)
        path = []
        current_key = final_rounds[0]
        while current_key:
            current_task = task_graph[current_key]
            path.append(current_task['subtask'])
            current_key = current_task['depends_on']
        return list(reversed(path))

    def read_experience(self):
        plan_file = os.path.join(self.experience_path, 'task_graph.json')
        if os.path.exists(plan_file) and os.path.getsize(plan_file) > 0:
            try:
                with open(plan_file, 'r', encoding='utf-8') as f:
                    planner_experience = json.load(f)
            except json.JSONDecodeError:
                planner_experience = []
        else:
            planner_experience = []
        planner_experience_task_list = [item['task'] for item in planner_experience]
        planner_experience_taskgraph_list = [item['task_graph'] for item in planner_experience]
        planner_experience_sorted = {'task_list': planner_experience_task_list, 'taskgraph_list': planner_experience_taskgraph_list}
        return planner_experience_sorted

    def refresh_experience(self):
        self.planner_experience = self.read_experience()

    def select_experience(self, task, top_n=3):
        (task_idx_list, scores) = self.find_top_n_indices(task, self.planner_experience['task_list'], top_n)
        selected_tasks = [self.planner_experience['task_list'][i] for i in task_idx_list]
        selected_taskgraphs = [self.planner_experience['taskgraph_list'][i] for i in task_idx_list]
        taskgraph_list = []
        for (task_data, taskgraph_data) in zip(selected_tasks, selected_taskgraphs):
            taskgraph_path = self.extract_subtask_path(taskgraph_data)
            taskgraph = {'task': task_data, 'operations': taskgraph_path}
            taskgraph_list.append(taskgraph)
        return taskgraph_list

    async def _plan_and_act(self):
        task = self.shared_memory['task']
        semantic = self.shared_memory['target_semantic']
        img_info = self.shared_memory['img_info'] or 'None'
        self.trajectory_info.update(task=task, user_img_info=img_info, target_semantic=semantic)
        if 'current result' in self.shared_memory:
            self.round += 1
            self.task_graph[f'round {self.round}'] = {'subtask': self.shared_memory['current operation'], 'depends_on': self.shared_memory['current_dependance'], 'result': self.shared_memory.pop('current result')}
            self.history.extend([{'role': 'user', 'content': self.shared_memory['current_prompt']}, {'role': 'assistant', 'content': json.dumps(self.shared_memory['current_reply'], ensure_ascii=False)}])
            if self.round >= self.max_rounds or self.task_graph[f'round {self.round}']['result']['final goal score'] >= self.score_threshold:
                return None
        dependencies = [name for (name, node) in self.task_graph.items() if Path(node['result'].get('output_img_dir') or '').is_file()]
        if self.round == 0:
            prompt = active.PLANNER_INITIAL_PROMPT.format(task, img_info, semantic)
            if self.use_experience:
                experience = self.select_experience(task, self.n_planner_experience)
                prompt += '\n' + active.compact(PLANNER_EXPERIENCE_PROMPT.format(str(experience)))
        else:
            prompt = active.PLANNER_NEXT_PROMPT.format(task, semantic, json.dumps(self.task_graph, ensure_ascii=False))
        if self.prompt_profile == 'precise':


            graph = {name: {'operation': node['subtask'], 'depends_on': node['depends_on'],
                            'observation': node['result'].get('observation', node['result'].get('img_semantic')),
                            'score': node['result']['final goal score'],
                            'feedback': node['result']['final goal evaluation']}
                     for name, node in self.task_graph.items()}
            prompt = self.precision('planner', '', self.shared_memory.get('original_observation'), rounds=graph)
        result = await self.llm.arequest_json(prompt, schema=schemas.planner_step(dependencies),
                                             name='planner_step', history=self.history if self.prompt_profile == 'original' else None,
                                             system_msgs=[self.system_prompt],
                                             validate=schemas.validate_planner_step,
                                             **({'max_tokens': active.TOKEN_BUDGETS['planner_step']} if self.prompt_profile == 'precise' else {}))
        dependency = result['depend']
        if dependency != 'None' and dependency not in dependencies:
            raise schemas.AgentProtocolError(f'Planner selected an unavailable dependency: {dependency}')
        source = None if dependency == 'None' else self.task_graph[dependency]['result']['output_img_dir']
        if source is None:
            source = self.shared_memory.get('user img path')
        source_observation = (self.task_graph[dependency]['result'].get('observation')
                              if dependency != 'None' else self.shared_memory.get('original_observation'))
        task_info = (f'image path: {source}\n' if source else img_info + '\n') + 'task: ' + result['operation']
        self.shared_memory.update({'current_prompt': prompt, 'current_reply': result, 'current operation': result['operation'], 'current_dependance': None if dependency == 'None' else dependency, 'dependance score': 0.0 if dependency == 'None' else self.task_graph[dependency]['result']['final goal score'], 'woker_task_info': task_info, 'worker_input_image': source})
        self.shared_memory['worker_input_observation'] = source_observation
        logger.info('[Planner][round %d] operation=%s depend=%s input_image=%s available=%s',
                    self.round + 1, result['operation'], dependency, source, dependencies)
        return Message(content=task_info, role=self.profile, send_to='Worker')

class Worker(Role_AIGC):
    name: str = 'Worker'
    profile: str = 'Worker'

    def __init__(self, *, worker_mode='direct', **kwargs):
        super().__init__(**kwargs)
        if worker_mode not in {'direct', 'pasm'}:
            raise ValueError('worker_mode must be direct or pasm')
        self.worker_mode = worker_mode
        self.description = WORKER_DESCRIPTION
        self.system_prompt = self.get_system_prompt()
        self.n_candidate_tools = 2
        self.n_random = 1
        self.n_experience_tools = 5
        self.n_usage_experience = 3
        self.k = 0.5
        self.lr = 0.1
        (self.subtask_experience_sorted, self.tool_usage_experience_sorted) = self.read_experience()

    def read_experience(self):
        subtask_experience_sorted = {}
        subtask_file = os.path.join(self.experience_path, 'subtask_tool_select.json')
        if os.path.exists(subtask_file) and os.path.getsize(subtask_file) > 0:
            try:
                with open(subtask_file, 'r', encoding='utf-8') as f:
                    subtask_experience = json.load(f)
            except json.JSONDecodeError:
                subtask_experience = []
        else:
            subtask_experience = []
        subtask_experience_task_list = [item['task'] for item in subtask_experience]
        subtask_experience_tool_list = [item['tool_name'] for item in subtask_experience]
        subtask_experience_sorted['task_list'] = subtask_experience_task_list
        subtask_experience_sorted['tool_list'] = subtask_experience_tool_list
        usage_file = os.path.join(self.experience_path, 'tool_usage_experience.json')
        if os.path.exists(usage_file) and os.path.getsize(usage_file) > 0:
            try:
                with open(usage_file, 'r', encoding='utf-8') as f:
                    tool_use_experience = json.load(f)
            except json.JSONDecodeError:
                tool_use_experience = []
        else:
            tool_use_experience = []
        tool_usage_experience_sorted = {}
        for item in tool_use_experience:
            tool_name = item['tool_name']
            usage_data = {'task': item['subtask'], 'pre_tool': item['pre_tool'], 'conditions': item['conditions']}
            if tool_name in tool_usage_experience_sorted:
                tool_usage_experience_sorted[tool_name]['subtask_list'].append(item['subtask'])
                tool_usage_experience_sorted[tool_name]['usage'].append(usage_data)
            else:
                tool_usage_experience_sorted[tool_name] = {'subtask_list': [item['subtask']], 'usage': [usage_data]}
        return (subtask_experience_sorted, tool_usage_experience_sorted)

    def select_tools_from_experience(self, subtask: str, top_n: int):
        if not self.subtask_experience_sorted.get('task_list') or not self.subtask_experience_sorted.get('tool_list'):
            return []
        (tool_idx_list, scores) = self.find_top_n_indices(subtask, self.subtask_experience_sorted['task_list'], 1000)
        selected_tools = []
        selected_tasks = []
        seen_tools = set()
        for i in tool_idx_list:
            tool = self.subtask_experience_sorted['tool_list'][i]
            task = self.subtask_experience_sorted['task_list'][i]
            if tool not in seen_tools:
                selected_tools.append(tool)
                selected_tasks.append(task)
                seen_tools.add(tool)
            if len(selected_tools) == top_n:
                break
        select_experience = []
        for (task, tool) in zip(selected_tasks, selected_tools):
            select_experience.append({'similar task': task, 'experienced tool': tool})
        return select_experience

    def select_toolusage_from_experience(self, tool_name: str, subtask: str, top_n: int):
        if tool_name not in self.tool_usage_experience_sorted:
            return []
        tool_usage_experience = self.tool_usage_experience_sorted[tool_name]
        if not tool_usage_experience.get('subtask_list') or not tool_usage_experience.get('usage'):
            return []
        (usage_idx_list, scores) = self.find_top_n_indices(subtask, tool_usage_experience['subtask_list'], top_n)
        selected_usage = [tool_usage_experience['usage'][i] for i in usage_idx_list]
        return selected_usage

    def refresh_experience(self):
        (self.subtask_experience_sorted, self.tool_usage_experience_sorted) = self.read_experience()

    def get_preference_weight(self, result, tool_category):
        if tool_category == schemas.GENERATION_CATEGORY:
            dimensions = schemas.GEN_DIMS
        elif tool_category == schemas.EDITING_CATEGORY:
            dimensions = schemas.EDIT_DIMS
        else:
            raise schemas.AgentProtocolError(f'Unknown tool category: {tool_category}')
        weights = result['preference']
        if set(weights) != set(dimensions):
            raise schemas.AgentProtocolError('Tool preference dimensions do not match the selected category')
        if any((isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (not 0 <= value <= 1) for value in weights.values())):
            raise schemas.AgentProtocolError('Tool preferences must be finite values from 0 to 1')
        total = sum(weights.values())
        return {key: value / total for (key, value) in weights.items()} if total else {key: 1 / len(weights) for key in weights}

    async def _select_tool(self, task_info, input_image):
        if self.worker_mode == 'direct':
            def validate_route(result):
                if result['category'] == schemas.EDITING_CATEGORY and not input_image:
                    raise schemas.AgentProtocolError('Editing requires an existing dependency or input image')

            prompt = active.WORKER_ROUTE_PROMPT.format(task_info)
            if self.prompt_profile == 'precise':
                prompt = ('Route the operation: generate with FLUX, or edit the supplied image with Step1X. '
                          'Without an input image, choose generation.\n') + json.dumps(
                    {'operation': self.shared_memory['current operation'], 'input_image_available': bool(input_image)})
            selection = await self.request(prompt,
                                           schemas.worker_route(has_input_image=bool(input_image)), 'worker_route', validate=validate_route)
            category = selection['category']
            name = {schemas.GENERATION_CATEGORY: 'FLUX',
                    schemas.EDITING_CATEGORY: 'Step1X_Edit_ImageEditTool'}[category]
            if name not in self.tools.AIGC_Tools or self.tools.AIGC_Tools[name].category != category:
                raise schemas.AgentProtocolError(f'Two-tool deployment is missing {name}')
            logger.info('[Worker] route category=%s selected=%s input_image=%s', category, name, input_image)
            return name, {'mode': 'direct', 'category': category, 'tool': name}

        selection = await self.request(active.WORKER_PREFERENCES_PROMPT.format(task_info), schemas.WORKER_PREFERENCES, 'worker_preferences',
                                       validate=lambda result: self.get_preference_weight(result, result['category']))
        category = selection['category']
        weights = self.get_preference_weight(selection, category)
        ranked = []
        for (name, tool) in self.tools.AIGC_Tools.items():
            if tool.category == category:
                score = round(sum((tool.preference[key] * weights[key] for key in weights)), 4)
                ranked.append((name, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        if not ranked:
            raise schemas.AgentProtocolError(f'No registered tool for category {category}')
        name = ranked[0][0]
        return name, {'mode': 'pasm', 'category': category, 'weights': weights, 'ranking': ranked}

    async def _plan_and_act(self):
        operation = self.shared_memory['current operation']
        task_info = self.shared_memory['woker_task_info']
        input_image = self.shared_memory.get('worker_input_image')
        name, selection = await self._select_tool(task_info, input_image)
        tool = self.tools.AIGC_Tools[name]
        info = self.tools.get_info(name)
        host_inputs = {key: input_image for (key, meta) in info['conds_info'].items() if meta.get('context') == 'image' and input_image}
        self.shared_memory['tool_errors'] = []
        self.shared_memory['last_selection'] = selection
        specification = self.tools.function_spec(name, host_inputs=host_inputs)
        if self.prompt_profile == 'precise':
            for field in specification['function']['parameters']['properties'].values():
                if field.get('type') == 'string':
                    field['maxLength'] = min(field.get('maxLength', 1600), 1600)
        experience = self.select_toolusage_from_experience(name, operation, self.n_usage_experience)
        prompt = active.tool_conditions_prompt(task_info, name, tool.description, experience)
        stage = 'editing' if tool.category == schemas.EDITING_CATEGORY else 'generation'
        prompt = self.precision(stage, prompt, self.shared_memory.get('worker_input_observation'), operation=operation)
        if self.prompt_profile == 'precise' and input_image:
            prompt = [{'type': 'image', 'image': input_image}, {'type': 'text', 'text': prompt}]
        options = {'max_tokens': active.TOKEN_BUDGETS['function_call']} if self.prompt_profile == 'precise' else {}
        call = await self.llm.arequest_tool(prompt, tools=[specification], tool_choice={'type': 'function', 'function': {'name': name}}, system_msgs=[self.system_prompt], **options)
        if call['name'] != name:
            raise schemas.AgentProtocolError(f"Function call changed the routed tool: {call['name']}")
        conditions = self.tools.bind_function_arguments(name, call['arguments'], host_inputs=host_inputs)
        device = self.tools.device_for(name) if hasattr(self.tools, 'device_for') else None
        started = time.monotonic()
        logger.info('[Worker] selected=%s device=%s input_image=%s conditions=%s',
                    name, device, input_image, conditions)
        try:
            output = await self.tools.invoke_function(call, host_inputs=host_inputs)
            element = ET.fromstring(output)
            output_image = element.findtext('image')
            if element.tag != 'result' or not output_image or (not Path(output_image).is_file()):
                raise ValueError('Tool result must contain an existing image path')
        except Exception as exc:
            error = {'tool': name, 'category': 'tool_execution', 'error': f'{type(exc).__name__}: {exc}'}
            self.shared_memory['tool_errors'].append(error)
            self.shared_memory['current result'] = {'final goal score': 0, 'final goal evaluation': TOOL_ERROR_EVALUATION, 'suggestions': 'None', 'output_img_dir': 'None', 'tool_name': name}
            logger.error('[Worker] tool=%s failed=%s', name, exc)
            return Message(content=json.dumps(self.shared_memory['current result']), role=self.profile, send_to='Planner')
        execution_seconds = round(time.monotonic() - started, 3)
        logger.info('[Worker] tool=%s completed seconds=%.3f output_image=%s', name, execution_seconds, output_image)
        if self.prompt_profile == 'precise':
            semantic = await self.read_img.run([output_image])
            self.shared_memory['last_observation'] = self.read_img.last_observation
            (evaluation, score) = await self.evaluator.evaluate(operation, output_image, input_image, name)
        else:
            (evaluation, score) = await self.evaluator.evaluate(operation, output_image, input_image, name)
            semantic = await self.read_img.run([output_image])
        self.shared_memory['current result'] = {'final goal score': score, 'final goal evaluation': evaluation, 'output_img_dir': output_image, 'tool_name': name, 'device': device, 'execution_seconds': execution_seconds, 'evaluation': self.shared_memory.get('last_evaluation', {}), 'conditions': conditions, 'selection': self.shared_memory['last_selection'], 'img_semantic': semantic}
        if self.prompt_profile == 'precise':
            self.shared_memory['current result']['observation'] = self.read_img.last_observation
        self.clean_history()
        return Message(content=json.dumps(self.shared_memory['current result']), role=self.profile, send_to='Planner')

class WorkflowError(RuntimeError):
    pass
class _TracedLLM:
    def __init__(self, backend, role, events):
        self.backend = backend
        self.role = role
        self.events = events

    async def arequest_json(self, prompt, *, schema, name='agent_message', history=None, system_msgs=None, **kwargs):
        started = time.monotonic()
        event = {'role': self.role, 'kind': 'structured_output', 'name': name, 'prompt': prompt, 'schema': schema, 'system': system_msgs or [], 'history': list(history or [])}
        self.events.append(event)
        logger.info('[%s][%s] 请求模型...', self.role, name)
        try:
            attempts = []
            event['attempts'] = attempts
            result = await self.backend.arequest_json(prompt, schema=schema, name=name, history=history, system_msgs=system_msgs, protocol_events=attempts, **kwargs)
            schemas.validate(result, schema)
            if kwargs.get('validate'):
                kwargs['validate'](result)
            event['response'] = result
            logger.info('[%s][%s] response=%s', self.role, name, result)
            return result
        except Exception as exc:
            event.update(error=f'{type(exc).__name__}: {exc}', error_category=error_category(exc))
            raise
        finally:
            event['elapsed_seconds'] = round(time.monotonic() - started, 3)

    async def arequest_tool(self, prompt, *, tools, tool_choice, history=None, system_msgs=None, **kwargs):
        started = time.monotonic()
        event = {'role': self.role, 'kind': 'function_call', 'prompt': prompt, 'tools': tools, 'tool_choice': tool_choice, 'system': system_msgs or [], 'history': list(history or [])}
        self.events.append(event)
        logger.info('[%s][function_call] 生成工具调用条件...', self.role)
        try:
            attempts = []
            event['attempts'] = attempts
            result = await self.backend.arequest_tool(prompt, tools=tools, tool_choice=tool_choice, history=history, system_msgs=system_msgs, protocol_events=attempts, **kwargs)
            if not isinstance(result, dict) or set(result) != {'name', 'arguments'}:
                raise schemas.AgentProtocolError('Expected one validated function call')
            name = result['name']
            offered = {item['function']['name']: item['function']['parameters'] for item in tools}
            if name != tool_choice['function']['name'] or name not in offered:
                raise schemas.AgentProtocolError(f'Model selected an unoffered function: {name}')
            schemas.validate(result['arguments'], offered[name])
            event['response'] = result
            logger.info('[%s][function_call] tool=%s arguments=%s', self.role, result['name'], result['arguments'])
            return result
        except Exception as exc:
            event.update(error=f'{type(exc).__name__}: {exc}', error_category=error_category(exc))
            raise
        finally:
            event['elapsed_seconds'] = round(time.monotonic() - started, 3)

class Workflow:
    def __init__(self, llm, tools, *, output_dir='outputs', experience_dir='Experience', max_rounds=5, score_threshold=0.9, planner_use_experience=False, worker_mode='direct', prompt_profile='precise'):
        if prompt_profile not in {'original', 'precise'}:
            raise ValueError('prompt_profile must be original or precise')
        self.prompt_profile = prompt_profile
        if not isinstance(max_rounds, int) or not 1 <= max_rounds <= 100:
            raise ValueError('max_rounds must be an integer from 1 to 100')
        if not math.isfinite(score_threshold) or not 0 <= score_threshold <= 1:
            raise ValueError('score_threshold must be between 0 and 1')
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experience_dir = Path(experience_dir).expanduser().resolve()
        self.events = []
        self.rounds = []
        self.shared_memory = {}
        common = dict(tools=tools, shared_memory=self.shared_memory, experience_path=self.experience_dir, prompt_profile=prompt_profile)
        self.evaluator = Evaluator(llm=_TracedLLM(llm, 'Evaluator', self.events), **common)
        self.analyst = Analyst(llm=_TracedLLM(llm, 'Analyst', self.events), **common)
        self.planner = Planner(llm=_TracedLLM(llm, 'Planner', self.events), **common)
        self.worker = Worker(llm=_TracedLLM(llm, 'Worker', self.events), evaluator=self.evaluator, worker_mode=worker_mode, **common)
        self.worker.read_img = MLLM_read_img(_TracedLLM(llm, 'Analyst', self.events), prompt_profile)
        self.planner.max_rounds = max_rounds
        self.planner.score_threshold = score_threshold
        self.planner.use_experience = planner_use_experience
        self.trajectory_path = self.output_dir / 'trajectory.json'
        self._has_run = False

    @staticmethod
    def _validate_input(input_info):
        if not isinstance(input_info, dict):
            raise ValueError('Input must be a JSON object')
        texts = input_info.get('text')
        if not isinstance(texts, list) or not texts or (not isinstance(texts[0], str)) or (not texts[0].strip()):
            raise ValueError('Input requires a nonempty text list, for example {"text":["Generate a cat"]}')
        if len(texts) != 1:
            raise ValueError('One task per run is supported; provide exactly one text item')
        images = input_info.get('image') or []
        if not isinstance(images, list):
            raise ValueError('image must be a list of local image paths')
        images = [str(item) for item in images if item is not None and str(item).strip()]
        for image in images:
            if not Path(image).is_file():
                raise FileNotFoundError(f'Input image does not exist: {image}')
        return {'text': texts, **({'image': images} if images else {})}

    def _snapshot(self, status, **details):
        result = {'status': status, 'worker_mode': self.worker.worker_mode, 'prompt_profile': self.prompt_profile, 'task': self.shared_memory.get('task'), 'target_semantic': self.shared_memory.get('target_semantic'), 'goal_eval': self.shared_memory.get('goal_eval', {}), 'task_graph': self.planner.task_graph, 'rounds': self.rounds, 'calls': self.events, **details}
        tmp = self.trajectory_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(self.trajectory_path)
        return result

    async def run(self, input_info):
        if self._has_run:
            raise RuntimeError('Create a new Workflow for each independent task')
        self._has_run = True
        input_info = self._validate_input(input_info)
        self.analyst._inbox.append(Message(content=json.dumps(input_info, ensure_ascii=False)))
        try:
            await self.analyst._plan_and_act()
            self._snapshot('running', input=input_info)
            while True:
                message = await self.planner._plan_and_act()
                if message is None:
                    break
                await self.worker._plan_and_act()
                self.rounds.append({'round': len(self.rounds) + 1, 'operation': self.shared_memory['current operation'], 'depends_on': self.shared_memory['current_dependance'], 'result': dict(self.shared_memory['current result']), 'errors': list(self.shared_memory.get('tool_errors', []))})
                logger.info('[Workflow] round=%d operation=%s depend=%s output=%s score=%s',
                            len(self.rounds), self.rounds[-1]['operation'], self.rounds[-1]['depends_on'],
                            self.rounds[-1]['result'].get('output_img_dir'), self.rounds[-1]['result'].get('final goal score'))
                self._snapshot('running', input=input_info)
            valid_nodes = [(name, node) for (name, node) in self.planner.task_graph.items() if Path(node['result'].get('output_img_dir') or '').is_file()]
            if not valid_nodes:
                errors = [error for item in self.rounds for error in item['errors']]
                raise WorkflowError(f'Every tool invocation failed; errors: {errors}')
            (best_name, best) = max(valid_nodes, key=lambda item: item[1]['result']['final goal score'])
            source = Path(best['result']['output_img_dir'])
            final_path = self.output_dir / ('final_output' + source.suffix.lower())
            if source.resolve() != final_path.resolve():
                shutil.copy2(source, final_path)
            result = self._snapshot('completed', input=input_info, final_output=str(final_path), best_round=best_name, max_score=best['result']['final goal score'], met_score_threshold=best['result']['final goal score'] >= self.planner.score_threshold, stop_reason='score_threshold' if self.rounds[-1]['result']['final goal score'] >= self.planner.score_threshold else 'max_rounds')
            logger.info('Final output: %s (round: %s, score: %.4f)', final_path, best_name, result['max_score'])
            return result
        except Exception as exc:
            self._snapshot('failed', input=input_info, error=f'{type(exc).__name__}: {exc}', error_category=error_category(exc))
            raise
