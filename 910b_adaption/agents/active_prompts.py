from __future__ import annotations

import re
import textwrap
from .prompts import (
    ANALYST_EVAL_CONTEXT,
    ANALYST_TASK_SOURCE,
    EVALUATOR_INSTRUCTIONS,
    PLANNER_INITIAL_SOURCE,
    PLANNER_NEXT_SOURCE,
    WORKER_PREFERENCES_SOURCE,
)


def compact(text):
    text = re.sub(r"^[ ]+", "", textwrap.dedent(text), flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_output_format(text):
    text = compact(text)


    text = re.sub(r"={3,} The following is a reference example ={3,}.*?(?=\*\*Please pay attention)", "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^Example \d+:.*?(?=^Please provide your current round|^Please provide your answer|\Z)", "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^example:\n.*?</goal>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"Here are some sample references for your consideration:.*", "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^All the questions in the above questionnaire should be written in a single JSON file, like this:.*?(?=^If any item is not specified)", "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^.*(?:Output the task in the format:|Output the semantic in the format:|Please output in the xml format:|Present the output using the following format:|The output format is :|Output JSON data in XML format|specified format|Please provide your answer\.).*\n?", "", text)
    text = re.sub(r"(?m)^\s*<[^\n]*>[^\n]*\n?", "", text)
    text = text.replace('then output: <depend>None</depend>.', 'use None.')
    return compact(text)


ANALYST_TASK_PROMPT = clean_output_format(ANALYST_TASK_SOURCE)
ANALYST_GOALS_PROMPT = clean_output_format(ANALYST_EVAL_CONTEXT)


ANALYST_GOALS_PROMPT = ANALYST_GOALS_PROMPT.replace(
    'If not specified, write "null" for both "category" and "number".',
    'If not specified, use null for the entity category and quantity information.')
PLANNER_INITIAL_PROMPT = clean_output_format(PLANNER_INITIAL_SOURCE)
PLANNER_NEXT_PROMPT = clean_output_format(PLANNER_NEXT_SOURCE)
WORKER_PREFERENCES_PROMPT = clean_output_format(WORKER_PREFERENCES_SOURCE)

WORKER_ROUTE_PROMPT = WORKER_PREFERENCES_PROMPT.split('\n2.', 1)[0] + (
    '\n\nGenerate a new image with FLUX, or edit the supplied dependency image with '
    'Step1X-Edit. Editing requires an existing input image; without one, use generation.')


def tool_conditions_prompt(task_info, tool_name, description, experience):


    return compact(f"Task : {task_info}\n\nYou have selected {tool_name} to complete the task.\n"
                   f"Tool destription: {description}\n\n"
                   f"The following is the most relevant experience retrieved that used {tool_name} "
                   f"to solve a similar task, for reference only:\n{experience}\n\n"
                   "The given tool conditions should use concise text and avoid excessive complexity.")



PRECISION_RULES = {
    'analyst_task': """Fill task with the complete requested operation and semantic with the
complete desired final scene. Both must retain exact object counts, shapes,
colors, positions, sizes, layout, visibility, exclusions and edit-preservation
requirements. A generic summary is insufficient. Do not invent details.""",
    'analyst_goals': """Extract the request's evaluable goals. category_number: one record per
distinct object category, with its exact count; never merge different categories
as 'objects'. 0 means absent, -1 unspecified. identity: null unless a specific
reference image's identity/face/pose must be preserved. position: complete layout,
spacing and visibility statements. attribute-binding: complete color, shape,
size and preservation statements. Include explicit style/background; otherwise null.""",
    'planner': """Plan the next executable visual subtask. operation is the full instruction
the image tool will execute; include the objects, exact changes and constraints.
Example operation: 'Turn the car blue while preserving the street and other cars.'
Set depend to a useful existing round ID, or 'None' to start over.
Use observed objects to identify the actual mismatch; feedback may miscount.
Specify the correction and desired final state, preserving correct content.
Avoid repeating failed instructions.""",
    'generation': """Describe one complete final scene for FLUX: exact object counts,
shapes, colors, sizes, layout, background and exclusions from the request.
For rows, specify individually visible objects with gaps, no stacking or overlap.
Keep concrete visual details; omit commentary and decorative additions.""",
    'editing': """Write one focused Step1X instruction: identify the object and location,
state the exact change and final count/placement, then preserve unaffected objects,
colors, geometry, camera and background. Make relative size explicit; 'small'
means clearly smaller than the main objects. Do not describe an unrelated new scene.""",
    'evaluator': """Score AFTER against the full request and goals; BEFORE is only for
preservation. Cross-check the independent observation against the image. Count
visible instances by shape AND color, never by the requested count. Check layout,
size, overlaps and preserved content. Use 100=exact, 80=minor mismatch, 60=partial,
40=major mismatch, 20=poor, 0=absent. Any unmet explicit constraint limits semantic
to 80. State observed/required counts and concrete errors; do not call a mismatch
perfect. Give one specific correction. Use at most 20 words per explanation.""",
}


def precise_prompt(stage, user_request, observation=None, **context):
    import json
    text = compact(PRECISION_RULES[stage]) + '\nUser request:\n' + user_request
    data = dict(context)
    if observation:
        data['observation'] = observation
    if data:
        text += '\nContext:\n' + json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    return text


TOKEN_BUDGETS = {'analyst_task': 512, 'analyst_goals': 1024, 'planner_step': 384,
                 'worker_route': 64, 'read_image': 768, 'evaluator_scores': 1024,
                 'function_call': 512}


BLIND_IMAGE_PROMPT = """For each visible foreground group, report shape/category, color/material,
count, and brief position/relative size. Count visible instances once; exclude
reflections and hidden objects. Summarize layout, overlaps and background in one
sentence; note uncertainty. No target scene is supplied."""


def clean_evaluator_format(text):
    text = re.sub(r"(?m)^.*Please provide your reasoning process first within.*\n?", "", text)
    text = re.sub(r"(?m)^.*Output format:.*\n?", "", text)
    text = re.sub(r"(?m)^\s*<[^\n]*>[^\n]*\n?", "", text)
    text = text.replace('all the above <evaluation_xxx> results', 'all the above evaluation results')
    return compact(text)


def evaluation_prompt(task, goal_eval, original_semantic=''):
    tar_semantic = goal_eval.get('semantic', '')
    if original_semantic:
        evaluation_reasoning = f'\n            task: {task}\n            original/reference image Semantic: {original_semantic}\n            target image semantic: {tar_semantic}\n\n            The following is the evaluation and analysis result after executing the operation:\n\n            '
    else:
        evaluation_reasoning = f'\n            task: {task}\n            target image semantic: {tar_semantic}\n\n            The following is the evaluation and analysis result after executing the operation:\n\n            '
    if 'category_number' in goal_eval:
        evaluation_reasoning += f"\n            [Category Number]\n            Please analyze whether the following objects exist in the given image, and if they do, whether the number of objects matches the given target: {goal_eval['category_number']}\n            Objects with a quantity of **-1** are those with an unspecified number; their exact quantity does not need to be determined, only their presence is required.  \n            If the quantity of the target object is **0**, then the object should not be present in the image.\n\n            Give a score from 0 to 100 based on the degree of consistency between the image and the object quantity requirements:  \n            100: All specified objects are present or absent exactly as required, and all quantities match perfectly.  \n            80: Most object quantities match, with only minor inconsistencies (e.g., one object missing or extra) that do not significantly affect the overall semantic accuracy.  \n            60: Some object quantities or presence conditions are incorrect, impacting the overall alignment with the task.  \n            40: Many object quantities are incorrect or missing, leading to a notable deviation from the task requirements.  \n            20: The image fails to meet the object presence or quantity requirements in most respects, making the content largely irrelevant to the task.\n\n            If there are unreasonable objects in the image, please indicate the location of the object that is unreasonable.\n\n            Provide your **score** and **explanation** within 20 words:\n            "
    if 'position' in goal_eval:
        pos_list = '\n'.join((f'- {o_p}' for o_p in goal_eval['position']))
        evaluation_reasoning += f'\n            [Position]\n            Please analyze if the positional relationships between the objects conform to the following criteria:\n            {pos_list}\n\n            Give a score from 0 to 100 based on the degree of consistency in spatial relationships:  \n            100: All objects are positioned exactly as required, with clear and precise spatial arrangements that fully match the target criteria.  \n            80: Most positional relationships are correct, with only minor deviations that do not significantly affect the overall scene coherence.  \n            60: Some objects are misplaced or relationships are inconsistent, affecting the clarity or accuracy of the scene.  \n            40: Many positional relationships are incorrect or violated, reducing the overall alignment with the task.  \n            20: The image fails to reflect the required spatial arrangements; object positions are largely irrelevant to the task.\n\n            Provide score and explanation within 20 words:\n            '
    if 'attribute-binding' in goal_eval:
        attr_list = '\n'.join((f'- {o_a}' for o_a in goal_eval['attribute-binding']))
        evaluation_reasoning += f'\n            [Attribute Binding]\n            Please analyze if the attribute bindings of the objects conform to the following requirements:\n            {attr_list}\n\n            Give a score from 0 to 100 based on the degree of attribute binding consistency:  \n            100: All object attributes (e.g., color, texture, shape, material) perfectly match the specified requirements, with no discrepancies.  \n            80: Most object attributes are correct, with only minor deviations that do not significantly affect the overall semantic accuracy.  \n            60: Some object attributes are incorrect or inconsistent, affecting the clarity or completeness of the scene.  \n            40: Many object attributes are incorrect or misassigned, reducing the overall alignment with the task requirements.  \n            20: The image fails to reflect the required object attributes; most or all attribute bindings are incorrect or missing.\n\n            Provide score and explanation within 20 words:\n            '
    if 'style' in goal_eval:
        evaluation_reasoning += f"\n            [Style]\n            Please analyze whether the style of the image is: {goal_eval['style']}\n\n            Give a score from 0 to 100 based on the degree of style consistency with the target description:  \n            100: The image perfectly matches the target style, including overall visual appearance, texture, color palette, and artistic or photographic effects.  \n            80: The image mostly matches the target style, with only minor deviations that do not significantly affect the perception of style.  \n            60: The image partially matches the target style, but noticeable inconsistencies in visual elements, color, or texture reduce stylistic accuracy.  \n            40: The image has significant style deviations from the target, making it difficult to recognize the intended artistic or visual style.  \n            20: The image fails to reflect the intended style; overall appearance and visual characteristics are completely inconsistent with the target.\n\n            Provide score and explanation within 20 words:\n            "
    if 'background' in goal_eval:
        evaluation_reasoning += f"\n            [Background]\n            Please analyze whether the scene background is: {goal_eval['background']}\n\n            Give a score from 0 to 100 based on the degree of background consistency with the target description:  \n            100: The background perfectly matches the target scene, including environment, context, and relevant details, fully supporting the task semantics.  \n            80: The background mostly matches the target scene, with only minor inconsistencies that do not significantly affect overall scene understanding.  \n            60: The background partially matches the target scene, but some key contextual or environmental elements are missing or incorrect.  \n            40: The background has significant deviations from the target, making the scene contextually inconsistent or confusing.  \n            20: The background completely fails to match the target scene, providing irrelevant or misleading context.\n\n            Provide score and explanation within 20 words:\n            "
    if 'semantic' in goal_eval:
        evaluation_reasoning += f"\n            [Semantic]\n            Please analyze whether the overall image semantics align with the given semantics: {goal_eval['semantic']}\n\n            Give a score from 0 to 100 based on the degree of semantic consistency with the given semantics:  \n            100: The image semantics perfectly match the given semantics, fully covering all key objects, attributes, and relationships.  \n            80: The image semantics mostly match the given semantics, with only minor deviations not affecting overall understanding.  \n            60: The image semantics partially match the given semantics, with noticeable omissions or inaccuracies in key elements.  \n            40: The image semantics have significant mismatches with the given semantics, leading to semantic confusion.  \n            20: The image semantics completely fail to match the given semantics, missing core meaning and context.\n\n            Provide score and explanation within 20 words:\n            "
    evaluation_reasoning += clean_evaluator_format(EVALUATOR_INSTRUCTIONS)
    evaluation_reasoning = textwrap.dedent(evaluation_reasoning)
    evaluation_reasoning = re.sub('^[ ]+', '', evaluation_reasoning, flags=re.MULTILINE)
    return compact(evaluation_reasoning)
