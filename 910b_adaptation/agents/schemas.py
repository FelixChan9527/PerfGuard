from __future__ import annotations
import math
import re

from jsonschema import Draft202012Validator


class AgentProtocolError(ValueError):
    pass

def validate(value, schema):
    def finite(item):
        if isinstance(item, float) and not math.isfinite(item):
            raise AgentProtocolError('Structured response contains a nonfinite number')
        if isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    error = next(Draft202012Validator(schema).iter_errors(value), None)
    if error:
        location = '.'.join(map(str,error.absolute_path)) or '<root>'
        raise AgentProtocolError(f'Response schema violation at {location}: {error.message}')
    return value

GENERATION_CATEGORY = 'text2image-generation tool'
EDITING_CATEGORY = 'image-editing tool'
GEN_DIMS = ('color', 'shape', 'texture', '2D-spatial', '3D-spatial', 'numeracy', 'non-spatial')
EDIT_DIMS = ('addition', 'removement', 'replacement', 'attribute-alter', 'motion-change', 'style-transfer', 'background-change')
DIMENSIONS = {
    'category_number': 'object category & number',
    'position': 'positional relationships',
    'attribute-binding': 'attribute binding',
    'style': 'style consistency',
    'background': 'background match',
    'semantic': 'semantic',
}


def object_schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def nullable(schema):
    return {'anyOf': [schema, {'type': 'null'}]}


def string(description=None):



    result = {'type': 'string', 'minLength': 1, 'pattern': r'[\s\S]*\S[\s\S]*'}
    if description:
        result['description'] = description
    return result


READ_IMAGE = object_schema({'semantic': string('Concise description of the input image.')})
IMAGE_OBSERVATION = object_schema({
    'objects': {'type': 'array', 'items': object_schema({
        'category': string('Object category or shape, e.g. cube; without color words.'),
        'attributes': string('Color and material, e.g. red matte; separate differently colored groups.'),
        'count': {'type': 'integer', 'minimum': 1},
        'location': string('Visible position, arrangement and relative size.'),
    })},
    'semantic': string('Description grounded in visible objects; include uncertainty if necessary.'),
})
ANALYST_TASK = object_schema({'task': string(), 'semantic': string()})
ANALYST_GOALS = object_schema({
    'category_number': nullable({'type': 'array', 'items': object_schema({
        'category': string('Entity category without attribute words.'),
        'number': {'type': 'integer', 'minimum': -1,
                   'description': 'Required entity count; -1 means unspecified quantity, 0 means absent.'}})}),
    'identity': nullable({'type': 'array', 'items': object_schema({
        'category': string(), 'image': string(),
        'type': {'type': 'string', 'enum': ['face', 'pose', 'identity']}})}),
    'position': nullable({'type': 'array', 'items': string()}),
    'attribute-binding': nullable({'type': 'array', 'items': string()}),
    'style': nullable(string()),
    'background': nullable(string()),
})


def planner_step(dependencies):
    return object_schema({
        'operation': string('Complete visual instruction naming the objects, requested changes, and content to preserve.'),
        'depend': {'type': 'string', 'enum': ['None', *dependencies]},
    })


def validate_planner_step(response):
    operation = response['operation'].strip()
    if operation.lower() in {'none', 'null', 'n/a'}:
        raise AgentProtocolError('operation must describe a concrete generation or editing action; only depend may be None')
    if re.fullmatch(r'[A-Za-z0-9_:-]+', operation):
        raise AgentProtocolError('operation must be a complete visual instruction naming objects and exact changes; an action label is insufficient')


def _preferences(dimensions):
    return object_schema({key: {'type': 'number', 'minimum': 0, 'maximum': 1} for key in dimensions})


WORKER_ROUTE = object_schema({
    'category': {'type': 'string', 'enum': [GENERATION_CATEGORY, EDITING_CATEGORY]},
})


def worker_route(*, has_input_image):
    if has_input_image:
        return WORKER_ROUTE
    return object_schema({'category': {'type': 'string', 'enum': [GENERATION_CATEGORY]}})

WORKER_PREFERENCES = object_schema({
    'category': {'type': 'string', 'enum': [GENERATION_CATEGORY, EDITING_CATEGORY]},
    'preference': {'anyOf': [_preferences(GEN_DIMS), _preferences(EDIT_DIMS)]},
})


def evaluator_scores(goal_eval):
    return object_schema({
        'dimensions': object_schema({
            key: object_schema({'score': {'type': 'number', 'minimum': 0, 'maximum': 100},
                                'explanation': string()})
            for key in DIMENSIONS if key in goal_eval
        }),
        'final_evaluation': string('Overall evaluation highlighting errors and missing elements.'),
    })


def original_goals(response):
    result = dict(response)
    if result['category_number'] is not None:
        items = result['category_number']
        categories = [item['category'] for item in items]
        if len(set(categories)) != len(categories):
            raise AgentProtocolError('Analyst returned duplicate entity categories')
        result['category_number'] = {item['category']: item['number'] for item in items}
    if result['identity'] is not None:
        result['identity'] = [{item['category']: item['image'], 'type': item['type']}
                              for item in result['identity']]
    return result
