from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone
import inspect
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--prompt', help='Generation or editing task')
    source.add_argument('--input-json', type=Path, help='JSON file containing original {text:[], image:[]} input')
    parser.add_argument('--image', action='append', default=[], help='Local input image; repeat for multiple images')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--experience-dir', type=Path, default=PROJECT_ROOT / 'Experience')
    parser.add_argument('--llm-config', type=Path, default=PROJECT_ROOT / 'config/llm.yaml')
    parser.add_argument('--tool-config', type=Path, default=PROJECT_ROOT / 'ToolSets/config.flux_step1x.yaml')
    parser.add_argument('--worker-mode', choices=['direct', 'pasm'], default='direct')
    parser.add_argument('--prompt-profile', choices=['original', 'precise'], default='precise')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-rounds', type=int, default=5)
    parser.add_argument('--score-threshold', type=float, default=0.9)
    parser.add_argument('--planner-use-experience', action='store_true')
    parser.add_argument('--check-llm', action='store_true', help='Send an API request; with --image, also test actual image understanding')
    args = parser.parse_args(argv)
    if not args.check_llm and not (args.prompt or args.input_json):
        parser.error('Provide --prompt or --input-json, or use --check-llm')
    if args.input_json and args.image:
        parser.error('Place image paths inside --input-json instead of combining it with --image')
    return args


async def main(args):
    from utils.llm import MLLM
    llm = MLLM(config_path=args.llm_config)
    try:
        return await _run(args, llm)
    finally:
        llm.close()


async def _run(args, llm):
    if args.check_llm:
        content = [{'type': 'text', 'text': 'Reply with OK.'}]
        if args.image:
            content = [{'type': 'text', 'text': 'Describe the main objects and colors in this image.'}]
            content.extend({'type': 'image', 'image': path} for path in args.image)
        reply, _ = await llm.achat(content)
        print(reply)
        return

    from agents import Workflow
    from ToolSets.toolset_utils.toolset import ToolsetBase
    timestamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S-%f')
    output = (args.output_dir or PROJECT_ROOT / 'outputs' / timestamp).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(output / 'inference.log', encoding='utf-8')])
    if args.input_json:
        input_info = json.loads(args.input_json.read_text(encoding='utf-8'))
    else:
        input_info = {'text': [args.prompt], **({'image': args.image} if args.image else {})}
    tools = ToolsetBase(config_path=args.tool_config, device=args.device)
    try:
        workflow = Workflow(llm, tools, output_dir=output, experience_dir=args.experience_dir,
                            max_rounds=args.max_rounds, score_threshold=args.score_threshold,
                            planner_use_experience=args.planner_use_experience, worker_mode=args.worker_mode,
                            prompt_profile=args.prompt_profile)
        result = await workflow.run(input_info)
        print(json.dumps({key: result[key] for key in ('status', 'final_output', 'best_round', 'max_score', 'met_score_threshold')}, ensure_ascii=False, indent=2))
    finally:
        result = tools.close()
        if inspect.isawaitable(result):
            await result


if __name__ == '__main__':
    asyncio.run(main(parse_args()))
