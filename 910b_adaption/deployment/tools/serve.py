#!/usr/bin/env python3
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    from ToolSets.toolset_utils.resident import ResidentTool, start, status, stop
    from ToolSets.toolset_utils.toolset import ToolsetBase
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['start', 'status', 'stop'])
    parser.add_argument('--config', type=Path, default=ROOT / 'ToolSets/config.flux_step1x.yaml')
    parser.add_argument('--tool', action='append')
    parser.add_argument('--wait', action='store_true', help='Wait for weight loading to finish')
    args = parser.parse_args()
    with ToolsetBase(config_path=args.config) as registry:
        names = args.tool or list(registry.tool_configs)
        def run(name):
            info, device = registry.get_info(name), registry.device_for(name)
            if args.action == 'status':
                return status(name)
            if args.action == 'stop':
                return stop(name)
            if args.wait:
                ResidentTool(name, info, device).server_load_model()
            else:
                start(name, info, device)
            return status(name)
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            results = list(pool.map(run, names))
    print(json.dumps({'tools': results}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
