from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


UPSTREAM_COMMIT = "e2261a2fc10dfe1ca5e54496e4319d588576e701"
PATCH_VERSION = 1
SOURCE_HASHES = {
    "inference.py": "1419e3f6f4aefd229c08d65559c72ca40a67ea430a06f8a7dadd2e665abcf740",
    "sampling.py": "179b8d59cbff1509218c5520771e38d825fc2e0c2d10289a183d4808cba5798f",
    "modules/autoencoder.py": "684c164a6d1d59043923793ae79854f218111543eca3924ceedd9cf3d7c8b2bc",
    "modules/conditioner.py": "302876cd4d67fc048a81e757fa39908b314cf9c05972e535f37eb1e4dc5b0e24",
    "modules/model_edit.py": "2bcb1e6389b52620980aebf6b41119e8781142476fa88eafb94bd72ef27b6ed1",
    "modules/connector_edit.py": "b1586bed9e68f98b9f6baadf1a15d3906e547f150b090df9e5e903aeaaebf0af",
    "modules/layers.py": "88fd23a7a58d7acc39b41ce3e0edb612997bf5e0f27e72c740746cb555ed7ac8",
    "modules/attention.py": "68aacd3b1cb7792b8dfad208e74dd2f4d0819f4eba7f74808039e7571163f17f",
    "library/custom_offloading_utils.py": "b4adebbf61973d57163650741e088677f345d03de3fa31f5052fddd0fad88a65",
    "library/device_utils.py": "640332a5aa2b72dc1e9e641d4e7f11da1295ef45c6682cf5408ce96fb635abb4",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _replace(source: str, old: str, new: str, count: int = 1) -> str:
    observed = source.count(old)
    if observed != count:
        raise ValueError(f"Step1X compatibility patch expected {count} occurrences, found {observed}: {old[:100]!r}")
    return source.replace(old, new)


def _drop_nodes(source: str, predicate) -> str:
    lines = source.splitlines(keepends=True)
    for node in reversed(ast.parse(source).body):
        if predicate(node):
            del lines[node.lineno - 1 : node.end_lineno]
    return "".join(lines)


def _inference(source: str) -> str:
    source = _drop_nodes(
        source,
        lambda node: (
            isinstance(node, ast.ImportFrom)
            and node.module in {"modules.multigpu", "xfuser.core.distributed"}
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name in {"cfg_usp_level_setting", "teacache_init", "main"}
        )
        or (isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"),
    )
    source = _replace(
        source,
        'def cudagc():\n    torch.cuda.empty_cache()\n    torch.cuda.ipc_collect()',
        'def cudagc(device):\n'
        '    device = torch.device(device)\n'
        '    if device.type == "cuda":\n'
        '        torch.cuda.empty_cache()\n'
        '        torch.cuda.ipc_collect()\n'
        '    elif device.type == "npu":\n'
        '        torch.npu.empty_cache()',
    )
    source = source.replace("cudagc()", "cudagc(self.device)")
    source = _replace(
        source,
        '        if os.getenv("TORCHELASTIC_RUN_ID") is not None:\n'
        '            local_rank = get_world_group().local_rank\n'
        '            torch.cuda.set_device(local_rank)\n'
        '            self.device = torch.device(f"cuda:{local_rank}")\n'
        '        else:\n'
        '            self.device = torch.device(device)',
        '        if os.getenv("TORCHELASTIC_RUN_ID") is not None:\n'
        '            raise RuntimeError("This Step1X adapter supports single-process inference only; unset TORCHELASTIC_RUN_ID.")\n'
        '        self.device = torch.device(device)',
    )
    source = _replace(
        source,
        'torch.cuda.amp.autocast(dtype=torch.bfloat16)',
        'torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)',
    )


    source = source.replace('mode="flash"', 'mode="torch"')
    return source


def _conditioner(source: str) -> str:

    source = _drop_nodes(
        source,
        lambda node: isinstance(node, ast.ClassDef) and node.name == "Qwen25VL_7b_Embedder_backup",
    )
    source = source.replace("torch.cuda.current_device()", "self.device")
    source = source.replace('.to("cuda")', '.to(self.device)')
    return source


def _layers(source: str) -> str:
    source = _replace(source, "from liger_kernel.ops.rms_norm import LigerRMSNormFunction\n", "")
    source = _replace(
        source,
        '        return LigerRMSNormFunction.apply(\n'
        '            x,\n'
        '            weight,\n'
        '            eps,\n'
        '            0.0,\n'
        '            "gemma",\n'
        '            True,\n'
        '        )',
        '        # Liger v0.5.4: casting_mode="gemma", offset=0.0.\n'
        '        # Compute variance, normalization and weight product in FP32,\n'
        '        # then cast the final result back to the input dtype.\n'
        '        value = x.float()\n'
        '        rrms = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)\n'
        '        return (value * rrms * weight.float()).to(x.dtype)',
    )
    source = _replace(
        source,
        '@torch.compile(mode="max-autotune-no-cudagraphs", dynamic=True)\n',
        "",
        count=3,
    )
    return source


def _attention(source: str) -> str:
    return _replace(
        source,
        "from xfuser.model_executor.layers.usp import USP\n",
        "# xFuser USP is only needed by the unsupported distributed xdit mode.\n",
    )


def prepare_source(source_path, device, cache_path=None) -> str:
    source_path = Path(source_path).expanduser().resolve()
    kind = str(device).split(":", 1)[0]
    if kind not in {"npu", "cuda", "cpu"}:
        raise ValueError(f"Explicit cpu/npu/cuda device required, got {device!r}")
    originals = {}
    for filename, expected in SOURCE_HASHES.items():
        path = source_path / filename
        data = path.read_bytes()
        observed = _sha(data)
        if observed != expected:
            raise ValueError(
                f"Step1X upstream source changed: {path}; expected sha256={expected}, got {observed}. "
                f"Restore commit {UPSTREAM_COMMIT} before applying compatibility patches."
            )
        originals[filename] = data.decode("utf-8")

    signature = _sha(json.dumps({"version": PATCH_VERSION, "hashes": SOURCE_HASHES}, sort_keys=True).encode())[:16]
    target = (
        Path(cache_path).expanduser().resolve()
        if cache_path is not None
        else source_path.parent.parent / "deployment" / "prepared" / f"step1x-v1p1-{signature}"
    )
    if target == source_path or source_path in target.parents:
        raise ValueError("Prepared Step1X source must be outside the original checkout")
    transforms = {
        "inference.py": _inference,
        "modules/conditioner.py": _conditioner,
        "modules/layers.py": _layers,
        "modules/attention.py": _attention,
    }
    prepared = {
        filename: transforms.get(filename, lambda text: text)(text)
        for filename, text in originals.items()
    }
    for filename, text in prepared.items():
        compile(text, filename, "exec")
    hashes = {filename: _sha(text.encode()) for filename, text in prepared.items()}
    manifest = {
        "upstream_repository": "https://github.com/stepfun-ai/Step1X-Edit",
        "upstream_commit": UPSTREAM_COMMIT,
        "model_version": "v1.1",
        "patch_version": PATCH_VERSION,
        "original_sha256": SOURCE_HASHES,
        "prepared_sha256": hashes,
        "supported": "single-process, mode=torch, no quantization, no LoRA, no block swapping",
        "changes": [
            "Drop unused distributed xFuser imports, CLI and TeaCache helpers",
            "Select encoder tensors and AMP/cache operations using the requested device",
            "Use existing upstream torch SDPA attention mode",
            "Replace Triton Liger RMSNorm with the same gemma/zero-offset FP32 equation",
            "Run three elementwise helper functions eagerly instead of CUDA torch.compile",
            "Drop unused backup conditioning class",
        ],
    }
    if (target / "compat_manifest.json").is_file():
        try:
            actual = json.loads((target / "compat_manifest.json").read_text())
            if actual == manifest and all(
                (target / filename).is_file() and _sha((target / filename).read_bytes()) == expected
                for filename, expected in hashes.items()
            ):
                return str(target)
        except (OSError, ValueError):
            pass
        raise ValueError(f"Prepared Step1X source was modified: {target}; remove this cache directory explicitly to regenerate it.")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing Step1X preparation directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        patches = []
        for filename, text in prepared.items():
            path = temporary / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            patches.extend(difflib.unified_diff(
                originals[filename].splitlines(keepends=True),
                text.splitlines(keepends=True),
                fromfile=f"upstream/{filename}",
                tofile=f"prepared/{filename}",
            ))
        for package in ("modules", "library"):
            (temporary / package / "__init__.py").write_text("", encoding="utf-8")
        if (source_path / "LICENSE").is_file():
            shutil.copyfile(source_path / "LICENSE", temporary / "LICENSE")
        (temporary / "compat.patch").write_text("".join(patches), encoding="utf-8")
        (temporary / "compat_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return str(target)
