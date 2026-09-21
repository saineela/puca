"""Merge Casper's PEFT adapter into the matching HF base model.

The merge is deliberately shard-wise: it never loads the 4B base or the
merged output as one in-memory object. The result is a normal indexed
Safetensors directory that Ollama can import and quantize.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(os.environ.get("CASPER_BASE", ROOT / "models" / "qwen3.5-4b-hf"))
ADAPTER = Path(os.environ.get(
    "CASPER_ADAPTER",
    ROOT / "models" / "nixlm" / "casper-puca-qlora-v5",
))
OUT = Path(os.environ.get(
    "CASPER_MERGED_OUT",
    ROOT / "models" / "qwen35-casper-puca-v5-merged",
))
SHARD_BYTES = int(os.environ.get("CASPER_MERGE_SHARD_BYTES", str(350 * 1024**2)))


def _adapter_tensors() -> dict[str, torch.Tensor]:
    path = ADAPTER / "adapter_model.safetensors"
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt", device="cpu") as source:
        for key in source.keys():
            tensors[key] = source.get_tensor(key).float()
    return tensors


def _delta_for(base_key: str, adapters: dict[str, torch.Tensor]) -> torch.Tensor | None:
    # HF's PEFT wrapper names the same language layers as
    # base_model.model.model.layers.N; the base checkpoint prefixes them with
    # model.language_model.layers.N.
    if not base_key.startswith("model.language_model.layers."):
        return None
    suffix = base_key[len("model.language_model."):]
    # The base key ends in ``.weight`` while PEFT inserts the LoRA names
    # before that suffix: ``...q_proj.lora_A.weight``.
    module = suffix.removesuffix(".weight")
    prefix = "base_model.model.model." + module
    a_key = prefix + ".lora_A.weight"
    b_key = prefix + ".lora_B.weight"
    if a_key not in adapters or b_key not in adapters:
        return None
    config = json.loads((ADAPTER / "adapter_config.json").read_text())
    scale = float(config.get("lora_alpha", 1)) / float(config.get("r", 1))
    return (adapters[b_key] @ adapters[a_key] * scale)


def main() -> None:
    index_path = BASE / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    source_shards = sorted(set(weight_map.values()))
    if OUT.exists() and any(OUT.glob("model-*.safetensors")):
        raise RuntimeError(f"Refusing to overwrite existing merged output: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    for source_name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "merges.txt", "vocab.json"):
        source = BASE / source_name
        if source.exists():
            shutil.copy2(source, OUT / source_name)

    adapters = _adapter_tensors()
    merged_map: dict[str, str] = {}
    output_shards: list[str] = []
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        name = f"model-{len(output_shards) + 1:05d}-of-PLACEHOLDER.safetensors"
        output_shards.append(name)
        save_file(pending, str(OUT / name), metadata={"format": "pt"})
        for key in pending:
            merged_map[key] = name
        print(f"wrote {name}: {pending_bytes / 1024**2:.1f} MiB", flush=True)
        pending = {}
        pending_bytes = 0

    for source_name in source_shards:
        with safe_open(str(BASE / source_name), framework="pt", device="cpu") as source:
            for key in source.keys():
                tensor = source.get_tensor(key)
                delta = _delta_for(key, adapters)
                if delta is not None:
                    if tuple(delta.shape) != tuple(tensor.shape):
                        raise RuntimeError(
                            f"shape mismatch for {key}: base={tuple(tensor.shape)} delta={tuple(delta.shape)}"
                        )
                    tensor = tensor.float().add(delta).to(tensor.dtype)
                    print(f"merged {key}", flush=True)
                pending[key] = tensor
                pending_bytes += tensor.numel() * tensor.element_size()
                if pending_bytes >= SHARD_BYTES:
                    flush()
    flush()

    total = len(output_shards)
    renamed: list[str] = []
    for old in output_shards:
        new = old.replace("PLACEHOLDER", f"{total:05d}")
        (OUT / old).rename(OUT / new)
        renamed.append(new)
    for key, old in list(merged_map.items()):
        merged_map[key] = old.replace("PLACEHOLDER", f"{total:05d}")

    (OUT / "model.safetensors.index.json").write_text(json.dumps({
        "metadata": {"total_size": sum(
            (OUT / name).stat().st_size for name in renamed
        )},
        "weight_map": merged_map,
    }, indent=2))
    (OUT / "casper_merge.json").write_text(json.dumps({
        "base": str(BASE),
        "adapter": str(ADAPTER),
        "lora_alpha": json.loads((ADAPTER / "adapter_config.json").read_text()).get("lora_alpha"),
        "rank": json.loads((ADAPTER / "adapter_config.json").read_text()).get("r"),
        "shards": total,
        "shard_bytes": SHARD_BYTES,
    }, indent=2))
    print(f"merged Casper model written to {OUT} ({total} shards)")


if __name__ == "__main__":
    main()
