#!/usr/bin/env python3
"""Verify the ZGCM contract and mixed F16/BF16 payload against a local HF snapshot."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gguf-py"))
import gguf


PATTERN = [
    True, True, True, True, True, False,
    True, True, True, True, True, False,
    True, True, True, True, True, False,
    True, True, True, True, True, False,
    True, True, True, True, True, False,
    True, True,
]
ARRAYS = {
    "tokenizer.ggml.tokens": ("STRING", 155136, "7dc430e0f40a2172e2031fb34093ddac55853d39f913b0bf49662b56441f61f2"),
    "tokenizer.ggml.token_type": ("INT32", 155136, "d29e0d8c61f7ac437e77147e9ffb8d7e37b2babe06a0f0300abc55886c27afa4"),
    "tokenizer.ggml.merges": ("STRING", 321649, "00a3309332f4906c13e63d35d6292a445ae52bdb207fa7944de227d704f64c08"),
}
TEMPLATE_SHA = "df60c73492112e649ed643aef30520dcd953ee435dee612ee5b1d69277a3d2b5"
BF16_TENSORS = {f"blk.{i}.ffn_down.weight" for i in range(32)}


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
    print(f"PASS {message}", flush=True)


def inventory() -> dict[str, tuple[str, tuple[int, ...]]]:
    result = {
        "output_norm.weight": ("model.norm.weight", (4096,)),
        "token_embd.weight": ("model.embed_tokens.weight", (155136, 4096)),
        "output.weight": ("lm_head.weight", (155136, 4096)),
    }
    layers = {
        "attn_norm": ("post_attention_layernorm", (4096,)),
        "ffn_norm": ("post_feedforward_layernorm", (4096,)),
        "attn_q_norm": ("self_attn.q_norm", (128,)),
        "attn_k_norm": ("self_attn.k_norm", (128,)),
        "attn_q": ("self_attn.q_proj", (4096, 4096)),
        "attn_k": ("self_attn.k_proj", (1024, 4096)),
        "attn_v": ("self_attn.v_proj", (1024, 4096)),
        "attn_output": ("self_attn.o_proj", (4096, 4096)),
        "ffn_gate": ("mlp.gate_proj", (11008, 4096)),
        "ffn_up": ("mlp.up_proj", (11008, 4096)),
        "ffn_down": ("mlp.down_proj", (4096, 11008)),
    }
    for i, local in enumerate(PATTERN):
        for target, (source, shape) in layers.items():
            result[f"blk.{i}.{target}.weight"] = (f"model.layers.{i}.{source}.weight", shape)
        if local:
            result[f"blk.{i}.attn_gate.weight"] = (f"model.layers.{i}.self_attn.g_proj.weight", (4096, 4096))
    return result


def metadata(reader: gguf.GGUFReader, snapshot: Path) -> None:
    check(reader.endianess == gguf.GGUFEndian.LITTLE, "header little-endian")
    check(reader.fields["GGUF.version"].contents() == 3, "header GGUF version 3")
    check(reader.alignment == 32, "header tensor alignment 32")
    scalars = {
        "general.architecture": ("STRING", "zgcm"),
        "general.type": ("STRING", "model"),
        "general.name": ("STRING", "ZGCM-1-7B"),
        "general.file_type": ("UINT32", 1),
        "general.quantization_version": ("UINT32", 2),
        "zgcm.block_count": ("UINT32", 32),
        "zgcm.context_length": ("UINT32", 262144),
        "zgcm.embedding_length": ("UINT32", 4096),
        "zgcm.feed_forward_length": ("UINT32", 11008),
        "zgcm.vocab_size": ("UINT32", 155136),
        "zgcm.attention.causal": ("BOOL", True),
        "zgcm.attention.head_count": ("UINT32", 32),
        "zgcm.attention.head_count_kv": ("UINT32", 8),
        "zgcm.attention.key_length": ("UINT32", 128),
        "zgcm.attention.value_length": ("UINT32", 128),
        "zgcm.attention.layer_norm_rms_epsilon": ("FLOAT32", struct.unpack("<f", bytes.fromhex("bd378635"))[0]),
        "zgcm.attention.sliding_window": ("UINT32", 128),
        "zgcm.rope.freq_base": ("FLOAT32", 10000000.0),
        "zgcm.rope.dimension_count": ("UINT32", 42),
        "tokenizer.ggml.model": ("STRING", "gpt2"),
        "tokenizer.ggml.pre": ("STRING", "zgcm"),
        "tokenizer.ggml.add_bos_token": ("BOOL", False),
        "tokenizer.ggml.add_eos_token": ("BOOL", False),
        "tokenizer.ggml.eos_token_id": ("UINT32", 154820),
        "tokenizer.ggml.padding_token_id": ("UINT32", 154820),
    }
    for key, (kind, expected) in scalars.items():
        field = reader.fields.get(key)
        check(field is not None, f"KV {key} present")
        check(field.types == [gguf.GGUFValueType[kind]], f"KV {key} type {kind}")
        check(field.contents() == expected, f"KV {key} value {expected!r}; actual {field.contents()!r}")
    key = "zgcm.attention.sliding_window_pattern"
    field = reader.fields.get(key)
    check(field is not None, f"KV {key} present")
    check(field.types == [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.BOOL], f"KV {key} type ARRAY BOOL")
    pattern = field.contents()
    check(len(pattern) == 32, f"KV {key} length 32; actual {len(pattern)}")
    for i, expected in enumerate(PATTERN):
        check(pattern[i] == expected, f"KV {key}[{i}] expected {expected}; actual {pattern[i]}")
    globals_ = [i for i, local in enumerate(pattern) if not local]
    check(globals_ == [5, 11, 17, 23, 29], f"global layers [5, 11, 17, 23, 29]; actual {globals_}")
    for key, (kind, length, expected) in ARRAYS.items():
        field = reader.fields.get(key)
        check(field is not None, f"KV {key} present")
        check(field.types == [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType[kind]], f"KV {key} type ARRAY {kind}")
        value = field.contents()
        check(len(value) == length, f"KV {key} length {length}; actual {len(value)}")
        digest = hashlib.sha256(json.dumps(value, ensure_ascii=True, separators=(",", ":")).encode("ascii")).hexdigest()
        check(digest == expected, f"KV {key} SHA256 {expected}; actual {digest}")
    key = "tokenizer.chat_template"
    field = reader.fields.get(key)
    check(field is not None, f"KV {key} present")
    check(field.types == [gguf.GGUFValueType.STRING], f"KV {key} type STRING")
    value = field.contents().encode("utf-8")
    check(value == (snapshot / "chat_template.jinja").read_bytes(), f"KV {key} exact snapshot bytes")
    check(hashlib.sha256(value).hexdigest() == TEMPLATE_SHA, f"KV {key} SHA256 {TEMPLATE_SHA}")
    absent = [
        "zgcm.rope.scaling.*", "zgcm.rope.freq_base_swa", "zgcm.rope.dimension_count_swa", "zgcm.rope.dimension_sections",
        "zgcm.attention.rope_pattern", "zgcm.attention.gate*", "zgcm.*activation*", "zgcm.*tie*",
        "tokenizer.ggml.bos_token_id", "tokenizer.ggml.unknown_token_id", "tokenizer.ggml.seperator_token_id", "tokenizer.ggml.separator_token_id",
        "tokenizer.ggml.mask_token_id", "tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id",
        "tokenizer.ggml.fim_*", "tokenizer.ggml.prefix_token_id", "tokenizer.ggml.suffix_token_id", "tokenizer.ggml.middle_token_id",
        "tokenizer.ggml.scores", "tokenizer.ggml.token_type_count", "tokenizer.ggml.add_sep_token",
        "tokenizer.ggml.add_space_prefix", "tokenizer.ggml.remove_extra_whitespaces", "tokenizer.ggml.normalizer*",
        "tokenizer.ggml.precompiled_charsmap", "tokenizer.ggml.*suppress*", "tokenizer.huggingface.json",
        "tokenizer.chat_templates", "tokenizer.chat_template.*", "tokenizer.ggml.ignore_merges",
    ]
    for pattern in absent:
        matches = sorted(k for k in reader.fields if fnmatch.fnmatchcase(k, pattern))
        check(not matches, f"absent {pattern}; matches {matches}")
    canonical = set(scalars) | set(ARRAYS) | {"zgcm.attention.sliding_window_pattern", "tokenizer.chat_template"}
    descriptive = {"general.basename", "general.size_label", "general.license", "general.tags", "general.languages"}
    actual = set(reader.fields) - {"GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"}
    extras = actual - canonical - descriptive
    check(not extras, f"no extra inference-affecting or unreviewed KV keys; extras {sorted(extras)}")
    for key in sorted(actual & descriptive):
        expected_types = [gguf.GGUFValueType.ARRAY, gguf.GGUFValueType.STRING] if key in ("general.tags", "general.languages") else [gguf.GGUFValueType.STRING]
        check(reader.fields[key].types == expected_types, f"descriptive-only KV {key} valid type")


def payloads(reader: gguf.GGUFReader, snapshot: Path, expected: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    sources = {}
    for part in sorted(set(index.values())):
        path = snapshot / part
        with path.open("rb") as f:
            header_length = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(header_length))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            check(name not in sources and index.get(name) == part, f"source index/header {name}")
            sources[name] = (path, 8 + header_length, info)
    check(set(index) == set(sources) == {v[0] for v in expected.values()}, "source inventory exactly the 382 C4 tensors")
    tensors = {t.name: t for t in reader.tensors}
    for target, (source, shape) in expected.items():
        path, offset, info = sources[source]
        count = int(np.prod(shape))
        start, end = info["data_offsets"]
        check(info["dtype"] == "BF16" and tuple(info["shape"]) == shape and end - start == count * 2, f"source {source} BF16 shape {shape}")
        words = np.memmap(path, mode="r", dtype="<u2", offset=offset + start, shape=(count,))
        bf16 = target in BF16_TENSORS
        actual = tensors[target].data.view("<u2").reshape(-1) if bf16 else tensors[target].data.reshape(-1)
        dtype = "BF16" if bf16 else "<f4" if len(shape) == 1 else "<f2"
        for begin in range(0, count, 4 * 1024 * 1024):
            stop = min(count, begin + 4 * 1024 * 1024)
            if bf16:
                converted = words[begin:stop]
            else:
                expanded = np.left_shift(words[begin:stop].astype("<u4"), 16).view("<f4")
                converted = expanded.astype(dtype)
            if converted.tobytes() != actual[begin:stop].tobytes():
                raise ValueError(f"payload mismatch: {target} expected {source}; converted BF16 bytes differ in elements [{begin},{stop})")
        print(f"PASS payload {target} <- {source}: {count} values bit-identical after {dtype} conversion", flush=True)
        del words


def verify(path: Path, snapshot: Path) -> None:
    reader = gguf.GGUFReader(path)
    metadata(reader, snapshot)
    expected = inventory()
    tensors = {t.name: t for t in reader.tensors}
    for i, local in enumerate(PATTERN):
        name = f"blk.{i}.attn_gate.weight"
        check((name in tensors) == local, f"{'required local gate' if local else 'absent global gate'} {name}")
    check(len(reader.tensors) == 382, f"tensor count 382; actual {len(reader.tensors)}")
    check(set(tensors) == set(expected), f"exact tensor inventory; missing {sorted(set(expected) - set(tensors))}; unknown {sorted(set(tensors) - set(expected))}")
    for name, (_, shape) in expected.items():
        tensor = tensors[name]
        check(tuple(tensor.shape) == shape[::-1], f"tensor {name} GGML shape {shape[::-1]}; actual {tuple(int(n) for n in tensor.shape)}")
        dtype = gguf.GGMLQuantizationType.BF16 if name in BF16_TENSORS else gguf.GGMLQuantizationType.F32 if len(shape) == 1 else gguf.GGMLQuantizationType.F16
        check(tensor.tensor_type == dtype, f"tensor {name} type {dtype.name}; actual {tensor.tensor_type.name}")
    payloads(reader, snapshot, expected)
    print("PASS ZGCM contract: all metadata, 382 tensors, 27 local gates, and all converted payload bytes", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--hf-model", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(args.gguf, args.hf_model)
    except (ValueError, KeyError, OSError, TypeError, IndexError) as exc:
        print(f"FAIL {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
