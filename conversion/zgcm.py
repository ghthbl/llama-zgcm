from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("ZgcmForCausalLM")
@ModelBase.example("zgcagi/ZGCM-1-7B")
class ZgcmModel(TextModel):
    model_arch = gguf.MODEL_ARCH.ZGCM
    bf16_tensors = tuple(f"blk.{i}.ffn_down.weight" for i in range(32))

    def _validate_config(self) -> list[bool]:
        expected = {
            "vocab_size": 155136, "hidden_size": 4096, "intermediate_size": 11008,
            "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8,
            "head_dim": 128, "hidden_act": "silu", "rms_norm_eps": 1e-6,
            "rope_theta": 10000000.0, "partial_rotary_factor": 0.334,
            "max_position_embeddings": 262144, "sliding_window": 128,
            "window_attn_skip_freq": 6, "attention_bias": False, "tie_word_embeddings": False,
            "eos_token_id": 154820, "pad_token_id": 154820, "bos_token_id": None,
        }
        for key, value in expected.items():
            if key not in self.hparams or self.hparams[key] != value:
                raise ValueError(f"ZGCM requires {key}={value!r}, got {self.hparams.get(key)!r}")
        layer_types = self.hparams["layer_types"]
        if any(t not in ("sliding_attention", "full_attention") for t in layer_types):
            raise ValueError("ZGCM has an unknown attention layer type")
        pattern = [t == "sliding_attention" for t in layer_types]
        literal = [
            True, True, True, True, True, False,
            True, True, True, True, True, False,
            True, True, True, True, True, False,
            True, True, True, True, True, False,
            True, True, True, True, True, False,
            True, True,
        ]
        if pattern != literal or self.hparams.get("attention_gate_layers") != literal:
            raise ValueError("ZGCM requires 27 gated local layers and global layers [5,11,17,23,29]")
        if self.hparams.get("rope_scaling") or self.hparams.get("rope_parameters") or self.hparams.get("quantization_config"):
            raise ValueError("ZGCM checkpoint must have unscaled RoPE and unquantized weights")
        return pattern

    def _tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        shapes = {
            "model.embed_tokens.weight": (155136, 4096),
            "model.norm.weight": (4096,),
            "lm_head.weight": (155136, 4096),
        }
        layer_shapes = {
            "post_attention_layernorm": (4096,),
            "post_feedforward_layernorm": (4096,),
            "self_attn.q_proj": (4096, 4096),
            "self_attn.k_proj": (1024, 4096),
            "self_attn.v_proj": (1024, 4096),
            "self_attn.q_norm": (128,),
            "self_attn.k_norm": (128,),
            "self_attn.o_proj": (4096, 4096),
            "mlp.gate_proj": (11008, 4096),
            "mlp.up_proj": (11008, 4096),
            "mlp.down_proj": (4096, 11008),
        }
        for i, local in enumerate(self._validate_config()):
            for name, shape in layer_shapes.items():
                shapes[f"model.layers.{i}.{name}.weight"] = shape
            if local:
                shapes[f"model.layers.{i}.self_attn.g_proj.weight"] = (4096, 4096)
        return shapes

    @staticmethod
    def _unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"ZGCM duplicate JSON key: {key}")
            result[key] = value
        return result

    def index_tensors(self, remote_hf_model_id: str | None = None):
        if remote_hf_model_id is not None:
            raise ValueError("ZGCM conversion requires a complete local checkpoint")
        expected = self._tensor_shapes()
        index_path = self.dir_model / "model.safetensors.index.json"
        index = json.loads(index_path.read_text(encoding="utf-8"), object_pairs_hook=self._unique_object)["weight_map"]
        parts = sorted(set(index.values()))
        if parts != ModelBase.get_model_part_names(self.dir_model, "model", ".safetensors"):
            raise ValueError("ZGCM shard files do not match the weight index")
        seen = set()
        for part in parts:
            path = self.dir_model / part
            with path.open("rb") as f:
                header_length = int.from_bytes(f.read(8), "little")
                json.loads(f.read(header_length), object_pairs_hook=self._unique_object)
            with gguf.utility.SafetensorsLocal(path) as tensors:
                for name, tensor in tensors.items():
                    if name in seen:
                        raise ValueError(f"ZGCM duplicate tensor: {name}")
                    if name.endswith(".self_attn.g_proj.weight") and name not in expected:
                        raise ValueError(f"ZGCM gate on a global or unknown layer: {name}")
                    if name not in expected:
                        raise ValueError(f"ZGCM unknown tensor: {name}")
                    if tensor.shape != expected[name] or tensor.dtype != "BF16":
                        raise ValueError(f"ZGCM wrong shape or dtype: {name}: {tensor.shape}, {tensor.dtype}")
                    if index.get(name) != part:
                        raise ValueError(f"ZGCM tensor index mismatch: {name}")
                    seen.add(name)
        missing = expected.keys() - seen
        missing_gates = sorted(n for n in missing if n.endswith(".g_proj.weight"))
        if missing_gates:
            raise ValueError(f"ZGCM missing local gates: {missing_gates}")
        if missing or set(index) != seen:
            raise ValueError(f"ZGCM missing tensors or index mismatch: {sorted(missing)}")
        logger.info("ZGCM checkpoint: 382 tensors, 27 local gates, global layers [5,11,17,23,29]")
        return super().index_tensors(remote_hf_model_id)

    def prepare_tensors(self):
        if self.ftype != gguf.LlamaFileType.MOSTLY_F16 or self.is_big_endian or self.fuse_qkv:
            raise ValueError("ZGCM parity conversion requires little-endian F16 and separate Q/K/V")
        self._consumed: set[str] = set()
        self._outputs: set[str] = set()
        super().prepare_tensors()
        missing = self._tensor_shapes().keys() - self._consumed
        if missing or len(self._outputs) != 382:
            raise ValueError(f"ZGCM unconsumed tensors: {sorted(missing)}")

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if new_name in self.bf16_tensors:
            # BF16 Metal operands avoid F16 staging of every SwiGLU down input.
            return gguf.GGMLQuantizationType.BF16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name in self._consumed:
            raise ValueError(f"ZGCM consumed tensor twice: {name}")
        if name.endswith(".post_attention_layernorm.weight"):
            # HF name says post_attention, but forward applies it before attention.
            target = self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_NORM, bid)
        elif name.endswith(".post_feedforward_layernorm.weight"):
            # HF name says post_feedforward, but forward applies it before the MLP.
            target = self.format_tensor_name(gguf.MODEL_TENSOR.FFN_NORM, bid)
        else:
            target = self.map_tensor_name(name)
        if target in self._outputs:
            raise ValueError(f"ZGCM duplicate output tensor: {target}")
        self._consumed.add(name)
        self._outputs.add(target)
        yield target, data_torch

    def set_gguf_parameters(self) -> None:
        pattern = self._validate_config()
        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        self.gguf_writer.add_causal_attention(True)
        self.gguf_writer.add_key_length(self.hparams["head_dim"])
        self.gguf_writer.add_value_length(self.hparams["head_dim"])
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(pattern)
        self.gguf_writer.add_rope_dimension_count(int(self.hparams["head_dim"] * self.hparams["partial_rotary_factor"]))

    def set_vocab(self) -> None:
        data = json.loads((self.dir_model / "tokenizer.json").read_text(encoding="utf-8"), object_pairs_hook=self._unique_object)
        if data["model"]["type"] != "BPE" or data["model"].get("ignore_merges") is not True or data.get("normalizer") is not None:
            raise ValueError("ZGCM requires byte-level BPE with ignore_merges and no normalizer")
        vocab = data["model"]["vocab"]
        added = data["added_tokens"]
        if len(vocab) != 154820 or set(vocab.values()) != set(range(154820)) or [t["id"] for t in added] != list(range(154820, 154856)):
            raise ValueError("ZGCM tokenizer vocabulary ids differ from the checkpoint contract")
        texts = set(vocab) | {t["content"] for t in added}
        if len(texts) != 154856:
            raise ValueError("ZGCM duplicate token text")
        tokens = [""] * self.hparams["vocab_size"]
        types = [gguf.TokenType.NORMAL] * len(tokens)
        for text, token_id in vocab.items():
            tokens[token_id] = text
        for token in added:
            token_id = token["id"]
            if token["special"] != (token_id < 154838):
                raise ValueError(f"ZGCM added-token special flag mismatch: {token_id}")
            tokens[token_id] = token["content"]
            types[token_id] = gguf.TokenType.CONTROL if token["special"] else gguf.TokenType.USER_DEFINED
        for token_id in range(154856, len(tokens)):
            text = f"[PAD{token_id}]"
            if text in texts:
                raise ValueError(f"ZGCM unused token collides with vocabulary: {text}")
            tokens[token_id] = text
            types[token_id] = gguf.TokenType.UNUSED
        self.gguf_writer.add_tokenizer_model("gpt2")
        # The probe hash collides with glm4, which has different ignore_merges behavior.
        self.gguf_writer.add_tokenizer_pre("zgcm")
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(types)
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        template = (self.dir_model / "chat_template.jinja").read_text(encoding="utf-8")
        if special_vocab.special_token_ids != {"eos": 154820, "pad": 154820} or special_vocab.chat_template != template:
            raise ValueError("ZGCM special-token roles or chat template differ from the checkpoint contract")
        if special_vocab.add_special_token:
            raise ValueError("ZGCM checkpoint must not configure automatic special-token insertion")
        special_vocab.add_to_gguf(self.gguf_writer)
        self.gguf_writer.add_add_bos_token(False)
        self.gguf_writer.add_add_eos_token(False)
