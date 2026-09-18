"""Real engine differential test with a small, locally generated MLA checkpoint."""

import json
import math
import os
import re
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


def make_checkpoint(
    path: Path,
    *,
    q_lora_rank: int = 0,
    moe: bool = False,
    num_experts: int = 4,
    experts_per_token: int = 2,
    hidden_size: int = 256,
):
    # Keep the original dense/no-Q-LoRA fixture as the quick regression.
    # FP8 MoE's native ep_gather requires hidden_size divisible by 512.
    hidden, intermediate, heads, layers, vocab = hidden_size, 512, 16, 2, 256
    if hidden <= 0:
        raise ValueError("hidden_size must be positive")
    if moe and not (1 <= experts_per_token <= num_experts):
        raise ValueError("experts_per_token must be between 1 and num_experts")
    config = {
        "architectures": ["DeepseekV2ForCausalLM"],
        "model_type": "deepseek_v2",
        "hidden_size": hidden,
        "intermediate_size": intermediate,
        "num_attention_heads": heads,
        "num_key_value_heads": heads,
        "num_hidden_layers": layers,
        "vocab_size": vocab,
        "max_position_embeddings": 4096,
        "q_lora_rank": q_lora_rank or None,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "first_k_dense_replace": 1 if moe else layers,
        "n_routed_experts": num_experts if moe else 2,
        "n_shared_experts": 1,
        "num_experts_per_tok": experts_per_token if moe else 1,
        "moe_intermediate_size": intermediate,
        "routed_scaling_factor": 1.0,
        "torch_dtype": "bfloat16",
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
    }
    (path / "config.json").write_text(json.dumps(config))
    generator = torch.Generator(device="cpu").manual_seed(42)

    def weight(*shape):
        return (torch.randn(shape, generator=generator, device="cpu") * 0.02).bfloat16()

    weights = {
        "model.embed_tokens.weight": weight(vocab, hidden),
        "model.norm.weight": torch.ones(hidden, dtype=torch.bfloat16, device="cpu"),
        "lm_head.weight": weight(vocab, hidden),
    }
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        for name in ("input_layernorm", "post_attention_layernorm"):
            weights[prefix + name + ".weight"] = torch.ones(
                hidden, dtype=torch.bfloat16, device="cpu"
            )
        shapes = {
            "self_attn.q_proj": (heads * 192, hidden),
            "self_attn.kv_a_proj_with_mqa": (576, hidden),
            "self_attn.kv_b_proj": (heads * 256, 512),
            "self_attn.o_proj": (hidden, heads * 128),
            "mlp.gate_proj": (intermediate, hidden),
            "mlp.up_proj": (intermediate, hidden),
            "mlp.down_proj": (hidden, intermediate),
        }
        if q_lora_rank:
            shapes.pop("self_attn.q_proj")
            shapes["self_attn.q_a_proj"] = (q_lora_rank, hidden)
            shapes["self_attn.q_b_proj"] = (heads * 192, q_lora_rank)
            weights[prefix + "self_attn.q_a_layernorm.weight"] = torch.ones(
                q_lora_rank, dtype=torch.bfloat16, device="cpu"
            )
        if moe and layer >= config["first_k_dense_replace"]:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                shape = shapes.pop("mlp." + projection)
                shapes["mlp.shared_experts." + projection] = shape
                for expert in range(config["n_routed_experts"]):
                    shapes[f"mlp.experts.{expert}.{projection}"] = shape
            shapes["mlp.gate"] = (config["n_routed_experts"], hidden)
        for name, shape in shapes.items():
            weights[prefix + name + ".weight"] = weight(*shape)
        weights[prefix + "self_attn.kv_a_layernorm.weight"] = torch.ones(
            512, dtype=torch.bfloat16, device="cpu"
        )
    save_file(weights, str(path / "model.safetensors"))
    vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2}
    vocabulary.update({f"t{i}": i for i in range(3, vocab)})
    tokenizer = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    fast.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    fast.save_pretrained(path)


class ChunkedMlaEngineTest(unittest.TestCase):
    def _prompt(self, length):
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is None:
            return " ".join(f"t{3 + i % 253}" for i in range(length))
        # Preserve real tokenizer/BOS behavior while varying token content.
        paragraph = (
            "The old bridge crosses a quiet river. A red bird lands on a branch "
            "while two children count the boats. The library opens in the morning "
            "and closes after sunset. "
        )
        ids = tokenizer.encode(paragraph * (length // 8 + 2))[:length]
        self.assertEqual(len(ids), length)
        prompt = tokenizer.decode(ids, skip_special_tokens=True)
        self.assertEqual(tokenizer.encode(prompt), ids)
        return prompt

    def _request(
        self,
        manager,
        length,
        budget,
        concurrent=False,
        *,
        prompt=None,
        expected_reuse=0,
        return_frames=False,
    ):
        prompt = self._prompt(length) if prompt is None else prompt
        if expected_reuse is None:
            expected_reuse = 0
            if self._cache_active:
                prompt_ids = (
                    self.tokenizer.encode(prompt)
                    if self.tokenizer is not None
                    else [int(token[1:]) for token in prompt.split()]
                )
                for cached in self._cached_prompts:
                    common = 0
                    for current, previous in zip(prompt_ids[:-1], cached):
                        if current != previous:
                            break
                        common += 1
                    expected_reuse = max(
                        expected_reuse,
                        common // self._cache_block_size * self._cache_block_size,
                    )
        payload = {
            "prompt": prompt,
            "yield_generator": True,
            "generate_config": {
                "max_new_tokens": 6,
                "min_new_tokens": 6,
                "top_k": 1,
                "return_output_ids": True,
                "reuse_cache": True,
                "is_streaming": True,
            },
        }
        frames = []
        with requests.post(
            f"http://127.0.0.1:{manager.port}/", json=payload, stream=True, timeout=120
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                text = line.decode().removeprefix("data:").strip()
                if text.lower() == "[done]":
                    continue
                frames.append(json.loads(text))
        report = os.environ.get("MLA_TEST_RESPONSE_LOG")
        if report:
            with open(report, "a") as output:
                output.write(
                    json.dumps(
                        {
                            "budget": budget,
                            "length": length,
                            "concurrent": concurrent,
                            "reuse": getattr(self, "_cache_active", False),
                            "graph": getattr(self, "_graph_active", False),
                            "frames": frames,
                        }
                    )
                    + "\n"
                )
        # No user-visible output for intermediate prefill chunks; then six decode outputs.
        self.assertEqual(len(frames), 6, frames)
        for step, frame in enumerate(frames, 1):
            self.assertEqual(frame["aux_info"]["input_len"], length)
            self.assertEqual(frame["aux_info"]["output_len"], step)
            self.assertEqual(frame["aux_info"]["reuse_len"], expected_reuse)
            remaining = length - expected_reuse
            iterations = (math.ceil(remaining / budget) if budget else 1) + step - 1
            if concurrent:
                self.assertGreaterEqual(frame["aux_info"]["iter_count"], iterations)
            else:
                self.assertEqual(frame["aux_info"]["iter_count"], iterations)
        self.assertTrue(frames[-1]["finished"])
        if getattr(self, "_cache_active", False):
            self._cached_prompts.append(
                self.tokenizer.encode(prompt)
                if self.tokenizer is not None
                else [int(token[1:]) for token in prompt.split()]
            )
        return frames if return_frames else [frame["output_ids"] for frame in frames]

    def test_full_vs_chunked_prefill_and_decode(self):
        tp_size = int(os.environ.get("MLA_TEST_TP_SIZE", "1"))
        dp_size = int(os.environ.get("MLA_TEST_DP_SIZE", "1"))
        ep_size = int(os.environ.get("MLA_TEST_EP_SIZE", "1"))
        block_size = int(os.environ.get("MLA_TEST_BLOCK_SIZE", "64"))
        kernel_block_size = int(os.environ.get("MLA_TEST_KERNEL_BLOCK_SIZE", "64"))
        fp8_kv_cache = int(os.environ.get("MLA_TEST_FP8_KV_CACHE", "0"))
        quantization = os.environ.get("MLA_TEST_QUANTIZATION", "")
        quantization_args = f" --quantization {quantization}" if quantization else ""
        budgets = tuple(
            int(x)
            for x in os.environ.get("MLA_TEST_BUDGETS", "0,64,128,256").split(",")
        )
        self.assertEqual(
            budgets[0], 0, "every configuration needs an unchunked reference"
        )
        with tempfile.TemporaryDirectory() as generated_directory:
            directory = os.environ.get("MLA_TEST_MODEL_PATH", generated_directory)
            if directory == generated_directory:
                make_checkpoint(
                    Path(directory),
                    q_lora_rank=int(os.environ.get("MLA_TEST_Q_LORA_RANK", "0")),
                    moe=os.environ.get("MLA_TEST_MOE", "0") == "1",
                    num_experts=int(os.environ.get("MLA_TEST_MOE_EXPERTS", "4")),
                    experts_per_token=int(os.environ.get("MLA_TEST_MOE_TOP_K", "2")),
                    hidden_size=int(os.environ.get("MLA_TEST_HIDDEN_SIZE", "256")),
                )
                self.tokenizer = None
            else:
                self.assertTrue((Path(directory) / "config.json").is_file())
                # Use the service's post-processor/BOS compatibility handling.
                self.tokenizer = TokenizerFactory.create(
                    directory, directory, "deepseek2"
                )
            reference = {}
            check_reuse = os.environ.get("MLA_TEST_CACHE_REUSE", "0") == "1"
            check_graph = os.environ.get("MLA_TEST_CUDA_GRAPH", "0") == "1"
            capture_sizes = tuple(
                int(x)
                for x in os.environ.get("MLA_TEST_DECODE_CAPTURE_CONFIG", "1,2").split(
                    ","
                )
            )
            self.assertTrue(capture_sizes and all(x > 0 for x in capture_sizes))
            capture_config = ",".join(str(x) for x in capture_sizes)
            graph_log_config = Path(generated_directory) / "graph_alog.conf"
            if check_graph:
                # initLogger reloads alog after the logger constructor has read
                # LOG_LEVEL. Preserve DEBUG in that later configuration too.
                alog = Path(__file__).resolve().parents[2] / "config" / "alog.conf"
                graph_log_config.write_text(
                    alog.read_text()
                    .replace("alog.rootLogger=INFO,", "alog.rootLogger=DEBUG,")
                    .replace("alog.logger.console=INFO,", "alog.logger.console=DEBUG,")
                )
                phases = [(False, False, 0)] + [
                    (True, check_reuse, budget) for budget in budgets
                ]
            elif check_reuse:
                phases = [(False, False, 0)] + [
                    (False, True, budget) for budget in budgets
                ]
            else:
                phases = [(False, False, budget) for budget in budgets]
            cache_args = (
                " --enable_device_cache 1 --enable_memory_cache 0 --enable_remote_cache 0"
                if check_reuse
                else ""
            )
            self._cache_block_size = block_size
            for phase, (graph, reuse, budget) in enumerate(phases):
                self._graph_active = graph
                self._cache_active = reuse
                self._cached_prompts = []
                manager = MagaServerManager(
                    env_args={
                        "DETERMINISTIC_GEMM": "1",
                        "ENABLE_STABLE_SCATTER_ADD": "ON",
                        **({"LOG_LEVEL": "DEBUG"} if graph else {}),
                    },
                    role_name=(
                        f"mla_graph_{int(graph)}_reuse_{int(reuse)}_chunk_{budget}"
                        if check_graph
                        else (
                            f"mla_reuse_{int(reuse)}_chunk_{budget}"
                            if check_reuse
                            else f"mla_chunk_{budget}"
                        )
                    ),
                    smoke_args_str=(
                        f"--act_type bf16 --tp_size {tp_size} --dp_size {dp_size} --ep_size {ep_size} --world_size {tp_size * dp_size} "
                        f"--role_type PDFUSION --enable_cuda_graph {int(graph)} --reuse_cache {int(reuse)} "
                        f"--fp8_kv_cache {fp8_kv_cache} --seq_size_per_block {block_size} --kernel_seq_size_per_block {kernel_block_size} "
                        "--test_block_num 128 --max_seq_len 2048 --max_context_batch_size 2 "
                        "--warm_up 0 --frontend_server_count 1 --shutdown_timeout 5 "
                        f"--prefill_chunk_size {budget}{quantization_args}{cache_args}"
                        + (
                            f" --decode_capture_config {capture_config} --concurrency_limit 3"
                            f" --ft_alog_conf_path {graph_log_config}"
                            if graph
                            else ""
                        )
                    ),
                )
                try:
                    self.assertTrue(
                        manager.start_server(
                            model_path=directory,
                            tokenizer_path=directory,
                            model_type="deepseek2",
                            timeout=300,
                        )
                    )
                    # Exhaustive block boundaries are covered by mla_reuse_cache_test.
                    for length in (128, 130, 257, 1025):
                        with self.subTest(budget=budget, length=length):
                            output_ids = self._request(
                                manager, length, budget, expected_reuse=None
                            )
                            if phase == 0:
                                reference[length] = output_ids
                            else:
                                self.assertEqual(output_ids, reference[length])
                        if (
                            length == 130
                            and os.environ.get("MLA_TEST_CAUSAL_SUFFIX") == "1"
                        ):
                            original = self._prompt(length)
                            if self.tokenizer is None:
                                changed = original.split()
                                changed[64:] = [
                                    f"t{253 - i % 127}" for i in range(length - 64)
                                ]
                                changed_prompt = " ".join(changed)
                            else:
                                original_ids = self.tokenizer.encode(original)
                                changed_prompt = (
                                    "A scientist measures the temperature of a blue star. "
                                    * length
                                )
                                # Change a suffix, retaining the first 64 actual token IDs.
                                changed_ids = self.tokenizer.encode(changed_prompt)[
                                    :length
                                ]
                                changed_ids = original_ids[:64] + changed_ids[64:]
                                self.assertNotEqual(changed_ids[64:], original_ids[64:])
                                changed_prompt = self.tokenizer.decode(
                                    changed_ids, skip_special_tokens=True
                                )
                                self.assertEqual(
                                    self.tokenizer.encode(changed_prompt), changed_ids
                                )
                            causal_ids = self._request(
                                manager,
                                length,
                                budget,
                                prompt=changed_prompt,
                                expected_reuse=None,
                            )
                            if phase == 0:
                                reference["causal"] = causal_ids
                            else:
                                self.assertEqual(causal_ids, reference["causal"])
                    # The optional audit barrier queues both requests behind a primer.
                    # FIFO admits new prefill rows only at an empty execution boundary.
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        primer = None
                        if os.environ.get("MLA_TEST_SYNC_CONCURRENCY") == "1":
                            ready = (
                                Path(os.environ["RTP_HOT_HOOK_DUMP_DIR"])
                                / f"b{budget}.batch_ready"
                            )
                            ready.unlink(missing_ok=True)
                            ready.with_suffix(".batch_trigger").write_text("primer")
                            primer = pool.submit(
                                self._request,
                                manager,
                                128,
                                budget,
                                True,
                                expected_reuse=None,
                            )
                            deadline = time.monotonic() + 20
                            while not ready.exists():
                                if primer.done():
                                    primer.result()
                                    self.fail(
                                        "primer returned before the audit barrier"
                                    )
                                self.assertLess(
                                    time.monotonic(), deadline, "primer barrier timeout"
                                )
                                time.sleep(0.01)
                        futures = {
                            length: pool.submit(
                                self._request,
                                manager,
                                length,
                                budget,
                                True,
                                expected_reuse=None,
                            )
                            for length in (257, 130)
                        }
                        for length, future in futures.items():
                            with self.subTest(budget=budget, concurrent_length=length):
                                self.assertEqual(future.result(), reference[length])
                        if primer is not None:
                            with self.subTest(budget=budget, primer=True):
                                self.assertEqual(primer.result(), reference[128])
                    if graph:
                        process_log = Path(manager.log_file_path).read_text()
                        self.assertIn("Capture Decode End", process_log)
                        self.assertIn("Replay End", process_log)
                        self.assertIn("batch size used in replay: 1", process_log)
                        if os.environ.get("MLA_TEST_SYNC_CONCURRENCY") == "1":
                            if max(capture_sizes) >= 2:
                                self.assertIn(
                                    "batch size used in replay: 2", process_log
                                )
                            else:
                                self.assertIn(
                                    "decode batch size 2 exceeds max captured 1, fallback to normal run",
                                    process_log,
                                )
                        self.assertNotIn("Capture Prefill Start", process_log)
                        self.assertNotIn("Buffer reallocation required", process_log)
                        report = os.environ.get("MLA_TEST_RESPONSE_LOG")
                        if report:
                            with open(report + ".graph.jsonl", "a") as output:
                                output.write(
                                    json.dumps(
                                        {
                                            "budget": budget,
                                            "reuse": reuse,
                                            "capture_sizes": capture_sizes,
                                            "batch2_fallback_count": process_log.count(
                                                "decode batch size 2 exceeds max captured 1, fallback to normal run"
                                            ),
                                            "capture_end_count": process_log.count(
                                                "Capture Decode End"
                                            ),
                                            "replay_end_count": process_log.count(
                                                "Replay End"
                                            ),
                                            "replay_batches": sorted(
                                                set(
                                                    int(x)
                                                    for x in re.findall(
                                                        r"batch size used in replay: (\d+)",
                                                        process_log,
                                                    )
                                                )
                                            ),
                                        }
                                    )
                                    + "\n"
                                )
                finally:
                    manager.stop_server()


if __name__ == "__main__":
    unittest.main()
