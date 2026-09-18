import itertools
import math
import os
import random
import sys
from typing import Any, Dict, List, Optional
from unittest import SkipTest, TestCase, main

import torch
import torch.nn.functional as F

device = torch.device(f"cuda")

import flashinfer.page as page

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models.rotary_embedding.deepseek_rotary_embedding import (
    DeepseekV3YarnRotaryEmbedding,
)
from rtp_llm.models_py.modules import LinearFactory
from rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper import (
    MlaFlashInferPrefillImpl,
)
from rtp_llm.models_py.modules.hybrid.test.mla_attention_ref import attention_ref
from rtp_llm.ops import FMHAConfig, ParallelismConfig, compute_ops
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, rtp_llm_ops
from rtp_llm.utils.model_weight import W


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads):
    bs_page_num, page_size, ckv_dim = ckv.shape
    page_num = bs_page_num // batch_size
    _, _, kpe_dim = kpe.shape
    ckv = ckv.view(batch_size, page_num * page_size, ckv_dim)
    kpe = kpe.view(batch_size, page_num * page_size, kpe_dim)
    ckv = ckv[:, :kv_len, :]
    kpe = kpe[:, :kv_len, :]
    k = (
        torch.cat([ckv, kpe], dim=-1)
        .view(-1, 1, ckv_dim + kpe_dim)
        .repeat_interleave(num_heads, dim=1)
    )
    v = ckv.repeat_interleave(num_heads, dim=1)

    return k, v


def create_cos_sin_cache():
    rotary_emb = DeepseekV3YarnRotaryEmbedding(
        64,
        163840,
        10000,
        scaling_factor=1.0,
        original_max_position_embeddings=4096,
        beta_fast=32,
        beta_slow=1,
        mscale=0.707,
        mscale_all_dim=0.707,
    )
    half_rope_dim = 64 // 2
    cos_cache = rotary_emb.cos_cached[:, :half_rope_dim]
    sin_cache = rotary_emb.sin_cached[:, :half_rope_dim]
    # cos sin cache must be float32
    cos_sin_cache = (
        torch.cat([cos_cache, sin_cache], dim=-1)
        .contiguous()
        .to(device)
        .to(torch.float32)
    )
    return cos_sin_cache


class MLATest(TestCase):
    NUM_TOKENS = [7, 2000]
    HIDDEN_SIZES = [2048]
    PAGE_SIZE = [64]
    REUSE_LEN = [0, 128]

    def setUp(self) -> None:
        if not torch.cuda.is_available():
            raise SkipTest("CUDA is not available")
        torch.set_default_device(device)

    def _run_mla_test(
        self, num_tokens: int, hidden_size: int, page_size: int, reuse_len: int
    ):

        input_lengths = [num_tokens]
        mock_page_num = 2048
        page_num = math.ceil((reuse_len + num_tokens + page_size - 1) / page_size)
        block_list = [i for i in range(1, page_num + 1)]
        # print(f"block_list: {block_list}")
        kvcache_block_id = torch.tensor(
            [block_list],
            dtype=torch.int32,
            device=torch.device("cpu"),
        )

        self.config = ModelConfig()
        self.config.attn_config.head_num = 16
        self.config.hidden_size = hidden_size
        self.config.attn_config.nope_head_dim = 128
        self.config.attn_config.rope_head_dim = 64
        self.config.attn_config.kv_lora_rank = 512
        self.config.attn_config.v_head_dim = 128
        self.config.attn_config.q_lora_rank = 0
        self.config.attn_config.tokens_per_block = 64
        self.config.attn_config.kernel_tokens_per_block = 64
        self.config.attn_config.softmax_extra_scale = 1.0
        self.config.attn_config.use_mla = True
        self.config.attn_config.size_per_head = 192
        self.scaling = (
            self.config.attn_config.nope_head_dim
            + self.config.attn_config.rope_head_dim
        ) ** (-0.5)

        self.parallelism_config = ParallelismConfig()
        self.parallelism_config.tp_size = 1
        self.parallelism_config.tp_rank = 0

        torch.manual_seed(0)
        input_lengths_t = torch.tensor(
            input_lengths, dtype=torch.int32, device=torch.device("cpu")
        )
        prefix_lengths_t = torch.tensor(
            [reuse_len],
            dtype=torch.int32,
            device=torch.device("cpu"),
        )

        attn_inputs: PyAttentionInputs = PyAttentionInputs()
        attn_inputs.is_prefill = True
        attn_inputs.prefix_lengths = prefix_lengths_t
        attn_inputs.sequence_lengths = torch.tensor(
            [], dtype=torch.int32, device=torch.device("cpu")
        )
        attn_inputs.input_lengths = input_lengths_t
        attn_inputs.kv_cache_block_id = kvcache_block_id
        attn_inputs.kv_cache_block_id_device = kvcache_block_id.to(device)
        attn_inputs.kv_cache_kernel_block_id = kvcache_block_id
        attn_inputs.kv_cache_kernel_block_id_device = kvcache_block_id.to(device)

        weights = self._create_weights(self.config, hidden_size)
        layer_weights: List[Dict[str, torch.Tensor]] = [weights]

        cos_sin_cache = create_cos_sin_cache()

        fmha_impl = MlaFlashInferPrefillImpl(
            self.config.attn_config,
            attn_inputs,
            layer_weights,
            cos_sin_cache,
            quant_config=self.config.quant_config,
        )

        q = torch.randn(
            [
                num_tokens,
                self.config.attn_config.head_num,
                self.config.attn_config.nope_head_dim
                + self.config.attn_config.rope_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        compressed_kv = torch.randn(
            [num_tokens, self.config.attn_config.kv_lora_rank],
            dtype=torch.bfloat16,
            device=device,
        )

        k_pe = torch.randn(
            [num_tokens, self.config.attn_config.rope_head_dim],
            dtype=torch.bfloat16,
            device=device,
        )

        cache = torch.randn(
            [
                mock_page_num,
                page_size,
                self.config.attn_config.kv_lora_rank
                + self.config.attn_config.rope_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        kv_cache: Optional[LayerKVCache] = LayerKVCache()
        kv_cache.kv_cache_base = cache

        k_cache, v_cache = torch.split(
            kv_cache.kv_cache_base,
            [
                self.config.attn_config.kv_lora_rank,
                self.config.attn_config.rope_head_dim,
            ],
            dim=-1,
        )
        # NewMlaRotaryEmbeddingParams: flashinfer params live under .params
        compute_ops.concat_and_cache_mla(
            compressed_kv,
            k_pe,
            kv_cache.kv_cache_base,
            fmha_impl.rope_params.slot_mapping,
            "auto",
            torch.tensor(1.0, dtype=torch.float32, device=device),
        )

        out = fmha_impl.compute_prefill_context(q, compressed_kv, k_pe, kv_cache, 0)

        index_list = torch.empty(0, dtype=torch.int32, device=device)
        if fmha_impl.fmha_impl.reuse_cache_page_indice is not None:
            index_list = fmha_impl.fmha_impl.reuse_cache_page_indice.clone()
        selected_blocks = cache[index_list]
        selected_blocks = selected_blocks.view(-1, selected_blocks.size(-1))

        compressed_kv = torch.cat(
            [selected_blocks[:, : compressed_kv.size(1)], compressed_kv], dim=0
        )
        k_pe = k_pe.view(-1, self.config.attn_config.rope_head_dim)
        k_pe = torch.cat([selected_blocks[:, compressed_kv.size(1) :], k_pe], dim=0)

        k_pe = k_pe.view(-1, 1, self.config.attn_config.rope_head_dim)
        self.kv_b_proj = LinearFactory.create_linear_from_weights(
            layer_weights[0], W.mla_kv_b_w, W.mla_kv_b_s, None
        )

        kv = self.kv_b_proj(compressed_kv)
        kv = kv.view(
            -1,
            self.config.attn_config.head_num,
            self.config.attn_config.nope_head_dim + self.config.attn_config.v_head_dim,
        )
        k_nope = kv[:, :, : self.config.attn_config.nope_head_dim].contiguous()
        value_states = kv[:, :, self.config.attn_config.nope_head_dim :].contiguous()

        k = k_pe.new_empty(
            k_pe.size(0),
            self.config.attn_config.head_num,
            self.config.attn_config.rope_head_dim
            + self.config.attn_config.nope_head_dim,
        )
        k[..., : self.config.attn_config.nope_head_dim] = k_nope
        k[..., self.config.attn_config.nope_head_dim :] = k_pe
        out_ref, _ = attention_ref(
            1,
            q,
            k,
            value_states,
            causal=True,
            sm_scale=self.scaling,
        )
        out_norm = out / (torch.norm(out) + 1e-8)
        out_ref_norm = out_ref / (torch.norm(out_ref) + 1e-8)
        self.assertTrue(torch.allclose(out_norm, out_ref_norm, atol=0.01, rtol=0.01))
        out_flat = out.flatten()
        out_ref_flat = out_ref.flatten()
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(
            out_flat.unsqueeze(0), out_ref_flat.unsqueeze(0), dim=1
        )
        self.assertTrue(
            torch.allclose(
                torch.tensor(1.0).to(device).to(cosine_sim.dtype),
                cosine_sim,
                atol=0.01,
                rtol=0.01,
            )
        )

    def _create_weights(self, config, hidden_size):
        """创建测试权重"""
        weights = {}
        weights[W.mla_fusedqkrope_no_lora_w] = torch.randn(
            [
                config.hidden_size,
                config.attn_config.size_per_head * config.attn_config.head_num
                + config.attn_config.kv_lora_rank
                + config.attn_config.rope_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_kv_a_ln_gamma] = torch.randn(
            [config.attn_config.kv_lora_rank], dtype=torch.bfloat16, device=device
        )

        weights[W.mla_kc] = torch.randn(
            [
                config.attn_config.head_num,
                config.attn_config.nope_head_dim,
                config.attn_config.kv_lora_rank,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_vc] = torch.randn(
            [
                config.attn_config.head_num,
                config.attn_config.kv_lora_rank,
                config.attn_config.v_head_dim,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        weights[W.mla_kv_b_w] = torch.randn(
            [
                config.attn_config.kv_lora_rank,
                config.attn_config.head_num
                * (config.attn_config.nope_head_dim + config.attn_config.v_head_dim),
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        kv_b = weights[W.mla_kv_b_w].view(
            config.attn_config.kv_lora_rank,
            config.attn_config.head_num,
            config.attn_config.nope_head_dim + config.attn_config.v_head_dim,
        )
        weights[W.mla_kc] = (
            kv_b[:, :, : config.attn_config.nope_head_dim].permute(1, 2, 0).contiguous()
        )
        weights[W.mla_vc] = (
            kv_b[:, :, config.attn_config.nope_head_dim :].transpose(0, 1).contiguous()
        )

        weights[W.attn_o_w] = torch.randn(
            [
                config.attn_config.head_num * config.attn_config.v_head_dim,
                config.hidden_size,
            ],
            dtype=torch.bfloat16,
            device=device,
        )

        return weights

    def test_mlp(self):
        for params in itertools.product(
            self.NUM_TOKENS, self.HIDDEN_SIZES, self.PAGE_SIZE, self.REUSE_LEN
        ):
            with self.subTest(
                num_tokens=params[0],
                hidden_size=params[1],
                page_size=params[2],
                reuse_len=params[3],
            ):
                self._run_mla_test(*params)

    @staticmethod
    def _chunk_inputs(lengths, prefixes, blocks):
        inputs = PyAttentionInputs()
        inputs.is_prefill = True
        inputs.input_lengths = torch.tensor(lengths, dtype=torch.int32, device="cpu")
        inputs.prefix_lengths = torch.tensor(prefixes, dtype=torch.int32, device="cpu")
        inputs.sequence_lengths = torch.empty(0, dtype=torch.int32, device="cpu")
        inputs.kv_cache_block_id = torch.tensor(blocks, dtype=torch.int32, device="cpu")
        inputs.kv_cache_block_id_device = inputs.kv_cache_block_id.to(device)
        inputs.kv_cache_kernel_block_id = inputs.kv_cache_block_id
        inputs.kv_cache_kernel_block_id_device = inputs.kv_cache_block_id_device
        return inputs

    def test_chunk_metadata(self):
        params = rtp_llm_ops.FlashInferMlaAttnParams()
        # Reuse one params object, including a growing then shrinking batch.
        cases = [
            ([64], [0], [[5, 2, 9]], [0, 64], [64], list(range(320, 384))),
            ([64], [64], [[5, 2, 9]], [0, 64], [128], list(range(128, 192))),
            (
                [64, 2],
                [64, 128],
                [[5, 2, 9], [1, 3, 7]],
                [0, 64, 66],
                [128, 130],
                list(range(128, 192)) + [448, 449],
            ),
            ([2], [128], [[5, 2, 9]], [0, 2], [130], [576, 577]),
        ]
        for lengths, prefixes, blocks, q_indptr, kv_lengths, slots in cases:
            with self.subTest(lengths=lengths, prefixes=prefixes):
                inputs = self._chunk_inputs(lengths, prefixes, blocks)
                params.fill_params(
                    inputs.prefix_lengths,
                    inputs.sequence_lengths,
                    inputs.input_lengths,
                    inputs.kv_cache_kernel_block_id,
                    64,
                )
                self.assertEqual(params.qo_indptr_h.tolist(), q_indptr)
                self.assertEqual(params.kvlen_h.tolist(), kv_lengths)
                self.assertEqual(
                    params.prefill_ragged_kv_len_indptr_d.cpu().tolist(),
                    [0] + list(itertools.accumulate(kv_lengths)),
                )
                self.assertEqual(params.slot_mapping.cpu().tolist(), slots)
                positions = [p + i for p, n in zip(prefixes, lengths) for i in range(n)]
                self.assertEqual(params.positions_h.tolist(), positions)

    def _chunk_fixture(self):
        config = ModelConfig()
        attn = config.attn_config
        attn.head_num = 16
        attn.nope_head_dim = 128
        attn.rope_head_dim = 64
        attn.kv_lora_rank = 512
        attn.v_head_dim = 128
        attn.size_per_head = 192
        attn.tokens_per_block = attn.kernel_tokens_per_block = 64
        attn.use_mla = True
        config.hidden_size = 2048
        torch.manual_seed(17)
        weights = self._create_weights(config, config.hidden_size)
        # Fan-in scaling keeps logits and raw outputs in the usual BF16 range.
        for name in (W.mla_kv_b_w, W.mla_kc, W.mla_vc):
            weights[name] = weights[name] / math.sqrt(attn.kv_lora_rank)
        cos_sin = create_cos_sin_cache()
        return attn, weights, cos_sin

    def test_chunked_forward(self):
        attn, weights, cos_sin = self._chunk_fixture()
        for length, budget, absorb_len in itertools.chain(
            itertools.product(
                (1, 63, 64, 65, 127, 128, 129, 130, 257, 1025),
                (64, 128, 256),
                (0, 1024),
            ),
            # With a cached prefix, final Q lengths straddle the absorb threshold.
            ((2047, 1024, 1024), (2048, 1024, 1024), (3073, 2048, 1024)),
        ):
            with self.subTest(length=length, budget=budget, absorb_len=absorb_len):
                page_count = math.ceil(length / 64)
                # Fragmented physical pages, with unowned pages between them.
                blocks = [5, 2, 9] + list(range(11, 11 + max(0, page_count - 3)))
                blocks = blocks[:page_count]
                q = torch.randn(length, 16, 192, dtype=torch.bfloat16, device=device)
                ckv = torch.randn(length, 512, dtype=torch.bfloat16, device=device)
                kpe = torch.randn(length, 64, dtype=torch.bfloat16, device=device)
                reference_cache = LayerKVCache()
                reference_cache.kv_cache_base = torch.full(
                    (max(blocks) + 2, 64, 576), -7, dtype=torch.bfloat16, device=device
                )
                chunk_cache = LayerKVCache()
                chunk_cache.kv_cache_base = reference_cache.kv_cache_base.clone()
                fmha = FMHAConfig()
                fmha.absorb_opt_len = absorb_len
                reference_impl = MlaFlashInferPrefillImpl(
                    attn,
                    self._chunk_inputs([length], [0], [blocks]),
                    [weights],
                    cos_sin,
                    fmha_config=fmha,
                )
                reference = reference_impl.forward(
                    q.clone(), ckv, kpe.clone(), reference_cache, 0
                )
                outputs = []
                written = torch.zeros(
                    chunk_cache.kv_cache_base.shape[:2], dtype=torch.bool, device=device
                )
                for start in range(0, length, budget):
                    end = min(start + budget, length)
                    before = chunk_cache.kv_cache_base.clone()
                    impl = MlaFlashInferPrefillImpl(
                        attn,
                        self._chunk_inputs([end - start], [start], [blocks]),
                        [weights],
                        cos_sin,
                        fmha_config=fmha,
                    )
                    self.assertEqual(
                        impl.absorb_fmha is not None,
                        start > 0 and end - start < absorb_len,
                    )
                    outputs.append(
                        impl.forward(
                            q[start:end].clone(),
                            ckv[start:end],
                            kpe[start:end].clone(),
                            chunk_cache,
                            0,
                        )
                    )
                    current = torch.zeros_like(written)
                    for pos in range(start, end):
                        current[blocks[pos // 64], pos % 64] = True
                    torch.testing.assert_close(
                        chunk_cache.kv_cache_base[~current],
                        before[~current],
                        rtol=0,
                        atol=0,
                    )
                    written |= current
                    torch.testing.assert_close(
                        chunk_cache.kv_cache_base[written],
                        reference_cache.kv_cache_base[written],
                        rtol=0,
                        atol=0,
                    )
                # Compare raw values, including amplitude, not normalized directions.
                torch.testing.assert_close(
                    torch.cat(outputs), reference, rtol=0.01, atol=0.01
                )

    def test_fp8_weight_prefill_keeps_quantized_kv_projection(self):
        from rtp_llm.config.quant_config import init_quant_config
        from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import (
            is_deep_gemm_e8m0_used,
        )
        from rtp_llm.models_py.kernels.cuda.fp8_kernel import requant_weight_ue8m0
        from rtp_llm.test.utils.numeric_util import per_block_cast_to_fp8

        attn, weights, cos_sin = self._chunk_fixture()
        quant = init_quant_config("FP8_PER_BLOCK")
        # Match load-time quantization: KV-B is quantized, while the absorbed
        # matrices keep the original BF16 checkpoint values.
        weight = weights[W.mla_kv_b_w].t().contiguous()
        quant_weight, scales = per_block_cast_to_fp8(weight, use_ue8m0=False)
        if is_deep_gemm_e8m0_used():
            quant_weight, scales = requant_weight_ue8m0(quant_weight, scales)
        else:
            quant_weight = quant_weight.reshape(weight.shape[1], weight.shape[0])
            scales = scales.reshape(scales.shape[1], scales.shape[0])
        weights[W.mla_kv_b_w] = quant_weight
        weights[W.mla_kv_b_s] = scales
        fmha = FMHAConfig()
        fmha.absorb_opt_len = 4096
        for length, budget in itertools.product((130, 257), (64, 128)):
            with self.subTest(length=length, budget=budget):
                blocks = [5, 2, 9, 1, 7][: math.ceil(length / 64)]
                q = torch.randn(length, 16, 192, dtype=torch.bfloat16, device=device)
                ckv = torch.randn(length, 512, dtype=torch.bfloat16, device=device)
                kpe = torch.randn(length, 64, dtype=torch.bfloat16, device=device)

                def cache():
                    result = LayerKVCache()
                    result.kv_cache_base = torch.full(
                        (12, 64, 576), -7, dtype=torch.bfloat16, device=device
                    )
                    return result

                reference_cache = cache()
                reference_impl = MlaFlashInferPrefillImpl(
                    attn,
                    self._chunk_inputs([length], [0], [blocks]),
                    [weights],
                    cos_sin,
                    fmha_config=fmha,
                    quant_config=quant,
                )
                reference = reference_impl.forward(
                    q.clone(), ckv, kpe.clone(), reference_cache, 0
                )
                chunk_cache = cache()
                outputs = []
                for start in range(0, length, budget):
                    end = min(start + budget, length)
                    impl = MlaFlashInferPrefillImpl(
                        attn,
                        self._chunk_inputs([end - start], [start], [blocks]),
                        [weights],
                        cos_sin,
                        fmha_config=fmha,
                        quant_config=quant,
                    )
                    self.assertIsNone(
                        impl.absorb_fmha,
                        "FP8 KV-B activation quantization cannot be absorbed into BF16 weights",
                    )
                    before = chunk_cache.kv_cache_base.clone()
                    outputs.append(
                        impl.forward(
                            q[start:end].clone(),
                            ckv[start:end],
                            kpe[start:end].clone(),
                            chunk_cache,
                            0,
                        )
                    )
                    slots = torch.tensor(
                        [blocks[p // 64] * 64 + p % 64 for p in range(start, end)],
                        dtype=torch.long,
                        device=device,
                    )
                    flat = chunk_cache.kv_cache_base.reshape(-1, 576)
                    keep = torch.ones(flat.shape[0], dtype=torch.bool, device=device)
                    keep[slots] = False
                    self.assertTrue(
                        torch.equal(
                            flat[slots],
                            reference_cache.kv_cache_base.reshape(-1, 576)[slots],
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            flat[keep].view(torch.uint8),
                            before.reshape(-1, 576)[keep].view(torch.uint8),
                        )
                    )
                torch.testing.assert_close(
                    torch.cat(outputs), reference, atol=0.01, rtol=0.01
                )

    def test_native_block_sizes_and_causal_suffix(self):
        for block_size, absorb_len in itertools.product((32, 64, 128), (0, 1024)):
            with self.subTest(block_size=block_size, absorb_len=absorb_len):
                attn, weights, cos_sin = self._chunk_fixture()
                attn.tokens_per_block = attn.kernel_tokens_per_block = block_size
                length = 2 * block_size + 2
                blocks = [5, 2, 9]
                q = torch.randn(length, 16, 192, dtype=torch.bfloat16, device=device)
                ckv = torch.randn(length, 512, dtype=torch.bfloat16, device=device)
                kpe = torch.randn(length, 64, dtype=torch.bfloat16, device=device)
                fmha = FMHAConfig()
                fmha.absorb_opt_len = absorb_len

                def cache():
                    result = LayerKVCache()
                    result.kv_cache_base = torch.full(
                        (11, block_size, 576), -7, dtype=torch.bfloat16, device=device
                    )
                    return result

                def forward(start, end, q_input, ckv_input, kpe_input, kv):
                    impl = MlaFlashInferPrefillImpl(
                        attn,
                        self._chunk_inputs([end - start], [start], [blocks]),
                        [weights],
                        cos_sin,
                        fmha_config=fmha,
                    )
                    expected_slots = [
                        blocks[p // block_size] * block_size + p % block_size
                        for p in range(start, end)
                    ]
                    self.assertEqual(
                        impl.fmha_params.slot_mapping.cpu().tolist(), expected_slots
                    )
                    return impl.forward(
                        q_input.clone(), ckv_input.clone(), kpe_input.clone(), kv, 0
                    )

                reference_cache = cache()
                reference = forward(0, length, q, ckv, kpe, reference_cache)
                actual_cache = cache()
                outputs = []
                for start in range(0, length, block_size):
                    end = min(start + block_size, length)
                    before = actual_cache.kv_cache_base.clone()
                    outputs.append(
                        forward(
                            start,
                            end,
                            q[start:end],
                            ckv[start:end],
                            kpe[start:end],
                            actual_cache,
                        )
                    )
                    written = torch.zeros(
                        (11, block_size), dtype=torch.bool, device=device
                    )
                    for pos in range(start, end):
                        written[blocks[pos // block_size], pos % block_size] = True
                    torch.testing.assert_close(
                        actual_cache.kv_cache_base[~written],
                        before[~written],
                        atol=0,
                        rtol=0,
                    )
                    torch.testing.assert_close(
                        actual_cache.kv_cache_base[written],
                        reference_cache.kv_cache_base[written],
                        atol=0,
                        rtol=0,
                    )
                torch.testing.assert_close(
                    torch.cat(outputs), reference, atol=0.01, rtol=0.01
                )

                # Future KV must not affect queries before the changed suffix.
                future_q, future_ckv, future_kpe = q.clone(), ckv.clone(), kpe.clone()
                future_q[block_size:] = -future_q[block_size:]
                future_ckv[block_size:] = -future_ckv[block_size:]
                future_kpe[block_size:] = -future_kpe[block_size:]
                changed = forward(0, length, future_q, future_ckv, future_kpe, cache())
                torch.testing.assert_close(
                    changed[:block_size], reference[:block_size], atol=0.01, rtol=0.01
                )

    def test_chunked_batched_forward(self):
        attn, weights, cos_sin = self._chunk_fixture()
        lengths = (130, 193)
        blocks = ([5, 2, 9, 0], [1, 4, 8, 7])
        queries = [
            torch.randn(n, 16, 192, dtype=torch.bfloat16, device=device)
            for n in lengths
        ]
        compressed = [
            torch.randn(n, 512, dtype=torch.bfloat16, device=device) for n in lengths
        ]
        keys = [
            torch.randn(n, 64, dtype=torch.bfloat16, device=device) for n in lengths
        ]
        for absorb_len in (0, 1024):
            with self.subTest(absorb_len=absorb_len):
                fmha = FMHAConfig()
                fmha.absorb_opt_len = absorb_len
                cache = LayerKVCache()
                cache.kv_cache_base = torch.full(
                    (11, 64, 576), -7, dtype=torch.bfloat16, device=device
                )
                reference_cache = LayerKVCache()
                reference_cache.kv_cache_base = cache.kv_cache_base.clone()
                references = []
                for row, length in enumerate(lengths):
                    impl = MlaFlashInferPrefillImpl(
                        attn,
                        self._chunk_inputs([length], [0], [blocks[row]]),
                        [weights],
                        cos_sin,
                        fmha_config=fmha,
                    )
                    references.append(
                        impl.forward(
                            queries[row].clone(),
                            compressed[row],
                            keys[row].clone(),
                            reference_cache,
                            0,
                        )
                    )
                prefixes = [0, 0]
                # Vary grants and batch membership; the third call has different prefixes.
                for batch in (
                    [(0, 64), (1, 64)],
                    [(1, 128)],
                    [(1, 1), (0, 64)],
                    [(0, 2)],
                ):
                    inputs = self._chunk_inputs(
                        [n for _, n in batch],
                        [prefixes[row] for row, _ in batch],
                        [blocks[row] for row, _ in batch],
                    )
                    impl = MlaFlashInferPrefillImpl(
                        attn, inputs, [weights], cos_sin, fmha_config=fmha
                    )
                    slices = [
                        (row, slice(prefixes[row], prefixes[row] + n))
                        for row, n in batch
                    ]
                    before = cache.kv_cache_base.clone()
                    output = impl.forward(
                        torch.cat([queries[row][s] for row, s in slices]),
                        torch.cat([compressed[row][s] for row, s in slices]),
                        torch.cat([keys[row][s] for row, s in slices]),
                        cache,
                        0,
                    )
                    expected = torch.cat([references[row][s] for row, s in slices])
                    torch.testing.assert_close(output, expected, atol=0.01, rtol=0.01)
                    written = torch.zeros((11, 64), dtype=torch.bool, device=device)
                    for row, count in batch:
                        for pos in range(prefixes[row], prefixes[row] + count):
                            written[blocks[row][pos // 64], pos % 64] = True
                        prefixes[row] += count
                    torch.testing.assert_close(
                        cache.kv_cache_base[~written], before[~written], atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        cache.kv_cache_base[written],
                        reference_cache.kv_cache_base[written],
                        atol=0,
                        rtol=0,
                    )


if __name__ == "__main__":
    main()
