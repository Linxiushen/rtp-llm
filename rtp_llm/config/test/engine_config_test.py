import unittest
from unittest import TestCase
from unittest.mock import patch

from rtp_llm.config.engine_config import finalize_scheduler_config
from rtp_llm.device.device_type import DeviceType
from rtp_llm.ops import RoleType


class DummyFIFOSchedulerConfig:
    def __init__(self):
        self.max_context_batch_size = 2
        self.max_batch_tokens_size = 0
        self.prefill_chunk_size = 0


class EngineConfigTest(TestCase):
    def _finalize(self, chunk_size=0, **overrides):
        cfg = DummyFIFOSchedulerConfig()
        cfg.prefill_chunk_size = chunk_size
        args = {
            "max_seq_len": 1024,
            "use_hybrid_attention": False,
            "role_type": RoleType.PREFILL,
            "use_batch_decode_scheduler": False,
            "seq_size_per_block": 64,
        }
        args.update(overrides)
        finalize_scheduler_config(cfg, **args)
        return cfg

    def test_finalize_scheduler_config_disabled_by_default(self):
        # prefill_chunk_size <= 0 => chunked prefill disabled, no validation runs.
        cfg = self._finalize(
            use_hybrid_attention=True,  # would raise if chunked prefill were enabled
        )

        self.assertEqual(cfg.max_batch_tokens_size, 2048)
        self.assertEqual(cfg.prefill_chunk_size, 0)

    def test_finalize_scheduler_config_rejects_chunk_size_smaller_than_one_block(self):
        with self.assertRaises(ValueError):
            self._finalize(chunk_size=17)

    def test_finalize_scheduler_config_floor_aligns_chunk_size(self):
        requested_chunk_size = 130
        cfg = self._finalize(chunk_size=requested_chunk_size)

        self.assertEqual(cfg.prefill_chunk_size, 128)
        self.assertLessEqual(cfg.prefill_chunk_size, requested_chunk_size)

    def test_finalize_scheduler_config_allows_int_max_chunk_size(self):
        cfg = self._finalize(
            chunk_size=2**31 - 1,
            seq_size_per_block=1,
        )

        self.assertEqual(cfg.prefill_chunk_size, 2**31 - 1)

    def test_finalize_scheduler_config_rejects_chunk_size_above_int_max(self):
        with self.assertRaises(ValueError):
            self._finalize(
                chunk_size=2**31,
                seq_size_per_block=1,
            )

    def test_finalize_scheduler_config_rejects_hybrid_attention(self):
        with self.assertRaisesRegex(ValueError, "hybrid"):
            self._finalize(chunk_size=64, use_hybrid_attention=True)

    def test_finalize_scheduler_config_disables_chunked_prefill_for_unsupported_role(
        self,
    ):
        # Roles other than PREFILL / PDFUSION never activate chunked prefill in C++; config
        # finalization should not reject their model combination just because the shared env
        # var is present, and it should silently zero prefill_chunk_size so downstream sees a
        # disabled config.
        cfg = self._finalize(
            chunk_size=64,
            use_hybrid_attention=True,
            role_type=RoleType.DECODE,
            use_batch_decode_scheduler=True,
        )

        self.assertEqual(cfg.max_batch_tokens_size, 2048)
        self.assertEqual(cfg.prefill_chunk_size, 0)

    def test_finalize_scheduler_config_rejects_batch_decode_scheduler(self):
        with self.assertRaisesRegex(ValueError, "use_batch_decode_scheduler=True"):
            self._finalize(
                chunk_size=64,
                use_batch_decode_scheduler=True,
            )

    def test_finalize_scheduler_config_allows_supported_roles(self):
        # Both roles execute prefill locally and share the same chunked-prefill gate.
        for role_type in (RoleType.PREFILL, RoleType.PDFUSION):
            with self.subTest(role_type=role_type):
                cfg = self._finalize(
                    chunk_size=64,
                    role_type=role_type,
                )
                self.assertEqual(cfg.prefill_chunk_size, 64)


class MlaChunkedConfigTest(TestCase):
    def setUp(self):
        from rtp_llm.model_factory import ModelFactory

        self.update = ModelFactory.update_engine_config_from_model_config
        self.engine, self.model = self._make_configs()
        cuda = patch(
            "rtp_llm.device.device_type.get_device_type", return_value=DeviceType.Cuda
        )
        cuda.start()
        self.addCleanup(cuda.stop)

    @staticmethod
    def _make_configs():
        from rtp_llm.config.engine_config import EngineConfig
        from rtp_llm.config.model_config import ModelConfig
        from rtp_llm.config.py_config_modules import PyEnvConfigs

        engine = EngineConfig.create(PyEnvConfigs())
        engine.pd_sep_config.role_type = RoleType.PDFUSION
        engine.runtime_config.fifo_scheduler_config.prefill_chunk_size = 64
        engine.hw_kernel_config.enable_cuda_graph = False
        engine.kv_cache_config.reuse_cache = False
        model = ModelConfig()
        model.max_seq_len = 2048
        model.data_type = "bf16"
        model.attn_config.use_mla = True
        model.attn_config.head_num = 16
        model.attn_config.kv_head_num = 16
        model.attn_config.tokens_per_block = 64
        model.attn_config.kernel_tokens_per_block = 64
        return engine, model

    def test_allows_supported_mla_configurations(self):
        from rtp_llm.config.quant_config import Fp8BlockWiseQuantConfig

        fusion, prefill = RoleType.PDFUSION, RoleType.PREFILL
        # TP, FP8 weights, prefix reuse, decode graph, role.
        cases = [
            (1, False, False, False, fusion),
            (2, False, False, False, fusion),
            (4, False, False, False, fusion),
            (8, False, False, False, fusion),
            (16, False, False, False, fusion),
            (1, False, True, False, fusion),
            (2, False, True, False, fusion),
            (1, True, False, False, fusion),
            (2, True, True, False, fusion),
            (2, False, False, True, fusion),
            (2, False, True, True, fusion),
            (2, True, False, True, fusion),
            (2, True, True, True, fusion),
            (1, False, False, False, prefill),
            (1, False, True, False, prefill),
            (1, True, False, False, prefill),
            (1, True, True, False, prefill),
        ]
        for tp, fp8, reuse, graph, role in cases:
            with self.subTest(tp=tp, fp8=fp8, reuse=reuse, graph=graph, role=role):
                engine, model = self._make_configs()
                engine.parallelism_config.tp_size = tp
                engine.parallelism_config.world_size = tp
                engine.pd_sep_config.role_type = role
                engine.kv_cache_config.reuse_cache = reuse
                engine.hw_kernel_config.enable_cuda_graph = graph
                quant = Fp8BlockWiseQuantConfig() if fp8 else None
                model.quant_config = quant
                self.update(engine, model)
                self.assertEqual(
                    engine.runtime_config.fifo_scheduler_config.prefill_chunk_size, 64
                )
                self.assertEqual(engine.pd_sep_config.role_type, role)
                self.assertEqual(engine.kv_cache_config.reuse_cache, reuse)
                self.assertEqual(engine.hw_kernel_config.enable_cuda_graph, graph)
                self.assertIs(model.quant_config, quant)

    def test_allows_native_mla_block_layouts(self):
        for logical, kernel in ((32, 32), (64, 32), (128, 64), (128, 128)):
            with self.subTest(logical=logical, kernel=kernel):
                engine, model = self._make_configs()
                engine.runtime_config.fifo_scheduler_config.prefill_chunk_size = (
                    2 * logical
                )
                engine.kv_cache_config.seq_size_per_block = logical
                engine.kv_cache_config.kernel_seq_size_per_block = kernel
                model.attn_config.tokens_per_block = logical
                model.attn_config.kernel_tokens_per_block = kernel
                self.update(engine, model)
                self.assertEqual(model.attn_config.tokens_per_block, logical)
                self.assertEqual(model.attn_config.kernel_tokens_per_block, kernel)

    def test_invalid_head_partition_still_uses_loader_validation(self):
        from types import SimpleNamespace

        from rtp_llm.model_loader.loader import get_model_loader

        for budget in (0, 64):
            with self.subTest(budget=budget):
                engine, model = self._make_configs()
                engine.parallelism_config.tp_size = 3
                engine.parallelism_config.world_size = 3
                engine.runtime_config.fifo_scheduler_config.prefill_chunk_size = budget
                self.update(engine, model)
                with self.assertRaisesRegex(
                    Exception, "invalid tp_size 3 for config.head_num 16"
                ):
                    get_model_loader(
                        model, SimpleNamespace(_head_num=16, tp_size=3), None, None
                    )

    def test_rejects_incompatible_combinations(self):
        from rtp_llm.ops import CPRotateMethod, KvCacheDataType, SpeculativeType

        cases = [
            ("model.attn_config.is_sparse", True, "sparse"),
            ("model.data_type", "fp16", "BF16"),
            ("model.attn_config.kv_cache_dtype", KvCacheDataType.FP8, "KV"),
            (
                "engine.parallelism_config.prefill_cp_config.method",
                CPRotateMethod.ALL_GATHER,
                "CP",
            ),
            (
                "engine.parallelism_config.prefill_cp_config.method",
                CPRotateMethod.PREFILL_CP,
                "CP",
            ),
            ("engine.sp_config.type", SpeculativeType.MTP, "speculative"),
        ]
        for path, value, message in cases:
            with self.subTest(path=path, value=value):
                self.engine, self.model = self._make_configs()
                obj = self
                parts = path.split(".")
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
                with self.assertRaisesRegex(ValueError, message):
                    self.update(self.engine, self.model)
        for platform in (DeviceType.Cpu, DeviceType.ROCm, DeviceType.Ppu):
            with self.subTest(platform=platform):
                self.engine, self.model = self._make_configs()
                with patch(
                    "rtp_llm.device.device_type.get_device_type", return_value=platform
                ):
                    with self.assertRaisesRegex(ValueError, "CUDA"):
                        self.update(self.engine, self.model)

    def test_inactive_chunking_does_not_restrict_mla(self):
        for role, budget, dtype in (
            (RoleType.PDFUSION, 0, "fp16"),
            (RoleType.DECODE, 64, "bf16"),
        ):
            with self.subTest(role=role, budget=budget):
                engine, model = self._make_configs()
                engine.pd_sep_config.role_type = role
                engine.runtime_config.fifo_scheduler_config.prefill_chunk_size = budget
                model.attn_config.is_sparse = True
                model.data_type = dtype
                self.update(engine, model)
                self.assertEqual(
                    engine.runtime_config.fifo_scheduler_config.prefill_chunk_size, 0
                )


if __name__ == "__main__":
    unittest.main()
