"""Cross-request device-cache differential checks using the MLA engine fixture."""

import http.client
import json
import os
import socket
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
import torch

from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import StatusVersionPB
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2_grpc import RpcServiceStub
from rtp_llm.test.smoke import chunked_mla_engine_test as fixture
from rtp_llm.test.utils.maga_server_manager import MagaServerManager


def tokens(length, seed):
    return [f"t{3 + (seed + i) % 253}" for i in range(length)]


# These hooks run only in this test's server processes. They observe real KV
# writes and stop at the next chunk boundary until the test releases the event.
_saved_windows = {}
_model_case = None


def lifecycle_model_before(ctx):
    # Keep one label for every layer of a model call, even when cancellation
    # lets the driver enqueue a follow-up before in-flight bookkeeping drains.
    global _model_case
    _model_case = json.loads(Path(os.environ["MLA_TEST_LIFECYCLE_STATE"]).read_text())[
        "case"
    ]


def lifecycle_before(ctx):
    impl, q, ckv, kpe, cache, layer = ctx.args[:6]
    state_file = Path(os.environ["MLA_TEST_LIFECYCLE_STATE"])
    assert _model_case is not None, "model-entry lifecycle hook must run first"
    metadata = {
        "case": _model_case,
        "time": time.time(),
        "pid": os.getpid(),
        "layer": layer,
        "absorb": impl.absorb_fmha is not None,
        "lengths": impl.attn_inputs.input_lengths.tolist(),
        "prefixes": impl.attn_inputs.prefix_lengths.tolist(),
        "blocks": impl.attn_inputs.kv_cache_kernel_block_id.tolist(),
    }
    _saved_windows[(id(impl), layer)] = (metadata, cache.kv_cache_base.clone())
    trigger = state_file.with_suffix(".trigger")
    if layer != 0 or not trigger.exists():
        return
    try:
        command = json.loads(trigger.read_text())
    except FileNotFoundError:
        # Another TP rank may claim the trigger after the existence check.
        return
    if metadata["prefixes"] != [command["prefix"]]:
        return
    try:
        trigger.rename(trigger.with_suffix(".claimed"))
    except FileNotFoundError:
        return
    with socket.create_connection(
        ("127.0.0.1", command["port"]), timeout=30
    ) as connection:
        connection.sendall((json.dumps(metadata) + "\n").encode())
        if connection.recv(1) != b"G":
            raise RuntimeError("lifecycle barrier was not released")


def lifecycle_after(ctx):
    impl, q, ckv, kpe, cache, layer = ctx.args[:6]
    metadata, before = _saved_windows.pop((id(impl), layer))
    width = impl.attn_configs.kv_lora_rank + impl.attn_configs.rope_head_dim
    flat = cache.kv_cache_base.reshape(-1, width)
    slots = impl.fmha_params.slot_mapping.long()
    block_size = impl.attn_configs.kernel_tokens_per_block
    expected_slots = [
        blocks[position // block_size] * block_size + position % block_size
        for blocks, prefix, length in zip(
            metadata["blocks"], metadata["prefixes"], metadata["lengths"]
        )
        for position in range(prefix, prefix + length)
    ]
    metadata["slots_exact"] = slots.cpu().tolist() == expected_slots
    untouched = torch.ones(flat.shape[0], dtype=torch.bool, device=flat.device)
    untouched[slots] = False
    metadata["unwritten_exact"] = torch.equal(
        flat[untouched].view(torch.uint8),
        before.reshape(-1, width)[untouched].view(torch.uint8),
    )
    metadata["written_exact"] = torch.equal(flat[slots], torch.cat((ckv, kpe), dim=-1))
    with open(os.environ["MLA_TEST_LIFECYCLE_EVENTS"], "a") as output:
        output.write(json.dumps(metadata) + "\n")


def wait_for_enqueued(manager, count, timeout=10):
    """Status registration follows engine enqueue; it proves both requests arrived."""
    deadline = time.monotonic() + timeout
    with grpc.insecure_channel(f"127.0.0.1:{manager.port + 1}") as channel:
        stub = RpcServiceStub(channel)
        while True:
            status = stub.GetWorkerStatus(
                StatusVersionPB(latest_finished_version=-1), timeout=2
            )
            active = list(status.running_task_info)
            if len(active) >= count:
                return [
                    {
                        "request_id": x.request_id,
                        "length": x.input_length,
                        "phase": x.phase,
                        "waiting": x.is_waiting,
                    }
                    for x in active
                ]
            if time.monotonic() >= deadline:
                raise AssertionError(f"only {len(active)} of {count} requests enqueued")
            time.sleep(0.01)


def raw_request(port, prompt, timeout_ms=0):
    payload = json.dumps(
        {
            "prompt": prompt,
            "yield_generator": True,
            "generate_config": {
                "max_new_tokens": 6,
                "min_new_tokens": 6,
                "top_k": 1,
                "return_output_ids": True,
                "reuse_cache": True,
                "is_streaming": True,
                "timeout_ms": timeout_ms,
            },
        }
    ).encode()
    connection = socket.create_connection(("127.0.0.1", port), timeout=30)
    headers = (
        f"POST / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(payload)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode()
    connection.sendall(headers + payload)
    return connection


def read_raw_response(connection):
    response = http.client.HTTPResponse(connection)
    response.begin()
    return response.status, response.read().decode()


class ChunkedMlaCacheLifecycleTest(unittest.TestCase):
    @staticmethod
    def _make_checkpoint(path):
        fixture.make_checkpoint(
            path,
            q_lora_rank=int(os.environ.get("MLA_TEST_Q_LORA_RANK", "0")),
            moe=os.environ.get("MLA_TEST_MOE", "0") == "1",
            num_experts=int(os.environ.get("MLA_TEST_MOE_EXPERTS", "4")),
            experts_per_token=int(os.environ.get("MLA_TEST_MOE_TOP_K", "2")),
            hidden_size=int(os.environ.get("MLA_TEST_HIDDEN_SIZE", "256")),
        )

    def test_prefix_hits_forks_and_released_pages(self):
        tp_size = int(os.environ.get("MLA_TEST_TP_SIZE", "1"))
        driver = fixture.ChunkedMlaEngineTest()
        reference = {}
        with tempfile.TemporaryDirectory() as directory:
            self._make_checkpoint(Path(directory))
            # Every reuse/chunk combination has the same checkpoint and topology.
            for reuse, budget in (
                (False, 0),
                (True, 0),
                (True, 64),
                (True, 128),
                (True, 256),
            ):
                with self.subTest(reuse=reuse, budget=budget):
                    state = Path(directory) / f"prefix_state_{int(reuse)}_{budget}.json"
                    events = state.with_suffix(".events.jsonl")
                    config = state.with_suffix(".hooks.json")
                    state.write_text(json.dumps({"case": "prefix"}))
                    config.write_text(
                        json.dumps(
                            {
                                "case": "mla_prefix_lifecycle",
                                "function_hooks": [
                                    {
                                        "target": "rtp_llm.models_py.model_desc.generic_moe.GenericMoeModel.forward",
                                        "before": "lifecycle_model_before",
                                    },
                                    {
                                        "target": "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper.MlaFlashInferPrefillImpl.forward",
                                        "before": "lifecycle_before",
                                        "after": "lifecycle_after",
                                    },
                                ],
                            }
                        )
                    )
                    manager = MagaServerManager(
                        env_args={
                            "DETERMINISTIC_GEMM": "1",
                            "ENABLE_STABLE_SCATTER_ADD": "ON",
                            "RTP_HOT_HOOK": "1",
                            "RTP_HOT_HOOK_FILE": os.environ.get(
                                "MLA_TEST_LIFECYCLE_HOOK_FILE",
                                str(Path(__file__).resolve()),
                            ),
                            "RTP_HOT_HOOK_CONFIG": os.environ.get(
                                "MLA_TEST_LIFECYCLE_HOOK_CONFIG", str(config)
                            ),
                            "MLA_TEST_LIFECYCLE_STATE": str(state),
                            "MLA_TEST_LIFECYCLE_EVENTS": str(events),
                        },
                        role_name=f"mla_reuse_{int(reuse)}_chunk_{budget}",
                        smoke_args_str=(
                            f"--act_type bf16 --tp_size {tp_size} --dp_size 1 --ep_size 1 "
                            f"--world_size {tp_size} --reuse_cache {int(reuse)} "
                            "--enable_device_cache 1 --enable_memory_cache 0 --enable_remote_cache 0 "
                            "--role_type PDFUSION --enable_cuda_graph 0 --fp8_kv_cache 0 "
                            "--seq_size_per_block 64 --kernel_seq_size_per_block 64 "
                            "--test_block_num 32 --max_seq_len 2048 --max_context_batch_size 2 "
                            "--warm_up 0 --frontend_server_count 1 --shutdown_timeout 5 "
                            f"--prefill_chunk_size {budget}"
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

                        def request(key, prompt_tokens, hit=0, concurrent=False):
                            started = time.time()
                            frames = driver._request(
                                manager,
                                len(prompt_tokens),
                                budget,
                                concurrent,
                                prompt=" ".join(prompt_tokens),
                                expected_reuse=hit if reuse else 0,
                                return_frames=True,
                            )
                            result = [frame["output_ids"] for frame in frames]
                            report = os.environ.get("MLA_TEST_LIFECYCLE_LOG")
                            if report:
                                with open(report, "a") as output:
                                    output.write(
                                        json.dumps(
                                            {
                                                "key": key,
                                                "reuse": reuse,
                                                "budget": budget,
                                                "started": started,
                                                "finished": time.time(),
                                                "frames": frames,
                                            }
                                        )
                                        + "\n"
                                    )
                            if not reuse:
                                reference[key] = result
                            else:
                                self.assertEqual(result, reference[key], key)
                            return result

                        def synchronized_requests(case, requests, primer_seed):
                            # Hold a primer until both target requests are registered after
                            # engine enqueue, so they compete at the same admission boundary.
                            state.write_text(json.dumps({"case": case}))
                            with socket.socket() as listener:
                                listener.bind(("127.0.0.1", 0))
                                listener.listen(1)
                                listener.settimeout(30)
                                state.with_suffix(".trigger").write_text(
                                    json.dumps(
                                        {
                                            "prefix": 0,
                                            "port": listener.getsockname()[1],
                                        }
                                    )
                                )
                                primer = raw_request(
                                    manager.port, " ".join(tokens(64, primer_seed))
                                )
                                barrier, _ = listener.accept()
                                barrier.settimeout(30)
                                with barrier.makefile("rb") as stream:
                                    observed = json.loads(stream.readline())
                                self.assertEqual(observed["prefixes"], [0])
                                with ThreadPoolExecutor(max_workers=2) as pool:
                                    futures = [
                                        pool.submit(request, key, value, hit, True)
                                        for key, value, hit in requests
                                    ]
                                    try:
                                        active = wait_for_enqueued(manager, 3)
                                        report = os.environ.get(
                                            "MLA_TEST_LIFECYCLE_LOG"
                                        )
                                        if report:
                                            with open(
                                                report + ".batch.jsonl", "a"
                                            ) as out:
                                                out.write(
                                                    json.dumps(
                                                        {
                                                            "reuse": reuse,
                                                            "budget": budget,
                                                            "case": case,
                                                            "tasks": active,
                                                        }
                                                    )
                                                    + "\n"
                                                )
                                    finally:
                                        barrier.sendall(b"G")
                                        barrier.close()
                                    status, body = read_raw_response(primer)
                                    primer.close()
                                    self.assertEqual(status, 200, body)
                                    for future in futures:
                                        future.result()
                            state.write_text(json.dumps({"case": "prefix"}))

                        # A's incomplete last page is not available for B to reuse.
                        for index, shared in enumerate((0, 64, 128)):
                            a = tokens(130, 23 + index * 41)
                            b = a[:shared] + tokens(257 - shared, 149 + index * 19)
                            request((shared, "a"), a)
                            request((shared, "b"), b, shared)

                        # Identical prompts still recompute the uncached final token(s).
                        a = tokens(130, 12)
                        request("tail_a", a)
                        request("tail_b", a, 128)
                        b = tokens(257, 216)
                        request("same_a", b)
                        request("same_b", b, 256)

                        # Two branches reuse A while consuming a shared chunk budget.
                        prefix = tokens(128, 65)
                        request("fork_a", prefix)
                        branches = {
                            "fork_b": prefix + tokens(129, 181),
                            "fork_c": prefix + tokens(129, 202),
                        }
                        synchronized_requests(
                            "fork",
                            [(key, value, 128) for key, value in branches.items()],
                            0,
                        )

                        # A cache hit and a miss enter together after earlier pages release.
                        prefix = tokens(128, 238)
                        request("mixed_a", prefix)
                        synchronized_requests(
                            "mixed",
                            [
                                ("mixed_hit", prefix + tokens(129, 88), 128),
                                ("mixed_miss", tokens(257, 109), 0),
                            ],
                            1,
                        )
                        rows = [
                            json.loads(line) for line in events.read_text().splitlines()
                        ]
                        self.assertTrue(rows)
                        self.assertTrue(
                            all(
                                x["slots_exact"]
                                and x["written_exact"]
                                and x["unwritten_exact"]
                                for x in rows
                            )
                        )
                        if budget == 256:
                            for case in ("fork", "mixed"):
                                self.assertTrue(
                                    any(
                                        x["case"] == case and len(x["lengths"]) == 2
                                        for x in rows
                                    ),
                                    f"{case} must execute a real two-request prefill batch",
                                )
                        if budget:
                            self.assertTrue(
                                all(sum(x["lengths"]) <= budget for x in rows)
                            )
                    finally:
                        manager.stop_server()
                        destination = os.environ.get("MLA_TEST_LIFECYCLE_LOG")
                        if destination and events.exists():
                            Path(
                                destination
                                + f".reuse{int(reuse)}.b{budget}.events.jsonl"
                            ).write_text(events.read_text())

    def test_interrupted_chunks_and_allocation_retry(self):
        tp = int(os.environ.get("MLA_TEST_TP_SIZE", "1"))
        driver = fixture.ChunkedMlaEngineTest()
        reference = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_checkpoint(root)
            config = root / "hooks.json"
            config.write_text(
                json.dumps(
                    {
                        "case": "mla_lifecycle",
                        "function_hooks": [
                            {
                                "target": "rtp_llm.models_py.model_desc.generic_moe.GenericMoeModel.forward",
                                "before": "lifecycle_model_before",
                            },
                            {
                                "target": "rtp_llm.models_py.modules.factory.attention.cuda_mla_impl.flashinfer_mla_wrapper.MlaFlashInferPrefillImpl.forward",
                                "before": "lifecycle_before",
                                "after": "lifecycle_after",
                            },
                        ],
                    }
                )
            )
            for budget in (0, 64):
                state = root / f"state_{budget}.json"
                events = root / f"events_{budget}.jsonl"
                state.write_text(json.dumps({"case": "warmup"}))
                manager = MagaServerManager(
                    env_args={
                        "DETERMINISTIC_GEMM": "1",
                        "ENABLE_STABLE_SCATTER_ADD": "ON",
                        "RTP_HOT_HOOK": "1",
                        "RTP_HOT_HOOK_FILE": os.environ.get(
                            "MLA_TEST_LIFECYCLE_HOOK_FILE",
                            str(Path(__file__).resolve()),
                        ),
                        "RTP_HOT_HOOK_CONFIG": os.environ.get(
                            "MLA_TEST_LIFECYCLE_HOOK_CONFIG", str(config)
                        ),
                        "MLA_TEST_LIFECYCLE_STATE": str(state),
                        "MLA_TEST_LIFECYCLE_EVENTS": str(events),
                    },
                    role_name=f"mla_interrupt_chunk_{budget}",
                    smoke_args_str=(
                        f"--act_type bf16 --tp_size {tp} --world_size {tp} --dp_size 1 --ep_size 1 "
                        "--role_type PDFUSION --enable_cuda_graph 0 --reuse_cache 1 "
                        "--enable_device_cache 1 --enable_memory_cache 0 --enable_remote_cache 0 "
                        "--fp8_kv_cache 0 --seq_size_per_block 64 --kernel_seq_size_per_block 64 "
                        "--test_block_num 8 --max_seq_len 2048 --max_context_batch_size 2 "
                        "--warm_up 0 --frontend_server_count 1 --shutdown_timeout 5 "
                        f"--prefill_chunk_size {budget}"
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
                    driver._request(
                        manager, 130, budget, prompt=" ".join(tokens(130, 6))
                    )
                    for case, seed in (
                        ("cancel", 77),
                        ("timeout", 133),
                        ("pressure", 199),
                    ):
                        state.write_text(
                            json.dumps(
                                {
                                    "case": (
                                        "pressure_setup" if case == "pressure" else case
                                    )
                                }
                            )
                        )
                        prompt = " ".join(tokens(257, seed))
                        with socket.socket() as listener:
                            listener.bind(("127.0.0.1", 0))
                            listener.listen(1)
                            listener.settimeout(30)
                            state.with_suffix(".trigger").write_text(
                                json.dumps(
                                    {
                                        "prefix": (
                                            64 if budget and case != "pressure" else 0
                                        ),
                                        "port": listener.getsockname()[1],
                                    }
                                )
                            )
                            start = time.monotonic()
                            connection = raw_request(
                                manager.port,
                                (
                                    " ".join(tokens(64, 17))
                                    if case == "pressure"
                                    else prompt
                                ),
                                2000 if case == "timeout" else 0,
                            )
                            barrier, _ = listener.accept()
                            barrier.settimeout(30)
                            with barrier.makefile("rb") as stream:
                                observed = json.loads(stream.readline())
                            self.assertEqual(
                                observed["prefixes"],
                                [64 if budget and case != "pressure" else 0],
                            )
                            if budget and case != "pressure":
                                prior = [
                                    json.loads(line)
                                    for line in events.read_text().splitlines()
                                ]
                                self.assertTrue(
                                    any(
                                        x["case"] == case
                                        and x["layer"] == 1
                                        and x["prefixes"] == [0]
                                        and x["lengths"] == [64]
                                        for x in prior
                                    ),
                                    "first chunk must have completed both attention layers",
                                )
                            if case == "cancel":
                                connection.shutdown(socket.SHUT_RDWR)
                                connection.close()
                                # Wait on the backend cancellation event when it can be published
                                # while computation is suspended; release is always bounded.
                                deadline = time.monotonic() + 3
                                while time.monotonic() < deadline:
                                    if "cancelled by user" in Path(
                                        manager.log_file_path
                                    ).read_text(errors="replace"):
                                        break
                                    time.sleep(0.02)
                            elif case == "timeout":
                                # The boundary is known. Wait for this request's deadline,
                                # not for an assumed amount of GPU progress.
                                while time.monotonic() < start + 2.2:
                                    time.sleep(0.02)
                            else:
                                # Queue both 5-page requests behind a held primer. After
                                # it drains, only one fits in the 8-page device cache.
                                primer = connection
                                connection = raw_request(manager.port, prompt)
                                other = raw_request(
                                    manager.port, " ".join(tokens(257, 29))
                                )
                                active = wait_for_enqueued(manager, 3)
                                state.write_text(json.dumps({"case": case}))
                                self.assertTrue(
                                    any(x["waiting"] for x in active), active
                                )
                                report = os.environ.get("MLA_TEST_LIFECYCLE_LOG")
                                if report:
                                    with open(report + ".pressure.jsonl", "a") as out:
                                        out.write(
                                            json.dumps(
                                                {"budget": budget, "tasks": active}
                                            )
                                            + "\n"
                                        )
                            barrier.sendall(b"G")
                            barrier.close()
                            if case == "cancel":
                                deadline = time.monotonic() + 10
                                while "cancelled by user" not in Path(
                                    manager.log_file_path
                                ).read_text(errors="replace"):
                                    self.assertLess(
                                        time.monotonic(),
                                        deadline,
                                        "backend must observe client cancellation",
                                    )
                                    time.sleep(0.02)
                            if case == "timeout":
                                status, body = read_raw_response(connection)
                                connection.close()
                                self.assertTrue(
                                    "603" in body or "timeout" in body.lower(),
                                    (status, body),
                                )
                            elif case == "pressure":
                                primer_status, primer_body = read_raw_response(primer)
                                primer.close()
                                self.assertEqual(primer_status, 200, primer_body)
                                with ThreadPoolExecutor(max_workers=2) as pool:
                                    responses = list(
                                        pool.map(read_raw_response, (connection, other))
                                    )
                                connection.close()
                                other.close()
                                for key, (status, body) in zip(
                                    ("pressure_a", "pressure_b"), responses
                                ):
                                    self.assertEqual(status, 200, body)
                                    frames = [
                                        json.loads(line.removeprefix("data:").strip())
                                        for line in body.splitlines()
                                        if line.startswith("data:")
                                        and line.removeprefix("data:").strip().lower()
                                        != "[done]"
                                    ]
                                    self.assertEqual(len(frames), 6, body)
                                    self.assertTrue(frames[-1]["finished"])
                                    self.assertEqual(
                                        frames[-1]["aux_info"]["reuse_len"], 0
                                    )
                                    ids = [x["output_ids"] for x in frames]
                                    if budget == 0:
                                        reference[key] = ids
                                    else:
                                        self.assertEqual(ids, reference[key])
                        if case in ("cancel", "timeout"):
                            state.write_text(json.dumps({"case": case + "_followup"}))
                            ids = driver._request(
                                manager, 257, budget, prompt=prompt, expected_reuse=0
                            )
                            if budget == 0:
                                reference[case] = ids
                            else:
                                self.assertEqual(ids, reference[case])
                            observed_events = [
                                json.loads(line)
                                for line in events.read_text().splitlines()
                            ]
                            aborted = [x for x in observed_events if x["case"] == case]
                            resumed = [
                                x
                                for x in observed_events
                                if x["case"] == case + "_followup"
                            ]
                            if budget:
                                self.assertTrue(aborted)
                                self.assertTrue(
                                    all(
                                        p + q < 257
                                        for x in aborted
                                        for p, q in zip(x["prefixes"], x["lengths"])
                                    )
                                )

                            def written_pages(records):
                                return {
                                    page
                                    for x in records
                                    for row, prefix, length in zip(
                                        x["blocks"], x["prefixes"], x["lengths"]
                                    )
                                    for page in row[: (prefix + length + 63) // 64]
                                }

                            ranks = {x["pid"] for x in aborted}
                            self.assertEqual(len(ranks), tp)
                            for pid in ranks:
                                old_pages = written_pages(
                                    [x for x in aborted if x["pid"] == pid]
                                )
                                new_pages = written_pages(
                                    [x for x in resumed if x["pid"] == pid]
                                )
                                self.assertTrue(
                                    old_pages & new_pages,
                                    f"follow-up must reuse released physical pages on pid {pid}",
                                )
                    rows = [
                        json.loads(line) for line in events.read_text().splitlines()
                    ]
                    self.assertTrue(rows)
                    self.assertTrue(
                        all(
                            x["slots_exact"]
                            and x["written_exact"]
                            and x["unwritten_exact"]
                            for x in rows
                        )
                    )
                finally:
                    manager.stop_server()
                    destination = os.environ.get("MLA_TEST_LIFECYCLE_LOG")
                    if destination and events.exists():
                        Path(destination + f".b{budget}.events.jsonl").write_text(
                            events.read_text()
                        )


if __name__ == "__main__":
    unittest.main()
