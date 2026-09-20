"""Benchmark the NCCL all_gather allocation change on one multi-GPU node.

Run with ``torchrun --standalone --nproc_per_node=8 all_gather_benchmark.py``.
The RTP process group uses ``--rtp-port`` independently of torchrun's store.
Both paths allocate their output inside the timed region; the input is reused.
CUDA events measure stream latency (including collective rank skew and launch
gaps), not the sum of individual GPU kernel durations.
"""

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist

from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    _get_group,
    all_gather,
    destroy_distributed_environment,
    init_distributed_environment,
)
from rtp_llm.ops import NcclCommConfig, ParallelismConfig


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def summarize(samples):
    return {
        "median_us": statistics.median(samples),
        "p90_us": percentile(samples, 0.9),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def baseline_zeroed(local, process_group, world_size):
    output = torch.zeros(
        (world_size * local.shape[0], *local.shape[1:]),
        dtype=local.dtype,
        device=local.device,
    )
    dist.all_gather_into_tensor(output, local, group=process_group)
    return output


def check_output(output, rows, hidden, rank, world_size, row_values):
    assert output.shape == (world_size * rows, hidden)
    assert output.dtype == torch.bfloat16
    for source_rank, chunk in enumerate(output.split(rows)):
        expected = (row_values[:, None] + source_rank * 100).to(torch.bfloat16)
        assert torch.equal(
            chunk, expected.expand(rows, hidden)
        ), f"rank {rank}, source {source_rank}: gathered data mismatch"


def measure(path, local, process_group, local_rank):
    torch.cuda.synchronize()
    dist.barrier(group=process_group, device_ids=[local_rank])
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = path(local)
    end.record()
    torch.cuda.synchronize()
    duration_us = start.elapsed_time(end) * 1000.0
    del output
    return duration_us


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, nargs="+", default=[257, 8191, 8192])
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rtp-port", type=int, default=29611)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.rows) <= 0 or args.hidden <= 0 or args.iterations <= 0:
        parser.error("rows, hidden, and iterations must be positive")
    if args.warmup < 0:
        parser.error("warmup must be nonnegative")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    config = ParallelismConfig()
    config.world_rank = rank
    config.world_size = world_size
    config.local_rank = local_rank
    config.tp_size = world_size
    config.dp_size = 1
    base_port = args.rtp_port + 11
    # torchrun's agent store uses its own rendezvous port. RTP initializes an
    # independent TCPStore below, so rank 0 must host that store for this run.
    os.environ["TORCHELASTIC_USE_AGENT_STORE"] = "False"
    init_distributed_environment(
        config,
        nccl_comm_config=NcclCommConfig(
            nccl_ip="127.0.0.1",
            tp_nccl_port=base_port - 2,
            dp_tp_nccl_port=base_port - 10,
            ffn_tp_nccl_port=base_port - 5,
        ),
        nccl_init_port=args.rtp_port,
        backend="nccl",
        timeout=120,
    )
    try:
        # With TP=world_size, DP_AND_TP resolves to the same NCCL ranks as TP,
        # while bypassing the optional TP symmetric-memory implementation.
        process_group = _get_group(Group.DP_AND_TP)
        results = []
        for rows in args.rows:
            row_values = torch.arange(rows, device=f"cuda:{local_rank}") % 97
            local = (row_values[:, None] + rank * 100).to(torch.bfloat16)
            local = local.expand(rows, args.hidden).contiguous()
            paths = {
                "baseline_zeros": lambda x: baseline_zeroed(
                    x, process_group, world_size
                ),
                "candidate_empty": lambda x: all_gather(x, Group.DP_AND_TP),
            }
            for name, path in paths.items():
                check_output(
                    path(local), rows, args.hidden, rank, world_size, row_values
                )
                torch.cuda.synchronize()
                for _ in range(args.warmup):
                    output = path(local)
                    torch.cuda.synchronize()
                    del output
            samples = {name: [] for name in paths}
            for iteration in range(args.iterations):
                order = tuple(paths) if iteration % 2 == 0 else tuple(reversed(paths))
                for name in order:
                    samples[name].append(
                        measure(paths[name], local, process_group, local_rank)
                    )
            local_result = {
                "rank": rank,
                "rows_per_rank": rows,
                "hidden": args.hidden,
                "output_bytes_per_rank": world_size * rows * args.hidden * 2,
                "paths": {name: summarize(data) for name, data in samples.items()},
                "samples_us": samples,
            }
            gathered = [None] * world_size
            dist.all_gather_object(gathered, local_result, group=process_group)
            if rank == 0:
                critical_path = {
                    name: summarize(
                        [
                            max(item["samples_us"][name][i] for item in gathered)
                            for i in range(args.iterations)
                        ]
                    )
                    for name in paths
                }
                for item in gathered:
                    del item["samples_us"]
                results.append(
                    {
                        "shape": [rows, args.hidden],
                        "critical_path_us": critical_path,
                        "ranks": gathered,
                    }
                )
            dist.barrier(group=process_group, device_ids=[local_rank])
        if rank == 0:
            report = {
                "meta": {
                    "gpu": torch.cuda.get_device_name(local_rank),
                    "world_size": world_size,
                    "dtype": "bfloat16",
                    "baseline": "torch.zeros + all_gather_into_tensor",
                    "candidate": "RTP all_gather (torch.empty + all_gather_into_tensor)",
                    "torch_compile": False,
                    "input_allocation_timed": False,
                    "output_allocation_timed": True,
                    "timing": "CUDA event stream latency, not pure kernel time",
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "warmup": args.warmup,
                    "iterations": args.iterations,
                },
                "results": results,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
    finally:
        destroy_distributed_environment()


if __name__ == "__main__":
    main()
