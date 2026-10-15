#!/usr/bin/env python3
"""Regression for allocations created between TMS pause and resume."""

import argparse

import torch
from torch_memory_saver import torch_memory_saver


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("hook_mode", nargs="?", default="hook")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    torch_memory_saver.hook_mode = args.hook_mode
    torch.npu.set_device(args.device)

    original = torch.full((1_000_000,), 17, dtype=torch.int32, device="npu")
    torch.npu.synchronize()

    torch_memory_saver.pause()

    pool = torch.npu.MemPool()
    with torch.npu.use_mem_pool(pool):
        allocated_while_paused = torch.full(
            (1_000_000,), 23, dtype=torch.int32, device="npu"
        )
        torch.npu.synchronize()

        torch_memory_saver.resume()
        torch.npu.synchronize()

        assert original[:4].tolist() == [17] * 4
        assert allocated_while_paused[:4].tolist() == [23] * 4

    print("pause-then-allocate-resume: PASS")


if __name__ == "__main__":
    main()
