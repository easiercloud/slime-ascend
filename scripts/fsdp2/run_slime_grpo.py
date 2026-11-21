"""Run Slime's real Ray loop with fixed GRPO samples and a cold HF receiver."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=2)
    options = parser.parse_args()
    if options.rollouts < 2:
        parser.error("At least two GRPO updates are required")
    options.output_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="slime-grpo-", dir=options.output_root)).resolve()
    repo = Path(__file__).resolve().parents[2]
    paths = [str(repo / "slime"), str(repo / "scripts/fsdp2")]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    print(f"Slime GRPO run directory: {run}", flush=True)

    import ray
    from train import main as train_main
    from slime.observability.logging_utils import configure_logger
    from slime.utils.arguments import parse_args

    arguments = ["--train-backend", "fsdp2", "--fsdp-fixed-rollout", "--fsdp-random-init",
                 "--fsdp-model-config", str(repo / "scripts/fsdp2/qwen3_8b_2layer.json"),
                 "--rollout-batch-size", "4", "--n-samples-per-prompt", "4",
                 "--global-batch-size", "16", "--micro-batch-size", "1",
                 "--num-rollout", str(options.rollouts), "--update-weight-disk-dir", str(run / "weights")]
    sys.argv = [str(repo / "slime/train.py"), *arguments]
    args, restore_plan = parse_args(return_restore_plan=True)
    args.fsdp_report_dir = str(run / "rank-reports")
    (run / "arguments.json").write_text(json.dumps(vars(args), default=str, indent=2) + "\n")
    configure_logger()
    ray_dir = tempfile.mkdtemp(prefix="slime-ray-", dir="/tmp")
    try:
        ray.init(address="local", num_cpus=16, resources={"NPU": 8},
                 include_dashboard=False, object_store_memory=512 * 1024**2, _temp_dir=ray_dir)
        train_main(args, restore_plan)
    finally:
        ray.shutdown()
    ranks = [json.loads((run / "rank-reports" / f"rank{rank:02d}.json").read_text()) for rank in range(8)]
    for rank, report in enumerate(ranks):
        if report["rank"] != rank or report["data_domain"] != 8 or report["mesh"] != [1, 8]:
            raise RuntimeError("Rank identity or FSDP mesh differs from the eight-rank plan")
        if report["global_shape"] != [4096, 4096] or report["local_shape"] != [512, 4096]:
            raise RuntimeError("The Q projection is not really sharded eight ways")
        if len(report["steps"]) != options.rollouts:
            raise RuntimeError("Missing GRPO optimizer steps")
        for index, step in enumerate(report["steps"]):
            if (step["rollout_id"] != index or step["optimizer_step"] != index + 1
                    or step["rank"] != rank or step["microbatches"] != 2):
                raise RuntimeError("Incorrect GRPO step, data partition or accumulation schedule")
            if not all(math.isfinite(step[key]) for key in ("loss", "global_grad_norm", "q_projection_update_l2")):
                raise RuntimeError("Nonfinite GRPO metric")
            if step["global_grad_norm"] <= 0 or step["q_projection_update_l2"] <= 0:
                raise RuntimeError("No real GRPO parameter update")
    receipts = [json.loads((run / "weights" / f"weight_v{version:06d}" / "receiver-receipt.json").read_text())
                for version in range(options.rollouts + 1)]
    if any(receipt["weight_version"] != version or not receipt["verified"]
           for version, receipt in enumerate(receipts)):
        raise RuntimeError("HF weight receiver did not accept every version")
    summary = {"scope": "Slime fixed-rollout FSDP2 GRPO and offline HF cold-load receiver",
               "online_generation": False, "completed_ranks": 8, "rollouts": options.rollouts,
               "optimizer_steps_per_rank": options.rollouts, "steps": ranks[0]["steps"],
               "received_weight_versions": [receipt["weight_version"] for receipt in receipts],
               "receiver_receipts": receipts, "ray_temp_dir": ray_dir}
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Slime GRPO report: {run / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
