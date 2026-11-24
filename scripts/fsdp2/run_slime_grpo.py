"""Run Slime's real Ray loop with fixed GRPO samples and a cold HF receiver."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--precision-audit", action="store_true")
    parser.add_argument("--operator-audit", action="store_true")
    return parser


def validate_options(parser, options):
    if options.rollouts < 2:
        parser.error("At least two GRPO updates are required")
    if options.operator_audit and not options.precision_audit:
        parser.error("--operator-audit requires --precision-audit")
    return options


def expected_parameter_shapes(config):
    if (config.get("model_type") != "qwen3" or config.get("num_hidden_layers") != 2
            or config.get("attention_bias", False) or config.get("tie_word_embeddings", False)):
        raise ValueError("The measured backend requires the dense two-layer, 25-parameter Qwen3 configuration")
    hidden, heads = int(config["hidden_size"]), int(config["num_attention_heads"])
    head = int(config.get("head_dim", hidden // heads))
    kv = int(config.get("num_key_value_heads", heads)) * head
    intermediate, vocab = int(config["intermediate_size"]), int(config["vocab_size"])
    q = heads * head
    shapes = {"model.embed_tokens.weight": [vocab, hidden], "model.norm.weight": [hidden],
              "lm_head.weight": [vocab, hidden]}
    for layer in range(2):
        prefix = f"model.layers.{layer}."
        shapes.update({prefix + name: shape for name, shape in {
            "self_attn.q_proj.weight": [q, hidden], "self_attn.k_proj.weight": [kv, hidden],
            "self_attn.v_proj.weight": [kv, hidden], "self_attn.o_proj.weight": [hidden, q],
            "self_attn.q_norm.weight": [head], "self_attn.k_norm.weight": [head],
            "mlp.gate_proj.weight": [intermediate, hidden], "mlp.up_proj.weight": [intermediate, hidden],
            "mlp.down_proj.weight": [hidden, intermediate], "input_layernorm.weight": [hidden],
            "post_attention_layernorm.weight": [hidden]}.items()})
    if len(shapes) != 25 or any(shape[0] <= 0 or shape[0] % 8 or any(d <= 0 for d in shape) for shape in shapes.values()):
        raise ValueError("Every one of the 25 parameter shapes must support eight even shards")
    return shapes


def build_arguments(repo, run, options, config_path):
    env = {}
    if options.precision_audit:
        env.update(SLIME_FSDP2_PRECISION_AUDIT="1", SLIME_FSDP2_AUDIT_DIR=str(run / "precision-audit"))
    if options.operator_audit:
        env["SLIME_FSDP2_OPERATOR_AUDIT"] = "1"
    return ["--train-backend", "fsdp2", "--fsdp-fixed-rollout", "--fsdp-random-init",
            "--fsdp-model-config", str(config_path), "--rollout-batch-size", "4", "--n-samples-per-prompt", "4",
            "--global-batch-size", "16", "--micro-batch-size", "1", "--num-rollout", str(options.rollouts),
            "--update-weight-disk-dir", str(run / "weights"), "--train-env-vars", json.dumps(env, sort_keys=True)]


def audit_export_storage(model_path, expected_shapes=None):
    """Check actual headers and, when enabled, stream all saved tensor bytes."""
    root = Path(model_path)
    manifest = json.loads((root / "manifest.json").read_text())
    if expected_shapes is not None and {name: value["shape"] for name, value in manifest["state"].items()} != expected_shapes:
        raise RuntimeError("Export inventory/shapes differ from the supplied model configuration")
    hashes = manifest.get("precision_audit", {}).get("full_parameter_sha256")
    if hashes is not None and set(hashes) != set(manifest["state"]):
        raise RuntimeError("Saved full tensor SHA inventory is incomplete")
    tensors = {}
    for name in manifest["files"]:
        if not name.endswith(".safetensors"):
            continue
        with (root / name).open("rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            if not 0 < length < 8 * 1024**2:
                raise RuntimeError("Unexpected safetensors header length")
            header = json.loads(stream.read(length))
            payload_start = 8 + length
            for key, metadata in header.items():
                if key == "__metadata__":
                    continue
                if key in tensors or key not in manifest["state"] or metadata["dtype"] != "F32":
                    raise RuntimeError("Duplicate/unexpected tensor or non-FP32 checkpoint storage")
                if (metadata["shape"] != manifest["state"][key]["shape"]
                        or manifest["state"][key]["dtype"] != "torch.float32"):
                    raise RuntimeError("Safetensors storage shape disagrees with the manifest")
                result = {"dtype": metadata["dtype"], "shape": metadata["shape"]}
                if hashes is not None:
                    begin, end = metadata["data_offsets"]
                    if begin < 0 or end - begin != 4 * math.prod(metadata["shape"]):
                        raise RuntimeError("Invalid FP32 safetensors payload size")
                    stream.seek(payload_start + begin)
                    digest = hashlib.sha256()
                    remaining = end - begin
                    while remaining:
                        block = stream.read(min(8 * 1024**2, remaining))
                        if not block:
                            raise RuntimeError("Truncated saved tensor payload")
                        digest.update(block)
                        remaining -= len(block)
                    if digest.hexdigest() != hashes[key]:
                        raise RuntimeError("Actual saved tensor bytes differ from the live/gathered/cold-loaded SHA")
                    result["sha256"] = digest.hexdigest()
                tensors[key] = result
    if set(tensors) != set(manifest["state"]):
        raise RuntimeError("Safetensors storage is not a complete model")
    return {"weight_version": manifest["weight_version"], "tensors": tensors,
            "full_tensor_bytes_verified": hashes is not None}


def _read(path):
    return json.loads(Path(path).read_text())


def _tensor(record, shape, *, dtype="torch.float32"):
    digest = record.get("sha256")
    if (record.get("dtype") != dtype or record.get("shape") != shape or record.get("finite") is not True
            or not isinstance(digest, str) or len(digest) != 64 or any(x not in "0123456789abcdef" for x in digest)):
        raise RuntimeError("An actual tensor audit has invalid dtype/shape/finite/SHA coverage")


def _shards(records, shapes, rank):
    if set(records) != set(shapes):
        raise RuntimeError("Live parameter/shard audit does not cover all 25 model parameters")
    for name, shape in shapes.items():
        record = records[name]
        _tensor(record, [shape[0] // 8, *shape[1:]])
        if (not record.get("sharded") or record.get("global_shape") != shape
                or record.get("mesh_shape") != [1, 8] or record.get("mesh_coordinate") != [0, rank]
                or record.get("placement_specs") != [{"kind": "Replicate"}, {"kind": "Shard", "dim": 0}]):
            raise RuntimeError("An actual parameter placement differs from the eight-rank native FSDP plan")


def _buffers(records):
    if set(records) != {"model.rotary_emb.inv_freq", "model.rotary_emb.original_inv_freq"}:
        raise RuntimeError("Sensitive RoPE buffer inventory is incomplete")
    for record in records.values():
        _tensor(record, record["shape"])


def verify_precision_audits(run, rollouts, shapes, manifests, receipts, *, operator_audit=False):
    root = Path(run) / "precision-audit"
    steps = []
    observed_samples = {step: [] for step in range(1, rollouts + 1)}
    for rank in range(8):
        initial = _read(root / f"rank{rank:02d}-initial.json")
        _shards(initial["parameters"], shapes, rank)
        _buffers(initial["buffers"])
        if set(initial["optimizer"]) != set(shapes) or any(initial["optimizer"].values()):
            raise RuntimeError("Initial Adam state inventory must be complete and uninitialized")
        rank_parameters = [initial["parameters"]]
        for step in range(1, rollouts + 1):
            record = _read(root / f"rank{rank:02d}-step{step:02d}.json")
            if record["rank"] != rank or record["world_size"] != 8 or record["step"] != step:
                raise RuntimeError("Incorrect actual precision-audit rank or step")
            _shards(record["parameters"], shapes, rank)
            _buffers(record["buffers"])
            for category in ("gradients_pre_clip", "gradients_post_clip"):
                if set(record[category]) != set(shapes):
                    raise RuntimeError("Actual gradient audit does not cover all 25 trainable parameters")
                for name, value in record[category].items():
                    _tensor(value, [shapes[name][0] // 8, *shapes[name][1:]])
                    if value.get("missing") is not False:
                        raise RuntimeError("A trainable parameter has no actual gradient")
            if set(record["optimizer"]) != set(shapes):
                raise RuntimeError("Actual Adam audit does not cover all 25 parameters")
            for name, state in record["optimizer"].items():
                if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise RuntimeError("Actual Adam tensor inventory is incomplete")
                _tensor(state["step"], [])
                if state["step"].get("value") != step:
                    raise RuntimeError("Actual Adam step tensor differs from the optimizer step")
                for key in ("exp_avg", "exp_avg_sq"):
                    _tensor(state[key], [shapes[name][0] // 8, *shapes[name][1:]])
            current_phases = {f"step{step:02d}_current_mb{mb:02d}" for mb in range(2)}
            allowed_phases = current_phases | {f"step{step:02d}_old_mb{mb:02d}" for mb in range(2)}
            event_groups = [record["module_probes"], record["collective_probes"]]
            if operator_audit:
                event_groups.append(record.get("operator_probes") or [])
            if any(event.get("phase") not in allowed_phases for events in event_groups for event in events):
                raise RuntimeError("Actual precision observation has an unexpected step/microbatch phase")
            if record["observed_policy"].get("passed") is not True:
                raise RuntimeError("Actual compute/wire policy did not accept the step")
            for phase in sorted(current_phases):
                seen = set()
                for event in record["module_probes"]:
                    if event["phase"] != phase or event.get("event") != "input":
                        continue
                    for key, value in event["parameters"].items():
                        name = event["module"] + "." + key
                        if (name not in shapes or value["dtype"] != "torch.bfloat16" or value["finite"] is not True
                                or value["shape"] != shapes[name]):
                            raise RuntimeError("Actual forward compute parameter differs from finite full BF16")
                        seen.add(name)
                if seen != set(shapes):
                    raise RuntimeError("An actual current microbatch did not observe all 25 compute parameters")
                collectives = [event for event in record["collective_probes"] if event["phase"] == phase]
                for name, dtype in (("all_gather_into_tensor", "torch.bfloat16"), ("reduce_scatter_tensor", "torch.float32")):
                    calls = [event for event in collectives if event["collective"] == name]
                    if not calls or any(not event["tensors"] or any(value["dtype"] != dtype for value in event["tensors"]) for event in calls):
                        raise RuntimeError("An actual current microbatch lacks BF16 gather/FP32 reduce wire evidence")
                if operator_audit:
                    operators = [event for event in record["operator_probes"] if event["phase"] == phase]
                    if not {"mean", "rsqrt", "softmax", "log_softmax", "sin", "cos"} <= {event["family"] for event in operators}:
                        raise RuntimeError("An actual current microbatch lacks sensitive operator evidence")
                    for event in operators:
                        parts = event["operator"].split(".")
                        if len(parts) < 3 or parts[0] != "aten" or parts[1].lstrip("_") != event["family"]:
                            raise RuntimeError("Sensitive operator name does not match the actual ATen family")
                        tensors = event["inputs"] + event["outputs"]
                        if not tensors or any(value["dtype"] != "torch.float32" for value in tensors):
                            raise RuntimeError("An actual sensitive forward operator did not execute in FP32")
                    if (record.get("observed_sensitive_operator_policy") or {}).get("passed") is not True:
                        raise RuntimeError("Sensitive operator audit did not accept the actual step")
            logprobs = record["response_logprob_microbatches"]
            if len(logprobs) != 2 or {value["microbatch"] for value in logprobs} != {0, 1}:
                raise RuntimeError("Actual forward logprob observations are missing a microbatch")
            for observation in logprobs:
                standalone = _read(root / f"rank{rank:02d}-step{step:02d}-mb{observation['microbatch']:02d}-logprobs.json")
                if (standalone != observation or observation["rank"] != rank or observation["step"] != step
                        or observation["accepted_weight_version"] != step - 1
                        or observation["old_logprob_source"] != "current_policy_step_start_snapshot"
                        or observation["accepted_manifest"] != manifests[step - 1]
                        or len(observation["samples"]) != 1):
                    raise RuntimeError("Actual fixed-rollout LP observation has inconsistent identity or source")
                sample = observation["samples"][0]
                observed_samples[step].append((sample["rollout_position"], sample["sample_rollout_id"], sample["sample_index"]))
                length = len(sample["response_token_ids"])
                fields = ("current_log_probs", "old_log_probs", "difference", "ratio", "loss_mask")
                if (not length or not sample["prompt_token_ids"]
                        or any(len(sample[field]) != length for field in fields)
                        or any(not math.isfinite(value) for field in fields for value in sample[field])
                        or sample["current_dtype"] != "torch.float32" or sample["old_dtype_used_by_loss"] != "torch.float32"
                        or observation["numeric_acceptance"] != "not_evaluated_without_independent_HF_calibration"):
                    raise RuntimeError("Actual response LP/mask/ratio evidence is incomplete or falsely labeled")
            global_lp = record["global_response_logprob_observations"]
            if global_lp["samples"] != 16 or len(global_lp["ranks"]) != 8 or global_lp["valid_tokens"] <= 0:
                raise RuntimeError("Actual LP observation did not cover the full global batch")
            source_ranks = global_lp["ranks"]
            source_ids = [identity for source in source_ranks for identity in source["sample_rollout_ids"]]
            if ({source["rank"] for source in source_ranks} != set(range(8)) or len(source_ids) != 16
                    or len(set(source_ids)) != 16
                    or global_lp["numeric_acceptance"] != "not_evaluated_without_independent_HF_calibration"):
                raise RuntimeError("Global LP observation contains missing/duplicate samples or an unsupported numeric claim")
            rank_parameters.append(record["parameters"])
            steps.append({"rank": rank, "step": step, "parameters": 25, "gradient_tensors": 50,
                          "adam_tensors": 75, "operator_fp32_verified": operator_audit})
        for version, manifest in enumerate(manifests):
            live = _read(root / f"rank{rank:02d}-weight-v{version:06d}-live-shards.json")
            _shards(live, shapes, rank)
            audit = manifest["precision_audit"]
            if (audit["world_size"] != 8 or audit["parameter_count"] != 25 or audit["shard_count"] != 200
                    or audit["gather_verified"] is not True or len(audit["rank_shards"]) != 8
                    or audit["rank_shards"][rank] != live
                    or any(live[name]["sha256"] != rank_parameters[version][name]["sha256"] for name in shapes)):
                raise RuntimeError("Published/gathered shards differ from the actual initial or updated parameter bytes")
    for identities in observed_samples.values():
        if (len(identities) != 16 or {value[0] for value in identities} != set(range(16))
                or len({value[1] for value in identities}) != 16 or len({value[2] for value in identities}) != 16
                or any(value[1] is None or value[2] is None for value in identities)):
            raise RuntimeError("Actual response LP records contain missing/duplicate sample identities")
    for receipt in receipts:
        audit = receipt.get("precision_audit", {})
        if audit.get("verified") is not True or audit.get("parameter_count") != 25 or audit.get("shard_count") != 200:
            raise RuntimeError("HF cold-load receiver did not verify all 25 x 8 actual shard byte hashes")
    q_name = "model.layers.0.self_attn.q_proj.weight"
    if any(manifests[version]["precision_audit"]["full_parameter_sha256"][q_name]
           == manifests[version - 1]["precision_audit"]["full_parameter_sha256"][q_name]
           for version in range(1, len(manifests))):
        raise RuntimeError("Published Q projection tensor bytes did not change after a claimed optimizer update")
    return {"verified": True, "ranks": 8, "steps": steps, "parameter_count": 25,
            "published_versions": len(manifests), "shards_verified_per_version": 200,
            "operator_audit_enabled": operator_audit,
            "scope": "Actual dtype/finite/tensor coverage and complete live/gathered/saved/coldloaded bytes; no numeric-oracle claim"}


def verify_run(run, rollouts, config, *, precision_audit=False, operator_audit=False):
    run = Path(run)
    shapes = expected_parameter_shapes(config)
    q_shape = shapes["model.layers.0.self_attn.q_proj.weight"]
    ranks = [_read(run / "rank-reports" / f"rank{rank:02d}.json") for rank in range(8)]
    for rank, report in enumerate(ranks):
        if report["rank"] != rank or report["data_domain"] != 8 or report["mesh"] != [1, 8]:
            raise RuntimeError("Rank identity or FSDP mesh differs from the eight-rank plan")
        if report["global_shape"] != q_shape or report["local_shape"] != [q_shape[0] // 8, q_shape[1]]:
            raise RuntimeError("The configured Q projection is not really sharded eight ways")
        if len(report["steps"]) != rollouts:
            raise RuntimeError("Missing GRPO optimizer steps")
        for index, step in enumerate(report["steps"]):
            if (step["rollout_id"] != index or step["optimizer_step"] != index + 1
                    or step["rank"] != rank or step["microbatches"] != 2):
                raise RuntimeError("Incorrect GRPO step, data partition or accumulation schedule")
            if not all(math.isfinite(step[key]) for key in ("loss", "global_grad_norm", "q_projection_update_l2")):
                raise RuntimeError("Nonfinite GRPO metric")
            if step["global_grad_norm"] <= 0 or step["q_projection_update_l2"] <= 0:
                raise RuntimeError("No real GRPO parameter update")
    receipts = [_read(run / "weights" / f"weight_v{version:06d}" / "receiver-receipt.json") for version in range(rollouts + 1)]
    if any(receipt["weight_version"] != version or receipt.get("verified") is not True
           or Path(receipt["model_path"]).resolve() != (run / "weights" / f"weight_v{version:06d}").resolve()
           for version, receipt in enumerate(receipts)):
        raise RuntimeError("HF weight receiver did not accept every exact version/path")
    manifests = [_read(Path(receipt["model_path"]) / "manifest.json") for receipt in receipts]
    if any(manifest["weight_version"] != version for version, manifest in enumerate(manifests)):
        raise RuntimeError("Export manifest has an inconsistent version")
    storage = [audit_export_storage(receipt["model_path"], shapes) for receipt in receipts]
    (run / "export-storage-audit.json").write_text(json.dumps(storage, indent=2) + "\n")
    precision = (verify_precision_audits(run, rollouts, shapes, manifests, receipts, operator_audit=operator_audit)
                 if precision_audit else None)
    return {"scope": "Slime fixed-rollout FSDP2 GRPO and offline HF cold-load receiver",
            "online_generation": False, "completed_ranks": 8, "rollouts": rollouts,
            "optimizer_steps_per_rank": rollouts, "steps": ranks[0]["steps"],
            "received_weight_versions": [receipt["weight_version"] for receipt in receipts],
            "receiver_receipts": receipts, "storage_audited": True, "precision_audit": precision,
            "runtime": manifests[0].get("runtime", {}), "model_parameter_shapes": shapes}


def main():
    parser = get_parser()
    options = validate_options(parser, parser.parse_args())
    repo = Path(__file__).resolve().parents[2]
    config_path = (options.model_config or repo / "scripts/fsdp2/qwen3_8b_2layer.json").resolve()
    config = _read(config_path)
    expected_parameter_shapes(config)
    options.output_root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="slime-grpo-", dir=options.output_root)).resolve()
    paths = [str(repo / "slime"), str(repo / "scripts/fsdp2")]
    sys.path[:0] = paths
    os.environ["PYTHONPATH"] = os.pathsep.join(paths + [os.environ.get("PYTHONPATH", "")])
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    print(f"Slime GRPO run directory: {run}", flush=True)

    import ray
    from train import main as train_main
    from slime.observability.logging_utils import configure_logger
    from slime.utils.arguments import parse_args

    arguments = build_arguments(repo, run, options, config_path)
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
    summary = verify_run(run, options.rollouts, config, precision_audit=options.precision_audit,
                         operator_audit=options.operator_audit)
    summary["ray_temp_dir"] = ray_dir
    (run / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Slime GRPO report: {run / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
