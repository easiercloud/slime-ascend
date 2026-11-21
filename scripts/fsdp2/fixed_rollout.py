"""Deterministic offline Samples for testing the real Slime GRPO data path."""

import json
import random
from pathlib import Path


class FixedDataSource:
    """Data-source lifecycle for synthetic samples that need no tokenizer."""

    def __init__(self, args):
        self.args = args
        self.consumers = {}

    def load(self, rollout_id):
        return None

    def save(self, rollout_id):
        raise NotImplementedError("The fixed fixture does not implement resumable checkpoints")

    def close(self):
        return None


def generate_rollout(args, rollout_id, data_source, *, evaluation=False):
    """Return fixed grouped responses; old logprobs are measured by the actor."""
    from slime.rollout.base_types import RolloutFnTrainOutput
    from slime.utils.types import Sample

    if evaluation:
        raise NotImplementedError("The offline fixture has no evaluation generator")
    config = json.loads(Path(args.fsdp_model_config).read_text())
    groups = []
    for group in range(args.rollout_batch_size):
        samples = []
        prompt_rng = random.Random(1234 + group)
        prompt = [prompt_rng.randrange(10, config["vocab_size"]) for _ in range(64)]
        for response in range(args.n_samples_per_prompt):
            index = group * args.n_samples_per_prompt + response
            rng = random.Random(5678 + index)
            tokens = [rng.randrange(10, config["vocab_size"]) for _ in range(32)]
            mask = [1] * 32
            if response % 2:
                mask[-2:] = [0, 0]
            samples.append(Sample(
                group_index=group,
                index=index,
                rollout_id=index,
                prompt="fixed offline prompt",
                tokens=prompt + tokens,
                response="fixed offline response",
                response_length=len(tokens),
                reward=float(response),
                loss_mask=mask,
                status=Sample.Status.COMPLETED,
                weight_versions=[str(args.fsdp_weight_version)],
            ))
        groups.append(samples)
    return RolloutFnTrainOutput(samples=groups, metrics={"offline_fixed_samples": 1})
