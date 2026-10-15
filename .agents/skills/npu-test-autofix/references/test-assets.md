# Local Test Assets

Use local HF weights and datasets. Do not download assets from Hugging Face while running the NPU test workflow.

## Inventory Before Asking

Run the inventory command before requesting any local path:

```bash
python3 .agents/skills/npu-test-autofix/scripts/prepare_test_resources.py \
  --repo-root . \
  --test-root slime/tests \
  --test-root tests \
  --inventory
```

Read `.npu-test-autofix/asset-requirements.json`, show its complete model and dataset requirement list to the user, and ask where those assets are stored locally. Do not infer paths from old NPU tests because locations such as `/efs_rl`, `/home/...`, and `/root/...` may belong to a different machine.

This question is mandatory when any requirement remains unresolved. An unset
environment variable, an empty conventional directory, or a failed filesystem
search does not prove the user has no usable assets elsewhere. Do not finalize
the run or mark affected tests `UNEXECUTED_MISSING_ASSET` before receiving the
answer, unless the user explicitly requested non-interactive execution.

The preferred answer supplies one HF weights root containing a child directory for each model name, one datasets root preserving the listed logical paths, and an existing Megatron checkpoint root when converted checkpoints already exist. If no Megatron root exists, ask the user to confirm that locally available HF weights may be converted into the run's cache. If files are stored elsewhere, collect explicit per-asset paths and adapt only the affected tests.

After the user responds, verify root readability and every required relative
path. For Megatron checkpoints, also verify
`latest_checkpointed_iteration.txt` and its expected `release` or numeric
content. Report only the remaining gaps and ask a focused follow-up instead of
repeating requirements already resolved. A missing-asset classification is
valid only after explicit user confirmation of unavailability, invalid paths
that remain unresolved after follow-up, or an explicitly non-interactive run.

## Required Environment

Set these only after the user confirms the paths:

```bash
export SLIME_TEST_HF_WEIGHTS_ROOT=/local/path/to/hf-weights
export SLIME_TEST_DATASETS_ROOT=/local/path/to/datasets
```

The preparation script creates `SLIME_TEST_MEGATRON_WEIGHTS_ROOT`. For the complete `tests` tree and for suites below it, the default cache is `tests/weights-MA`.

## Canonical NPU Test Form

Adapt each selected NPU test to read paths from the environment:

```python
HF_WEIGHTS_ROOT = os.environ["SLIME_TEST_HF_WEIGHTS_ROOT"]
DATASETS_ROOT = os.environ["SLIME_TEST_DATASETS_ROOT"]
MEGATRON_WEIGHTS_ROOT = os.environ["SLIME_TEST_MEGATRON_WEIGHTS_ROOT"]

hf_checkpoint = os.path.join(HF_WEIGHTS_ROOT, MODEL_NAME)
megatron_checkpoint = os.path.join(
    MEGATRON_WEIGHTS_ROOT, f"{MODEL_NAME}_torch_dist"
)

ckpt_args = (
    f"--hf-checkpoint {hf_checkpoint} "
    f"--ref-load {megatron_checkpoint} "
)
rollout_args = (
    f"--prompt-data {DATASETS_ROOT}/dapo-math-17k/dapo-math-17k.jsonl "
)
```

Preserve dataset-relative suffixes such as:

- `dapo-math-17k/dapo-math-17k.jsonl`
- `gsm8k/train.parquet` and `gsm8k/test.parquet`
- `aime-2024/aime-2024.jsonl`

Remove `prepare()` calls that create `/root/models`, create `/root/datasets`, invoke `hf download`, or download datasets. Keep unrelated preparation such as temporary-file cleanup.

Do not hardcode a developer home directory, `/root/models`, `/root/datasets`, or shared-cluster paths such as `/efs_rl`.

## Conversion Cache

Run `prepare_test_resources.py` once before the baseline. It:

- Scans static `MODEL_NAME`, `MODEL_TYPE`, and `NUM_GPUS` declarations.
- Deduplicates models across tests.
- Reuses an existing `<MODEL_NAME>_torch_dist` directory only after its saved
  TP/PP/EP/world-size layout is confirmed compatible with the requesting test.
- Converts a missing model once per compatible parallel layout and writes it
  under `weights-MA`.
- Uses a per-model lock to prevent concurrent duplicate conversion.
- Writes the resolved paths and conversion status to `.npu-test-autofix/test-assets.json`.

If a test uses non-static model metadata, an unknown checkpoint origin topology,
or a layout different from `<MODEL_NAME>_torch_dist`, do not guess. Ask for the
saved TP/PP/EP/world size, record it in the preparation report, and adapt it
explicitly before running.
