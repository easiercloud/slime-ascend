# NPU validation matrix (2026-10-26)

Baseline: branch `slime-latest-20261022`, Slime `18868a79b0b083a093798a94ed0f8cb4f71b9c14`, SGLang v0.5.21, Megatron Core `4b4acac9a1d28ea6829c8d4f566d75698a21249d`.

Current validation host: x86_64 / CANN 9.1.0. The target A3 ARM64 / CANN 9.0.0 image has not been built or tested.

| Tier | Case | NPUs | Status |
| --- | --- | ---: | --- |
| ut | `test_npu_accelerator_cache.py` | CPU | PASS |
| ut | `test_ppo_utils_npu.py` | CPU | PASS |
| st | `test_qwen2.5_0.5B_debug_rollout_then_train_npu.py` | 8 | PENDING |
| st | `test_qwen2.5_0.5B_gsm8k_async_short_npu.py` | 8 | PASS |
| st | `test_qwen2.5_0.5B_opd_sglang_npu.py` | 8 | PENDING |
| nightly_CI | `test_mimo_7B_mtp_only_grad_npu.py` | 16 | PENDING |
| nightly_CI | `test_ppo_logprob_entropy_gpu_npu.py` | 2 | PENDING |
| nightly_CI | `test_quick_start_glm4_9B_npu.py` | 8 | PENDING |
| nightly_CI | `test_qwen2.5_0.5B_fanout_short_npu.py` | 4 | PENDING |
| nightly_CI | `test_qwen2.5_0.5B_fully_async_short_npu.py` | 4 | PENDING |
| nightly_CI | `test_qwen2.5_0.5B_sglang_config_distributed_npu.py` | 8 | PENDING |
| nightly_CI | `test_qwen2.5_0.5B_sglang_config_npu.py` | 8 | PENDING |
| nightly_CI | `test_qwen2.5_0.5B_short_npu.py` | 4 | PASS |
| nightly_CI | `test_qwen3_0.6B_parallel_check_npu.py` | 8 | PENDING |
| nightly_CI | `test_qwen3_30B_A3B_npu.py` | 16 | PENDING |
| nightly_CI | `test_qwen3_4B_ckpt_npu.py` | 8 | PENDING |
| nightly_CI | `test_qwen3_4B_external_pd_npu.py` | 6 | PENDING |
| nightly_CI | `test_qwen3_4B_ppo_disaggregate_npu.py` | 8 | PENDING |
| nightly_CI | `test_qwen3_4B_ppo_npu.py` | 8 | PENDING |
| nightly_CI | `test_qwen3_4B_ppo_train_critic_only_npu.py` | 8 | PENDING |
| nightly_CI | `test_qwen3_4B_streaming_partial_rollout_npu.py` | 8 | PENDING |
| nightly_CI | `test_sglang_config_mixed_offload_ft_npu.py` | 8 | PENDING |
| nightly_CI | `test_sglang_config_mixed_offload_npu.py` | 8 | PENDING |

UT evidence: 11 current NPU tests passed in `/tmp/slime-latest-npu-ut-transformers5121-20261026.log`; 113 selected patched-upstream CPU tests and 41 SGLang compatibility tests also passed.

ST evidence: `/tmp/slime-latest-qwen05-async-st-sglang0521-npu8-15-20261026.log` (Ray job succeeded, training step 2 and weight updates).

Nightly evidence: `/tmp/slime-latest-qwen05-nightly-pinned-tms-npu8-11-20261026.log` (Ray job succeeded, training step 0 and weight update).

At this audit all 16 physical NPUs are occupied by another `mindspeed_bridge` training run (PIDs 455469-455491); no NPU test was started. Do not stop that task.

Asset audit: `/home/data/weights` contains the HF model directories, and `/home/data/datasets` contains the listed datasets. The default CI Megatron checkpoint root `/home/slime-ascend_CI/slime-ascend/weights-MA` is absent; only the Qwen2.5-0.5B checkpoint exists at `/home/c0088099/slime-v040-work/weights-MA/Qwen2.5-0.5B-Instruct_torch_dist`. Other cases need their exact checkpoints created or configured before execution.

Run remaining ST cases before remaining Nightly cases when the cards are free, selecting isolated NPU IDs, Ray ports and HCCL ports. Preserve logs and update this matrix case by case. Do not use `st_run.sh` because it contains broad process cleanup.
