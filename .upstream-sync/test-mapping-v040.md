# v0.4.0 变化测试映射初稿

- Base `5bae5bb7928d65906e24e99f5b5d3f99a4daaf93` → target `0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e`。
- 自动差异：新增 29、修改 34、删除 1；只列实际 `test_*.py` 文件，helper 单独作为依赖。
- GPU CI membership 依据 target `.github/workflows/pr-test.yml` 的 `test_file` matrix 精确比对；`NUM_GPUS` 来自 target 测试源码。
- 映射状态只是候选，行为契约/参数化 topology 需逐项审查后才能转成 NPU 用例。

| Diff | 上游测试 | GPU CI matrix | NUM_GPUS | 现有 NPU 对应 / 后续决定 |
|---|---|---:|---:|---|
| MODIFIED | tests/plugin_contracts/test_plugin_generate_contracts.py | no / helper or non-matrix | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/plugin_contracts/test_plugin_path_loading_contracts.py | no / helper or non-matrix | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/plugin_contracts/test_plugin_rollout_contracts.py | no / helper or non-matrix | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/plugin_contracts/test_plugin_runtime_hook_contracts.py | no / helper or non-matrix | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_accelerator.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_disk_delta_checkpoint_index.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_disk_delta_recovery.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_distributed_rollout.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_fully_async_rollout.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_glm52_6layer_deterministic_e2e.py | yes | 8 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_hf_to_megatron.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_layerwise_alignment.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_megatron_argument_validation.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_megatron_checkpoint.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_megatron_tokenizer_init.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_mimo_7B_mtp_only_grad.py | yes | 8 | tests/tests_npu/nightly_CI/test_mimo_7B_mtp_only_grad_npu.py |
| MODIFIED | tests/test_model_provider_freeze.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_moonlight_16B_A3B_r3.py | yes | 8 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_optional_straw.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_pipeline_rl.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_published_payload_retry.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_queue_sample_codec.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_qwen2.5_0.5B_debug_rollout_then_train.py | yes | 8 | tests/tests_npu/st/test_qwen2.5_0.5B_debug_rollout_then_train_npu.py |
| MODIFIED | tests/test_qwen2.5_0.5B_fanout_short.py | yes | 4 | tests/tests_npu/nightly_CI/test_qwen2.5_0.5B_fanout_short_npu.py |
| MODIFIED | tests/test_qwen2.5_0.5B_fully_async_short.py | yes | 4 | tests/tests_npu/nightly_CI/test_qwen2.5_0.5B_fully_async_short_npu.py |
| NEW | tests/test_qwen2.5_0.5B_pipeline_rl.py | yes | 4 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_qwen2.5_0.5B_rollout_health.py | yes | 4 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_qwen2.5_0.5B_score_centering.py | yes | 2 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_qwen2.5_0.5B_sglang_config_distributed.py | yes | 8 | tests/tests_npu/nightly_CI/test_qwen2.5_0.5B_sglang_config_distributed_npu.py |
| NEW | tests/test_qwen2.5_0.5B_training_recovery.py | yes | 4 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_qwen3.6_35B_A3B_pd_mooncake.py | yes | 8 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_qwen3_0.6B_parallel_check.py | yes | 8 | tests/tests_npu/nightly_CI/test_qwen3_0.6B_parallel_check_npu.py |
| MODIFIED | tests/test_qwen3_30B_A3B_r3.py | yes | 8 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_qwen3_30B_A3B_training_recovery.py | yes | 8 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_qwen3_4B_ckpt.py | yes | 8 | tests/tests_npu/nightly_CI/test_qwen3_4B_ckpt_npu.py |
| MODIFIED | tests/test_qwen3_4B_ppo.py | yes | 8 | tests/tests_npu/nightly_CI/test_qwen3_4B_ppo_npu.py |
| DELETED | tests/test_r3_spill.py | no / helper or non-matrix | unknown | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_reference_top_p_replay.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_release_train.py | yes | 4 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_rollout_buffer_order.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_rollout_manager_recovery.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_rollout_metadata_index.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_rollout_sample_hooks.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_rollout_straw_transport.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_routed_experts_disk_layout.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_score_centering.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_score_centering_binary.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_score_centering_transport.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_server_control.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_sglang_config_mixed_offload_ft.py | yes | 8 | tests/tests_npu/nightly_CI/test_sglang_config_mixed_offload_ft_npu.py |
| MODIFIED | tests/test_stateless_adam.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_straw_checkpoint_fork.py | yes | 4 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_straw_fully_async_recovery.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_straw_r3.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_streaming_rollout.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| NEW | tests/test_training_recovery.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/test_update_weight_factory.py | yes | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/utils/test_megatron_role_config.py | no / helper or non-matrix | 0 | no existing exact counterpart; review contract and create only if applicable |
| MODIFIED | tests/utils/test_sglang_config.py | no / helper or non-matrix | 0 | no existing exact counterpart; review contract and create only if applicable |


## 2026-10-15 纯 UT 执行与迁移筛选

- NPU UT：`tests/tests_npu/ut/test_npu_accelerator_cache.py` + `test_ppo_utils_npu.py`，**11 passed**，112.89s；14 条 torch.jit 弃用 warning。
- v0.4 新增纯 CPU/无模型候选：`test_megatron_checkpoint.py`、`test_megatron_tokenizer_init.py`、`test_disk_delta_checkpoint_index.py`、`test_reference_top_p_replay.py`、`test_rollout_metadata_index.py`、`test_published_payload_retry.py`、`test_stateless_adam.py`、`test_update_weight_factory.py`、`test_megatron_argument_validation.py`。
- 含 GPU/外部服务/权重检查点的修改测试继续映射到对应 ST/Nightly，不因资源存在而默认放宽断言。纯 UT 需先在上游 replay 树运行，确认无 GPU hard-code，再转换到 `tests/tests_npu/ut` 并保持 NUM_GPUS=0。
- 此前 adapter patch 正向重放检查曾因 replay tree 已预先包含 adapter 改动而失败；随后反向移除两份已应用 patch、确认 clean 状态后正向重放，两个 patch 的 `git apply --check` 与应用均通过。


### NPU overlay UT 迁移结果

- 新增 NPU UT 映射文件：`test_megatron_checkpoint_v040_npu.py`、`test_megatron_tokenizer_init_v040_npu.py`、`test_disk_delta_checkpoint_index_v040_npu.py`、`test_stateless_adam_v040_npu.py`、`test_update_weight_factory_v040_npu.py`、`test_megatron_argument_validation_v040_npu.py`。复用上游纯 CPU 测试逻辑，源码解析路径固定指向 clean replay tree；覆盖面属于上游 CPU contract，不声称已执行 NPU kernel。
- 目标目录复验：原 Slime NPU UT 11 passed；新 NPU 映射 UT 35 passed + 6 passed + 参数验证 57 passed（另 12 项 Straw/PD-Mooncake 子集留待依赖环境）；合计 **109 passed**。`test_reference_top_p_replay.py` 因入口自动 import 当前 `/workspace/MegatronAdaptor` 与旧 `/workspace/Megatron-LM` 组合无法收集；`test_rollout_metadata_index.py`、`test_published_payload_retry.py` 需要未安装的 Straw。
- `test_stateless_adam` 首次在 NPU 目录复跑有 3 个 checkpoint 案例找错代码路径，修正为 clean replay 源后 6 项全过。
- UT 测试适配添加 `_install_npu_cuda_compat` AST no-op，避免隔离流程误调用设备初始化；生产代码未作此回归相关改动。
- 上游纯 UT 收集环境 `/workspace` 的 `PYTHONPATH` 会导入旧工作区 Slime/MegatronAdaptor。目标测试运行需将 clean replay 放在 `PYTHONPATH` 前；该目录的导入隔离仍需作为 CI 启动器配置处理。


### 2026-10-15 更新：Straw 与 top-p 映射完成

- 安装 `straw-queue 0.1.2` 后，上游 metadata/payload/argument-validation 三组 85 passed；映射到 NPU UT 的同三组亦 85 passed。
- `test_reference_top_p_replay_v040_npu.py` 在真实目标 Megatron-Core 导入路径下 21 passed。
- 整个 `tests/tests_npu/ut` 最终合并门槛 155 passed，50 warnings，118.70s；此前记录的 Straw/top-p 阻塞已解除。
