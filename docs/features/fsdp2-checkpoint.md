# FSDPTurbo 训练检查点验收

本阶段在 Slime 的 `FSDPTrainRayActor.save_model()` 和 `init()` 中接入训练状态保存与恢复。
模型和 AdamW 状态使用 PyTorch Distributed Checkpoint（DCP）按 FSDP 分片保存，保留 FP32 主权重，
不经过 BF16 推理快照。接口方向参考 Miles 的 FSDP checkpoint 实现。

## 当前边界

- 单机 Qwen3-0.6B、两个训练 rank、replica-DP=1、FSDP=2、TP/CP=1，固定 AdamW 学习率。
- 保存模型、AdamW 一阶/二阶动量和 step、累计 optimizer_steps、下一 rollout ID、权重版本计数。
- 每个 rank 分别保存 Python、NumPy、Torch CPU、当前 NPU 的随机数状态。
- 仅在完整 rollout 更新结束后同步保存；先写隐藏临时目录，再原子重命名发布。
- 清单记录配置、运行时版本和各文件 SHA256；缺文件、损坏、配置不匹配直接失败。
- 只支持相同拓扑和配置恢复；不提供重分片、异步保存、自动最新检查点选择或降级为重新训练。

这是训练 actor 检查点验收。在线数据源游标、BatchBuilder 状态、SGLang 状态及全流程重启尚未纳入。
因此只在 `--fsdp-checkpoint-check` 中开放，不能据此宣称在线 GRPO 已支持完整断点续训。
HF 发布快照仍只用于推理，不能作为训练检查点加载。

## 参数与目录

使用 `--fsdp-checkpoint-save ROOT` 指定保存根目录，
使用 `--fsdp-checkpoint-load ROOT/iter_0000000` 指定一个完整的检查点目录。
编号沿用 Slime 的零起始 rollout ID；`iter_0000000` 表示完成 rollout 0，下一轮从 1 开始。
恢复始终加载优化器与 RNG，不使用 Slime 的 Megatron `--load/--save` 或通用微调回退逻辑。
原有这些参数仍在 FSDP 参数校验阶段拒绝。

```text
checkpoints/iter_0000000/
  COMMITTED.json
  rng-rank-0.pt
  rng-rank-1.pt
  state/.metadata
  state/*.distcp
```

已经存在的检查点不会被覆盖。失败时可留下隐藏临时目录用于排障，该目录不能被恢复。
需要共享 POSIX 文件系统，所有 rank 可读写同一路径；DCP collective 故障由分布式超时处理。
只加载可信训练任务生成的 DCP 文件。RNG 文件使用 `weights_only=True` 读取。

## 验收命令

保持 driver 使用镜像原 Python，仅训练 actor 使用独立 venv：

```bash
export FSDP_PYTHON=/workspace/slime-fsdp2/envs/fsdp2/bin/python
export HF_CHECKPOINT=/workspace/slime-fsdp2/models/Qwen3-0.6B
export FSDP_OUTPUT_DIR=/absolute/new-checkpoint-check
bash examples/fsdp2/run-checkpoint-check.sh
```

流程通过现有 Slime placement group、BatchBuilder、object store、RayTrainGroup 和 actor.train：

1. 固定 GRPO 输入完成第一轮（两次优化器更新），保存检查点。
2. 原 actor 继续第二轮，记录模型、动量、RNG 样本和指标。
3. 销毁两个 actor 并重建，恢复第一轮检查点。
4. 比较恢复状态，再用相同输入继续第二轮，与第 2 步逐项精确比较。
5. 再保存第二个检查点，验证继续训练后的再次保存。

`result.json` 记录每个参数分片及动量的 SHA256、进程 PID、随机数样本和逐步指标。
必须同时满足进程重建、训练进度恢复、权重/优化器/RNG 一致、下一轮指标一致才通过；不使用数值容差。
固定输入用于隔离训练恢复误差，不代表在线生成结果或收敛质量的等价性。

如需从已有检查点启动整个验收进程，可设置：

```bash
export FSDP_CHECKPOINT_LOAD=/absolute/previous-check/checkpoints/iter_0000001
export FSDP_OUTPUT_DIR=/absolute/new-restored-check
bash examples/fsdp2/run-checkpoint-check.sh
```

必须使用新的输出目录。报告中的 `initial_start_rollout_id` 应为 2。
本验收只使用两张训练卡，其余两张卡不启动推理引擎。

## 交付

补丁 `patches/features/0005-fsdpturbo-checkpoint.patch` 追加到前四个 FSDPTurbo 功能补丁后，
由 `patches/MANIFEST.toml` 校验和官方 `build.sh` 应用。不能在开发中的子模块目录直接重放构建。
CPU 回归包括真实 DCP 的 AdamW 冷启动恢复，以及检查点完整性和参数隔离测试。
两卡验收必须另外运行，CPU DCP 测试不能替代 NPU 分片验证。
