# 固定 rollout 的 GRPO 训练验收

本阶段接通 `BatchBuilder.convert/split_by_dp → Ray object store → ActorGroup.async_train → FSDPTrainRayActor.train`，
训练 Actor 使用 FSDPTurbo native FSDP2 和本地 Qwen3-0.6B。固定 token 样本替代在线生成；
本阶段不代表 SGLang 生成、训练、权重回传的完整闭环已完成。

## 已接入的数据和计算

- 使用 Slime 已生成的 DP 分区和微批调度，按 `num_microbatches` 划分 optimizer step。
- 使用 BatchBuilder 按唯一 rollout、prompt group 归一化后的 `rewards`，不重复归一化。
- 独立序列右侧 padding；response 第一个 token 对齐 prompt 最后位置的 logits，包含末尾 EOS。
- logits 转 FP32 后除以 `rollout_temperature`，计算 response log-prob。
- 默认在任何更新前计算本轮全部旧策略 log-prob 并冻结。训练原语也可读取 `rollout_log_probs`；
  该分支通过 CPU 测试，固定双卡验收不模拟 sampler log-prob。
- PPO clipped GRPO loss 支持不对称 clip 和可选 dual clip；每个 segment 使用全 rollout 的
  `rollout_mask_sums` 作分母。反向乘 `world_size / global_batch_size`，梯度累积不再除以微批数量。
- FP32 主权重、BF16 计算、FP32 梯度规约；全局梯度裁剪，分片 AdamW 状态，连续更新保留动量。
- 模型 collective 前校验各 rank 的 token、mask、调度、分区、rollout 总数和全局 mask 分母。

## 双卡验收

```bash
export FSDP_PYTHON=/absolute/fsdp-venv/bin/python
export HF_CHECKPOINT=/absolute/models/Qwen3-0.6B
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
bash examples/fsdp2/run-grpo-check.sh
```

Driver 使用原推理环境，训练 Actor 使用独立 venv。2 张卡训练，另 2 张预留。
示例自行启动并释放私有 Ray 集群，支持 `SLIME_SOURCE_DIR`、`FSDP_OUTPUT_DIR`、`BASE_PYTHON`
和 `FSDP_GRPO_TIMEOUT_SECONDS` 覆盖。超时默认 2400 秒，首次 provider 加载可能触发算子编译。

固定数据包含 2 个 prompt group、8 个唯一 rollout、12 个训练片段：前 4 个 rollout 分别有 3、1、2、2 个片段，
后 4 个各 1 个。第一个 rollout 跨 rank 分布，验证全 rollout 分母。
使用 global batch 4、micro batch 2，执行两次 optimizer step，覆盖不同序列长度、
非连续 mask、一个完全屏蔽片段、截断样本和梯度累积。学习率固定 `1e-4`，温度 `0.8`。

参考实现是每个 rank 上未分片的 HF BF16 模型：独立使用 log_softmax/gather 和 GRPO 表达式，
每个微批梯度以 FP32 显式跨 rank 平均并在 CPU 累积；未分片的 NPU FP32 主权重和 AdamW 提供参数更新参考。
梯度范数以 CPU FP64 累加独立计算。每步将参考模型的输入权重对齐 Actor 当前主权重，
参考 AdamW 动量独立连续保留，旧策略 log-prob 始终使用初始策略。
这是逐步相同输入上的计算与更新对照，不是两条独立多步训练轨迹完全等价的证明。
初次独立轨迹实验中，极小的 FP32 更新差异跨过 BF16 转换边界，导致后一步误差放大；
因此采用逐步对照定位后端差异，原误差阈值保持不变。
逐步比较全部唯一参数的原始梯度及更新差值，而非只比较最终权重。参考实现保留多个 CPU 快照，
此验收需要数十 GB 主机内存；正式训练路径不创建这些快照。

预设验收阈值：log-prob 最大绝对误差 `1e-3`、loss 绝对误差 `1e-3`、梯度相对 L2 `3%`、
梯度方向余弦至少 `0.999`、梯度范数相对误差 `3%`、参数更新差值相对 L2 `10%`。
同时要求初始权重完全一致、分片参数存储为一半、优化器状态分片且两步更新均非零。
成功证据为退出码 0、`result.json` 的 `passed: true` 和 `FSDP_GRPO_CHECK_PASSED`。

## 支持边界

当前仅开放显式 `--fsdp-grpo-check` 验收入口。完整在线 train 入口保持拒绝，避免进入尚未实现的
权重发布或 checkpoint 流程。仅支持单机、dense Qwen3、固定微批、完整词表分布。
熵正则、reference/KL、OPD、PPO critic、score centering、TIS、自定义 loss/reducer、
动态 batching、packing、offload、训练推理同卡、checkpoint 均不在本阶段范围内。
`--reset-optimizer-states` 和 stateless Adam 也明确拒绝。

新增功能通过 `patches/features/0003-fsdpturbo-grpo.patch` 交付到 slime-ascend，
在前两份 feature patch 后应用。不得只提交修改后的 slime 子模块指针。
在独立干净目录运行官方 `build.sh` 重放补丁，再对重放源码运行验收。
单测入口：`tests/test_fsdp_grpo.py`、`tests/test_fsdp_entry.py`、`tests/test_fsdp_model.py`。
下一阶段接通 HF 权重导出及推理端更新，验证真实生成和训练闭环，随后补 checkpoint。
