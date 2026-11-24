# FSDPTurbo 同步训练与冷启动续训

`--fsdp-sync-train` 进入 Slime 原有 `train.py` 主循环，使用 object-store、同步 SGLang rollout、GRPO 和完整 HF 磁盘权重发布。
首期限定单机 Qwen3、2 张训练卡和 2 张推理卡，固定并行规模、常量学习率、内置数据源和单个本地 JSONL 文件。

## 保存和恢复

主循环完成当前 rollout 的训练后，依次保存分片模型/AdamW/各 rank RNG，以及原生数据源的游标。
两者都成功后写入 `rollout/fsdp_committed_0000000.json`，最后原子更新 `fsdp_latest.json`。
模型单独存在不能用于同步续训。恢复会校验文件哈希、训练进度、数据集及 tokenizer 内容、采样配置；不完整或不匹配时直接报错。
模型分片的完整性和训练配置还会由训练 Actor 校验。

`--save` 是训练检查点根目录；`--load` 指定要恢复的根目录。
默认读取该目录的 `fsdp_latest.json`，也可用 `--ckpt-step 0` 选择已完整提交的 rollout 0。
`--num-rollout` 是从零开始的总轮数：恢复 rollout 0 后设置为 2，实际执行 rollout 1。
恢复较早的检查点时，如保存目录存在该轮或更晚的输出，需要指定新的 `--save` 目录。
若失败发生在模型写入后、联合标记写入前，同样使用新保存目录从上一完整边界继续。

训练状态目录与推理权重目录必须分开。保存发生在当前轮的权重发布之前；重启会扫描已有 `weight_v*`，
从更高版本重新发布恢复后的权重，获得所有引擎确认后才开始生成，避免覆盖已有快照或使用旧权重。
HF 快照用于推理，不能代替包含优化器和 RNG 的训练检查点。

## 两个独立 driver 的示例

先按 [环境说明](fsdp2.md) 准备依赖。以下命令使用同一数据集、检查点根目录和推理快照目录：

```bash
export FSDP_PYTHON=/absolute/fsdp-venv/bin/python
export HF_CHECKPOINT=/absolute/models/Qwen3-0.6B
export FSDP_SAVE_DIR=/absolute/run/checkpoints
export FSDP_WEIGHT_DIR=/absolute/run/weights
export FSDP_OUTPUT_DIR=/absolute/run/first
export FSDP_NUM_ROLLOUT=1
bash examples/fsdp2/run-sync-train.sh

export FSDP_LOAD_DIR=$FSDP_SAVE_DIR
export FSDP_OUTPUT_DIR=/absolute/run/resumed
export FSDP_NUM_ROLLOUT=2
bash examples/fsdp2/run-sync-train.sh
```

示例每次创建并关闭自己的 Ray 集群。`result.json` 记录训练起点、各 rank 步数、发布版本与权重 SHA256；
`rollout-*.pt` 保存生成数据用于验收，联合标记记录数据游标。示例使用四条不同提示词便于检查续训没有回到第一批。
`result.json` 的 `passed` 表示训练主循环走完，进程退出码为 0 才表示 driver 正常结束。

## 支持边界

- 重启后保留训练状态和数据读取位置，不承诺重新启动的 SGLang 生成结果与不中断运行逐 token 相同。
- 不接入 Straw、异步/partial rollout、动态过滤、额外预取、定制数据源、参考模型/KL、多节点、offload 或在线扩缩容。
- 当前仅支持新建推理会话的冷启动；失败时清理本次会话的资源，不支持保留推理服务的热重连。
- 检查点目录只支持单个 driver 写入；文件系统需支持同目录原子替换、硬链接和 fsync。读取可信本地检查点。
- 验收证明恢复边界和后端衔接，不能据此判断长程训练收敛或超参数适用性。

补丁为 `patches/features/0006-fsdpturbo-sync-resume.patch`，依赖前五个 FSDPTurbo 功能补丁。
共享保存函数只在显式开启 `--fsdp-sync-train` 时增加联合提交步骤，Megatron 路径保持原行为。
