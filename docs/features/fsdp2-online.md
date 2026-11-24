# FSDPTurbo 真实生成、GRPO 与权重同步验收

本阶段使用 Slime 原有 RolloutManager、SGLang TP2 引擎、BatchBuilder 和 RayTrainGroup。
训练 actor 在独立 venv 中使用 FSDPTurbo native FSDP2，训练卡与推理卡分开。
`--fsdp-online-check` 是有明确边界的两轮集成检查，不代表全部常规训练生命周期已支持。

## 运行

```bash
export FSDP_PYTHON=/absolute/fsdp-venv/bin/python
export HF_CHECKPOINT=/absolute/Qwen3-0.6B
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
bash examples/fsdp2/run-online-check.sh
```

支持 `BASE_PYTHON`、`SLIME_SOURCE_DIR`、`FSDP_OUTPUT_DIR`、`FSDP_ROUTER_PORT` 和
`FSDP_ONLINE_TIMEOUT_SECONDS`。每次使用新的输出目录；禁止覆盖已发布版本。
默认端口搜索起点 30000；SGLang 还需要连续 31 个空闲端口，必须按服务器情况调整。
HCCL 端口沿用 60400–60499 / 61400–61499，可由环境变量覆盖。

示例执行：发布初始版本 1 → 真实生成 8 个回答 → GRPO 更新 → 发布版本 2 →
再次生成 8 个回答 → GRPO 更新 → 发布版本 3。每次发布后另用固定 token 输入检查概率。
问答样例使用 Slime 内置 F1 奖励，与参考答案进行 token 重叠比较；不是数学正确率测试。
关闭 weight decay，要求实际优势和梯度非零，避免仅靠衰减产生权重变化。
首期重算并冻结每批训练 actor 的 old log-prob，不使用采样器 log-prob，不启用 TIS。

## 发布协议

所有训练 rank 按相同 HF state_dict 顺序还原 DTensor；仅 rank 0 保留完整 CPU BF16 权重。
保留 Qwen3 原始 HF 参数名，复制 tied embedding/head，写 safetensors 和配置/分词器文件。
在同目录临时路径完成写入后 rename 为 `weight_vNNNNNN`，记录 SHA256 和 optimizer step。
推理端暂停生成、清空缓存、调用现有 update_weights_from_disk，再核对成功回执和版本号。
任一环节失败不恢复生成，也不把它记作发布成功。

这些文件只用于推理同步，**不包含 optimizer/RNG 状态，不能恢复训练**。
首期保留全部版本用于验收，需要足够的共享磁盘和 rank 0 主机内存；不声称高吞吐。
集体通信的进程故障依赖分布式超时终止，未实现在线容错。

## 验收证据

`result.json` 记录训练解释器、设备、梯度、实际优势、各版本 SHA256、推理回执和固定输入概率变化。
`rollout-0.pt` / `rollout-1.pt` 保存真实生成和奖励，`weights/` 保存已发布 HF 文件。
成功要求两轮均有非零 policy gradient、BF16 权重变化、推理版本正确且概率变化，退出码为 0。
没有奖励差异的生成批次会使检查失败；不能为通过检查而伪造奖励。

不以不同生成文本证明同步正确，也不把此检查等同于数值精度对齐或长期收敛测试。
固定 rollout 的逐步数值 oracle 继续由 `run-grpo-check.sh` 验证。
