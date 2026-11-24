# FSDPTurbo 模型阶段验收

本阶段在 Slime 的真实 FSDP Ray Actor 内加载本地 Qwen3-0.6B，使用 FSDPTurbo
native FSDP2 完成 masked causal CE 前向、反向、梯度裁剪和一次 AdamW 更新。
它是后续 GRPO 集成的模型基础；还不能运行 FSDP GRPO、发布推理权重或保存 checkpoint。
Megatron 基线继续使用原入口。

## 固定配置

- 两个训练 rank，副本 DP=1、FSDP=2、TP/CP/EP=1。
- FP32 主参数、BF16 计算、FP32 梯度归约；HF eager attention，关闭 KV cache。
- 28 个 decoder layer 分别分片，root 管理 embedding、最终 norm 和 lm_head。
  共享 embedding/lm_head 权重保持同一参数，不单独包装两次。
- 不开启 offload、重计算、量化、packing、fused attention 或预取。
- 验收固定 AdamW：lr=1e-4、betas=(0.9,0.999)、eps=1e-6、weight_decay=0.01，
  全局梯度范数裁剪上限为 1.0。这是测试配置，不代表已完成生产优化器参数适配。
- checkpoint 从本地加载，不执行 remote code；拒绝不完整权重、残留 meta 参数及非 dense Qwen3。

已验证环境与 provider 固定提交见 [环境说明](fsdp2.md)。模型使用 Qwen3-0.6B
revision `c1899de289a04d12100db370d81485cdf75e47ca`；更换模型需重新验收。

## 运行

使用原镜像 Python 启动 driver，训练 Actor 使用独立 venv：

```bash
export FSDP_PYTHON=/absolute/fsdp-venv/bin/python
export HF_CHECKPOINT=/absolute/models/Qwen3-0.6B
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
bash examples/fsdp2/run-model-check.sh
```

`BASE_PYTHON`、`SLIME_SOURCE_DIR`、`FSDP_OUTPUT_DIR` 可覆盖解释器、待验证源码和日志路径。
该脚本复用独立 Ray 集群 launcher；只占用两个训练 NPU，另两个预留给后续推理。
SGLang 仅在 CPU worker 检查解释器和 ServerArgs 导入，不启动推理引擎。

## 数值对照方法

两个 rank 使用固定的不同 token 序列，有效长度分别为 23、32，补齐至 32。
next-token 目标总数为 22+31=53。每个 rank 采用
`local_ce_sum * world_size / global_valid_tokens`，补偿 FSDP 平均梯度归约。
单测另用独立拼接后的全局 CE 验证公式与 padding 梯度屏蔽。

参考侧每个 rank 使用未分片 BF16 HF 模型；显式 FP32 all-reduce 平均梯度，
再通过 CPU FP32 AdamW 计算参数更新。被测侧通过 FSDPTurbo 执行分片训练。
比较每个唯一参数的初值、完整梯度和更新量；不以更新后权重接近代替更新量检查。

验收检查：

- 初始完整参数完全一致，所有主参数/梯度 FP32，无缺失梯度和非有限数值。
- 每个 rank 的参数存储恰好为全模型一半，FSDP unit 数为层数+1，共享权重保持。
  AdamW 两个动量状态均为与参数分片形状一致的 DTensor。
- 有效 token logits 相对 L2 ≤2%，loss 相对误差 ≤1%。
- 裁剪前后梯度相对 L2 均 ≤3%，方向余弦 ≥0.999，全局范数相对误差 ≤3%。
- 更新量相对 L2 ≤10%，实际更新非零。这些 BF16 容差在实测前固定。

成功输出 `FSDP_MODEL_CHECK_PASSED`，日志目录含 `result.json`、`run.log`、
`exit-code.txt`。报告记录逐 rank 的指标、容差、分片大小和优化器配置。
本次短序列 CE 验收不衡量收敛、吞吐、长序列稳定性或 GRPO 正确性。

## 回归

```bash
PYTHONPATH="$PWD/slime${PYTHONPATH:+:$PYTHONPATH}" python -m pytest \
  slime/tests/test_fsdp_entry.py slime/tests/test_fsdp_model.py \
  slime/tests/test_megatron_argument_validation.py slime/tests/utils/test_sglang_arguments.py -q
bash examples/fsdp2/run-init.sh
```

交付使用 `0002-fsdpturbo-qwen3-model.patch`，在 `0001` 后应用。
应在独立干净目录通过 `build.sh` 回放并运行验收，不在有未提交修改的开发目录重建。
