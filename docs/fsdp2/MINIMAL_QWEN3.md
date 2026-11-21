# Qwen3随机初始化八卡验证

当前入口使用Slime仓库A2 Dockerfile指定的核心训练版本：Python3.12、CANN9.1.0、torch2.10.0、torch_npu2.10.0.post6、torchvision0.25.0、Transformers5.12.1。基础镜像和digest、Turbo源码及补丁hash在`scripts/fsdp2/pinned-runtime.json`。这不是完整Megatron/SGLang镜像的整镜复现。

## 实测环境

2026-11-21在Docker26的新建自有容器验证：两层Qwen3随机初始化，1,630,556,672参数，8个NPU rank，固定合成token，3次AdamW更新，训练退出0。报告目录`qwen3-fsdp8-YE0LC1Ba`。源码使用Turbo262f基线，仅回移两处上游torch2.9+的NPU设备接口修复，形成可重放参考提交20409188。较新Turbo要求更高Transformers/Triton版本，因此没有直接覆盖Slime的固定依赖。

本入口不再调用旧Docker系统调用兼容helper，该文件已删除。历史提交9ea9278记录的是另一套torch2.7环境的实测，不可套用其版本结论。

## 启动

在已准备好的自有容器、源码根目录运行：

```bash
python3 scripts/fsdp2/check_pinned_runtime.py
bash scripts/fsdp2/run_qwen3_random_8npu.sh /workspace/experiment/runs
```

先检查核心版本、实际Turbo Git提交及device.py指纹，再检查NPU0–7明确无计算进程。默认3步、每rank batch1、序列128、FP32主参数、BF16前向、FP32梯度归约、AdamW。输出每次使用新目录。

共享机器上默认HCCL端口16666可能仍被其他通信上下文占用。启动器在未显式指定时将HCCL_NPU_SOCKET_PORT_RANGE和HCCL_HOST_SOCKET_PORT_RANGE设为auto，不停止其他人的上下文。该配置来源见[HCCL NPU端口官方说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_env/HCCL_NPU_SOCKET_PORT_RANGE.md)。

## 验收边界

每rank的JSON记录实际设备、版本、mesh、placements与参数global/local shape；汇总要求8rank、真实8分片、有限loss和梯度、非零参数变化。随机数据不要求loss单调下降。首轮pinned运行中出现默认HCCL端口占用，配置独立动态端口后通过；保留原失败记录。

该脚本验证Turbo原生FSDP8，Slime actor/GRPO及文件权重接收链路另行验收，不将SFT型labels loss当作GRPO。
