# 通用 GPU→NPU 转换规则

先保留测试目的、断言、输入数据含义、随机性控制、checkpoint 语义、样本数、
失败条件和分布式不变量。设备迁移不能成为删减功能或降低断言的理由。

## 证据驱动映射

| GPU/CUDA 关注点 | NPU 转换要求 |
| --- | --- |
| CUDA 设备 API | 使用框架已有设备抽象或已验证的 `torch_npu` 等价调用。 |
| NCCL 配置 | 使用框架实际支持的 HCCL/集合通信配置；不做全局字符串替换。 |
| CUDA 可见卡 | 尊重调用方的 Ascend 可见设备变量和测试实际拓扑。 |
| `torchrun`/Ray/DeepSpeed/FSDP | 保留原框架启动器；仅替换已验证的后端、资源和环境传播。 |
| CUDA 图、编译和 kernel 调优 | 先确认 NPU 支持；调优不能被表述为语义转换。 |
| 在线模型/数据下载 | 用用户确认的本地或内部源替换，并验证同一模型/数据 schema。 |

## 分布式和资源不变量

转换前后检查 world size、DP/TP/PP/EP、梯度累积、全局 batch、micro batch、
rollout workers、采样数、seed、checkpoint 分片和端口/服务依赖。资源缩减必须使
被测路径仍可到达，并在交付物中说明。

## 静态审计

至少检查目标和启动 helper 中的：

```bash
rg -n 'torch\.cuda|cuda:|nccl|CUDA_VISIBLE_DEVICES|hf download|wget |curl .*huggingface|/root/(models|datasets)|/home/|pkill|ray stop' <target>
```

每个命中必须删除或按框架 profile 说明。不得自动终止共享进程。不得把未执行、跳过、
缺资产或不支持的路径记为 PASS。

## 新 profile 的准入要求

为新框架新增 profile 前，提供：仓库标记、测试发现方式、Python/脚本/分布式启动器、
资产定位方式、NPU 环境传播方式、至少一个成功和一个失败场景，以及对应自动化测试。
