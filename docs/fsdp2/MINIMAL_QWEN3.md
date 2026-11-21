# Qwen3随机初始化八卡冒烟

首轮目标改为slime-ascend已支持的稠密Qwen3-8B，保留hidden、head、vocab等结构，仅减到2层。随机初始化不读取预训练权重；固定合成token直接提供labels，无需下载tokenizer和数据集。

已在910B3八卡完成首轮3次实际更新：CANN9.1、torch2.7.1、torch_npu2.7.1.post8、Transformers5.2、Turbo262f5ba。8份rank报告记录了`[1,8]` mesh、`R,S(0)` placements，Q投影从`[4096,4096]`分成每rank`[512,4096]`。3步loss分别为12.78125、12.7421875、12.8046875，梯度有限且参数确实变化。随机数据不要求loss单调下降。

这一步验收FSDPTurbo原生FSDP2的8卡前后向和优化器更新。它通过后仍需实现Slime actor与固定rollout，不能当作Slime GRPO后端已接通。

## 最小启动

优先复用现有镜像。当前已装Turbo `262f5ba`能在CANN9.1、torch2.7.1、Transformers5.2环境导入并构造FSDP8配置，Qwen3的CPU小模型前后向也已通过。该Turbo提交的README和pyproject对torch最低版本写法不一致，因此是否可用于NPU仍以实际运行判断，不预先宣称官方兼容，也不因GLM文档先升级CANN9.2。

在选定的自有训练容器内，确认所选Turbo已可导入后执行：

```bash
python3 scripts/fsdp2/smoke_qwen3_random.py --inspect
bash scripts/fsdp2/run_qwen3_random_8npu.sh /workspace/experiment/slime-fsdp2-minimal/runs
```

启动器先读取`npu-smi info`，只有明确报告0–7卡均无计算进程时才启动8个rank。它不停止其他进程、不清理Ray、不安装依赖。每次新建独立结果目录，默认3步、每rank batch=1、序列128、FP32主参数、BF16前向、FP32梯度归约、AdamW。

旧Docker19的默认过滤器可能把`faccessat2`返回为EPERM，导致CANN的tiling库注册失败。启动器通过`compat_faccessat2.py`仅在探测到该情况时，对当前进程及子进程追加返回ENOSYS的拒绝过滤，让libc使用原本允许的旧接口；现代环境不需要该过滤时直接执行。它不关闭seccomp、不增加capability、不修改主机。w22同一镜像单卡`ones.sum=16`已验证该兼容路径。

模型配置保存在`scripts/fsdp2/qwen3_8b_2layer.json`，结构来自现有Qwen3-8B公开配置，仅修改层数、对应窗口层数和cache开关。可以用`--config`明确传入其他Qwen3配置；本轮不启用TP、EP、CP、量化、offload、MTP或推理服务。

## 通过证据

每rank写`rankXX.json`，包含实际device、mesh、DTensor placements和参数global/local shape。要求被观察的Q投影真实分为8片，loss及全局梯度范数有限、梯度非零、每次优化后Q投影权重有实际变化。全部8rank完成后，rank0写`summary.json`；`torchrun.log`保留原始错误。

合成token训练只能证明训练链路可运行，不能证明语料收敛质量、GRPO数值、推理权重交接、EP或DCP恢复。后续再复用现有数据、tokenizer和固定rollout，不在首轮引入这些依赖。

## 机器和环境选择

先比较全部可登录机器的完整空闲8卡、主机可用内存、持久磁盘、Docker/runtime和已缓存镜像。完整8卡可用比下载新版本更重要。已有CANN9.1/torch2.7环境若NPU初始化和FSDP2都通过，则继续使用；否则优先使用已缓存的CANN9.1/torch2.10镜像，再考虑下载。

旧w25 `ywy-glm`的单设备探测返回aclInit 507899，因此它的CPU导入成功不代表旧容器NPU可用。首轮在w22独立自有容器验证了缓存Ubuntu镜像和进程兼容处理；使用当地工号目录`/home/y00985004/slime-fsdp2`，未修改其他人的容器。

机器长期选择仍按总迁移成本评估：w12有相同缓存镜像、现代Docker、约1.9Ti可用RAM和379G持久余量；w25已有个人工程，约1.3Ti可用RAM和282G余量；w24有约1.8Ti可用RAM和1.2T余量但需要迁入已验证环境。w22是本次已验证的运行点，约910Gi可用RAM和176G余量，不能仅因当时八卡空闲就称为长期综合最优。

首轮日志在进程组清理时有非致命hooks警告，退出码为0，8rank和3次更新报告齐全。接Slime生命周期时仍需核对进程组的创建与销毁顺序。
