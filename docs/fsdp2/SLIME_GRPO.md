# Slime固定rollout GRPO与文件权重链路

本后端以补丁管理，保持Slime基线18868a79。build.sh按npu-compat、upstream-fixes、features回放；新增actor和入口分别由features下两份补丁承载，不提交Slime子模块的生成结果。

## 实现路径

- train-backend=fsdp2使用独立严格参数校验与actor factory，默认Megatron路径保留。
- 实际RayTrainGroup创建8个FSDP2TrainRayActor。Slime基类只初始化一次WORLD，Turbo复用并安装自身NPU设备代理。样本data domain是8，HSDP replicate维度是1。
- 使用真实RolloutManager、ServingCluster生命周期和BatchBuilder。固定generator产生4组×4响应=16个样本；每rank两个microbatch，每个rollout只做一次optimizer更新。
- 复用Slime policy loss。response token按t-1取logits，FP32 log-softmax/gather；reward在全局组中归一化，随后按8路分配。mask贡献除以整条rollout的mask总量，再乘WORLD/global_rollout_count，微批累积不额外平均。
- 默认old logprob来自更新前整批冻结策略快照。固定token不是在线采样，零附近的scalar loss不代表零梯度；实际参数变化另验。clip边界由独立非1 ratio数值测试覆盖。
- update_weights全rank收集完整FP32 CPU状态，由rank0原子发布HF/safetensors目录。全部rank必须返回同一版本与路径，再由RolloutManager独立CPU冷加载、校验文件hash、完整key/shape/dtype/有限值与logits探针，成功才接受版本。

精度协议使用manifest schema_version=2，forward_dtype必填；旧实验schema1保留作历史记录，本接收器不接收。探针明确记录forward_dtype。文件、完整加载、key/shape/dtype与有限值先按FP32 master校验，然后接收端按训练BF16前向比较并保留RoPE等浮点buffer原精度；保持0.02绝对容差加0.02相对容差。最终报告还直接审计全部safetensors文件header，确认实际存储均为F32，不以加载时强制上转FP32代替存储证据。不能将BF16训练前向直接与CPU全FP32前向当作同一计算。

版本路径为初始v0、更新后v1/v2，每版含manifest.json、COMPLETE及receiver-receipt.json。这是实际文件接收链路，尚未接在线SGLang。

## 启动

先在宿主机确认8卡使用窗口；容器内PID命名空间可能看不到其他容器中的进程，不能只凭容器内npu-smi判断主机空闲。

在pinned-runtime.json对应容器、源码根目录执行：

```bash
bash build.sh --skip-submodule
bash scripts/fsdp2/run_slime_grpo_8npu.sh /workspace/experiment/runs
```

skip-submodule只用于干净的固定Slime基线。构建前保存自己的生成目录修改，回放补丁会改写该目录。启动器使用独立Ray local实例、16CPU、独立临时目录和512MiB对象存储，finally清理自己的实例，不运行全局ray stop。

## 验收状态

训练提交9cdf46c在2026-11-21完成实际Slime FSDP8固定rollout GRPO：[脱敏验收记录](evidence/qwen3_grpo_8npu.json)。

- 8个不同rank，mesh为[1,8]，Q投影全局[4096,4096]、每rank[512,4096]。
- 每rank两轮optimizer更新，每轮两个microbatch。全局梯度范数为13.07187444、13.06958341；Q投影更新L2为0.00409229、0.00407336，均有限且非零。
- v0、v1、v2全部通过独立HF冷加载，25个张量key/shape/dtype齐全；三个版本safetensors headers全部F32，权重payload hash逐版变化。
- BF16前向探针最大绝对误差依次为0.015625、0.01953125、0.01171875，原0.02绝对加0.02相对容差保持不变。
- 14项入口/数据流CPU测试通过；修复后的10项GRPO数值/HF导出冷加载测试也在目标torch2.10/Transformers5.12.1容器通过。

最初189561f完成v0导出，但CPU FP32前向与NPU BF16前向直接比较导致误拒绝。独立同种子重建的25个初始张量全部逐元素相同；9cdf46c修复精度契约后，完整两轮训练及三版权重链路已完成。中心化组reward与更新前snapshot会使scalar loss接近零，不能据此判定没有梯度，也不能把它当作训练质量提升结论。

本次SSH断线，未捕获主进程最终退出码；完整driver summary在train.main返回和Ray shutdown后落盘，8份rank报告、3份接收回执及实际存储header审计已取回，后置验收程序exit0。不能将其表述为已捕获训练exit0。原生FSDP8的3步训练exit0另已记录于f6a86a1。

## 当前边界

仅稠密随机Qwen3、FSDP8、TP/CP/PP/EP=1、固定object-store数据、单样本单rollout和full disk权重发布。严格拒绝critic、KL/OPD、DCP/optimizer/RNG恢复、offload、分段rollout、动态batch、在线生成和其他未实现路径，不静默忽略参数。HF全量导出用于独立模型加载，不能冒充可恢复优化器训练的checkpoint。
