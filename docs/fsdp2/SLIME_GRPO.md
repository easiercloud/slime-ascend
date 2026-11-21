# Slime固定rollout GRPO与文件权重链路

本后端以补丁管理，保持Slime基线18868a79。build.sh按npu-compat、upstream-fixes、features回放；新增actor和入口分别由features下两份补丁承载，不提交Slime子模块的生成结果。

## 实现路径

- train-backend=fsdp2使用独立严格参数校验与actor factory，默认Megatron路径保留。
- 实际RayTrainGroup创建8个FSDP2TrainRayActor。Slime基类只初始化一次WORLD，Turbo复用并安装自身NPU设备代理。样本data domain是8，HSDP replicate维度是1。
- 使用真实RolloutManager、ServingCluster生命周期和BatchBuilder。固定generator产生4组×4响应=16个样本；每rank两个microbatch，每个rollout只做一次optimizer更新。
- 复用Slime policy loss。response token按t-1取logits，FP32 log-softmax/gather；reward在全局组中归一化，随后按8路分配。mask贡献除以整条rollout的mask总量，再乘WORLD/global_rollout_count，微批累积不额外平均。
- 默认old logprob来自更新前整批冻结策略快照。固定token不是在线采样，零附近的scalar loss不代表零梯度；实际参数变化另验。clip边界由独立非1 ratio数值测试覆盖。
- update_weights全rank收集完整FP32 CPU状态，由rank0原子发布HF/safetensors目录。全部rank必须返回同一版本与路径，再由RolloutManager独立CPU冷加载、校验文件hash、完整key/shape/dtype/有限值与logits探针，成功才接受版本。

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

- 核心固定环境的原生8卡3步已通过，见提交f6a86a1。
- 14项入口/数据流CPU检查通过。
- 6项GRPO数值与真实HF导出冷加载测试，在固定torch2.10/Transformers5.12.1容器通过。
- 同容器真实Ray local资源预检通过；只检查CPU任务和逻辑资源，不等于分配8张NPU。
- Slime完整八卡GRPO待实机验收。首次启动被宿主机其他训练进程检查拦住，未分配NPU，不能当作GRPO运行失败或成功。

最终summary要求8个不同rank、(1,8)mesh、Q投影全局[4096,4096]与每rank[512,4096]、连续两次optimizer更新、每rank两微批、有限非零梯度和参数变化，以及v0/v1/v2全部版本接收。CPU测试不能替代它。

## 当前边界

仅稠密随机Qwen3、FSDP8、TP/CP/PP/EP=1、固定object-store数据、单样本单rollout和full disk权重发布。严格拒绝critic、KL/OPD、DCP/optimizer/RNG恢复、offload、分段rollout、动态batch、在线生成和其他未实现路径，不静默忽略参数。HF全量导出用于独立模型加载，不能冒充可恢复优化器训练的checkpoint。
