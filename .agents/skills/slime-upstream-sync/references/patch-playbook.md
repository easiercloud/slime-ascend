# Ascend 补丁故障手册

本手册记录 `patches/MANIFEST.toml` 当前补丁的意图和复用边界。先匹配首个具体
错误、目标仓库和依赖版本，再复用方案；补丁能应用不等于运行时问题仍存在。

## 使用顺序

1. 从 Manifest 确认 `target`，不得跨仓应用。
2. 用 `git apply --check`（普通 diff）或 `git am --3way`（邮件 patch）确认文本基线；
   成功不代表语义仍兼容。
3. target 组件基线变化时，把该 target 的全部 patch 都纳入语义审计，不只审计冲突项。
4. 匹配下表的正向特征并检查排除条件。
5. 优先验证补丁影响的最窄路径，再跑完整 NPU 用例。仍有必要且验证通过时记录
   `KEEP_VERIFIED`。
6. 上游已有等价修复时在无旧 patch 的干净源码上验证，随后删除补丁，不保留“双重修复”。

## 冲突迭代与停止条件

补丁无法应用时必须继续完成以下闭环，不能把 `patch does not apply` 当作外部阻塞：

1. 将首次失败记为 `PENDING_ANALYSIS`，保存 target、目标 commit、失败 hunk 和完整错误；
   只隔离失败 patch，继续检查其余独立 patch。
2. 先跑完整 best-effort baseline。受该 patch 或其依赖链影响而无法执行的用例记 NG，
   其他用例继续，不能在此时死磕一个 patch 或单个用例。
3. 在旧基线读取补丁前后语义，在目标 SHA 定位对应符号和上游相关 commit，并结合首轮
   运行证据判断旧问题是否仍可复现。
4. 选择并记录 `KEEP_VERIFIED`、`REWRITE`、`SPLIT`、`UPSTREAM_ABSORBED` 或
   `DROP_UNSAFE`。clean apply 的补丁也必须用运行证据证明后才能 `KEEP_VERIFIED`。
5. `REWRITE/SPLIT` 必须从目标源码重新生成最小 patch；禁止只放宽上下文。
6. `UPSTREAM_ABSORBED` 必须有等价源码和测试证据；`DROP_UNSAFE` 必须有安全替代。
7. 更新 manifest 后从干净目标树重放，执行静态检查、最窄行为测试，并把受影响用例
   放回后续全量轮次。
8. 把结论写入 `.upstream-sync/patch-decisions.json` 和中文审计报告。发布前不得残留
   `PENDING_ANALYSIS` 或 `OPEN`。

只有缺失且无法在任务范围内获取的模型、数据集、设备、磁盘、外部服务或凭据可中止。
代码、依赖、编译、测试、launcher、patch 和 replay 故障必须进入问题台账并持续迭代。

## slime：NPU 运行和资源适配

| 补丁 | 解决的问题与思路 | 复用特征与边界 |
| --- | --- | --- |
| `command_utils.patch` | 为 Ray 注册 `NPU` 资源，向 worker 传播 Ascend/HCCL 环境，补齐 Megatron/Bridge `PYTHONPATH`，增加 NPU 启动入口。 | Ray 无法调度 NPU、worker 看不到设备或缺 Bridge 模块时检查。包含固定设备列表、进程清理和启动等待，复用前必须尊重调用方设备范围；共享机器清理进程需授权。 |
| `megatron_utils_actor.patch`、`megatron_utils_init.patch`、`train.patch`、`train_async.patch` | 在 NPU 路径尽早导入 `megatron_adaptor`，使 Megatron/MindSpeed patch 在建模前生效。 | 出现 CUDA 实现未被替换或 adaptor patch 未注册时检查；GPU 路径必须保持条件导入。 |
| `memory_utils.patch` | 用 `torch.npu` 查询/清理设备内存，避免 NPU 执行 CUDA host-cache 私有 API。 | `torch.cuda.mem_get_info`、`_host_emptyCache` 在 NPU 报错时复用；不要把 device cache 与 host cache 语义混为一谈。 |
| `ray_actor_group.patch`、`ray_placement_group.patch`、`ray_tran_actor.patch`、`rollout.patch`、`sglang_engine.patch` | 将 Ray bundle/actor 从 `GPU` 路由到 `NPU`，从 Ray accelerator ID 和 `ASCEND_RT_VISIBLE_DEVICES` 计算本地设备号。 | placement group pending、actor 错卡、物理卡到本地卡映射错误时检查。保持 GPU 分支，验证字符串 ID 转整数及可见卡重排。 |
| `tools_convert_hf_to_torch_dist.patch` | HF→Megatron 转换在 NPU 上选择设备、导入 adaptor，并使用 HCCL 初始化进程组。 | 转换脚本仍调用 CUDA/NCCL时复用；checkpoint 参数布局必须另行验证。 |
| `update_weight_common.patch` | NPU grouped-MoE `linear_fc1` 聚合后强制正确拼接维度。 | 权重更新出现 expert gate/up shape 或顺序不匹配时检查；只限已确认的 NPU grouped-MoE 布局。 |
| `update_weight_from_distributed.patch` | Actor→SGLang 权重同步组在 NPU 使用 HCCL，而非 NCCL。 | `/init_weights_update_group` 或分布式权重更新报 NCCL/device backend 错误时复用。 |
| THD CP/TP physical-only tail | 保持每条真实 sequence 的 `2*CP` 镜像分区不变，把 TP 对齐尾块建模为零真实 token 的独立 physical sequence。 | CP2+TP4 长序列才出现 SGLang/Megatron log-prob 分叉，且每个 dynamic microbatch 最后样本高风险。tokens/loss mask 必须共享 metadata 与 partition indices；不要扩展最后一个真实 sequence。 |

## slime：接口与模型兼容

| 补丁 | 解决的问题与思路 | 复用特征与边界 |
| --- | --- | --- |
| `cp_utils.patch` | `log_prob is None` 时构造零张量，避免 CP 数据整理立即崩溃。 | 仅在上游明确允许缺失 rollout log-prob 时使用；若行为契约要求真实 log-prob，这会隐藏生产缺陷，应改为追查数据源。 |
| `megatron_utils_init.patch` | 丢弃 Bridge/mbridge 新旧版本不接受的 `packed_seq_params`、`loss_mask` 兼容关键字。 | `forward() got an unexpected keyword argument` 且 CP/损失语义不依赖该参数时复用；CP>1 或参数有实际语义时禁止静默丢弃。 |
| `model_provider.patch` | 注册 critic 输出层映射，补齐 Bridge provider 参数，并为 Qwen3-VL critic 构造视觉+语言模型及权重加载 wrapper。 | Qwen3-VL critic 建模、参数映射或 provider 字段缺失时检查；必须核对当前 Bridge 模型类型和权重名称/shape。 |
| `qwen3_5.patch` | 完整处理 Qwen3.5 的视觉塔、MTP、线性注意力、MoE/shared expert、QKV 名称与张量重排。 | `Unknown parameter name`、SGLang refit 名称/shape 不匹配时逐分支核对。硬编码视觉维度与 split size 只适用于对应配置，换模型必须先验证 config。 |

## 第三方依赖

| target / 补丁 | 经验摘要 | 典型特征 |
| --- | --- | --- |
| `mbridge/0001` | safetensors FP8 反量化按可用性选择 NPU/CUDA。 | `safe_open(... device="cuda")` 在纯 NPU 环境失败。 |
| `megatron-bridge/0001` | 识别 MindSpeed TE 层类型、延迟导入可选 distill，并兼容 Qwen3-VL deepstack 可变参数。 | `AutoMapping` 未知 MindSpeed 模块、缺 modelopt、deepstack 参数数量错误。 |
| `megatron-bridge/0002` | 同时兼容 Transformers 4.57 tuple 与 5.3 ModelOutput，并适配新版 RoPE 配置。 | 视觉输出解包失败、`rope_theta` 移入 `rope_parameters`。 |
| `megatron-bridge/0003` | Qwen2/3 缺失 `rope_theta` 时使用默认值。 | 自定义 HF config 无 `rope_theta`；默认值必须与模型配置一致。 |
| `megatron/0001` | 大型基础兼容包：宽松 dist-checkpoint、NPU DDP/参数缓冲、TE/注意力、RoPE、MTP、优化器与 pipeline 适配。 | 只能按锁定 Megatron commit 整包验证；不要看到单一报错就复制整包。先按文件/算子拆分根因。 |
| `megatron/0002` | checkpoint CPU 写入禁用异步拷贝；避免错误覆盖 attention mask 类型。 | NPU 异步 checkpoint 不完整，或 attention mask 被错误改为 `padding_causal`。 |
| `sglang/0001` | Slime 0.3.0 与 SGLang 0.5.13 的大规模集成：分离式传输超时/清理、权重更新、profiling、NPU DSA/MoE、模型与多模态路径。 | 必须匹配锁定 SGLang 版本；先定位到具体文件和请求阶段，不整体套用到新版。 |
| `sglang/0002` | 将 NPU MoE 权重 transpose 放到调用点，修复 reshard 后 shape/layout。 | fused MoE `w13/w2` shape 错误或重复转置。 |
| `sglang/0003` | 去掉 Qwen3 extend/mixed 路径中的额外 causal 条件以修复精度。 | Qwen3/Qwen3-MoE NPU 精度回归；新版已有修复时删除。 |
| `sglang/0004` | GLM4.7 MTP 的 NPU attention block-table/head padding 兼容。 | FIA TND batch/head shape 不一致、GLM4.7 MTP 启动或 decode 失败。 |
| `sglang/0005` | NPU MoE EP 即使与 TP size 相同也创建独立 group，并让 dispatcher 使用 EP group。 | MoE EP collective 使用 TP group 导致 hang/错组。 |
| Actor DeepEP empty-token isolation | 保留 SGLang DeepEP，Actor 默认使用标准 alltoall，并保留独立 opt-in 开关复测 Actor DeepEP。 | Actor EP rank 出现 `[1,H]` placeholder 与 0-row routing map，报 `Permute not possible`；507015 只能作延迟症状，不能单独匹配。 |
| TMS `is_paused` lifecycle patch | 为 `torch_memory_saver` allocation 增加显式 pause 状态，避免 pause 期间新建 allocation 被错误 resume。 | `cpu_backup should not be nullptr` 出现在 colocate Actor wake-up；必须先确认 rollout 已完成且异常来自 `torch_memory_saver.resume()`。按锁定 SGLang/TMS 源码重建 wheel，并运行 `npu-test-autofix/scripts/check_tms_pause_lifecycle.py`；不要按包名 `0.0.8` 假设已包含修复。 |
| `sgl-kernel-npu` wheel rebuild | 当 TMS 源码补丁或依赖变更需要重新封装 wheel 时，必须在发布前重建 wheel 并把产物带入安装链路。 | 仅修改源码仓库而不重建 wheel 会让 quick install 继续安装旧产物，clean replay 通过但新机器复现失败。 |
| `transformer-engine-npu/0001` | CP 序列长度访问兼容、grouped-linear wgrad 条件和 RMSNorm 初始化修正。 | ring CP 的 dict/object 参数差异、wgrad 累积条件反转、RMSNorm 初始化异常。依赖删除仅适合由外部环境统一提供依赖的安装方式。 |

## 审计要求

- 记录 patch SHA、目标依赖 commit、首个失败签名和验证命令。
- 大型基础 patch 必须拆分记录实际命中的文件/符号，不能只写“应用 NPU patch”。
- 临时补丁必须有上游移除条件；验证通过后仍要检查上游是否已有等价实现。
- 任何跳过断言、用零值替代真实训练信号或丢弃参数的方案，都要重新核对测试行为契约。
- clean replay 通过前，必须先确认工作区里没有 `__pycache__`、日志、`*.pyc`、临时 inventory、`torch_compile_debug` 或手工试验脚本残留在可持久化路径里。
