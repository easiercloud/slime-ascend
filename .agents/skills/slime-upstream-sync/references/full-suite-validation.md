# 全量同步验证经验

## 测试覆盖口径

- 上游 `tests/**/test_*.py` 的完整清单与 GPU CI 清单不同；前者用于全量同步审计，后者仅用于 CI 对照。
- “跑全量测试用例”必须以目标 `slime/tests` 下递归发现的所有 `test_*.py` 为全集，包含根目录 UT、子目录 UT、GPU/ST 脚本、`tests/tests_npu/st`、`tests/tests_npu/nightly_CI` 和 `tests/tests_npu/ut`。不能只用 `tests/tests_npu` 的发现数量代表全量。
- 每次同步或自动升级 run 都必须在最终物化树上重新递归扫描 `slime/tests/**/test_*.py` 并生成当次 inventory 快照。旧的 `.upstream-sync/full-test-inventory*.json` 只能作为历史审计证据，禁止作为后续 run 的输入或固定分母。
- 若本次扫描数量与上一轮不同，报告必须明确列出新增、删除、移动和分类变化；如果扫描数量与执行数量不一致，未执行用例必须逐项进入 `unexecuted`，不能从分母中消失。
- 最终测试报告必须列出每一个发现的测试文件，并为每个用例记录测试特性/行为契约、分类、转换状态、执行状态、PASS/NG/未执行原因和日志路径。
- 静态 `NUM_GPUS > 0` 的上游测试生成 NPU 副本；设备无关测试保留原文件并独立进程执行，避免 fixture 向 `sys.modules` 注入的假 Megatron/Ray 模块污染后续测试。
- 原 `tests/tests_npu/{st,nightly_CI,ut}` 是独立基线，必须同样执行，不能被“新增转换数量”替代。
- `tests/tests_npu` 子集可作为 NPU baseline 单独报告，但报告标题和机器字段必须写清楚这是 NPU 子集；若 `mode=full`，`baseline_discovered` 应使用 `slime/tests` 全量发现数。

## 执行顺序约束

- 严格按同步 skill 的顺序执行：完成资产盘点后，先跑完整首轮并保存每个用例的 PASS/NG/未执行状态；首轮结束前不得围绕单个失败用例反复调参或改变其测试语义。
- 首轮完成后，按问题账本把失败按共同根因聚类，在至多五轮的边界内修复并复测。任何例外都必须由用户明确授权并写入日志。
- 全量首轮期间每三分钟巡检正在运行的用例：检查日志是否继续推进、Ray job 状态和 NPU 实际负载。若日志已出现明确失败且 Ray/父进程未退出，保存原始日志、记录运行器退出码误判风险，终止仅属于该用例的 job 并继续队列；不得空等到通用超时。
- 通用单项超时只是硬上限。若日志连续十分钟没有新增记录，立即视为疑似卡死并检查 Ray/NPU；确认无有效进展后停止该用例，标记 NG，并在问题账本记下日志路径、最后进展时间、Ray job id 与判定依据。

## 转换输出和资产规则

- 资产 inventory 或本机搜索未解析出完整路径时，必须主动向用户询问 HF 权重根目录、
  数据集根目录，以及已有 Megatron checkpoint 根目录或是否使用本轮转换缓存。环境变量未设置、
  约定目录为空和搜索无结果都不能单独作为“资产不可用”的最终证据。
- 收到用户路径后必须验证可读性、每个 inventory 相对路径和 checkpoint marker；只针对剩余缺口
  继续追问。除用户明确确认资产不可用、补充后的路径仍无效，或用户明确要求非交互运行外，
  不得把关联 ST 标为 `UNEXECUTED_MISSING_ASSET`，也不得在未询问的情况下结束本轮同步。
- 转换器只能用 AST/CST 定位修改位置，不能用 `ast.unparse` 输出整个文件；后者会丢失注释、空行、括号分组与可审查性。
- 从上游测试文本复制并做最小替换：HF 路径使用 `HF_WEIGHTS_ROOT`，`--ref-load` 使用 `MEGATRON_WEIGHTS_ROOT/<MODEL_NAME>_torch_dist`，数据集使用 `DATASETS_ROOT`。
- `--hf-checkpoint` 与 `--ref-load` 必须按参数分别映射；一个 f-string 同时含两者时，不能按整段字符串推断根目录。
- 任何执行前必须运行 `prepare_test_resources.py`。复用 checkpoint 前，至少验证 `<model>_torch_dist/latest_checkpointed_iteration.txt` 存在且内容为 `release` 或整数；目录存在不足以证明可用。
- 缺失 checkpoint 必须先转换并验证，再让关联用例进入 Ray。转换自身失败时记录为依赖/资源问题，关联测试保持未执行，不得标记 PASS。

## Qwen GDN 依赖

Qwen3.5 的 HF→Megatron 转换会在构建 `Qwen3_5GatedDeltaNet` 时加载 GDN 后端。若日志为
`Qwen GDN backend 'fla' requires flash-linear-attention`，先确认 `fla`、MindSpeed 与 NPU GDN 定制算子是否已按 Ascend 文档安装。不要创建空实现或绕过线性注意力来伪造 checkpoint；保存完整转换日志并把 Qwen3.5/Qwen3.6 关联用例列为未执行。

## TMS 嵌套区域

`torch_memory_saver` 的公开对象会延迟创建 `_impl`；当前 TMS 是否处于区域内只能在
`_ensure_initialized()` 后，通过 `_impl._binary_wrapper.cdll.tms_get_interesting_region()` 判定。
遇到 `assert not ...tms_get_interesting_region()` 时，Megatron 的 param/grad buffer 分配应在进入
`torch_memory_saver.region()` 前动态判定并回退到 `nullcontext()`。不得读取不存在的顶层
`torch_memory_saver._binary_wrapper`，否则检查恒为空、修复形同未生效。该项应作为共享
Megatron 兼容补丁复测，不以单个用例规避 TMS。

## torch_npu 初始化 OOM 与 TMS resume 生命周期

- Qwen3-4B colocate 在 `torch_npu 2.10.0` 下曾于 Megatron Actor 初始化申请 7.49 GiB 时 OOM；升级到
  `torch_npu 2.10.0.post4` 后该 OOM 消失并进入真实 rollout。该结论仅适用于匹配的 CANN/PyTorch/模型与
  allocator 签名，后续环境不得无条件升级套用。
- 升级后出现的 `torch_memory_saver.resume(): cpu_backup should not be nullptr` 是后续独立问题：TMS 0.0.8
  原生 allocation metadata 没有记录 allocation 是否真正 pause。pause 期间新建的 active allocation 被全量
  resume 错当成 offload allocation，因此访问不存在的 CPU backup。
- 修复必须在 TMS 原生层增加 per-allocation `is_paused`，并使 `malloc/pause/resume/free` 成为幂等状态机；仅关闭
  param/grad buffer CPU backup 不能修复根因。完整签名、源码基线、验证脚本和部署要求见
  `../../npu-test-autofix/references/tms-lifecycle.md`。
- 已保存 Qwen3-4B PPO 日志证明 wake-up、actor train 8/8 和后续 update_weights 通过；
  `test_qwen3_4B_streaming_partial_rollout_npu.py` 又由用户完整验证 PASS，覆盖 streaming abort、16 个 partial
  sample 回收、Actor resume/train 和下一轮生命周期。后者在保存独立成功日志前标记 `USER_VERIFIED`。

## 真实 rollout 的 NPU 图捕获与指标中间件

- `torch_npu` 在任一图捕获尚存时调用 `torch.cuda.empty_cache()` 会触发
  `captures_underway.empty()` 内部断言。权重更新路径中的 allocator 清理只是优化：NPU 分支保留
  IPC 回收、跳过 `empty_cache()`，CUDA 分支维持原操作；不要通过关闭 rollout 或修改测试参数规避。
- FastAPI 的已包含路由可在 `request.app.routes` 中以没有 `path` 的 `_IncludedRouter` 出现。
  记录 HTTP 指标时必须用 `getattr(route, "path", None)` 并回退到 `request.url.path`，否则指标
  中间件本身会使 `/v1/loads` 返回 500，掩盖已完成的训练进展。

## MiMo MTP + EAGLE + CUDA Graph 的 Ascend 适配经验

- 在 Ascend 图捕获期间，boolean indexing 往往会下沉为 `torch.nonzero()`；例如
  `hidden_states[positions == 0] = 0` 会触发 `aclnnNonzeroV2`，并在 SGLang CUDA Graph
  capture 中报 `Not allow to synchronize captured-stream`。遇到这类写法时，优先改成
  `torch.where`/逐元素 mask，而不是先关闭 graph。
- MiMo MTP 场景下，若已确认 SGLang `mimo_mtp.py` 存在上述 boolean indexing，应把它归为
  后端兼容补丁：用 `mask = (positions == 0).unsqueeze(-1)` 加 `torch.where(...)` 替代原地布尔索引。
- `slime/utils/memory_utils.py` 中的 `torch.cuda.empty_cache()` 在 NPU graph 尚未完全释放时可能触发
  `captures_underway.empty()` 断言。这属于运行时清理兼容问题，应对特定报错做保护并保留其他异常抛出，
  不要把它误归为测试参数问题。
- `test_mimo_7B_mtp_only_grad_npu.py` 这类用例若需要保留 `EAGLE + CUDA Graph`，测试侧只做“收敛到 Ascend
  支持区间”的调整：例如把 `--sglang-cuda-graph-max-bs` 从 8 收到 2、`--sglang-mem-fraction-static`
  从 0.8 收到 0.6，并把 `--sglang-speculative-eagle-topk` 固定为 1。Ascend backend 不支持
  `speculative_eagle_topk > 1` 时，不要继续尝试用更高 topk。
- 若 colocate 形态下出现疑似显存竞争、graph 生命周期、weight update 清理或 rollout/训练互扰问题，可把
  “关闭 colocate、改为非共卡” 作为优先排查项之一。它属于运行形态切换，不等同于根因修复；日志中要明确记录
  是否使用 colocate，以及关闭 colocate 后是问题消失、变成资源瓶颈，还是暴露出新的真实错误。
- 归档问题时把该类问题拆成三层记录：`sglang` 后端补丁、`slime` 运行时补丁、测试配置收敛。不要把三者混成
  “单用例调参通过”，否则后续复盘无法区分真正需要 upstream 的修复点。
- 验证通过标准不能只看训练不报错；至少记录 MTP grad check 结果，确认 “MTP 参数有非零梯度、非 MTP
  参数无梯度泄漏”，再把该用例从失败账本移出。

## 第 5 轮收敛经验

- 本地没有 `<MODEL>-FP8` HF 目录时，不能让 NPU 用例默认引用 `-FP8`。对于同时支持 FP8 与普通 HF
  权重的 Qwen3-30B 用例，默认使用普通 HF 目录，只有显式设置 `SLIME_TEST_USE_FP8_ROLLOUT=1` 时才使用
  `-FP8`，并在执行前验证该目录存在。
- SGLang `enable_memory_saver` 在 NPU 上可能触发 `torch_memory_saver.cpp aclrtMallocPhysical failed` 并造成
  Ray job idle。Slime launcher 应保持 CUDA 原行为，但 NPU 默认关闭该开关，仅通过
  `SLIME_NPU_ENABLE_SGLANG_MEMORY_SAVER=1` 显式打开；若关闭后变成 Megatron actor OOM，说明已越过旧共因，
  后续应按训练并行和显存资源继续归因。
- Qwen3-30B MoE/DeepEP 若出现 `CamMoeDispatchNormal do tiling failed`，必须继续读取同一日志中的
  `HCCL_BUFFSIZE is too SMALL` 细节。若 `NEEDED_HCCL_BUFFSIZE` 大于当前 `HCCL_BUFFSIZE`，优先把问题归为
  HCCL buffer 配置，而不是继续追 HF/Megatron 权重路径。
- `ModuleNotFoundError: No module named 'acl'` 在 SGLang 子进程早期探测中可能是噪声；外部分离式传输要以
  `/tmp/sglang_external_*.log` 和最终 RayTaskError 为主。`mooncake`、`nixl`、`fla` 缺失是外部依赖，不得用
  空模块或假实现绕过。
- mixed offload fault tolerance 用例若在清理 Ray 后仍复现 `EADDRINUSE`，需要记录具体端口和 Ray job id。
  若随后出现 `KeyError: 'weights'`，应同时检查 SGLang weight updater 的 offload tag 恢复幂等性。
- 全量模式第 5 轮后仍有失败时，`result.json` 必须保持 `mode: "full"` 且写 `status: "NG"`；不要为了通过 schema
  把全量失败伪装成 `trial_existing_assets`。`NG_AFTER_MAX_ROUNDS` 可作为说明字段或审计文字保留，但机器状态使用
  `NG`。因缺资产导致已经执行失败的用例计入 `ng`，同时可计入 `missing_resources`；校验关系应是
  `missing_resources <= ng + unexecuted`。
- clean replay 前必须清理工作树中的 `**/__pycache__/*.pyc` 等运行产物；不得把运行产物写入 patch。现存
  `overlay/` 只作为只读历史兼容输入。若验证树里有 `tests/__init__.py` 这类为了导入语义新增的源码文件，
  应生成 `/dev/null` new-file patch 并登记 Manifest，不得再写入 overlay，也不得在 replay 中排除。
- clean replay 的 `DIFFERENT <source-file>` 是未持久化源码差异证据。先判断该差异是否属于有效 NPU 适配；有效时必须
  重新生成最小 patch 并放回 manifest，例如 `tools/convert_hf_to_torch_dist.py` 的 NPU/HCCL 初始化差异；只有明确的
  日志、cache、`.pt`、`torch_compile_debug` 等运行副产物才允许作为 replay exclude。
- clean replay 必须在独立、干净的树上重放，且只接受 `patches/MANIFEST.toml` 中登记的 patch 与未变化的
  legacy overlay。任何只存在于本机工作区、临时脚本、日志目录或 `__pycache__` 的改动都不算可移植修复；
  所有源码和测试改动必须先沉淀到 patch，运行经验才写入 skill 参考。

## Moonlight-16B-A3B NPU 收敛经验

- Moonlight-16B-A3B 这类大模型 ST 若在 8 卡 colocate 形态下、Actor 与 SGLang 共卡启动，Megatron 很容易在
  `ParamAndGradBuffer` 初始化阶段直接 OOM。首要排查顺序不是继续压单个参数，而是先判断是否应扩到 16 卡并把
  Actor/Rollout 分卡运行。
- 从 colocate 改成非共卡后，必须同时显式设置 rollout 总卡数。仅保留
  `--rollout-num-gpus-per-engine` 而缺少 `--rollout-num-gpus` 会导致 placement 计算中出现
  `actor_num_gpus + args.rollout_num_gpus` 的 `NoneType` 错误。
- Moonlight 当前环境下，rollout 采用 `8 NPU / TP2 per engine` 可能导致 SGLang server 直接 crash；
  若出现 `Server process terminated unexpectedly`，优先回退到已验证更稳的 `8 NPU / 2 × TP4 engine`
  形态，并同步降低 `--rollout-max-response-len`，不要先把问题归因到权重或数据集。
- 若 SGLang server 已成功启动，但后台 NPU Graph 线程在 `graphs.update()`、`torch.npu.current_stream()`、
  `LazySetDevice` 处报 `507033`，并伴随 `Failed to start the device`、`ctx can not be NULL`、
  `rtSetDevice execution failed` 等 CANN 日志，应把它归类为当前环境的 NPU Graph 兼容问题。此时可直接用
  `--sglang-disable-cuda-graph` 继续验证训练主链路，并移除 `--sglang-cuda-graph-max-bs`。
- 当日志已进入 `actor train: 100%` 或 `optimizer.step()`，说明 rollout、forward、backward 主链路基本打通；
  若最终挂在 `TransformerEngineNPU fused_adam.py`、`exp_avg_sq.sqrt()` 等位置并表现为 NPU OOM，应把问题归为
  optimizer state / 中间计算显存压力，而不是重新回头调 rollout。
- Moonlight 这类 case 的有效收敛组合包括：`--use-distributed-optimizer`、
  `--optimizer-cpu-offload`、`--use-precision-aware-optimizer`，同时把训练压力收紧到
  `--micro-batch-size 1`、`--max-tokens-per-gpu 1024`。若此前还启用了
  `--accumulate-allreduce-grads-in-fp32`，应优先删除，因为它会显著放大 grad buffer 占用。
- 建议把该类问题链按以下顺序记录：`8卡 colocate Actor OOM` → `16卡 Actor/Rollout 8+8 分离` →
  `补 rollout 总卡数` → `SGLang TP2 crash` → `回退 TP4 + response 1024` → `NPU Graph 507033` →
  `disable cuda graph` → `FusedAdam optimizer.step OOM` → `CPU offload + precision-aware optimizer +
  micro-batch=1 + max-tokens=1024`。
- AIME eval 或其他长耗时 eval 在定位 optimizer OOM 时可以临时关闭，以缩短验证回路；但日志中必须明确这是
  调试态裁剪，最终是否恢复 eval 取决于该 ST 的正式契约。

## THD packed sequence + Context Parallel：Ascend 数据契约修复

- 状态：`FIXED / multi-parallel regression verified`。原先的 `TP4 + CP1 + log-probs-chunk-size 1024`
  仅作为历史 workaround 保留，不再代表当前正式解决方案。
- 原始现象是 CP-local tensor 已完成切分，但 THD sequence metadata 仍为 global 长度，典型表现为
  `Query T = N`、`sum(seqLen) = 2N`；NPU FlashAttention 的 TND tiling 或后续 RoPE 因 tensor physical
  length 与 metadata 不一致而失败。Qwen 和 GLM 出现同类报错时，应归为公共 THD+CP 数据路径问题。
- 根因位于 `slime/backends/megatron_utils/data.py` 的旧 THD+CP packing 逻辑未遵循
  Megatron/TransformerEngine packed-sequence CP 契约：
  - 没有独立维护表示真实 token 边界的 `cu_seqlens` 与表示 padding 后物理边界的
    `cu_seqlens_padded`；
  - CP 下先用 `slice_with_cp()` 自行切 sequence，再以 `cu_seqlens * cp_size` 恢复 metadata，语义错误；
  - `cu_seqlens_q_padded` 直接复用 `cu_seqlens_q`，且每条 sequence 未按 `2 * cp_size` 对齐 padding；
  - NPU 路径缺少 NVIDIA TransformerEngine 的 `thd_get_partitioned_indices` 等价实现；
  - `PackedSeqParams.max_seqlen_q/kv` 使用未 padding 的最大长度，导致 RoPE physical length 不一致。
- 修复时必须保持以下契约：
  - 每条 THD sequence 按 `2 * cp_size` 对齐 padding；
  - 分别生成 real `cu_seqlens` 和 physical `cu_seqlens_padded`；
  - 在 `cp_utils.py` 提供 NPU/PyTorch THD CP partition fallback，CP rank `r` 选择 block
    `[r, 2 * cp_size - r - 1]`；
  - `tokens`、`loss_mask` 等所有 token-level tensor 必须复用同一组 partition indices；
  - `PackedSeqParams.cu_seqlens_q/kv` 使用 padded metadata，`max_seqlen_q/kv` 使用 padded maximum
    sequence length。
- 验证不能只确认单个 CP2 case 启动成功，应覆盖 attention、RoPE、loss 和 gradient 一致性。已验证：
  - Qwen3-4B `TP2 + CP2` checkpoint case：PASS；
  - GLM 9B `TP2 + CP2`：PASS；
  - Qwen3-0.6B parallel-check：`CP2`、`TP2 + CP2`、`PP2 + CP2`、`TP2 + PP2 + CP2`、`CP4`、
    `calculate_per_token_loss=False/True` 及 grad norm comparison 全部 PASS。
- 因此该问题不能再标记为未修复。当前结论是 Ascend NPU 的 THD packed sequence Context
  Parallel 数据路径已修复，且已通过 CP4 与 TP/PP/CP 混合并行回归。
- 历史排障记录：在根因修复前，`test_quick_start_glm4_9B_npu.py` 曾用 `CP1 + TP4 +
  log-probs-chunk-size 1024` 绕开 CP 路径并缓解 `ref_log_probs` OOM。该配置可用于故障隔离，但不能代替
  CP 正确性回归，也不应据此回退正式用例的 CP 覆盖。

## Qwen3-30B-A3B：Actor DeepEP 与 CP2+TP4 尾部对齐

- 状态：完整 30B 用例 `USER_VERIFIED PASS`；成功日志尚未独立保存，下一轮需升级为 `LOG_VERIFIED`。
- 第一根因是 Actor DeepEP 空 token rank：hidden state 占位 shape 为 `[1, H]`，routing map 为 0 行，触发
  `Permute not possible`；异步执行可延迟表现为 507015。SGLang 保留 DeepEP，Actor 默认改标准 alltoall，
  `SLIME_TEST_USE_ACTOR_DEEPEP=1` 仅用于单独复测 Actor DeepEP。
- 第二根因是 TP4 所需额外尾部 padding 被并入最后一个真实 THD sequence，改变 CP2 镜像 chunk 边界；
  `get_responses()` 仍按真实长度提取，导致长序列 logits/token 错位。短序列正常而长序列 Megatron/SGLang
  log-prob 严重分叉是关键正向签名。
- 最终布局把 TP tail 建模为独立、零真实 token 的 physical-only sequence。真实 sequence 只做
  `2 * cp_size` 对齐，tokens/loss mask 共用 padding metadata 和 CP partition indices，CP-local 总长保持 TP4
  可整除。
- 已验证单条 4809 response、16 条/15 dynamic microbatch，以及完整 AIME + 两轮 16-sample/8192-token
  rollout/train/update_weights。完整诊断、排除条件与数值见
  `../../npu-test-autofix/references/moe-thd-cp-tp.md`。

## Qwen3-4B PPO disaggregate：传输依赖与 actor logprob OOM

- 状态：`FIXED / user-verified end-to-end PASS`。
- 必须按时间顺序拆成两个根因，不能把首次 NIXL 异常误判为 PPO 或 NPU OOM：
  1. GPU 源用例显式指定 `--rollout-data-transport nixl`，但 Ascend 测试环境没有 `nixl`。
     失败发生在 rollout 数据通过 `ray.put(..., _tensor_transport="nixl")` 返回 trainer 时；NPU 用例
     删除该参数，使用 Slime 默认 object-store transport。
  2. 绕过 NIXL 后已进入 actor PPO train，随后在 policy loss 的 logprob/entropy
     `vocab_parallel_softmax` 申请 4.35 GiB 并 OOM。长响应与 dynamic batching 使
     `[tokens, vocab_shard]` 临时张量成为显存峰值。
- 已验证配置保持 response length、PPO 阶段和 TP2/CP2 拓扑不变，只把
  `--max-tokens-per-gpu` 从 `16384` 降为 `8192`，并增加 `--log-probs-chunk-size 2048`。
- 验证信号必须包含 rollout、reward、critic/value、advantage/return、old log_probs、actor PPO
  train 和正常退出；仅看到 `actor train` 启动不能标记 PASS。
