# Slime-Ascend v0.4.0 同步交接

> 更新：2026-10-15（Asia/Shanghai）
> 状态：基线与 Diff 已核实；开始隔离适配，UT 已有 109 项通过，ST/Nightly 尚未启动
> 工作区：`/home/c0088099/slime-ascend-v040`（容器 `sync-slime-0.4.0-20261015`）
> 原工作区：`/workspace/slime-ascend`，本轮未修改、未清理、未推送

## 已完成：环境与同步基线

- 用户任务说明与交接模板来自 IDE 工作区 `slime-sync/task.md`、`handoff.md`，已阅读。容器原仓库根目录没有这两个文件，因此在本隔离副本创建本交接文档。
- 容器有 16 张 NPU；Python 3.12.13、PyTorch `2.10.0+cpu`、torch_npu `2.10.0.post4` 可导入；CANN 为 9.1.0。`npu-smi info` 在当前命令输出中未产生内容，需进一步确认卡型号/状态。
- 原 Slime-Ascend：分支 `sync/slime-20260911`，HEAD `3e9011d3eddd5439c3631cc8ffec5236e1686af1`，领先同名远端 5 个提交。工作树存在大量未跟踪运行产物，Slime 子模块含现存 NPU overlay 修改；均未触碰。
- 原同步记录 `d9e7e64` 标注了 `4c193f1`，但当前已提交 gitlink 与子模块 HEAD 实际为 `5bae5bb7928d65906e24e99f5b5d3f99a4daaf93`。依照技能首次运行规则，以当前 gitlink 作为本轮变更评估 base。
- 上游 tag `v0.4.0` 解析为 `0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e`。base→target 共 14 个提交、198 个路径变化；自动评估记录在 `.upstream-sync/change-assessment.json`。
- 主要上游变化：score centering/top-p replay；分布式 fully-async 与 Straw 数据/队列/归档；pipeline RL 与 cache flush；训练恢复、保留服务重启、checkpoint shard 权重同步；参考 logprob top-p 修复；CP>1 linear-attention 梯度修复；queue/source API 清理；新增 Slime Ascend accelerator backend。不能整体覆盖 NPU 代码，须逐项对照上游新增 accelerator 实现和当前补丁。
- 测试差异：上游 tests 下新增 29、修改 34、删除 1；自动评估扫描到目标版本 126 个测试文件；GPU CI 配置变化 3 项。完整列表位于 `.upstream-sync/change-assessment.json`。
- 现有测试布局：`tests/tests_npu/ut`、`st`、`nightly_CI` 和 `ci_check.py` 均存在；仓库含 upstream-sync、GPU→NPU 转换、NPU autofix 三套 Skill。`patches/MANIFEST.toml` 当前校验通过，但必须随 Slime 与各依赖基线变更重新审计。

## 隔离工作区

- 新建分支：`sync/slime-v0.4.0-npu`，基于原同步分支 HEAD `3e9011d`。
- 在新工作区 checkout 当前 Slime gitlink `5bae5bb7…`，并复制了该子模块中现存已跟踪 NPU 变更和两个未跟踪 NPU 源文件，便于迁移评审；原目录保留原样。
- 技能目录 `.agents/skills` 从原容器工作区只读复制到新工作区；同步脚本不能假定另一个路径。

## 初始依赖预检（NG 是迁移输入，不是最终结果）

- 通过：NPU Python 导入、CANN 9.1.0、router `0.3.2+slime`、sgl-kernel-npu `2026.9.0`、MemFabric `1.0.8`，以及 mbridge、Megatron-Bridge、MegatronAdaptor、TransformerEngineNPU 的安装声明/运行源码检查。
- 失败：目标上游 `docker/Dockerfile` 的 Megatron commit 为 `4b4acac9a1d28ea6829c8d4f566d75698a21249d`，当前 installer 仍声明 `1dcf0dafa884ad52ffb243625717a3471643e087`；SGLang runtime 检查失败（当前模块无 `__version__`，doctor 未确认其 editable checkout/目标基线）。
- 同步技能 Doctor 还报告 `command_codex` 缺失、无 `overlay/` 目录。当前 patches Manifest 与目标 Dockerfile 检查通过。完整证据：`.upstream-sync/doctor-v040-initial.json`。
- 目前不能把初始 Doctor 标为 PASS，也不能开始宣称测试验证通过；先处理同步编排器可用性、patch 持久化和依赖目标基线。

## 迁移顺序

1. 填充 `change-assessment.json` 的中文语义评估，按训练/rollout/data/算法/依赖/测试分组，证据指向精确提交与源码。
2. 把现存 Slime NPU overlay 逐项与 v0.4.0 的 Ascend accelerator backend 对照；为保留的 Slime NPU 改动生成 `patches/` patch 并更新 Manifest/hash，确认 patch 可从干净 v0.4.0 树重放。
3. 按 v0.4.0 authoritative Dockerfile 更新 NPU 依赖声明。重点审计 Megatron 4b4acac… 在 MA+TENPU 训练栈上的兼容性，并确认 SGLang 实际 checkout、installer 声明和 runtime 对齐；不引入 MindSpeed。
4. 生成 29 新增/34 修改/1 删除 GPU 测试到现有 NPU UT/ST/Nightly 的映射；先适配低成本可移植 UT，再 ST，再 Nightly，保留原断言和资源要求。
5. 完成资源盘点后运行 NPU UT → ST → Nightly；失败按首个异常和日志定位，最小修复并回归。每一阶段后更新本文件及审计记录。
6. 生成 patch decisions、干净 replay 与中文结果报告；本轮不推送、不合并。

## 尚未完成 / 风险

- 尚未执行源码同步、转换新 NPU 用例或产品测试；没有任何 PASS 结论。
- 初始 Doctor 为 NG；SGLang 与 Megatron 需迁移评审，SGLang工作树基线异常需单独确认。
- 测试资产根路径未配置；须先运行技能资产 inventory，再确认本机 HF 权重、数据集与 Megatron checkpoint 可用位置后执行依赖这些资源的 ST/Nightly。
- 容器内 `npu-smi` 命令未显示卡状态；需保存具体型号/状态证据。
- 不覆盖旧工作区已有未跟踪日志、checkpoint、kernel cache 或现存 NPU 补丁；不执行推送、合并或不可逆清理。

## 2026-10-15 容器恢复与资产缓存检查

- SSH 已恢复至 `192.168.13.185`（`node159x86`）。镜像 `slime-ascend:sync-20260911`，ID `sha256:98172429a18ae521d0b92e9f1e345f65a6351eb3fbaa0c2f55f9af770ec54fb3`。
- 已从该镜像启动独立容器 `slime-ascend-sync-v040-20261015`，复用原容器的特权、host 网络/IPC、Ascend driver/firmware、`/home` 与 `/data` 挂载；状态 running。新容器可访问 `/home/c0088099/slime-ascend-v040` 与 NPU。原容器保持运行，仓库隔离。
- 新容器 `npu-smi info` 可正常读取 Ascend910。当前 NPU HBM 已接近满载，执行需占用设备的 ST/Nightly 前须复核空闲量，避免与原容器任务争用。
- 检查 `/root/.cache/huggingface/hub`、`/home/c0088099/.cache/huggingface/hub`、`/data`、`/home/c0088099` 的常用缓存/模型/数据目录，未发现盘点清单所需 HF 权重、GSM8K/AIME/DAPO 数据或 Megatron checkpoint；`/data` 现有内容为其他项目目录。资产依赖测试仍待用户提供/准备资产，未启动相关 ST/Nightly。
- 已恢复此前 Megatron-LM 隔离仓库的 rebase 检查点，仍在首个 NPU 补丁提交的冲突解决阶段；尚无新测试结果。
- 按用户命名偏好，新容器已重命名为 `sync-slime-0.4.0-20261015`，状态 running；镜像和运行配置不变。

## 2026-10-15 `/home/slime-ascend_CI` 资产核对

更正上一节的初步缓存结论：CI 资产位于 `/home/slime-ascend_CI/slime-ascend`；`weights` 和 `datasets` 是指向 `/home/data/weights`、`/home/data/datasets` 的软链接，因此只看 `/data` 或不跟随链接会漏检。

- HF 权重：清单中的 10/10 个模型目录存在；均有 `config.json`、`tokenizer_config.json` 和非空 safetensors 权重。带 index 的 7 个模型均核对了 weight map，分片缺失数为 0。
  - GLM-4.7-Flash (48 shards)、GLM-Z1-9B-0414 (4)、MiMo-7B-RL (4)、Moonlight-16B-A3B-Instruct (27)、Qwen2.5-0.5B-Instruct (1)、Qwen3-0.6B (1)、Qwen3-30B-A3B (16)、Qwen3-4B (3)、Qwen3.5-0.8B (1)、Qwen3.6-35B-A3B (26)。
- 数据集：清单所需的四个文件全部存在且非空：`aime-2024/aime-2024.jsonl` (13,029 bytes)、`dapo-math-17k/dapo-math-17k.jsonl` (10,490,834 bytes)、`gsm8k/train.parquet` (3,347,161 bytes)、`gsm8k/test.parquet` (607,228 bytes)。
- 未找到 Megatron 转换 checkpoint 根目录。`/home/c0088099/weights-MA`、`/home/slime-ascend_CI/slime-ascend/weights-MA` 和 `/home/data/weights-MA` 均不存在；在 `/home/data` 和 `/home/c0088099` 的限定深度搜索也未发现 `latest_checkpointed_iteration.txt`。后续若 ST 要求 Megatron checkpoint，需要先运行 HF→Megatron 转换并确认缓存目录；HF 权重和数据集本身无需补充。
- 可复核扫描报告：`.upstream-sync/asset-scan-20261015.json`。


## 2026-10-15 Megatron 适配与 UT 进展

- 在隔离依赖仓 `/home/c0088099/slime-v040-deps/Megatron-LM` 将两个现存 Ascend 适配提交 rebase 到 Slime v0.4 Docker 指定的 Megatron `4b4acac9a1d28ea6829c8d4f566d75698a21249d`。提交为 `078809a4`、`1b7755fb`，分支 `sync/rebase-v040-npu`；无冲突路径保留目标上游实现。异步 checkpoint 冲突合并保留 v0.4 `non_blocking` 与释放 tensor 局部引用语义。
- 22 个适配文件 + 新依赖全量 Megatron Python 源码 compileall 通过；只见上游已有 `kv_block_allocator.py` 不合法转义序列 SyntaxWarning。补丁输出 `patches/third_party/megatron/megatron-npu-v040-rebased.patch`，manifest 增加对应条目；旧 v0.3.1 补丁移至 `.upstream-sync/legacy-megatron-patches/` 作为迁移来源，已从可应用 Manifest 移除，避免 installer 误应用。patch manifest validate 通过。
- 将 `docker/Dockerfile.a3.ubuntu22.04.cann90.latest`、`scripts/quick_install.sh` 的 Megatron pin 从旧 `1dcf0dafa` 对齐到目标完整 commit。当前镜像依赖仍是旧工作区装好的版本；未重建镜像或覆盖安装。
- Slime adapter 与 allocator capture 修复 patch 在 clean replay v0.4.0 树上做 reverse-clean + forward `git apply --check`/apply 均成功。
- NPU UT 执行：既有 UT 11 passed；迁移映射的 checkpoint/tokenizer/disk-delta/stateless-adam/update-weight/argument-validation 通过 98 项；总计 109 passed。测试适配文件位于 `tests/tests_npu/ut/*_v040_npu.py`，source AST 路径指向 clean replay，避免 `/workspace` 旧 Slime 导入污染。
- 12 个 Straw/PD-Mooncake 子集未运行；`straw-queue` 不在容器，top-p replay 用例会触发当前挂载 `/workspace/MegatronAdaptor` 与旧 Megatron 导入错配；Straw metadata/published-payload 2 个新增测试亦被缺少 `straw` 阻塞。Nightly/ST 尚未启动。
- 109 项 UT 结果不是 Megatron-NPU kernel 训练验证；仅覆盖 CPU/模拟边界和现有 NPU UT。转换 checkpoint 尚未生成，虽然 HF 权重和数据集已齐备。

### 下一步

1. 完成 top-p/Straw 新增测试导入隔离或依赖安装方案，不借由改动系统全局包来隐藏缺失。
2. 在独立环境构建/安装目标 Megatron `4b4acac…` + 两笔 NPU 适配补丁，做 MegatronAdaptor/TE-NPU import/单卡初始化 smoke；评估 Docker/installer 两处 pin 的来源与适用 CANN 版本。
3. 生成一份小模型 Megatron checkpoint，运行模型 ST（Qwen2.5-0.5B）并读取首个日志异常修复；确认 NPU HBM 空闲和旧容器争用后再逐项跑 ST。
4. 最后按 NIGHTLY 次序选择小模型到大模型；当前设备可见 16×Ascend910，每卡 HBM 约 64GiB。未推送、未合并、原工作区未修改。


## 2026-10-15 Qwen2.5 0.5B 转换推进与资源状态

- 在新容器以 editable `--no-deps` 安装隔离 Megatron-Core `0.19.2+1b7755fb4`，导入路径优先指向 `/home/c0088099/slime-v040-deps/Megatron-LM`；MegatronAdaptor 初始化、HCCL 单卡进程组、目标 GPTModel 构建均成功。
- HF→Megatron 转换逐次消除了结构配置错误：`num_layers`/`hidden_size`/attention heads、RoPE/RMSNorm、Qwen2 输出投影无 bias 但 QKV 有 bias、GQA KV heads=2。当前模型参数量为 389,437,312；权重形状校验已进入逐参数复制。临时实验的 `None` bias 映射已撤销，保留的是模型构建参数修正。
- 最新转换阻塞为卡 0 `torch.OutOfMemoryError`，拷贝权重时需再申请 260 MiB，但当时只余 11.86 MiB。复核 `npu-smi`：16 卡 HBM 均被其他运行占用约 59–60 GiB/64 GiB。未停止共享任务，也未声称 checkpoint 已生成；ST/Nightly 仍未启动。完整失败日志 `/tmp/slime-v040-qwen05-convert.log`。
- 将转换参数修复限定在 `tools/convert_hf_to_torch_dist.py`，形成 `patches/upstream-fixes/qwen2-hf-converter-v040.patch` 并加入 Manifest，校验通过；不会在通用训练参数默认函数中覆盖调用方提供的显式配置。
- 旧 Megatron 补丁实际已迁出 `patches/` 可应用目录到 `.upstream-sync/legacy-megatron-patches/`，Manifest 不再把它们列为可应用项。否则旧补丁会在新目标基线上被 `git am` 错误执行。
- 下一步：设备可用时先完成单卡 checkpoint 转换，核验 `latest_checkpointed_iteration.txt` 与分片，再执行 Qwen2.5 ST。空闲前继续处理纯 CPU 的上游测试映射、Slime 子模块目标树与补丁重放、安装脚本可复现性。

- 将迁移 UT 中绝对写死的 replay 路径改为 `SLIME_V040_UPSTREAM_ROOT` 可配置路径，默认读取本工作区的 `slime` 子模块；以环境变量指向 clean v0.4 replay 复验 tokenizer/checkpoint 两组 15 passed。

## 2026-10-15 隔离仓落地状态

- Slime 子模块已切至上游 `0b0c277d`，旧 NPU overlay 先保存到子模块 stash `pre-v040-isolated-overlay-20261015`；Manifest 中 target=slime 的三项补丁在该干净基线上按顺序应用成功。
- 子模块分支 `sync/slime-v0.4.0-npu`，提交 `2946c3d932d84ac63fc38231ccb0c78f5d2f0a78`，包含 v0.4 NPU adapter、allocator capture 修复和 Qwen2 HF 转换配置修复。源码 compileall、`git diff --check`、基于实际子模块的 15 项 tokenizer/checkpoint 映射 UT 均通过。
- 外层分支 `sync/slime-v0.4.0-npu` 的 gitlink 已指向此子模块提交；本轮未推送或合并主分支。


## 2026-10-15 UT 总门槛完成

- 新容器安装 Slime v0.4.0 `requirements.txt` 声明的 `straw-queue 0.1.2`。上游 Straw metadata、published payload retry、参数验证 85 passed。
- 将 `test_rollout_metadata_index.py`、`test_published_payload_retry.py`、`test_reference_top_p_replay.py` 映射到 `tests/tests_npu/ut/*_v040_npu.py`。top-p 迁移用例使用目标 Megatron-Core，不注入仅用于无 Megatron CPU CI 的 `_cp_dist_helpers` 假包，21 passed。
- 完整 `tests/tests_npu/ut` 在目标 Slime+Megatron 路径下运行：**155 passed，50 warnings，118.70s**。命令环境 `PYTHONPATH=/home/c0088099/slime-ascend-v040/slime:/home/c0088099/slime-v040-deps/Megatron-LM:$PYTHONPATH`，`SLIME_V040_UPSTREAM_ROOT=/home/c0088099/slime-ascend-v040/slime`；完整日志 `/tmp/slime-v040-all-npu-ut.log`。这是 UT gate，通过后才进入 ST。

## 2026-10-15 Qwen2.5 ST 启动与 Ray 资源状态

- 16 卡 HBM 占用下降后，Qwen2.5-0.5B-Instruct HF→Megatron 转换成功。缓存位于 `/home/c0088099/slime-v040-work/weights-MA/Qwen2.5-0.5B-Instruct_torch_dist`；包含 `latest_checkpointed_iteration.txt=release`、`release/.metadata`、`release/metadata.json` 和约 989 MB 分片。转换日志 `/tmp/slime-v040-qwen05-convert.log`。转换配置补齐 Qwen2 SwiGLU。
- 新容器内 `torchvision 0.25.0` 原为 CUDA wheel，与 CPU 版 PyTorch 不匹配，导致 ST import 阶段报 `torchvision::nms does not exist`；已在隔离容器重装 `torchvision==0.25.0+cpu`，`torchvision.io.decode_jpeg` 导入成功。
- 下一次 ST 发现 Ray runtime 仍将 `/workspace/Megatron-LM` 放在目标隔离 Megatron 前面，引发 `load_common` 参数类型错配。`command_utils.py` 两处路径改为从 `SLIME_MEGATRON_ROOT` 读取，默认值保留旧路径；相关补丁已登记。`py_compile` 通过。
- 第三次 ST 提交成功，但前一次失败运行遗留两个 4-NPU Ray placement group，使新 serving group 一直 PENDING。停止了本次提交的 job `raysubmit_WuTiLE5qmHUXaWU5`。准备按 creator_job_id 清理本次任务遗留组时，Ray GCS `192.168.13.185:6379` 无响应，宿主机报 CPU soft lockup；`ray status` 后续仍无法连接。未重启共享 Ray 集群，未删除其他任务资源。当前 **ST 未通过，Nightly 未开始**；日志 `/tmp/slime-v040-qwen05-async-st.log`。
- 下一步先确认宿主机及 GCS 恢复，重新列出 Ray job/actor/placement group，仅清理本任务 job `02000000` 与 `03000000` 的残留，然后重跑该 ST；按报错继续修复，再进入 Nightly。未推送或合并主分支。

## 2026-10-16 Ray 恢复与 Qwen2.5 ST 通过

- 隔离容器仍在运行，但 2026-10-15 的 GCS/raylet 已退出。首次重跑时 `ray status` 对失效 GCS 地址超时，`execute_train()` 未捕获 `TimeoutExpired`。现将该异常视为 Ray 未运行，随后启动新 head；`py_compile` 与 `git diff --check` 通过，补丁 `megatron-root-override.patch` 已更新。
- 重跑 `tests/tests_npu/st/test_qwen2.5_0.5B_gsm8k_async_short_npu.py`：**退出码 0，Ray job `raysubmit_ehrirgUisw48g5is` succeeded**。日志中确认 4 个 SGLang 引擎初始化、Megatron checkpoint 加载、训练 step 2 指标和权重同步。日志 `/tmp/slime-v040-qwen05-async-st-20261016b.log`。
- UT gate 仍为 155 passed；Nightly 下一步选 Qwen2.5-0.5B 短用例，使用隔离 checkpoint 缓存，运行后继续诊断。
