你正在独立工作区 `{{WORKSPACE}}` 中执行 slime 上游同步验证。

必须同时使用 `gpu-to-npu-test-converter` 和 `npu-test-autofix` 两个技能，并完整遵守它们的资产盘点、行为契约、最多五轮修复、问题台账和中文报告要求。

本轮参数：

- 上一次全量成功的上游 SHA：`{{PREVIOUS_SHA}}`
- 当前目标上游 SHA：`{{UPSTREAM_SHA}}`
- slime 子模块路径：`{{SUBMODULE_PATH}}`
- 运行模式：`{{RUN_MODE}}`

任务：

所有 NPU 技能脚本都以 `slime` 子模块为仓库根执行。相对测试路径必须以该仓库根为基准，例如在父仓库中调用 `python3 .agents/skills/npu-test-autofix/scripts/prepare_test_resources.py --repo-root slime --test-root tests --inventory`；禁止使用会解析成 `slime/slime/tests` 的 `--repo-root slime --test-root slime/tests`，也不要把父仓库误当成包含 `tests/` 的单体 slime 仓库。

0. 编排器已经在任何 patch 和依赖变更之前生成 `.upstream-sync/change-assessment.json`。先核对其中 `base_sha`、`target_sha`、commit 清单和 changed paths；目标 SHA 变化时必须逐项完成 `source_changes.feature_assessment`，每项写 `capability`、`change_type`、中文 `summary` 和源码/commit `evidence`。必须判断支持特性的新增、删除、默认值/CLI/配置/行为变化，比较上游三方依赖声明与构建来源，并核对 test 与 GPU CI 的新增、修改、删除、移动、参数化和分类变化。即使判断“无用户可见能力变化”也要给出有证据的明确条目，不能留空。

   然后运行 `.agents/skills/slime-upstream-sync/scripts/doctor.py`，以父仓库为 `--repo-root`，使用 `--check-sync-layout --check-upstream-dependencies --require-npu` 检查同步布局、目标上游依赖和 NPU，并把结果写入 `.upstream-sync/environment.json`。`full` 模式还必须使用 `--require-assets`。依赖契约以目标 `slime/docker/Dockerfile` 为唯一源头，逐项比较版本、源码仓/fork、commit/tag、wheel URL 与构建标识、extras、原生 wheel 坐标和上游运行时断言，不能只比较 pip 版本。每项必须分类为 `UPSTREAM_EXACT`、有完整映射和验证证据的 `NPU_EQUIVALENT`，或已证明 NPU 执行路径不需要的 `CUDA_ONLY_NOT_APPLICABLE`。例如普通 PyPI `sglang-router==0.3.2` 不等价于上游 slime 指定且 `sglang_router.__version__` 必须包含 `slime` 的定制 wheel。任何未分类项、fork/build/断言差异都按不匹配处理，必须实际同步上游并升级当前环境，不能以 NPU 差异为由忽略。同步时迁移对应第三方 patch，更新 quick install、NPU Dockerfile、quick-start 安装说明和原生 NPU wheel/kernel 坐标，再更新当前容器源码、重新安装并重建；只记录 NG、只报告 patch 冲突或只改版本字符串均不算完成。依赖 patch、编译和安装失败属于本轮必须迭代修复的代码问题，不是停止条件。升级后重新运行 doctor 覆盖 `.upstream-sync/environment.json`，只有全部 `dependency_installer_*` 和 `dependency_runtime_*` 为 PASS 才能继续 inventory、转换和后续测试；否则本轮不得生成可发布结果。最终 `component_inventory` 必须列出 Slime、SGLang、Megatron-LM、MegatronAdaptor、mbridge、Megatron-Bridge、TransformerEngineNPU、SGL Kernel NPU、router、torch/torch-npu、DeepEP、TorchMemorySaver、MemFabric、Triton Ascend、CANN/ATB 的 commit 或版本/build identity。

1. 检查准备脚本已经按照 `patches/MANIFEST.toml` 的权威顺序把 patches 正确物化到子模块；现有 `overlay/` 只作为只读历史兼容输入复制。manifest 与目录不一致时先修复 manifest；若 `.upstream-sync/materialize-failures.txt` 非空，不得仅写 BLOCKED 后停止，也不得在全量 baseline 前死磕该 patch。先将失败项记为 `PENDING_ANALYSIS`，只隔离失败 patch；存在显式依赖时把后续依赖 patch/用例标为受影响，继续其余独立 patch、转换和可执行测试。只要 target 组件基线发生变化，该 target 下 clean apply 的 patch 也必须重新做语义审计，因为 apply 成功不能证明旧补丁仍然需要。完整首轮结束后结合旧意图、目标源码/commit 和运行证据，逐项判断 `KEEP_VERIFIED`、`REWRITE`、`SPLIT`、`UPSTREAM_ABSORBED` 或 `DROP_UNSAFE`，写入 `.upstream-sync/patch-decisions.json`；判断 `UPSTREAM_ABSORBED` 时优先在不应用旧 patch 的干净源码上验证。更新 patch/manifest，并在后续轮次从干净树重试。发布前失败文件必须为空且所有 decision 为 `VERIFIED`；只有已证明无法取得的外部资源才允许停止全流程。
2. 识别目标上游 GPU CI 的全部唯一 ST 和参数化配置，再单独识别 `{{PREVIOUS_SHA}}..{{UPSTREAM_SHA}}` 范围内新增或修改的增量转换集合。若 `PREVIOUS_SHA` 为 `NONE`，从环境变量 `SLIME_SYNC_BASE_SHA` 读取基线分支原子模块 SHA。不要仅凭文件名判断。完整盘点现有 `tests/tests_npu/st`、`tests/tests_npu/nightly_CI` 和 `tests/tests_npu/ut`；不得用增量数量代替 `baseline_discovered`，不得遗漏 nightly。
3. 对缺失或过期的 NPU 对应用例执行 `gpu-to-npu-test-converter`，保留断言和行为契约。
4. 按 `npu-test-autofix` 做资产清单、依赖和 NPU 预检；需要的本地资产或 NPU 前置条件缺失时立即按失败关闭，不下载、不猜路径。
5. 执行完整 NPU 测试基线并最多修复五轮。补丁、编译、依赖、工作流、launcher、产品代码和 NPU 适配失败都必须继续诊断迭代，不得作为提前结束理由。只有明确缺失模型、数据集、NPU、磁盘容量、外部服务或凭据等无法在范围内补齐的资源问题才允许停止。未执行、跳过、超时、缺资源均不能算 PASS。
6. 把所有可持久化修改统一同步回父仓库的 `patches/`。修改上游已有文件和新增文件都必须生成最小 patch 并更新 `MANIFEST.toml`；新增文件使用 `/dev/null` new-file diff。禁止向 `overlay/` 新增、修改或删除内容，收尾脚本会验证 legacy overlay 前后完全一致。收尾脚本会在第二棵干净源码树中重放并做完整文件、符号链接和权限比对。不要提交、不要推送、不要更新成功标记。
7. 生成中文审计报告 `.upstream-sync/audit-report.zh-CN.md`。
   报告必须包含独立的“上游变化”和“测试变化”章节，展示 base/target SHA、commit 范围、支持特性影响、三方依赖声明变化、test/CI 增删改；还必须包含环境组件表，逐项展示 Slime、SGLang、Megatron-LM、MegatronAdaptor、mbridge、Megatron-Bridge、TransformerEngineNPU、SGL Kernel NPU、router、torch/torch-npu、DeepEP、TorchMemorySaver、MemFabric、Triton Ascend、CANN/ATB 的上游/base commit、patched HEAD 或二进制版本/build identity，并附列 MindSpeed、ModelOpt、Transformers、Ray、FastAPI、Mooncake/NIXL 的安装状态。增加“dev 分支集成”章节，记录 slime-ascend base branch/SHA、目标 Slime SHA、Manifest hash/patch 顺序、依赖 base 与 patched HEAD、安装器变更、clean replay、未解决问题和复核命令；门禁未通过时不得写成可合入。报告还必须包含动态扫描得到的 `slime/tests/**/test_*.py` 全量清单、每个测试用例的测试特性/行为契约、完整补丁决策表、完整 GPU ST、增量转换集合、现有 NPU ST/nightly/UT、每轮修复以及最终每用例状态。
   inventory 只能是本次 run 的审计快照，禁止沿用旧 inventory 作为发现源。
8. 最后生成 `.upstream-sync/result.json`，严格使用以下结构：

```json
{
  "schema_version": 2,
  "status": "PASS、NG 或 TRIAL_COMPLETE",
  "mode": "full 或 trial_existing_assets",
  "upstream_sha": "{{UPSTREAM_SHA}}",
  "baseline_discovered": 0,
  "executed": 0,
  "pass": 0,
  "ng": 0,
  "unexecuted": 0,
  "missing_resources": 0,
  "timeout": 0,
  "replay_verified": false,
  "report_path": ".upstream-sync/audit-report.zh-CN.md",
  "conversion_plan_path": ".upstream-sync/conversion-plan.json",
  "validation_result_path": ".upstream-sync/validation-result.json",
  "environment_path": ".upstream-sync/environment.json",
  "change_assessment_path": ".upstream-sync/change-assessment.json"
}
```

`full` 模式下，只有所有发现的完整 NPU 测试都实际执行且 PASS 时，`status` 才能写 `PASS`；任何无法完成的验证都写 `NG`。代理必须把 `replay_verified` 初始写为 `false`，收尾脚本完成独立重放后才会以确定性方式改为 `true`。

`trial_existing_assets` 模式下：

- 仍须先盘点全部 NPU 用例及其资产；
- 只执行当前 `SLIME_TEST_HF_WEIGHTS_ROOT`、`SLIME_TEST_DATASETS_ROOT` 和 `SLIME_TEST_MEGATRON_WEIGHTS_ROOT` 已满足的用例；
- 缺资产用例逐项记为 `UNEXECUTED_MISSING_ASSET`，计入 `unexecuted` 和 `missing_resources`，绝不算 PASS；
- 对实际执行的 NG 用例仍按 `npu-test-autofix` 最多修复五轮；
- 报告完整且 `pass + ng + unexecuted == baseline_discovered` 时写 `TRIAL_COMPLETE`，即使所有已执行用例通过也不能写正式 `PASS`。
- `mode` 必须写 `trial_existing_assets`；完整模式必须写 `full`。

不要进行任何 Git 推送。
