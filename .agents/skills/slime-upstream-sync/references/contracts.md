# 机器化契约

## 目录契约

所有相对测试路径都以 slime 仓库根为基准。外层仓库调用示例：

```bash
--repo-root slime --test-root tests
```

禁止使用 `--repo-root slime --test-root slime/tests`。

持久化边界：

- 修改上游已有文件：`patches/`。
- 新增源码、测试和配置文件：同样生成 `/dev/null` new-file patch 放入 `patches/`。
- `overlay/`：只读历史兼容输入。同步期间不得新增、修改或删除其内容；待历史内容全部
  转为 patch 后可整体移除。
- 执行状态和日志：`.upstream-sync/`、`.npu-test-autofix*/`，不得进入 patch 或 overlay。

`patches/MANIFEST.toml` 是所有持久化变更的唯一清单。新增文件 patch 也必须声明正确的
`target`、顺序和 hash；不得因为文件在上游不存在而绕过 Manifest 写入 overlay。

补丁清单中的 `target` 是应用域，不是说明文字：

- `slime`：由同步物化和 clean replay 按顺序应用。
- `sglang`、`megatron`、`megatron-bridge`、`mbridge`、
  `transformer-engine-npu`：登记为权威库存，由 `scripts/quick_install.sh` 在对应依赖仓应用，
  不得对 slime 子模块尝试应用。
- `sgl-kernel-npu`：作为 SGLang CUDA kernel 的 `NPU_EQUIVALENT` 登记在
  `scripts/quick_install.sh`，必须记录 release wheel 版本、硬件/torch/CANN 构建坐标、
  官方修复 commit 和 manifest patch。doctor 同时验证安装声明、运行包版本及关键源码断言；
  TMS 原生源码变化仍必须重建 wheel，不能用 Python 文件覆盖代替原生重建。
- `memfabric-hybrid`：作为 SGLang Ascend PD backend 的 `NPU_EQUIVALENT`，版本必须从
  当前 SGLang 源码的 `docker/npu.Dockerfile` 解析，并同时验证 quick installer 声明、
  已安装 distribution 版本和 `memfabric_hybrid` import；缺失时不得执行 PD 测试。

## conversion-plan.json

最小结构：

```json
{
  "schema_version": 1,
  "base_sha": "40-character SHA",
  "target_sha": "40-character SHA",
  "cases": [
    {
      "source": "tests/example.py",
      "target": "tests/tests_npu/st/test_example_npu.py",
      "change_type": "NEW",
      "test_type": "ST",
      "status": "PLANNED",
      "behavior_contract": {
        "capability": "...",
        "phases": ["..."],
        "success_criteria": ["..."]
      },
      "intentional_differences": [],
      "models": [],
      "datasets": [],
      "num_devices": 1
    }
  ]
}
```

允许状态：`PLANNED`、`STATIC_READY`、`BLOCKED`、`VALIDATED_PASS`、
`VALIDATED_NG`、`UNEXECUTED_MISSING_ASSET`。

`STATIC_READY` 只表示转换和静态审查完成，不能表示 NPU PASS。

## change-assessment.json

编排器必须在物化 patch、升级依赖和测试之前，比较上一次验证 SHA（首次运行使用当前
slime-ascend 子模块 SHA）与目标上游 SHA并生成该文件。机器采集内容包括完整 commit
列表、changed paths、按能力域分类的候选变化、依赖声明前后值、test/CI 增删改和目标
测试清单。

目标 SHA 发生变化时，执行代理必须补全 `source_changes.feature_assessment`。每项至少
包含 `capability`、`change_type`、中文 `summary` 和非空 `evidence`；若确认无用户可见
能力变化，也必须记录 `NO_USER_VISIBLE_CHANGE` 及源码证据，不能留空。最终报告的
“上游变化”和“测试变化”章节必须从该文件生成，不得凭记忆概述。

最终中文报告还必须包含 `dev 分支集成` 章节，记录 slime-ascend 基线分支/SHA、目标
Slime SHA、Manifest hash 和 patch 应用顺序、所有 Git 依赖的 base/patched HEAD、安装器
变化、clean replay、未解决问题及复核命令。该章节是后续同步到 dev 分支的交接信息，
不改变发布门禁；当前结果为 NG 时必须明确标为不可合入。

## patch-decisions.json

每个无法直接应用或被删除的补丁都必须记录，不限 `target`。当某个 target 的组件基线
发生变化时，该 target 下即使仍可 clean apply 的补丁也必须记录并重新做语义审计。
首次检查尚无足够源码或运行证据时允许使用临时决定 `PENDING_ANALYSIS`：

```json
{
  "schema_version": 1,
  "base_sha": "40-character SHA",
  "target_sha": "40-character SHA",
  "decisions": [
    {
      "patch": "patches/npu-compat/example.patch",
      "decision": "PENDING_ANALYSIS、KEEP_VERIFIED、REWRITE、SPLIT、UPSTREAM_ABSORBED 或 DROP_UNSAFE",
      "old_intent": "...",
      "target_evidence": ["文件、符号、commit 或错误签名"],
      "persistent_changes": ["更新、拆分或删除的 patch/manifest"],
      "verification": ["git apply --check、静态检查和测试命令"],
      "status": "OPEN 或 VERIFIED"
    }
  ]
}
```

`PENDING_ANALYSIS` 只允许存在于首轮应用、全量基线和集中诊断期间，表示该补丁已被
隔离且仍需源码与运行证据；它不是最终分类。补丁冲突不得只记录为 `BLOCKED`，也不得
因为出现 `PENDING_ANALYSIS` 而终止其余补丁或测试。必须从干净目标树迭代到所有
decision 均为 `KEEP_VERIFIED`、`REWRITE`、`SPLIT`、`UPSTREAM_ABSORBED` 或
`DROP_UNSAFE` 且 status 为 `VERIFIED`，或证明存在无法在当前范围获取的外部资源。
上游已吸收时要引用等价源码和测试；危险补丁被删除时要说明替代实现以及为何删除不会
丢失行为。

`KEEP_VERIFIED` 用于组件基线已变化、补丁仍然必要且已经
通过最窄行为测试的情况。`git apply --check` 或 `git am --3way` 成功仅证明文本能够
套用，不得直接写 `KEEP_VERIFIED`。`UPSTREAM_ABSORBED` 应尽量在不应用旧 patch 的
干净目标源码上验证等价实现，避免“旧 patch 仍能应用”掩盖重复或相反修改。

## 预检与补丁失败状态

预检同时输出“当前事实”和“是否允许继续测试”，不得把所有 NG 合并成一个立即终止
条件：

- `ENVIRONMENT_FAILED`：缺少命令、仓库布局、可用 NPU、必需资产、磁盘、凭据或外部
  服务等硬资源条件。只阻断实际受影响的执行；仍须保存 inventory 和审计证据。
- `DEPENDENCY_MIGRATION_REQUIRED`：目标依赖版本、commit、editable 源码或原生 wheel
  与上游基线不一致。属于迁移工作，不得停止可执行测试。
- `MANIFEST_FAILED`：Manifest 缺文件、多文件、hash 或 target 非法。先修复清单；这与
  某个源码 patch 无法套用是不同问题。
- `PATCH_DECISION_REQUIRED`：patch apply/check 失败或语义是否仍需要尚未确定。隔离失败
  patch，继续独立 patch 和完整 best-effort 基线。
- `PATCH_REPLAY_FAILED`、`DEPENDENCY_BUILD_FAILED`：作为本轮 NG 和问题台账输入，继续
  其余可执行测试与后续修复轮。

首次 baseline 的 patch/dependency NG 是诊断状态，不是提前结束条件。若失败 patch 的
缺失使某些用例无法启动，这些用例按 NG 记录 patch、target、日志和依赖链；其他用例
继续执行。最终发布仍采用 fail-closed：存在任何 `PENDING_ANALYSIS`、未验证 decision、
依赖基线不一致、构建失败、clean replay 失败或未通过用例时，最终结果必须为 NG。

## 完整测试清单

分别记录，不得混为一个数字：

- 目标 `slime/tests` 下递归发现的全部 `test_*.py`，包括上游原生 UT/ST、子目录测试和 `tests/tests_npu`；
- 目标上游 GPU CI 中全部唯一 ST 文件及参数化执行配置；
- 本次 SHA 区间内新增或修改的增量转换集合；
- `tests/tests_npu/st`、`tests/tests_npu/nightly_CI` 和 `tests/tests_npu/ut` 的完整现有 NPU 基线。

完整模式 `baseline_discovered` 来自最终物化树 `slime/tests` 下的完整 `test_*.py` discovery。不得用增量转换数量、GPU CI 数量或 `tests/tests_npu` 子集数量代替，也不得遗漏 nightly、UT 或子目录测试。

每次 run 的完整测试清单必须动态生成并作为该 run 的审计快照保存；清单文件不得手写、不得跨 run 复用、不得作为下一次 run 的发现源。下一次 run 必须重新扫描当前目标 SHA 的 `slime/tests`，并在审计中比较上一轮快照以发现新增、删除、移动和分类变化。

若只执行 `tests/tests_npu`，必须在结果中把其余 upstream 原生测试列为 `unexecuted`，并在报告中说明当前仅完成 NPU 子集。

## result.json v2

```json
{
  "schema_version": 2,
  "status": "PASS",
  "mode": "full",
  "upstream_sha": "40-character SHA",
  "baseline_discovered": 1,
  "executed": 1,
  "pass": 1,
  "ng": 0,
  "unexecuted": 0,
  "missing_resources": 0,
  "timeout": 0,
  "replay_verified": true,
  "report_path": ".upstream-sync/audit-report.zh-CN.md",
  "conversion_plan_path": ".upstream-sync/conversion-plan.json",
  "validation_result_path": ".upstream-sync/validation-result.json",
  "environment_path": ".upstream-sync/environment.json",
  "change_assessment_path": ".upstream-sync/change-assessment.json"
}
```

完整模式门禁：

- `status == NG` 表示全量验证已完成但未通过；不得推送或更新成功标记
- `status == PASS`
- `baseline_discovered > 0`
- `executed == pass == baseline_discovered`
- `ng == unexecuted == missing_resources == timeout == 0`
- `replay_verified == true`
- 所有 patch decision 已脱离 `PENDING_ANALYSIS`/`OPEN` 并达到 `VERIFIED`
- 所有目标依赖基线、patch 应用和原生构建验证均已通过
- `missing_resources` 统计因资源缺失导致的失败或未执行用例，因此必须满足
  `missing_resources <= ng + unexecuted`

试运行门禁：

- `status == TRIAL_COMPLETE`
- `pass + ng + unexecuted == baseline_discovered`
- `executed == pass + ng`
- `missing_resources <= unexecuted`
- 永不推送或更新成功标记

## 审计指纹

保存上游 SHA、外层仓库 SHA、patch manifest、只读 legacy overlay、两个子 skill、顶层
skill、Agent prompt、CANN、torch-npu、Python、设备型号和资产清单的哈希或版本。
凭据、代理口令和证书内容只记录“已配置”，不得记录值。

## 上游依赖契约

目标 `slime/docker/Dockerfile` 中的 `SGLANG_IMAGE_TAG` 和
`MEGATRON_COMMIT` 是本轮同步的权威依赖基线。`environment.json` 必须记录：

- 目标上游声明的 SGLang 版本和 Megatron commit；
- `scripts/quick_install.sh` 与所有维护中的 NPU Dockerfile 的声明值；
- 当前容器 editable 源码目录、HEAD、目标基线 commit 和祖先关系；
- NPU patches 在新基线上的应用/重写状态，以及需要重建的原生 wheel。
- Slime、SGLang、Megatron-LM、MegatronAdaptor、mbridge、Megatron-Bridge 和
  TransformerEngineNPU 的 remote、要求的 base commit、patched HEAD 和 dirty 状态；
- SGL Kernel NPU、router、torch/torch-npu、DeepEP、TorchMemorySaver、MemFabric、
  Triton Ascend 的 distribution 版本、安装来源和可取得的构建标识；
- CANN Toolkit/runtime 与 ATB 的版本、build identity 和 vendor 暴露的 commit。CANN
  二进制未暴露 core Git commit 时必须写 `null` 并保留版本/build 文件，禁止猜测 SHA。

声明不一致、目标 ref 无法解析、目标 commit 不是运行源码 HEAD 的祖先、patch
未迁移或原生扩展未重建，均为预检 NG。pip distribution version 相同不能替代源码
和 patch 基线校验。

依赖相等性包含版本之外的来源与行为：仓库/fork、commit/tag、wheel URL、构建标识、
extras、原生扩展坐标以及 Dockerfile 中的运行时断言。每项必须记录
`UPSTREAM_EXACT`、`NPU_EQUIVALENT` 或 `CUDA_ONLY_NOT_APPLICABLE`；后两者必须包含
替代/排除原因、源码证据和验证命令。版本号相同但构建来源或行为标识不同仍是不匹配。
例如上游要求 `"slime" in sglang_router.__version__` 时，普通 PyPI
`sglang-router==0.3.2` 不满足契约，必须安装上游定制构建或提供经验证的 NPU 等价构建。
发现差异后必须同步依赖并更新当前环境，不允许只记录差异后继续测试。

最终 `result.json` 契约必须读取 `environment_path`，要求最终 doctor 状态为 `PASS`，
并要求 `dependency_installer_sglang`、`dependency_installer_megatron`、
`dependency_runtime_sglang`、`dependency_runtime_megatron`，以及 mbridge、
Megatron-Bridge、MegatronAdaptor、TransformerEngineNPU、CANN 对应的 installer/runtime
检查全部存在且通过。初次预检
NG 可以触发依赖迁移，但不得作为最终环境证据；升级后必须覆盖写入最终预检结果。
