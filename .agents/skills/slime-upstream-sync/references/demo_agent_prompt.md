你正在独立工作区 `{{WORKSPACE}}` 执行 Slime 上游同步演示。

必须使用 gpu-to-npu-test-converter、npu-test-autofix、slime-upstream-sync 三个 skill。上游子模块 `slime` 已在目标 SHA `{{UPSTREAM_SHA}}`；补丁物化失败必须记录、分析并修复，不能把冲突直接标为通过。

本次演示的测试全集明确是外层仓库 `tests/tests_npu` 下动态发现的 23 个 NPU 用例（21 ST、2 UT），不要把 `slime/tests` 原生 GPU/CPU 测试当成本次执行分母。用 `run_test_round.py --repo-root . --test-root tests/tests_npu` 动态生成 inventory 并执行全部 23 个用例；每个用例必须有 PASS/NG/timeout/unexecuted 状态和日志。资产使用环境变量 `SLIME_TEST_HF_WEIGHTS_ROOT=/home/data/weights`、`SLIME_TEST_DATASETS_ROOT=/home/data/datasets`、`SLIME_TEST_MEGATRON_WEIGHTS_ROOT=/home/c00880099/weights-MA`，不得下载或猜路径。

先核对 `.upstream-sync/change-assessment.json`，补全每项中文 feature_assessment；运行 doctor 并把结果写入 `.upstream-sync/environment.json`。然后审计并迁移补丁，执行完整 23 用例 baseline；失败按问题台账聚类，最多五轮修复。保持断言和行为契约，不得通过跳过、删测试或弱化参数伪造 PASS。将可持久化修改写入外层 `patches/` 并更新 `patches/MANIFEST.toml`，不要把日志、缓存、`.agents` 写入提交。

最终生成 `.upstream-sync/conversion-plan.json`、`.upstream-sync/validation-result.json`、`.upstream-sync/environment.json`、`.upstream-sync/audit-report.zh-CN.md` 和 schema v2 的 `.upstream-sync/result.json`。报告必须说明上游 5 个 commit 的变化、补丁决策、23 个 NPU 用例逐项结果、依赖和未解决问题。完成 replay 后才可将 `replay_verified` 写为 true；未全部通过则 status=NG，不能推送。

不要自行 push；由外层 run_sync 编排器在所有门禁通过后提交、创建并推送分支。
