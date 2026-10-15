# 框架 Profile

使用 `scripts/detect_framework.py` 生成 `.npu-test-autofix/framework-profile.json`。
结果仅是起点，启动器、资产和服务依赖必须通过当前仓库代码和测试确认。

## Slime

- 使用现有 `tests/tests_npu`、UT/ST/nightly 约定。
- 使用 `prepare_test_resources.py`、`prepare_command_utils.py` 和
  `test-assets.md`，但仅在当前测试确实依赖对应组件时启用。
- 使用 Slime 的 Megatron/Ray/SGLang 失败案例前，必须匹配日志签名。

## VERL

- 从 `tests/`、pytest 配置、recipe/config 和当前 launcher 发现测试。
- 默认使用 pytest 收集 Python 测试；分布式/rollout 用例使用该仓库测试实际调用的
  launcher，不假设 Ray、Megatron、vLLM 或固定目录。
- 从配置、fixture 和环境变量盘点模型、tokenizer、dataset、checkpoint 和服务地址。
- 本地资产路径、NPU 环境传播和集体通信参数必须由当前 VERL 配置和运行时证据决定。
- 不运行 Slime 的 `prepare_command_utils.py` 或权重转换，除非 VERL 工作树实际引用这些
  模块并已确认兼容。

## Generic

- 显式指定测试根和收集/启动命令。
- 使用 `run_test_round.py` 记录常规 Python/shell 用例；非标准系统测试可使用框架命令，
  但必须把每个命令、退出状态、日志和最终分类写入相同 issue/report 结构。
- 未确认资产、服务、设备或框架 profile 时停止，不把无法执行的用例当 PASS。
