# 框架 Profile

先用自动识别结果，再根据仓库实际代码确认。自动识别不等于可直接执行。

## Slime

- 标记：`slime/`、`slime_plugins/`、`train.py` 或 `train_async.py` 的组合。
- 测试约定：优先遵循现有 `tests/tests_npu`、UT/ST/nightly 结构。
- 启动器：以测试和 helper 实际使用的 Python、Ray、Megatron、SGLang 为准。
- 资产：使用 `npu-test-autofix` 的 Slime asset 规则。
- 专用资料：`conversion-rules.md` 与 `grpo-npu-casebook.md`。

## VERL

- 标记：顶层 `verl/` Python package，或项目元数据声明 VERL。
- 测试约定：从仓库中的 `tests/`、pytest 配置和现有 NPU 用例推断；不要创建 Slime
  目录结构。
- 启动器：从当前 test、recipe/config 和 launcher 脚本确认。VERL 可组合 Ray、FSDP、
  Megatron、vLLM 或其他组件，不能预设其中任意一个。
- 资产：从 YAML/JSON/Hydra 配置、fixture 和环境读取；记录模型/数据集逻辑名及其本地
  映射，不使用 Slime 的静态常量或权重缓存格式。
- NPU 适配：确认 PyTorch NPU、集合通信、worker 环境传播和 rollout backend 的实际支持。

## Generic

未知或自研框架使用此 profile。必须先确认测试根、收集器、启动器、分布式拓扑、资产
配置和成功信号。任一项无法确认时停止并报告，不得套用 Slime 或 VERL 配方。
