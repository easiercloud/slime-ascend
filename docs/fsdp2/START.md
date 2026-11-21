# FSDP2 开发起点

本分支先增加独立的环境采集入口。2026-11-21已选择先试FSDPTurbo，并创建私有开发仓[easiercloud/slime-ascend](https://github.com/easiercloud/slime-ascend)。最新首轮目标改为[Qwen3-8B减层、随机初始化八卡最小训练](MINIMAL_QWEN3.md)，不再要求GLM-5.3-Flash、真实BF16权重或GLM专属的CANN9.2推荐环境。Slime FSDP2训练后端、DCP恢复和导出冷加载仍待实现与验证。原GLM候选资料保留在[FSDPTURBO.md](FSDPTURBO.md)，不作为本轮Qwen3冒烟的门禁。

## 已核对的仓基线

- 开发分支：`feat/fsdp2-bootstrap`。
- 外层仓：GitCode `Ascend/slime-ascend`，从 `main@0fb6282d0c61cb889b2e047ebdaaf33b2031a7a8` 起步。
- 2026-11-18 交接文档审阅的是 `e58ab12fd6d55952d03f3ef97367242f1435f321`。当前 main 相比它更新了 A2/A3 镜像、OPD 示例、补丁校验等内容。
- 当前 manifest 的 Slime 基线仍为 `18868a79b0b083a093798a94ed0f8cb4f71b9c14`。
- 新增环境采集脚本不更改子模块、训练参数、Megatron 入口或补丁清单。

## 环境信息采集

在计划训练的**容器内**执行。宿主机报告仅作宿主机信息，不能代替容器依赖。

```bash
python3 scripts/fsdp2/probe_env.py --output /tmp/fsdp2-env.json
```

报告采集卡型和占用的 `npu-smi info` 输出、Python 与包版本、顶层模块位置、CANN/驱动版本文件。它不导入 torch、不分配 NPU 张量、不初始化 HCCL、不安装依赖、不运行训练。源码安装且未登记 distribution 的库可能没有版本号，需另记实际 SHA。

先确认真实卡型与目标镜像的支持情况。现有显存建模实验资源的历史记录为 A2 / 910B3，不能直接套用材料中的 A3 / 950PR 示例。A2 的 GLM-5.3-Flash FSDP2 算子可用性、HBM 和 EP 拓扑需要实测。

## 下一轮代码边界

1. 锁定 adapter、可发布镜像 digest、CANN/PyTorch/torch_npu/算子版本，记录卡组与逻辑 Device 数。
2. 在固定 Slime 基线上回放现有 NPU 补丁，阅读实际 actor 工厂与参数分流。
3. 让选中 runtime 在目标卡上严格加载真实减层 BF16，先完成 EP 一步。
4. 接固定 rollout，核对 response shift、mask、GRPO loss 全局分母与参数更新。
5. 完成 BF16 导出冷加载，再加入 DCP 同拓扑恢复。

子模块内的正式改动按仓库 `patches/README.md` 和 `MANIFEST.toml` 交付；不要用新增目录占位来宣称训练后端已接通。图文、32K、MTP 与 draft 同步继续等待范围决定。

## 本分支可做的本地检查

```bash
python3 scripts/fsdp2/probe_env.py > /tmp/fsdp2-local-inventory.json
python3 scripts/check_patch_manifest.py
git diff --check
```

本地 macOS 采集成功只说明报告脚本可运行，不代表 NPU 训练通过。

采集脚本使用 Python 3.8+ 标准库。上游补丁检查器导入 `tomllib`，需使用 Python 3.11+；本次用随应用提供的 Python 完成了 22 个已登记补丁的校验。
