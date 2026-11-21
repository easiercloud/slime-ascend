# FSDPTurbo首验准备

2026-11-21，首选adapter为FSDPTurbo。先核对候选模型组件和真实环境，再接Slime固定rollout。当前没有Slime FSDP2训练通过结果。

## 候选源码

- FSDPTurbo：`79b9ae5f6db0e3c6753ad7ff6c0f1ed20d08b09a`，沿用交接文档审阅版本。
- 本次查询上游main为`0383285e5a7e75d269624354fcc958bd6dcda739`。GLM示例和模型目录在两者之间没有提交差异，因此本轮继续固定已审阅SHA。
- GLM模型入口：`fsdp_turbo/models/glm5_next/modeling_glm5_next.py`的`Glm5NextForConditionalGeneration`。
- 并行入口：`FSDPTurbo(config, model)`。Slime应负责设备及默认进程组生命周期，接入时先核对Turbo内部初始化的行为。
- 上游SFT示例会收集`loading_info`，接入真实BF16时仍需显式拒绝缺失、多余、错shape的权重，不能只因`from_pretrained`返回就判定严格加载成功。

## 文档推荐组合检查

候选GLM示例推荐Python3.12、torch2.10.0、torch_npu2.10.0.post4及以上、CANN9.2.0及以上、Transformers5.17.0、Triton及Triton-Ascend3.6.0、tiktoken0.13.0。依据为[固定SHA的GLM使用指南](https://gitcode.com/Ascend/FSDPTurbo/blob/79b9ae5f6db0e3c6753ad7ff6c0f1ed20d08b09a/examples/glm5.3_flash/README.md)。

```bash
python3 scripts/fsdp2/probe_env.py --output /tmp/turbo-env.json
python3 scripts/fsdp2/check_fsdp_turbo_env.py /tmp/turbo-env.json
```

第二个命令返回0表示元数据匹配文档候选组合，1表示存在版本、模块或源码SHA差异，2表示输入无效。检查器不会安装依赖、导入训练库或初始化NPU。包的本地build后缀如`+cpu`不会掩盖release差异，预发布版本不按稳定推荐版通过。

即使返回0，也还需要验证动态库加载、实际算子执行、设备可用性和真实权重。`fla_npu`等模块“可发现”不能代替可导入或可运行。源码以Git SHA核对，包版本`0.1.0`不足以确认GLM实现版本。

`scripts/fsdp2/fsdp_turbo_glm53_profile.json`记录这组文档推荐值。它还不是可复现镜像锁：镜像digest、各算子SHA和wheel哈希仍需补齐。上游README的FLA安装命名与pyproject依赖命名存在差异，安装前需核对实际wheel，不自动执行该安装段。

## 实机差异和首验条件

已读到的自有旧容器为CANN9.1、torch2.7.1、torch_npu2.7.1.post8、Transformers5.2.0；源码Turbo为`262f5ba4ae266d62b01c7e8a2e2bc46d46d1a0c3`。这些值与本轮文档组合不匹配，应使用独立环境准备，不在旧显存实验容器内直接升级依赖。

现有资源属于Atlas A2训练系列产品（910B3）。库的通用安装表列为支持，但GLM示例的算子构建命令面向Atlas A3训练系列产品，因此910B3上的GLM完整前后向仍需专项验证。首轮先用短序列、关闭CP和MTP，按实际分配的卡数设EP并验证专家数整除。减专家冒烟与正式专家规模分开记录。

进入真实训练前需要：

1. 明确分配的卡号和进程占用，锁定配套镜像、驱动、CANN及算子组合。
2. 制备同源真实减层BF16，记录层类型、层索引、专家索引和tensor schema。
3. 在原生Turbo中严格加载并完成EP一步，核对梯度和参数变化。
4. 再接Slime固定rollout，计算可反传的response logprob，核对shift、mask和全局分母。SFT的chunk loss不能直接替代GRPO损失。
5. 做版本化BF16导出冷加载，再做同拓扑DCP恢复。

本次没有修改FSDPTurbo上游源码。候选源码与现有环境分别保存，运行记录保存在本地交付包内，不提交服务器连接清单。
