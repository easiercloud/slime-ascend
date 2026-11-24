# FSDPTurbo 后端适配：初始化、模型和固定 rollout GRPO 验证

本交付实现 FSDP 参数入口、训练 Actor 选择、独立 Python 环境、双卡通信与资源释放。
选择 `--fsdp-init-only` 验证初始化，或 `--fsdp-model-check` 验证 Qwen3-0.6B
模型加载、FSDP2 分片、前后向和一次 AdamW 更新。模型阶段详见 [模型验收](fsdp2-model.md)。
`--fsdp-grpo-check` 接通固定 rollout 的数据读取、response log-prob、GRPO loss 和连续两次更新，
详见 [GRPO 验收](fsdp2-grpo.md)。`--fsdp-online-check` 增加真实 SGLang 生成、
GRPO 更新和 HF 磁盘权重同步，详见 [在线验收](fsdp2-online.md)。训练 checkpoint 保存仍待接入。

## 代码维护与构建

外层 slime-ascend 是交付仓库，`slime/` 是固定上游提交应用补丁后的运行代码。
`patches/features/0001-fsdpturbo-backend-entry.patch` 包含入口、Actor、预检和单测。
`patches/features/0002-fsdpturbo-qwen3-model.patch` 在其后添加模型加载、分片及数值验收。
`patches/features/0003-fsdpturbo-grpo.patch` 添加固定 rollout 训练链路和 GRPO 数值验收。
`patches/features/0004-fsdpturbo-online.patch` 添加 HF 权重发布和两轮真实 rollout 闭环验收。
`patches/npu-compat/bounded-port-search.patch` 单独保存现有基线需要的端口查找修复。
补丁由 `patches/MANIFEST.toml` 注册并校验 SHA256；`build.sh` 按
`npu-compat`、`upstream-fixes`、`features` 顺序应用，每类内遵循清单顺序。
不要仅提交子模块指针：功能源码通过补丁交付，环境脚本、示例和本文直接由外层仓库维护。

在干净检出中构建：

```bash
python scripts/check_patch_manifest.py
bash build.sh
```

`build.sh` 会清理子模块，不能直接在含有未保存修改的开发目录运行。
补丁生成和回放应在独立目录进行；开发者可直接编辑子模块，验收后生成 format-patch。

## 固定环境

FSDPTurbo 提交固定为 `0383285e5a7e75d269624354fcc958bd6dcda739`。
已验证镜像环境为 Python 3.12.9、Torch 2.10.0、TorchNPU 2.10.0.post7、CANN 9.2。
训练 venv 使用 Transformers 5.16.0；原推理环境保持 Transformers 5.12.1。
其他训练依赖见 `requirements/fsdp2.txt`。此文件不是镜像全部依赖的锁文件，
也不表示支持其他 Torch/CANN 组合。TorchAO 原生扩展当前不可用，首期不启用量化。

在原镜像 Python 中执行以下命令，目录按实际位置设置：

```bash
bash scripts/install-fsdp-environment.sh /absolute/FSDPTurbo /absolute/new-fsdp-venv
```

脚本要求 provider 提交匹配且工作区干净、venv 目录不存在。已有验收环境可以直接复用。
不要全局激活训练 venv；仅训练 Actor 使用它，driver 和 SGLang 使用原解释器。
脚本只安装训练依赖，不拉镜像、不修改宿主驱动和 CANN、不重新安装共享 Torch。

## 双卡初始化验收

容器应能访问四个 NPU；示例在其中两个设备上创建训练 Actor。SGLang 仅做 CPU
worker 中的 ServerArgs 导入和环境检查，另外两张卡本阶段不进行推理。
驱动 Python 必须能够导入 Ray、Slime 依赖和 SGLang；训练解释器需要导入相同版本 Ray。

```bash
export FSDP_PYTHON=/absolute/new-fsdp-venv/bin/python
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
bash examples/fsdp2/run-init.sh
```

可通过 `BASE_PYTHON`、`FSDP_OUTPUT_DIR`、`SLIME_SOURCE_DIR` 覆盖 driver、日志和待验收源码目录。
示例启动独立本地 Ray 集群并在结束时关闭自己的集群；不调用全局 `ray stop`。
默认使用 `SLIME_MIN_PORT=50600`、HCCL host 60400–60499、NPU 61400–61499，
共享服务器应为并发任务指定不同端口。选择空闲端口与实际监听之间仍存在竞争窗口，
端口冲突不等同于通信算法失败。首次 provider 加载可能编译算子；默认总超时600秒。

成功输出 `result.json`、`run.log`、`exit-code.txt`，退出码为0，报告 `passed: true`。
验证项目包括 FP32/BF16 all-reduce、all-gather、reduce-scatter 的数值、独立解释器、
无 Megatron/Adaptor/SGLang 导入及通信组销毁。Slime 入口不导入 TransformerEngine；
FSDPTurbo 自身注册可选 CUDA 算子时可能导入它，报告在 `optional_provider_imports`
中如实记录，不把它误报成 Slime 对 Megatron 的依赖，也不宣称完全没有 TE 导入。

## 回归与当前限制

```bash
PYTHONPATH="$PWD/slime${PYTHONPATH:+:$PYTHONPATH}" python -m pytest \
  slime/tests/test_fsdp_entry.py \
  slime/tests/test_fsdp_model.py slime/tests/test_fsdp_grpo.py slime/tests/test_fsdp_weight_sync.py \
  slime/tests/test_megatron_argument_validation.py \
  slime/tests/utils/test_sglang_arguments.py -q
```

默认训练后端仍为 Megatron。FSDP 初始化拒绝 offload、同卡训练推理、PPO critic、
reference/KL、OPD、容错、checkpoint、routing replay 等未实现组合。
在线检查限定单机 Qwen3、2 张训练卡加 2 张推理卡、两轮同步 GRPO，以及完整 BF16 磁盘发布。
尚未开放任意长度常规训练、恢复训练、异步 rollout、多节点、增量同步或零拷贝同步。
