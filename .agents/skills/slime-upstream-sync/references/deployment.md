# 跨机器部署

## 迁移内容

迁移整个外层仓库，至少包含：

- `.agents/skills/slime-upstream-sync`
- `.agents/skills/gpu-to-npu-test-converter`
- `.agents/skills/npu-test-autofix`
- `patches/MANIFEST.toml` 与所有登记 patch
- 完整只读 legacy `overlay/`（若尚未全部迁移为 patch）

不要只复制 `SKILL.md`。`tools/upstream_sync` 如存在，仅是兼容入口；唯一实现位于顶层 skill 的 `scripts/`。

## 机器配置

通过受限权限的环境文件配置，不在仓库中保存 token：

```bash
SLIME_ASCEND_REPO_URL=...
SLIME_ASCEND_BASE_BRANCH=refactor
SLIME_UPSTREAM_REPO_URL=...
SLIME_UPSTREAM_BRANCH=main
SLIME_UPSTREAM_SUBMODULE_PATH=slime
SLIME_SYNC_STATE_DIR=/data/slime-sync/state
SLIME_SYNC_WORK_ROOT=/data/slime-sync/work
SLIME_TEST_HF_WEIGHTS_ROOT=/data/models/hf
SLIME_TEST_DATASETS_ROOT=/data/datasets
SLIME_TEST_MEGATRON_WEIGHTS_ROOT=/data/models/megatron
```

路径不得假设为 `/workspace`、`/root/models` 或某个开发者 home。状态目录必须
持久化，工作目录必须有足够空间，资产目录建议只读挂载。

## 上线顺序

1. 运行 doctor，使用 `--check-sync-layout --check-upstream-dependencies`，不要求资产，确认仓库、依赖源码与软件布局。
2. 运行 doctor，保留 `--check-upstream-dependencies` 并增加 `--require-npu --require-assets`。
3. 执行 `--check-config`。
4. 执行 `--trial-existing-assets`。
5. 执行完整 `--no-push`。
6. 人工检查报告、重放结果和 Git diff。
7. 才启用完整自动同步与原子推送。

失败工作区默认保留。确认流程稳定后，才对成功工作区启用自动清理。

## 运行账户

优先使用专用服务账户和 systemd one-shot service/timer。账户只应拥有目标仓库、
状态目录、工作目录、NPU 设备和所需只读资产的权限。共享训练机上不要让自治 Agent
拥有不受限的宿主机权限。

## 双周自动升级 Agent

稳定上线后使用 `assets/slime-upstream-sync.cron` 或等价 systemd timer 每两周触发一次自动升级。默认 cron 在 Asia/Shanghai 时区每月 1 日和 16 日 02:00 执行，相当于双周节奏。

自动 run 的强制要求：

- 每次都从当前 upstream 目标 SHA 重新物化干净工作树。
- 每次都递归扫描当前 `slime/tests/**/test_*.py` 生成新的全量 inventory；不得复用上一次 inventory 作为测试分母。
- 审计报告必须包含本次全量测试数量、相对上一轮的新增/删除/移动/分类变化、每个用例的测试特性、转换状态和执行状态。
- 自动 run 仍遵守最多 5 轮修复、3 分钟巡检和 10 分钟无日志判死规则。
- release gate 未满足时禁止推送成功标记，但必须保留工作区、日志、inventory 和问题账本。
