# Slime-Ascend v0.4.0 同步交接

更新日期：2026-10-26
工作分支：`sync/slime-v040-dev-cleanup-20261020`

## 当前代码

- Slime 子模块指向公开的 v0.4.0 提交 `0b0c277d5b4cc66efc3b6db269e2e5184e8f3b1e`。训练依赖采用 MegatronAdaptor 和 TransformerEngineNPU。
- NPU 适配通过 `patches/MANIFEST.toml` 管理；已清理 Slime v0.4.0 自带能力所覆盖的重复补丁。`scripts/check_patch_manifest.py` 用于检查补丁路径和 SHA256。
- A3 ARM64 环境使用 CANN 9.0、Python 3.12、PyTorch 2.10.0 和 torch_npu 2.10.0.post4。`docker/Dockerfile.a3.ubuntu22.04.cann90.py312` 与 `scripts/quick_install.sh` 安装 CANN 9.0 / Python 3.12 的预编译 SGL 内核包；仅对带本仓补丁的 torch-memory-saver 重新构建 wheel。现有 `docs/get_started/quick_start.md` 已同步更新。
- 已将上游 `Ascend/slime-ascend:dev`（`2fd2677910ca33447fc6ef9449b95663ce6c2427`）合入工作分支并解决 PR 冲突，合并提交为 `14b3ea5`。
- PR 沿用上游 `dev` 的 NPU UT、ST 和 Nightly 用例；本分支新增的 9 个 `*_v040_npu.py` UT 已移除，原有 ST/Nightly 用例的本分支改动已撤回。Skill 仅保留 `.agents/skills/slime-upstream-sync/SKILL.md`，清理提交为 `ed7b351`。

## 已有检查记录

- 2026-10-20：从 Slime v0.4.0 tag 顺序回放 4 份 Slime 补丁成功；当时的补丁清单路径及 SHA256 检查通过。
- 2026-10-16：隔离的 CANN 9.1 环境中，旧测试集合记录过 155 项 UT 通过；Qwen2.5-0.5B 异步 ST 退出码为 0。日志分别位于 `/tmp/slime-v040-all-npu-ut.log` 和 `/tmp/slime-v040-qwen05-async-st-20261016b.log`。这些是历史实验记录，所用测试集合与本次精简后的 PR 不同。
- 2026-10-26：与上游 `dev` 比较，PR 不再新增或修改 `tests/tests_npu` 文件，且 `.agents/skills` 下仅有一份同步 Skill；GitCode 推送钩子通过。
