# 受控 edit 的漂移处理

本页是操作人员处理受控 `edit` 恢复冲突的运行手册。恢复与日志的总体约束见
[Workspace](workspace.md)，故障契约见 [Runtime slices](runtime-slices.md)。

文件既不符合 edit 意图中的改前状态，也不符合改后状态时，恢复会停止，并给出
`task_id`、`step_id` 以及 `inspect` / `adopt-current` 提示。停止恢复不会发起新的
工具写入。连续执行 `resume` 也不会解除这个阻塞。

## 查看证据

所有命令都使用同一个代码目录和任务存储目录：

```bash
uv run python -m runs.task_workspace --code-dir /path/to/code --store-dir /path/to/store \
  inspect TASK_ID
```

输出中的 `drift_reports` 按步骤提供 `path`、`before_sha256`、`after_sha256`、
`current_sha256`、文件存在性、`before_image_available`、`old_string_present`、
`new_string_present` 和两段 unified diff：`before_to_current_diff` 与
`after_to_current_diff`。

报告读取任务的 `snapshot-objects/<sha256>` 改前对象，检查对象内容哈希，并按意图
中的替换参数重建改后内容。只有重建内容的哈希等于意图中的 `after`，才展示第二段
diff；不匹配时，`after_diff_unavailable_reason` 为 `after_hash_mismatch`。
报告不改变步骤状态。

旧任务缺失改前对象时，`before_image_available=false`，两段 diff 均为 `null`，
`diff_unavailable_reason` 为 `before_image_missing`；哈希和字符串存在性仍然可用。
对象损坏时也降级，并报告 `before_image_hash_mismatch`。二进制或无法解码的内容
以 `binary=true` 标记，相关 diff 不展示。删除文件的当前哈希按空字节计算，
同时显示 `current_exists=false`，以区分缺失文件与空文件。

新 edit 会在提交意图前，将改前字节写入同一内容寻址目录。已有对象复用，新对象
通过临时文件、fsync 和原子替换发布；这些存储操作不会增加工作区文件的 edit 写入
次数，也不会改变 edit 的副作用与结果提交边界。

## 采用当前内容并恢复

确认当前文件应当保留后，复制报告中的 `step_id` 和完整 `current_sha256`：

```bash
uv run python -m runs.task_workspace --code-dir /path/to/code --store-dir /path/to/store \
  adopt-current TASK_ID --step STEP_ID --expected-sha CURRENT_SHA256 --note "已检查外部修改"
uv run python -m runs.task_workspace --code-dir /path/to/code --store-dir /path/to/store \
  resume TASK_ID
```

`adopt-current` 只接受状态为 `ambiguous`、原因为 `workspace drift` 的受控 edit。
执行前重新读取当前文件并校验 `--expected-sha`；文件再次变化时拒绝操作，必须重新
检查。该操作不会改写或重新创建工作区文件。

一个 `BEGIN IMMEDIATE` 事务保存以下事实：冲突步骤变为 `cancelled`，保留原始意图，
追加 `reason=adopted_external_content`、`sha256`、`note` 和错误工具结果；同批次中
排在其后且没有执行证据的步骤也变为 `cancelled`，提示重新规划。已有结果、意图、
重试或开始执行事件都算执行证据，已成功或失败的步骤保持原状态。仍有不确定效果的
步骤不会被这个操作自动解除，恢复时仍需对账。

冲突调用的结果明确告知模型：“文件在中断期间被外部修改，操作人员已采用当前内容；
本次编辑未执行，修改前请先重新读取该文件”。后续取消调用的结果为“因前序编辑被
取消而未执行，请重新规划”。采用外部内容不证明原编辑已经完成。

随后 `resume` 通过原有批次恢复流程按调用顺序提交一次结果 bundle，再请求模型。
取消步骤的结果具有原始 Tool Call ID，可以维持调用与结果配对。采用操作本身不会
打包 bundle，也不会调用模型。再次采用同一步骤会被拒绝，不会增加重复结果。

此流程沿用单机、单 task writer 和无并发外部写入的支持边界。`--expected-sha` 检查
覆盖操作人员查看报告之后、执行采用之前的变化；它不提供文件系统与 SQLite 之间的
跨系统事务。

## 验证

```bash
uv run --with pytest python -m pytest tests/unit/test_drift_resolution.py
uv run --with pytest python -m pytest tests/unit/test_runtime_crash_contract.py tests/unit/test_tool_recovery.py
uv run --with pytest python -m pytest tests/unit
```
