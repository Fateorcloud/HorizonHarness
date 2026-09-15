# HorizonHarness

面向长时程 Coding Agent 的 Durable / Replayable / Verifiable Runtime。

HorizonHarness 关注的不是让模型“更聪明”，而是把模型决策、工具副作用、
持久化执行事实和进程恢复组织成一条边界清晰、可以检查的执行链：模型已经作出的
决策必须先写入 Journal；进程崩溃后继续原来的决策；任务完成由证据决定，而不是由
模型的一句 “Done” 决定。

项目保持实现可读、路径明确，适合用于 Agent Backend、Agent Platform 和长任务
Coding Agent 的工程研究。

## 为什么现在做

Coding Agent 正从“生成一段代码”走向跨多轮、跨工具、跨上下文窗口的持续执行。
[OpenAI 官方模型指南](https://developers.openai.com/api/docs/guides/latest-model)
已经把长任务一致性、异步工具调用、compaction、assistant item replay 和多 Agent
编排列为重要的应用侧问题，并明确工具仍由应用执行、pending work 仍由应用管理。
[Claude Code](https://code.claude.com/docs/en/subagents) 也在公开产品中提供并行或可恢复的
subagent，并通过 [hooks](https://code.claude.com/docs/en/hooks-guide) 和
[file checkpointing](https://code.claude.com/docs/en/agent-sdk/file-checkpointing)
加强执行控制、观测与文件回退。

这些公开能力共同指向一个趋势：模型会越来越擅长规划和使用工具，但任务运行时间越长、
并发越多、外部副作用越多，Runtime 对状态、故障和完成证据的责任反而越清晰。
HorizonHarness 不复制 Codex 或 Claude Code 的产品功能，也不推测它们的内部实现；它选择
其中一个更窄、可以公开验证的工程问题：进程可能在模型决策、工具副作用和结果提交之间
任意崩溃时，系统怎样保存事实、恢复执行，并证明任务确实完成。

## 这个项目要展示什么

作为个人作品，HorizonHarness 不靠堆叠 Memory、MCP、Multi-Agent 或 Sandbox 等功能名
证明复杂度，而是用一个小而完整的闭环向面试官展示：

| 工程能力 | 项目中的落点 |
| --- | --- |
| 架构建模 | 分离 Journal、Task State、Active Context，并指定唯一执行推进者 |
| 可靠性设计 | 定义决策提交、工具副作用、结果提交之间的崩溃语义和恢复不变量 |
| 协议意识 | 保存可 round-trip 的 canonical message，维持 Provider Tool Call 配对 |
| 验证能力 | 用进程级故障注入、真实文件写入次数和 verifier receipt 形成证据链 |
| 工程判断 | 明确受控 `edit` 与任意外部副作用的保证边界，不夸大 exactly-once |

## 要解决的问题

一次普通 Agent Loop 很容易实现：调用模型、执行工具、把结果交回模型。长任务真正
困难的是跨越故障边界后，系统还能回答这些问题：

- 哪个模型决策已经提交，哪些工具调用仍待执行？
- 工具副作用已经发生、结果却尚未落盘时，重启后是否会重复执行？
- 下一次模型调用能否收到满足 Provider 配对协议的完整消息历史？
- 当前代码是否真的满足任务创建时冻结的验收条件？
- 如何回放、派生和比较两次执行，而不修改原任务的事实？

HorizonHarness 用一份追加式 Journal 保存执行事实，并从中重建任务状态、模型上下文、
回放结果和验证证据。

## 核心执行闭环

```text
TaskSession
  校验持久化历史，装配 Provider、Tools、Workspace 和恢复适配器
        |
        v
core.run()
  唯一推进 pending batch、工具执行、bundle 提交、turn end 和模型调用
        |
        v
模型决策提交 -> 工具执行/对账 -> 结果原子提交 -> 下一轮模型
        |
        v
Execution Journal -> Task State / Replay / Verification / Fork
```

运行时明确分开三类状态：

| 状态层 | 职责 |
| --- | --- |
| Session / Journal | 保存完整执行事实，是恢复与审计的事实来源 |
| Task State / Workspace | 保存当前任务的持久状态，是 Journal 的投影 |
| Active Context | 为本次模型调用构造满足 Provider 协议的动态上下文 |

核心不变量是：

```text
No tool side effect may occur before
the model decision authorizing it
has been durably committed.
```

## 当前有证据支持的能力

| 契约 | 当前证据 |
| --- | --- |
| 决策先于副作用持久化 | 子进程在决策提交后崩溃，重启不重新请求原决策 |
| 受控 `edit` 可跨崩溃对账 | 三调用 A–F 故障矩阵检查实际写入次数、顺序、漂移和重复恢复 |
| Provider history 可重建 | 恢复后的真实 `LLMRequest.messages` 保留原 Tool Call ID 和 A/B/C 顺序 |
| ToolResult bundle 原子且幂等 | bundle 以 batch identity 提交；崩溃重启后仍只有一份 |
| 完成由验证证据决定 | Acceptance Contract 在任务创建时冻结；Receipt 绑定任务版本、代码树、合同和验证器 |
| Replay 与 Fork 可继续检查 | 回放使用已提交事实；Fork 生成带 lineage、checkpoint 和 context manifest 的独立子任务 |

完整的契约—实现—测试对应关系和 A–F 故障点见
[`docs/runtime-slices.md`](docs/runtime-slices.md)。这里刻意区分：代码存在、测试通过和
工程契约已被对应证据证明。

## 先运行故障恢复 Demo

无需模型密钥：

```bash
uv sync
bash runs/demos/run_durable_runtime_demo.sh
```

Demo 会提交包含 A/B/C 三个 `edit` 调用的模型决策，在 Tool A 前注入进程退出，然后由
新进程恢复原批次。它会检查三个真实文件、下一次模型调用、bundle 提交次数、验证结果
和最终任务状态。故障退出码为 `91`，成功恢复后的关键结果为：

```json
{
  "crash_exit_code": 91,
  "files": {"a": "1", "b": "2", "c": "3"},
  "next_model_calls": 1,
  "bundle_commit_count": 1,
  "verification_outcome": "passed",
  "status": "completed"
}
```

## 两条使用路径

普通、进程内的 Agent Loop 使用 `AgentSession`：

```python
from horizon_harness import AgentSession, AgentSpec, Provider

provider = Provider(id="fake", api="fake", model="fake-model")
spec = AgentSpec(
    name="solver",
    capabilities=("bash", "read", "task"),
    workflow="single",
)

with AgentSession(provider, spec=spec, cwd=".") as session:
    state, events = session.run("检查仓库并报告结果。")
```

需要持久化任务、崩溃恢复、验证和回放时，使用
`horizon_harness.workspace.TaskSession`，或通过 CLI 操作：

```bash
uv run python -m runs.task_workspace --fake --store-dir evals/out/workspace-smoke \
  start "Return a short answer." --task-id smoke --pause-after-turns 1

uv run python -m runs.task_workspace \
  --store-dir evals/out/workspace-smoke inspect smoke

uv run python -m runs.task_workspace --fake --store-dir evals/out/workspace-smoke \
  resume smoke --pause-after-turns 1
```

同一入口还提供 `replay`、`diff`、`fork`、`verify`、`complete` 和显式的 legacy
inspect/import 命令。

## 保证边界

| 能力 | 保证范围 |
| --- | --- |
| Durable side-effect reconciliation | 目前只覆盖有 before/after fingerprint 契约的受控 `edit` |
| Shell、网络 API、任意外部系统 | 不声明通用 exactly-once；每类工具必须提供自己的对账契约 |
| Hooks | 是生命周期扩展点，不是持久化副作用执行器；进程退出后不会自动对账外部效果 |
| `diff_runs()` | 比较已提交事实的结构差异，不证明某个模型或配置具有因果上的性能优势 |
| Regression runner | 是确定性 preflight fixture runner，不是正式 Runtime 或模型 Benchmark |

受控 `edit` 的恢复判定如下：

```text
actual == before  -> 副作用尚未发生，可为同一 logical step 发起新 attempt
actual == after   -> 副作用已经发生，不重复写入，提交 reconciled result
otherwise         -> WORKSPACE_DIVERGED，停止恢复
```

因此，项目不会把一个工具上的可对账恢复包装成“任意外部副作用都 exactly-once”。

## 代码导航

| 路径 | 内容 |
| --- | --- |
| `src/horizon_harness/core.py` | 普通执行与恢复共用的 Agent 推进路径 |
| `src/horizon_harness/workspace/` | Journal、codec、受控执行、session、snapshot、replay、verification |
| `runs/task_workspace.py` | 持久化任务 CLI |
| `runs/demos/durable_runtime_demo.py` | 无密钥的崩溃—恢复—验证演示 |
| `tests/unit/test_runtime_crash_contract.py` | A–F 进程级故障矩阵 |
| `docs/runtime-slices.md` | Runtime 契约、证据台账、边界与历史失败记录 |
| `studio/trace-viewer/` | HorizonHarness Observatory 执行轨迹查看器 |

项目也保留上下文压缩、Trace、评测适配器和 Memory 实验能力，但当前发布主线是执行
一致性，不以功能数量作为完成标准。历史项目名称下的评测成绩不会自动继承；正式结果
必须从冻结提交重新运行。

## 开发与验证

项目使用 `uv` 管理固定环境：

```bash
uv run python -m unittest discover -s tests/unit
uv run python -m scripts.lint_docs
bash runs/dev/run_ci.sh
```

完整 CI 包含格式、静态检查、文档与架构检查、类型检查、单元测试和 Bash Demo。

## 许可证

项目使用 Apache-2.0 许可证，完整文本见 [`LICENSE`](LICENSE)。
