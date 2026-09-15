# HorizonHarness

长程任务 Agent Runtime 与评测框架。

HorizonHarness 将多轮模型调用和工具协作变成明确、可检查的执行过程。
项目保持足够小，便于教学和改造：一个 Agent Loop，加上类型化的
State/Message/Event、显式 Tool Runtime、上下文投影与压缩，以及 Trace 和
Eval 工具。

## Runtime 结构

```text
AgentSession(AgentSpec)
        |
        +-- Agent Loop：模型 -> 工具 -> Observation -> 下一轮
        +-- State / Message / Event
        +-- Context View 与压缩
        +-- Trace 与成本统计
        +-- Eval Runner 与 Recipes
        |
        +-- Provider Adapter 与外部工具
```

根包只稳定导出 `Agent`、`State`、`Message`、`Event`、`AgentTool`、
`ToolResult`、`AgentSpec`、`AgentSession`、`Provider` 和 `run`。消息构造器、
工具、上下文策略、Recipes、Trace、Memory、MCP 和 Benchmark 辅助函数从各自
子包显式导入。

## 标准路径

文档只展示一个普通 Agent Loop，固定使用 `bash`、`read`、`task` 三种能力，
并使用 `single` workflow。能力和 workflow 是两个独立维度，不再使用组合式
Flavor 字符串。

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
    tuple(events)
```

无需凭据即可运行确定性 Demo：

```bash
uv sync
bash runs/demos/run_bash_agent_demo.sh
```

真实 OpenAI-compatible Provider 统一使用根目录 `.env` 中的
`OPENAI_MODEL`、`OPENAI_AUTH_TOKEN` 和 `OPENAI_BASE_URL`；同一端口切换模型时
只需修改 `OPENAI_MODEL`。NVIDIA API Catalog hosted NIM 可直接使用
`https://integrate.api.nvidia.com/v1` 与 `openai-chat`。

## 明确的运行语义

- `State` 是追加式运行状态；`Event` 是类型化执行轨迹。
- `continue_session(state, followup)` 在调用方持有的内存 State 上追加 follow-up，
  只是会话续接，不是崩溃恢复。
- `chain context continuation` 在有序评测实例之间传递 active messages、task 和
  data，不恢复事件流、未完成 Tool Call 或执行游标。
- `batch reattachment` 表示新进程重新轮询和收集外部容器任务，不恢复 Agent Loop。
- `resume_run` 预留给未来的 Checkpoint 崩溃恢复 API，本阶段不实现。

本阶段唯一 GUI 是 HorizonHarness Observatory，不新增 TUI 或聊天界面。

## 评测与 Benchmark

- `src/horizon_harness/recipes/`：复用同一 Agent Loop 的 workflow Recipe。
- `src/horizon_harness/evals/`：容器、Chain、Memory 和 Benchmark 接缝。
- `studio/trace-viewer/`：HorizonHarness Observatory。
- `src/horizon_harness/evals/benchmark.py`：离线模型矩阵和 Run Manifest。

固定三模型矩阵为：`gpt-5.6-sol` + `openai-responses` + `reasoning=high`，
GLM-5.2 免费路由（`z-ai/glm-5.2:free`）+ `openai-chat`，以及 `deepseek-v4-flash` +
`openai-chat`。安装和 CI 不会启动正式 Terminal-Bench 2.1 或 SWE-bench Pro
运行。统一入口 `runs.run_bench` 会在调用既有 Benchmark Adapter 前写入 Run
Manifest，并在运行后将该启动记录复制到原有 `run_dir`。API Preflight 与现有
CI 共同承担 Runtime 可靠性验证；18 项清单只作为覆盖检查，不新增第二套 Runner。

重构前项目的历史成绩不继承；只有从冻结 Commit 重新运行后的结果才发布。

## 开发

所有 Python 命令使用锁定的 `uv` 环境：

```bash
uv run python -m unittest discover -s tests/unit
uv run python -m scripts.lint_docs
bash runs/dev/run_ci.sh
```

项目保留 Apache-2.0 许可证、已有归属和 Git 历史。
