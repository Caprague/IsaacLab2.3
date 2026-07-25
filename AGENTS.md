# AGENTS.md — AI 编程助手行为准则

本文件为 AI 编程助手（Claude Code、Cursor Agent 等）提供项目级的行为规范，
确保所有自动化代码贡献与 Isaac Lab 项目的人类开发者标准保持一致。

---

## 1. 项目概述

**Isaac Lab** 是 NVIDIA 开源的 GPU 加速机器人研究框架，基于 NVIDIA Isaac Sim 构建。
项目版本：**2.3.2**，依赖 Isaac Sim **5.1.0**。

- 仓库地址：https://github.com/isaac-sim/IsaacLab
- 主许可证：BSD-3-Clause（Mimic 组件使用 Apache-2.0）
- Python 版本：3.11

### 1.1 核心模块

| 目录 | 用途 |
|------|------|
| `source/isaaclab/` | 核心框架：环境、仿真、资产、传感器、地形、管理器 |
| `source/isaaclab_assets/` | 机器人资产定义 |
| `source/isaaclab_contrib/` | 社区贡献代码 |
| `source/isaaclab_rl/` | 强化学习组件 |
| `source/isaaclab_mimic/` | MimicGen 模仿学习（Apache-2.0 许可证） |
| `source/isaaclab_tasks/` | 预定义任务（Manipulation、Locomotion 等） |
| `scripts/` | 独立脚本：基准测试、演示、训练、教程 |
| `docs/` | Sphinx 文档 |

---

## 2. 代码风格与格式

### 2.1 格式化工具

项目使用 **ruff** 进行代码检查与格式化，配置在 `pyproject.toml`：

- **行宽**：120 字符
- **目标版本**：Python 3.10
- **文档字符串风格**：Google-style

AI 助手在生成或修改代码时，必须遵守与 `ruff` 一致的规则。关键规则：

| 规则 | 说明 |
|------|------|
| `E`, `W` | pycodestyle 错误与警告 |
| `F` | pyflakes 检查 |
| `I` | isort 导入排序 |
| `UP` | pyupgrade 现代化语法 |
| `C90` | McCabe 复杂度（上限 30） |
| `SIM` | flake8-simplify 简化建议 |

### 2.2 忽略的规则

以下规则在项目中**有意忽略**，AI 助手不应因这些模式而修改代码：

- `E402` — 模块级导入不必在文件顶部（Isaac Sim 扩展加载需要）
- `D401` — 文档字符串首行不强制祈使语气
- `RET504` — 允许不必要的变量赋值后 return
- `RET505` — 允许 return 后的 elif
- `SIM102/103/108/117/118` — 某些"简化"会使代码可读性降低
- `UP006/UP018` — 允许保留传统类型注解风格

### 2.3 导入顺序

导入必须严格按照以下顺序分组，每组之间用空行分隔：

1. `__future__` 导入
2. 标准库
3. 第三方库
4. Omniverse 扩展（`isaacsim`, `omni`, `pxr`, `carb`, `usdrt`, `curobo`）
5. `isaaclab` 核心
6. `isaaclab_contrib`
7. `isaaclab_rl`
8. `isaaclab_mimic`
9. `isaaclab_tasks`
10. `isaaclab_assets`
11. 本地文件夹

示例：
```python
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import numpy as np

import omni.kit.app

from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
```

### 2.4 许可证头

所有 `.py`、`.yaml`、`.yml` 文件必须包含许可证头（pre-commit 自动插入）：

```python
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
```

Mimic 相关文件（`source/isaaclab_mimic/`、`scripts/imitation_learning/isaaclab_mimic/`）使用 Apache-2.0 许可证头。

### 2.5 类型注解

- 使用 `from __future__ import annotations` 延迟求值
- 使用 Python 3.10+ 语法：`list[int]`、`dict[str, float]`、`tuple[int, ...]`、`X | None`
- 配置类使用 `@configclass` 装饰器
- 必需字段使用 `MISSING` 哨兵值
- 对外 API 必须有完整类型注解

### 2.6 Google 文档字符串

```python
def my_function(arg1: str, arg2: int = 0) -> bool:
    """一句话概述此函数的功能。

    Args:
        arg1: 参数一的描述。
        arg2: 参数二的描述。默认为 0。

    Returns:
        返回值的描述。

    Raises:
        ValueError: 异常条件的描述。

    Note:
        当需要突出显示重要说明时使用。
    """
```

---

## 3. 测试规范

### 3.1 测试框架

- 使用 **pytest**
- 测试位于 `source/isaaclab/test/` 下，按模块组织
- Isaac Sim CI 中的测试需添加 `@pytest.mark.isaacsim_ci` 标记

### 3.2 运行测试

```bash
# 运行所有测试
./isaaclab.sh -p source/isaaclab/test

# 运行特定模块测试
./isaaclab.sh -p source/isaaclab/test/actuators

# 仅运行 CI 标记的测试
./isaaclab.sh -p -m isaacsim_ci
```

### 3.3 AI 助手的测试规则

- 新增功能必须包含测试
- 修改现有行为必须同时更新相关测试
- 测试应覆盖正常路径与边界条件
- 未在 Isaac Sim 环境中时，至少保证纯 Python 逻辑的单元测试通过

---

## 4. 文档规范

- 使用 **Sphinx** + **reStructuredText** (`.rst`)
- 文档位于 `docs/source/` 下
- 代码中的 docstring 使用 Google 风格
- 文档中引用模块时使用 Sphinx 角色：`:class:`, `:attr:`, `:meth:`, `:mod:`
- 配置文档字符串使用 `"""Configuration for ..."""` 开头

---

## 5. Git 工作流

### 5.1 分支策略

- `main` — 主分支，稳定版本
- 功能分支命名：`feature/<描述>`、`fix/<描述>`、`docs/<描述>`

### 5.2 提交规范

- 提交信息使用**中文**描述（本项目偏好）
- 格式：`<类型>: <简短描述>`
- 类型：`feat`、`fix`、`docs`、`refactor`、`test`、`chore`
- 示例：`fix: 修复 IsaacSim 5.1 暂停/恢复后机器人可视化模型冻结问题`

### 5.3 提交签名

所有提交必须以以下行结尾：
```
Co-Authored-By: Claude <noreply@anthropic.com>
```

### 5.4 PR 规范

- 参考 `.github/PULL_REQUEST_TEMPLATE.md`
- PR 应小而聚焦
- 描述需包含变更摘要、动机、修复的 Issue 编号
- 文档更新 PR 需包含 `docs/` 下的相关修改

---

## 6. 行为准则

### 6.1 通用原则

1. **先读后写** — 修改文件前必须先阅读现有代码，理解项目模式
2. **保持一致** — 新代码的风格、命名、结构与周围代码一致
3. **最小改动** — 只改必要的部分，避免无关的"顺手重构"
4. **完整思考** — 考虑边界情况、错误处理、向后兼容性
5. **可验证** — 产出应该是可测试、可运行的

### 6.2 代码修改规则

- **不要**在没有理解现有模式的情况下引入新的依赖
- **不要**大规模重命名或重构，除非明确要求
- **不要**修改 `_isaac_sim` 符号链接或 Isaac Sim 的内部文件
- **不要**在不理解上下游关系的情况下修改 API 签名
- **要**在添加新类/函数时遵循现有的模块组织方式
- **要**使用 `@configclass` 定义配置类，而不是普通 dataclass

### 6.3 安全规则

- 不要执行任意 shell 代码或下载未经审查的外部资源
- 不要修改 `.github/workflows/` 中的 CI 配置，除非明确要求
- 不要泄露或记录敏感信息（密钥、令牌等）
- 参考 `SECURITY.md` 处理安全相关问题

### 6.4 Isaac Sim 特殊规则

- Isaac Sim 是一个大型二进制发行版，通过 `_isaac_sim` 符号链接引用
- **永远不要**修改 `_isaac_sim/` 下的文件
- Isaac Sim 必须在运行时环境中才能进行仿真测试
- 导入 `isaacsim`/`omni` 包的代码只能在 Isaac Sim 内核中运行
- 纯 Python 逻辑应与 Isaac Sim 依赖分离，以便独立测试

### 6.5 交互准则

- 用户要求用中文回答时，全程使用中文沟通
- 修改代码前，对于超过 3 个文件的变更，先概述方案再执行
- 遇到不确定的情况，主动询问而非猜测
- 对破坏性操作（删除文件、强制推送等）必须先确认

---

## 7. 常用命令参考

```bash
# 启动 Isaac Lab（进入 Isaac Sim Python 环境）
./isaaclab.sh

# 运行脚本
./isaaclab.sh -p scripts/demos/random_agent.py

# 代码检查
pre-commit run --all-files

# 仅检查
ruff check .

# 格式化
ruff format .

# 运行测试
./isaaclab.sh -p -m pytest source/isaaclab/test
```

---

## 8. 工具链

| 工具 | 用途 |
|------|------|
| ruff | 代码检查 + 格式化 |
| pyright | 类型检查 |
| codespell | 拼写检查 |
| pre-commit | 提交前自动检查 |
| pytest | 测试框架 |
| Sphinx | 文档生成 |
| isaacsim | NVIDIA Isaac Sim 运行时 |

---

> **最后更新**: 2026-07-25
> **维护者**: Isaac Lab Project Developers
