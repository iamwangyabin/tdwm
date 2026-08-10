# Agent 开发说明

## 关键规则

- 本项目必须基于已安装的 `stable-worldmodel` 包开发。不得克隆、内置或
  重复实现该包已经提供的框架能力。
- 依赖必须精确固定为 `stable-worldmodel[all]==0.1.1`。不得使用未固定版本，
  也不得在无关改动中顺带升级该依赖。
- 必须通过 `import stable_worldmodel as swm` 直接导入，并优先使用公开 API，
  不得依赖私有模块或复制上游脚本。
- 我们提出的方法必须与上游 baseline 保持独立，确保所有比较都可以审计，
  并使用相同的数据和评测流程。
- 没有经过受控、可复现的对比实验，不得声称我们的方法优于 baseline。

## 项目概述

TDWM 是一个研究 world model learning 任务的项目。项目使用
`stable-worldmodel` 作为可运行的研究平台，负责环境、数据集、数据采集、
规划、baseline、checkpoint 和评测。

本项目将实现一种新的 world model 方法，并与 Stable World Model 生态中
已有的强 baseline 进行比较。研究目标是在统一、公平的评测协议下，使提出的
方法优于这些 baseline。当前尚未确定具体方法，不得提前臆造方法设计。

## 必需依赖

本项目当前核实的包版本为 `0.1.1`，要求 Python 3.10 或更高版本。在已激活的
虚拟环境中安装完整研究依赖：

```bash
python -m pip install "stable-worldmodel[all]==0.1.1"
```

PyPI 分发包名称与 Python 导入名称不同：

```python
import stable_worldmodel as swm
```

项目依赖清单和 lockfile 创建后，必须在其中记录完全一致的精确版本。依赖版本
只能有一个事实来源，不得同时维护相互冲突的依赖声明。

## 环境运行方式

### 本地或普通开发容器

1. 使用 Python 3.10 或更高版本创建并激活独立虚拟环境。
2. 安装项目固定的完整依赖：

   ```bash
   python -m pip install "stable-worldmodel[all]==0.1.1"
   ```

3. 验证导入位置和 CLI：

   ```bash
   python -c "import stable_worldmodel as swm; print(swm.__file__)"
   swm envs
   swm datasets
   swm checkpoints
   ```

4. 项目脚本创建后，先运行 `python scripts/smoke_swm.py`，再启动数据采集、
   baseline 训练或评测。长时间训练前必须先用极小数据和极少 step 完成冒烟运行。
5. 如果需要自定义上游缓存位置，通过 `STABLEWM_HOME` 指定，不得把用户机器上的
   绝对路径写入代码或配置默认值。

### Gemini 云平台（按需启用）

本节不是本地开发的强制要求。只有当用户明确要求在 Gemini 云平台开展开发、
离线训练或推理服务时，才应用本节规则。开始云端任务前必须先判断运行模式，检查
相关环境变量和挂载目录是否真实存在，不得仅根据约定路径假定挂载已经完成。

#### 文件目录与存储

| 存储 | 内容与路径 | 环境变量 | 权限 | 大小 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 容器存储 | 除下列挂载路径以外的所有路径，可放临时代码、数据集、模型和结果 | 无 | 开发环境、离线训练、推理服务均可读写 | small：20G；medium：30G；large：50G；Xlarge：100G | 仅用于临时保存。可以把容器保存为包含当前数据的新镜像；容器关闭或重启后会被销毁，数据不保留。 |
| 代码 | `/gemini/code` | `$GEMINI_CODE` | 开发环境可读写；离线训练和推理服务只读 | 不限制 | 挂载在项目内并归属于项目。开启 SSH 或注入 JupyterLab 后，可以通过对应工具上传或下载。 |
| 数据集 1 | `/gemini/data-1` | `$GEMINI_DATA_IN1` | 只读 | 不限制 | 在平台“数据”栏上传，并在创建项目时选择挂载。 |
| 数据集 2 | `/gemini/data-2` | `$GEMINI_DATA_IN2` | 只读 | 不限制 | 同上。 |
| 数据集 3 | `/gemini/data-3` | `$GEMINI_DATA_IN3` | 只读 | 不限制 | 同上。 |
| 预训练模型 1 | `/gemini/pretrain` | `$GEMINI_PRETRAIN` | 只读 | 不限制 | 只能读取，不得原地修改。 |
| 预训练模型 2 | `/gemini/pretrain2` | `$GEMINI_PRETRAIN2` | 只读 | 不限制 | 只能读取，不得原地修改。 |
| 预训练模型 3 | `/gemini/pretrain3` | `$GEMINI_PRETRAIN3` | 只读 | 不限制 | 只能读取，不得原地修改。 |
| 结果集 | `/gemini/output` | `$GEMINI_DATA_OUT` | 仅离线训练提供，可读写 | 不限制 | 挂载在项目内并归属于项目，用于需要保留的训练结果。 |

#### 云平台执行规则

- 始终优先读取环境变量，例如 `$GEMINI_CODE` 和 `$GEMINI_DATA_OUT`；表中的固定
  路径只用于校验或环境变量缺失时的诊断，不得散落在业务代码中。
- 开始任务前检查所需变量非空、目录存在且权限符合当前运行模式。缺少必要挂载时
  立即报告，不得静默改用其他数据或模型。
- 开发环境中，代码改动写入 `$GEMINI_CODE`。容器存储可以用于可重新生成的缓存、
  解压文件和临时实验，但不得把唯一副本保存在容器存储中。
- 离线训练中，`$GEMINI_CODE`、`$GEMINI_DATA_IN1` 至 `$GEMINI_DATA_IN3` 以及
  `$GEMINI_PRETRAIN` 至 `$GEMINI_PRETRAIN3` 均视为只读。所有 checkpoint、日志、
  指标、视频和最终结果必须写入 `$GEMINI_DATA_OUT`。
- 离线训练时，建议设置：

  ```bash
  export TDWM_RUN_ROOT="$GEMINI_DATA_OUT/tdwm"
  export STABLEWM_HOME="$GEMINI_DATA_OUT/stable_worldmodel"
  mkdir -p "$TDWM_RUN_ROOT" "$STABLEWM_HOME"
  ```

  只有确认 `$GEMINI_DATA_OUT` 已挂载且可写后才能执行这些命令。
- 推理服务中，代码、数据集和预训练模型挂载均按只读处理。推理进程只能把临时文件
  写入容器存储；这些文件会随容器销毁，因此服务不得依赖它们持久保存状态。
- 数据集和预训练模型挂载目录禁止原地写入、重命名或删除。需要转换格式时，把
  转换产物写入离线训练结果目录，或在开发模式下写入明确的项目工作目录。
- 云端配置不得包含密钥、用户私有绝对路径或某次任务专用的挂载编号。数据集和
  模型来源必须通过配置项或环境变量注入。
- 启动正式离线训练前，先完成导入检查、数据集只读检查、小批量前向/反向计算、
  checkpoint 写入与恢复检查，并确认结果确实落在 `$GEMINI_DATA_OUT`。

## Stable World Model 使用方式

将该包作为项目的基础设施层使用：

- 使用 `swm.World(...)` 与向量化环境交互；
- 使用 `world.collect(...)` 采集 episode 数据集；
- 使用 `swm.data.load_dataset(...)` 加载训练和评测序列；
- 使用 `swm.data.convert(...)` 转换数据格式；
- 使用 `WorldModelPolicy` 和 `PlanConfig` 进行基于模型的控制；
- 使用 `stable_worldmodel.planning` 中的 CEM 等 solver 进行规划；
- 使用 `world.evaluate(...)` 进行策略评测并报告成功率；
- 使用 `swm` CLI 检查环境、变化因素、数据集和 checkpoint。

用于规划的方法应实现所选 solver 和 policy 所要求的接口。对于基于 cost 的
规划，这通常意味着接收当前信息字典和候选动作序列，并为每个候选返回一个
cost。编码前必须根据 `0.1.1` 版本核对准确接口，不得凭记忆猜测。

当前已知的上游对比方法包括 TD-MPC2、DINO-WM、PLDM、LeWM、GCBC、GCIVL
和 GCIQL。应使用其发布实现或包的公开 API，不得在本地维护 baseline 源码分叉。

## 项目架构

系统应保持以下三层结构：

1. **基础设施层：** 由 `stable_worldmodel` 负责环境、数据集读写、数据采集、
   规划组件、policy、baseline 组件和通用评测流程。
2. **研究方法层：** 本仓库只负责提出的模型、目标函数、训练集成，以及满足
   Stable World Model 公开接口所需的轻量 adapter。
3. **实验层：** 通过配置和脚本定义数据集、baseline、随机种子、训练实验、
   消融实验和公平比较，不得复制模型或框架代码。

## 目标仓库结构

项目开发过程中采用以下结构。不要为了凑目录而创建空模块；只有目录承担了
真实职责时才创建。

```text
tdwm/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── src/
│   └── tdwm/
│       ├── methods/          # 我们提出的 world model 方法（待定）
│       ├── adapters/         # stable_worldmodel 公开接口的轻量适配
│       ├── training/         # 项目特有的训练组装逻辑
│       └── evaluation/       # 指标计算和对比实验编排
├── configs/
│   ├── data/                 # 数据集和序列配置
│   ├── environment/          # World、环境和 FoV 配置
│   ├── baseline/             # 上游 baseline 实验配置
│   ├── method/               # 我们的方法配置
│   └── experiment/           # 完整、可复现的实验配置
├── scripts/
│   ├── smoke_swm.py          # 包和 API 冒烟测试
│   ├── collect_data.py       # 数据采集入口
│   ├── train.py              # baseline 和新方法共用的训练入口
│   ├── evaluate.py           # 共用评测入口
│   └── compare.py            # 汇总 baseline 对比结果
├── tests/
│   ├── unit/
│   └── integration/
├── reports/                  # 轻量结果表格和研究总结
├── data/                     # 本地数据集；不得提交大型文件
└── outputs/                  # 实验、checkpoint 和视频；默认不得提交
```

`pyproject.toml` 用于定义项目和开发工具。`requirements.txt` 应从选定的依赖
事实来源生成，不得将其作为第二份独立且可能冲突的依赖声明手工维护。

## 开发命令

安装固定版本后，可以立即使用以下命令：

```bash
python -c "import stable_worldmodel as swm; print(swm.__file__)"
swm envs
swm datasets
swm checkpoints
```

常用检查命令：

```bash
swm fovs swm/PushT-v1
swm inspect <dataset-name>
```

以下是项目计划采用的统一命令接口。对应脚本创建后再实现并记录其行为；在脚本
尚不存在时，不得声称这些命令已经可用：

```bash
python scripts/smoke_swm.py
python scripts/collect_data.py --config <config>
python scripts/train.py --config <config>
python scripts/evaluate.py --config <config>
python scripts/compare.py --config <config>
python -m pytest -q
```

## 提出的方法

待定。在明确研究假设之前，模型架构、损失函数、表示形式和训练目标全部保持
开放。不得根据仓库名称推断方法，也不得加入未经讨论的组件。

## 实现规则

待定。只有在明确提出的方法及其支持的输入、输出后，才添加方法特有的不变量
和实现约束。

## 评测协议

最终指标和目标环境仍然待定。在确定之前，遵循以下暂行原则：

- 优先使用基于数据集的 `world.evaluate(...)` 进行受控比较，因为它能够固定
  可达的起点和目标状态。
- 所有可比方法必须使用相同的数据集划分、episode 索引、起始 step、目标偏移、
  评测预算、随机种子、观测预处理和动作空间。
- 单独比较 world model 质量时，所有方法必须使用相同的规划 solver、规划时域、
  候选采样预算和重规划周期。
- 至少报告库返回的 `success_rate` 和逐 episode 成功结果。必须保留随机种子和
  原始逐 episode 结果，不能只保留平均值。
- 选定适合方法的预测指标后，分别报告训练目标、验证目标和长时预测退化情况。
- 选定目标环境后，同时开展分布内评测和基于变化因素的 OOD 评测。
- 需要时比较计算成本，包括参数量、训练时间、推理或规划时间，以及加速器峰值
  显存占用。
- 在声称性能提升前必须运行多个随机种子。具体种子数量和统计报告方式仍待确定。

初始 baseline 候选包括在线 model-based RL 方法 TD-MPC2，以及 JEPA 类 world
model：DINO-WM、PLDM 和 LeWM；在涉及 policy learning 的比较中，可以加入
GCBC、GCIVL 和 GCIQL。最终 baseline 集合仍待确定。

## 数据、Checkpoint 与可复现性

- 使用 Stable World Model 支持的数据格式和 loader；除非实验需要其他格式，
  训练数据优先使用 Lance。
- 上游数据集和 checkpoint 缓存应遵循 `STABLEWM_HOME`，不得硬编码用户机器上的
  绝对路径。
- 在 Gemini 离线训练环境中，持久化输出必须写入 `$GEMINI_DATA_OUT`；数据集和
  预训练模型挂载目录始终只读，容器存储只能保存可丢弃的临时文件。
- 不得提交数据集、checkpoint、视频、密钥或大型生成结果。必要时可以提交紧凑的
  配置、汇总指标和研究报告。
- 每个用于报告的实验都必须记录配置、随机种子、数据集标识及划分、包版本、
  代码版本、硬件信息，以及评测所用 checkpoint。
- 必须显式支持从 checkpoint 恢复，并在依赖长时间训练前测试恢复流程。

## 测试要求

- 开发研究功能前先添加 import 冒烟测试。
- proposed method 的单元测试不得依赖数据下载或网络访问。
- 为数据加载、模型接口兼容性、checkpoint 保存与加载，以及极小规模评测 rollout
  添加集成测试。
- 新功能或缺陷修复应为其改变的行为添加回归测试。
- 测试必须确定可复现，并尽可能使用小型合成数据或 fixture。

## Git 与变更安全

- 所有项目修改都必须与 GitHub 同步。开始修改前先从当前远程跟踪分支拉取最新
  代码并检查本地状态；完成修改和必要验证后，将本次代码变更提交并推送到对应
  远程分支。若存在未提交改动、分支分歧、合并冲突、远程缺失或认证失败，不得
  覆盖本地内容或强制推送，应先报告具体阻塞并等待处理。
- GitHub 只同步源代码、测试、配置、文档和经确认需要保存的轻量研究记录。
  中间文件、数据集、模型权重、checkpoint、视频、缓存、临时目录、原始运行日志、
  垃圾运行产物和大型生成文件不得提交或推送。每次提交前必须检查暂存清单和文件
  大小，并通过 `.gitignore` 阻止这些产物被误纳入版本控制。
- 每次得到可能需要长期保留的实验结果时，必须询问用户是否将该实验记录保存到
  GitHub。只有用户确认后，才可以提交用于审计和复现的轻量内容，例如实验配置、
  随机种子、代码版本、数据集标识与划分、汇总指标、结果表格、研究结论以及外部
  artifact 的路径或校验信息；不得借此上传模型、checkpoint、完整日志或大型结果。
- GitHub 同步授权仅限上述项目内容，不包含发布 release、上传大型 artifact、删除
  远程分支或改写远程历史；这些操作仍需用户另行明确授权。
- 不得修改已安装的 `stable_worldmodel` 包文件或依赖缓存。
- 依赖升级和删除数据集或 checkpoint 必须作为独立、经过明确确认的操作处理。
- 保留用户的无关改动和已有研究产物。

## 当前阶段的完成标准

当前启动阶段的工作在满足以下条件后完成：固定版本的包可以正常导入；可以检查
选定的环境和数据集；第一个 baseline 可以通过共用评测流程运行。性能优于 baseline
属于后续研究里程碑，必须由受控的多次实验支持，单次成功运行不能作为依据。

## 上游资料

- 文档：https://galilai-group.github.io/stable-worldmodel/
- 快速开始：https://galilai-group.github.io/stable-worldmodel/quick_start/
- 源码：https://github.com/galilai-group/stable-worldmodel
- PyPI：https://pypi.org/project/stable-worldmodel/
