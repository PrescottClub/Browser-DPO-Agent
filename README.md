<div align="center">
  <img src="https://img.shields.io/badge/🤖-DPO--Driver-blue?style=for-the-badge&logo=robot" alt="DPO-Driver Logo" width="200">

  <h1 align="center">🚀 DPO-Driver</h1>
  <p align="center">
    <strong>通过环境反馈进行直接偏好优化的轻量级AI Agent微调技术</strong>
    <br />
    <em>Lightweight AI Agent Fine-tuning via Environment Feedback DPO</em>
    <br />
    <br />
    <a href="./scripts/01_sft_training.py">
      <img src="https://img.shields.io/badge/训练-SFT%20%7C%20DPO-orange?style=flat-square" alt="模型训练">
    </a>
    <a href="./pyproject.toml">
      <img src="https://img.shields.io/badge/环境-Poetry-blue?style=flat-square" alt="环境依赖">
    </a>
    <a href="https://github.com/Qwen/Qwen2">
      <img src="https://img.shields.io/badge/模型-Qwen2--7B-green?style=flat-square" alt="基础模型">
    </a>
    <a href="./LICENSE">
      <img src="https://img.shields.io/badge/许可-MIT-lightgrey?style=flat-square" alt="开源许可">
    </a>
    <a href="./tests/">
      <img src="https://img.shields.io/badge/测试-21%20passed-brightgreen?style=flat-square" alt="测试状态">
    </a>
    <a href="./start_mlflow_ui.py">
      <img src="https://img.shields.io/badge/MLflow-实验追踪-blue?style=flat-square" alt="MLflow集成">
    </a>
    <a href="https://pytorch.org/">
      <img src="https://img.shields.io/badge/PyTorch-2.5%2B-red?style=flat-square" alt="PyTorch版本">
    </a>
    <a href="./pyproject.toml">
      <img src="https://img.shields.io/badge/Python-3.11%2B-blue?style=flat-square" alt="Python版本">
    </a>
  </p>

  <p align="center">
    <strong>📊 实验结果：SFT基线60% → DPO强化70% (+10%绝对提升)</strong>
  </p>
</div>

---

## 🌟 项目亮点

**DPO-Driver** 是一个突破性的AI Agent训练框架，它摒弃了传统RLHF的复杂流程，直接利用环境的二元成功/失败信号驱动DPO训练，实现了**资源友好**、**自动化**且**高效**的Agent对齐新范式。

### 🎯 核心创新

- **🔄 环境反馈DPO (EF-DPO)**：首次将环境的成功/失败信号直接作为偏好数据源
- **💡 免标注训练**：无需人类偏好标注，完全自动化的数据飞轮
- **⚡ 轻量级部署**：单张RTX 4060即可完成完整训练流程
- **📈 显著提升**：在MiniWoB++基准上实现+10%绝对性能增长

### 🏆 技术优势

| 传统RLHF | DPO-Driver |
|---------|------------|
| 需要大量人类标注 | 完全自动化数据收集 |
| 复杂的奖励模型训练 | 直接偏好优化 |
| 高计算资源需求 | 消费级GPU即可运行 |
| 难以规模化 | 易于扩展和复现 |

## 🔬 实验结果

### 📊 性能对比

```
模型版本                    平均成功率    提升幅度
SFT Baseline               60.00%        -
DPO Trained (EF-DPO)       70.00%       +10.00%
```

### 🎯 评估环境
- **基准**: MiniWoB++ 网页操作任务
- **模型**: Qwen2-7B-Instruct
- **硬件**: NVIDIA RTX 4060 (8GB VRAM)
- **任务**: 点击、输入、选择等常见网页操作

## 🚀 快速开始

### 1. 环境准备

**系统要求：**
- Python 3.11+
- NVIDIA GPU (推荐RTX 4060 8GB+)
- CUDA 12.1+

**安装Poetry (如果尚未安装)：**
```bash
# Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -

# Linux/macOS
curl -sSL https://install.python-poetry.org | python3 -
```

**克隆并安装：**
```bash
# 克隆项目
git clone https://github.com/your-repo/dpo-driver.git
cd dpo-driver

# 安装依赖 (Poetry会自动创建虚拟环境)
poetry install

# 验证安装
poetry run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 2. 验证环境

```bash
# 验证项目环境配置
poetry run python scripts/00_verify_setup.py
```

### 3. SFT基线训练

```bash
# 训练监督微调基线模型
poetry run python scripts/01_sft_training.py
```

### 4. 偏好数据收集

```bash
# 收集环境反馈偏好数据
poetry run python scripts/02_collect_preferences.py
```

### 5. DPO强化训练

```bash
# 执行直接偏好优化
poetry run python scripts/03_dpo_training.py
```

### 6. 性能评估

```bash
# 对比评估SFT vs DPO性能
poetry run python scripts/04_evaluate_agent.py
```

### 完整训练流程

| 步骤 | 脚本 | 作用 | 预期时间 | 输出 |
|------|------|------|----------|------|
| 0️⃣ | `00_verify_setup.py` | 环境验证 | 30秒 | 验证报告 |
| 1️⃣ | `01_sft_training.py` | SFT基线训练 | 10-15分钟 | `models/sft_v1_adapter/` |
| 2️⃣ | `02_collect_preferences.py` | 偏好数据收集 | 5-10分钟 | `data/preferences/dpo_v1_data.jsonl` |
| 3️⃣ | `03_dpo_training.py` | DPO强化训练 | 5-8分钟 | `models/dpo_v1_adapter/` |
| 4️⃣ | `04_evaluate_agent.py` | 性能评估 | 3-5分钟 | 性能对比报告 |

**一键运行完整流程：**
```bash
# 按顺序执行所有步骤
poetry run python scripts/00_verify_setup.py && poetry run python scripts/01_sft_training.py && poetry run python scripts/02_collect_preferences.py && poetry run python scripts/03_dpo_training.py && poetry run python scripts/04_evaluate_agent.py
```

## 📊 实验追踪与监控

本项目集成了MLflow进行实验管理和结果可视化：

### 启动MLflow UI
```bash
# 启动实验追踪界面
poetry run python start_mlflow_ui.py

# 在浏览器中访问：http://localhost:5000
```

### 清理缓存
如需清理项目缓存文件：
```bash
# 清理Python缓存
Get-ChildItem -Recurse -Directory "__pycache__" | Remove-Item -Recurse -Force

# 清理pytest缓存  
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
```

### 实验管理
- 🔬 **SFT实验**: 查看监督微调的损失曲线和模型性能
- 🎯 **DPO实验**: 对比偏好优化前后的性能提升  
- 📈 **评估结果**: 可视化不同模型在各任务上的成功率
- 🔄 **参数对比**: 追踪不同配置下的实验结果

每次运行训练脚本都会自动记录到MLflow，便于实验管理和结果复现。

## 🏗️ 系统架构

```mermaid
graph TB
    A[用户指令] --> B[Agent Core<br/>Qwen2-7B]
    B --> C[思考-行动<br/>Thought-Action]
    C --> D[环境接口<br/>Selenium]
    D --> E[MiniWoB++<br/>网页环境]
    E --> F[成功/失败<br/>二元反馈]
    F --> G[偏好数据<br/>Chosen/Rejected]
    G --> H[DPO训练<br/>直接偏好优化]
    H --> B
```

## 📁 项目结构

```
dpo-driver/
├── src/
│   ├── agent/          # Agent核心模块 (模型封装、DPO训练)
│   ├── environment/    # 环境接口 (Selenium集成)
│   ├── miniwob/       # MiniWoB++集成 (任务环境)
│   └── utils/         # 工具函数
├── scripts/           # 训练和评估脚本
│   ├── 00_verify_setup.py      # 环境验证
│   ├── 01_sft_training.py      # SFT基线训练
│   ├── 02_collect_preferences.py # 偏好数据收集
│   ├── 03_dpo_training.py      # DPO强化训练
│   └── 04_evaluate_agent.py    # 性能评估
├── data/             # 数据集
│   ├── preferences/   # DPO偏好数据
│   └── sft_golden_samples.jsonl # SFT训练数据
├── models/           # 模型存储
│   ├── sft_v1_adapter/  # SFT基线模型
│   └── dpo_v1_adapter/  # DPO强化模型
├── tests/            # 测试代码
├── README.md         # 项目文档
├── LICENSE           # 开源许可
├── CONTRIBUTING.md   # 贡献指南
└── pyproject.toml    # 项目配置
```

## 🔧 核心技术

### Agent架构
- **基础模型**: Qwen2-7B-Instruct
- **微调技术**: LoRA (Low-Rank Adaptation)
- **推理模式**: ReAct (Reasoning + Acting)

### DPO训练
- **学习率**: 5e-6
- **Beta值**: 0.1
- **训练步数**: 50步
- **优化器**: AdamW

### 环境集成
- **浏览器**: Selenium WebDriver
- **任务集**: MiniWoB++ 标准化基准
- **状态表示**: 简化DOM + 可交互元素

## 📈 性能分析

### 成功案例
- **文本输入任务**: 准确率提升15%
- **按钮点击任务**: 准确率提升8%
- **表单填写任务**: 准确率提升12%

### 技术洞察
1. **稀疏奖励有效性**: 证明了二元反馈足以驱动有效学习
2. **探索-利用平衡**: DPO在已知策略附近进行精炼优化
3. **收益递减现象**: 高基线下的边际改进成本递增

## ⚙️ 配置管理

项目使用 `config.yaml` 进行统一配置管理：

### 核心配置项
```yaml
model:
  base_model_name: "Qwen/Qwen2-7B-Instruct"  # 基础模型
  
training:
  sft:
    learning_rate: 2.0e-4    # SFT学习率
    max_steps: 100           # SFT训练步数
  dpo:
    learning_rate: 5.0e-6    # DPO学习率 (通常更小)
    beta: 0.1                # DPO beta参数
    max_steps: 50            # DPO训练步数
```

### 自定义配置
```bash
# 使用自定义配置文件
poetry run python scripts/01_sft_training.py --config_path my_config.yaml
```

配置文件支持热修改，无需重新安装依赖。

## 🛠️ 高级用法

### 自定义训练配置

```python
# 修改DPO训练参数
dpo_config = {
    "learning_rate": 5e-6,
    "beta": 0.1,
    "max_steps": 50,
    "batch_size": 4
}
```

### 扩展任务集

```python
# 添加新的MiniWoB++任务
task_list = [
    "click-button-v1",
    "enter-text-v1",
    "select-option-v1",
    # 添加更多任务...
]
```

### 使用示例

```python
from src.agent.model import AgentModel
from src.environment.interface import EnvironmentInterface

# 初始化Agent
agent = AgentModel(model_name="Qwen/Qwen2-7B-Instruct")
agent.load_adapter("./models/dpo_v1_adapter")

# 初始化环境
env = EnvironmentInterface()

# 执行任务
result = agent.predict("点击页面上的提交按钮")
print(f"Agent思考: {result['thought']}")
print(f"执行动作: {result['action']}")
```

## 🔍 技术原理

### 环境反馈DPO (EF-DPO)

传统的DPO需要人类标注的偏好对，而我们的EF-DPO直接利用环境反馈：

1. **数据收集**: Agent在环境中执行任务，记录轨迹
2. **偏好标注**: 成功轨迹标记为"chosen"，失败轨迹标记为"rejected"
3. **DPO训练**: 使用偏好对训练模型，提升决策质量

### 关键技术突破

- **稀疏奖励处理**: 将任务级成功/失败信号有效转化为轨迹级偏好
- **信用分配**: 通过对比学习隐式解决动作-结果的信用分配问题
- **探索-利用平衡**: DPO天然具备在已知策略附近精炼的特性

## 📚 相关工作

- **ReAct**: Reasoning and Acting in Language Models
- **DPO**: Direct Preference Optimization
- **MiniWoB++**: Web-based Interaction Benchmark
- **LoRA**: Low-Rank Adaptation of Large Language Models

## 🤝 贡献指南

我们欢迎社区贡献！请查看 [CONTRIBUTING.md](./CONTRIBUTING.md) 了解详细信息。

### 贡献方式
- 🐛 报告Bug和问题
- 💡 提出新功能建议
- 📝 改进文档
- 🔧 提交代码优化

## 📄 许可证

本项目基于 [MIT License](./LICENSE) 开源。

## 🙏 致谢

感谢以下开源项目的支持：
- [Qwen2](https://github.com/QwenLM/Qwen2) - 基础语言模型
- [TRL](https://github.com/huggingface/trl) - DPO训练框架
- [MiniWoB++](https://github.com/Farama-Foundation/miniwob-plusplus) - 评估基准
- [Transformers](https://github.com/huggingface/transformers) - 模型库

---

<div align="center">
  <p><strong>🌟 如果这个项目对您有帮助，请给我们一个Star！</strong></p>
  <p><em>让AI Agent拥有真正的决策智能</em></p>
</div>