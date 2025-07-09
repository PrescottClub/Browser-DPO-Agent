<div align="center">
  <a href="https://github.com/your-repo/browser-dpo-agent">
    <img src="https://path-to-your/awesome-logo.png" alt="Browser-DPO-Agent Logo" width="150">
  </a>
  <h1 align="center">Browser-DPO-Agent</h1>
  <p align="center">
    <strong>一个基于直接偏好优化（DPO）的、能够自主学习并执行复杂浏览器任务的AI智能体</strong>
    <br />
    <br />
    <a href="./scripts/01_sft_training.py">
      <img src="https://img.shields.io/badge/模型训练-SFT%20%7C%20DPO-9cf" alt="模型训练">
    </a>
    <a href="./pyproject.toml">
      <img src="https://img.shields.io/badge/环境-Poetry-blueviolet" alt="环境依赖">
    </a>
    <a href="https://github.com/your-repo/browser-dpo-agent/graphs/commit-activity">
      <img src="https://img.shields.io/badge/状态-活跃开发-brightgreen" alt="项目状态">
    </a>
    <a href="./LICENSE">
      <img src="https://img.shields.io/badge/许可-MIT-lightgrey" alt="开源许可">
    </a>
  </p>
</div>

---

**Browser-DPO-Agent** 不仅仅是一个浏览器自动化工具，它是一个具备**自主决策能力**的AI智能体。我们通过创新的 **SFT + DPO** 混合训练模式，使大语言模型（LLM）能够在真实、复杂的网页环境中，像人类一样“思考”和“选择”，最终完成指定任务。

## 核心理念：从“指令执行”到“决策智能”

传统的Web Agent依赖于精确的指令和CSS选择器，一旦页面结构发生变化或遇到预期外的状况，它们便束手无策。我们认为，真正的智能体应当具备适应性和决策能力。

- **`SFT` (监督微调)**: 我们首先通过高质量的 `(指令, "思考过程", 动作)` 数据集对模型进行微调，让它掌握基础的浏览器操作语言和“思考-行动”模式。这是智能体学习“说”和“做”的阶段。
- **`DPO` (直接偏好优化)**: 这是我们实现“决策智能”的关键。我们为模型提供包含 `(指令, 胜利动作, 失败动作)` 的偏好对，让它在多种可能性中，学习**选择更优、更高效的路径**。这赋予了模型在模糊和不确定场景下的决策能力。

## ✨ 特性

- **🧠 SFT+DPO混合训练**: 完美结合指令遵循与决策优化，打造更聪明的智能体。
- **🌐 端到端工作流**: 提供从数据准备、模型训练到环境交互的完整解决方案。
- **🔌 模块化与可扩展**: 基于`transformers`和`PEFT`，轻松更换模型、扩展动作集。
- **🎯 面向真实世界**: 专为解决复杂、动态的真实网页任务而设计。

## 🚀 快速开始

### 1. 环境准备

使用 [Poetry](https://python-poetry.org/) 管理项目依赖。

```bash
# 克隆仓库
git clone https://github.com/your-repo/browser-dpo-agent.git
cd browser-dpo-agent

# 安装依赖
poetry install

# （可选）配置Poetry在项目内创建虚拟环境
poetry config virtualenvs.in-project true
```

### 2. SFT基线模型训练

这是Agent学习基础操作的第一步。

```bash
# 执行SFT训练
poetry run python scripts/01_sft_training.py
```
训练完成后，一个经过LoRA微调的adapter将保存在`./models/sft_v1_adapter/`。

### 3. (即将到来) DPO偏好学习
*此功能正在积极开发中。*

## 🏗️ 项目架构

```
Browser-DPO-Agent/
├── data/
│   ├── preferences/          # (DPO) 偏好数据集
│   └── sft_golden_samples.jsonl # (SFT) 高质量指令样本
├── models/
│   └── sft_v1_adapter/       # 训练好的SFT LoRA adapter
├── scripts/
│   ├── 00_dpo_pressure_test.py # DPO压力测试脚本
│   └── 01_sft_training.py      # SFT训练主脚本
├── src/
│   ├── agent/
│   │   └── model.py          # Agent核心模型 (SFT+DPO)
│   └── environment/
│       └── interface.py      # 与浏览器环境的交互接口
├── config.yaml               # 全局配置文件
├── pyproject.toml            # Poetry依赖管理
└── README.md                 # 项目文档
```

## 🤝 贡献

我们热烈欢迎任何形式的社区贡献！无论是代码实现、文档改进还是问题反馈，都对我们至关重要。请参考我们的 [贡献指南](./CONTRIBUTING.md) 开始你的贡献之旅。

## 📜 开源许可

本项目基于 [MIT License](./LICENSE) 开源。

---
<p align="center">
  <em>让智能体拥有真正的决策力。</em>
</p> 