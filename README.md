<div align="center">
  <a href="https://github.com/your-repo/browser-dpo-agent">
    <img src="https://www.svgrepo.com/show/306500/openai.svg" alt="Browser-DPO-Agent Logo" width="120">
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

## 🎯 项目愿景：打造终极Web Agent

我们的目标是构建一个能够自主理解和执行任何网页任务的通用型智能体。它将能够：

- **处理不确定性**：在没有明确CSS选择器或遇到非预期页面结构时，依然能做出合理判断。
- **执行复杂任务链**：完成如“登录邮箱、查找最新订单邮件、提取订单号并填入CRM系统”等一系列复杂操作。
- **持续自我优化**：通过与环境的持续交互和DPO训练，不断提升决策的准确性和效率。

## 核心理念：从“指令执行”到“决策智能”

传统的Web Agent依赖于精确的指令和CSS选择器，一旦页面结构发生变化或遇到预期外的状况，它们便束手无策。我们认为，真正的智能体应当具备适应性和决策能力。

- **`SFT` (监督微调)**: 我们首先通过高质量的 `(指令, "思考过程", 动作)` 数据集对模型进行微调，让它掌握基础的浏览器操作语言和“思考-行动”模式。这是智能体学习“说”和“做”的阶段。
- **`DPO` (直接偏好优化)**: 这是我们实现“决策智能”的关键。我们为模型提供包含 `(指令, 胜利动作, 失败动作)` 的偏好对，让它在多种可能性中，学习**选择更优、更高效的路径**。这赋予了模型在模糊和不确定场景下的决策能力。

## ✨ 特性

- **🧠 SFT+DPO混合训练**: 完美结合指令遵循与决策优化，打造更聪明的智能体。
- **🌐 端到端工作流**: 提供从数据准备、模型训练、效果评估到真实环境交互的完整解决方案。
- **🤖 奖励模型（RM）集成**: DPO训练的核心，用于为智能体的行为提供偏好信号。
- **🔌 模块化与可扩展**: 基于`transformers`和`PEFT`，轻松更换模型、扩展动作集。
- **📊 专业评估套件**: 集成`MiniWoB++`等标准化测试环境，科学评估Agent的泛化和决策能力。

## 🚀 项目路线图 (Roadmap)

- **Phase 1: SFT基线闭环 (✅ 已完成)**
  - [x] 实现完整的SFT训练流程。
  - [x] 构建高质量的“黄金”SFT数据集。
  - [x] 产出具备基础“思考->行动”能力的v1模型。

- **Phase 2: DPO偏好学习闭环 (⏳ 进行中)**
  - [ ] 构建偏好数据集收集流程。
  - [ ] 实现奖励模型（Reward Model）的训练和集成。
  - [ ] 实现完整的DPO训练流程，优化Agent决策能力。

- **Phase 3: 自主探索与学习**
  - [ ] 让Agent在真实环境中自主探索，收集新的偏好数据。
  - [ ] 建立持续学习（Continual Learning）循环，使Agent能力不断进化。

- **Phase 4: 复杂任务执行与泛化**
  - [ ] 挑战需要多步推理和长期记忆的复杂任务。
  - [ ] 在更多、更复杂的真实网站上进行泛化能力测试。

## 🏗️ 最终架构展望

<div align="center">
  <img src="https://path-to-your/architecture-diagram.png" alt="项目最终架构图" width="700">
</div>

1. **用户指令 (User Prompt)**: 用户输入高级指令（如“帮我预订一张明天去上海的机票”）。
2. **Agent核心 (Agent Core)**:
   - **SFT基础模型**: 理解指令，生成初步的`thought`和`action`。
   - **DPO策略模型**: 在多个可能的`action`中，选择最优的一个。
3. **环境交互 (Environment Interaction)**: 在浏览器（如`MiniWoB++`或真实网站）中执行选定的`action`。
4. **状态反馈 (State Feedback)**: 浏览器返回新的页面状态（DOM、截图等）。
5. **奖励模型 (Reward Model)**: 评估`action`的好坏，生成偏好数据（`chosen` vs `rejected`）。
6. **经验回放与持续优化 (Experience Replay & Fine-tuning)**: 将新的偏好数据存入经验池，用于下一轮DPO训练，持续优化Agent的决策策略。

## 快速开始

### 1. 环境准备

使用 [Poetry](https://python-poetry.org/) 管理项目依赖。

```bash
# 克隆仓库
git clone https://github.com/your-repo/browser-dpo-agent.git
cd browser-dpo-agent

# 安装依赖
poetry install
```

### 2. SFT基线模型训练 (Phase 1)

```bash
# 执行SFT训练
poetry run python scripts/01_sft_training.py
```
训练完成后，一个经过LoRA微调的adapter将保存在`./models/sft_v1_adapter/`。

## 🤝 贡献

我们热烈欢迎任何形式的社区贡献！无论是代码实现、文档改进还是问题反馈，都对我们至关重要。请参考我们的 [贡献指南](./CONTRIBUTING.md) 开始你的贡献之旅。

## 📜 开源许可

本项目基于 [MIT License](./LICENSE) 开源。

---
<p align="center">
  <em>让智能体拥有真正的决策力。</em>
</p> 