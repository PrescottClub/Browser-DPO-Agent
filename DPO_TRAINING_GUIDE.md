# DPO训练与评估指南

本文档详细说明了如何执行DPO（Direct Preference Optimization）训练以及最终的模型评估。

## 📋 前置条件

在开始DPO训练之前，请确保：

1. **SFT基线模型已训练完成**
   - 运行过 `python scripts/01_sft_training.py`
   - 存在 `models/sft_v1_adapter/checkpoint-100/` 目录

2. **偏好数据已收集**
   - 运行过 `python scripts/02_collect_preferences.py`
   - 或者使用提供的示例数据 `data/preferences/dpo_v1_data.jsonl`

3. **环境依赖已安装**
   - 所有必要的Python包（transformers, trl, datasets等）

## 🧪 测试设置

在开始训练前，建议先运行测试脚本验证环境：

```bash
python scripts/test_dpo_setup.py
```

这个脚本会检查：
- 所有必要的导入是否正常
- 数据文件是否存在
- 偏好数据格式是否正确

## 🚀 执行DPO训练

### 步骤1: DPO训练

```bash
python scripts/03_dpo_training.py
```

**训练配置：**
- 基础模型：Qwen/Qwen2-7B-Instruct
- SFT起点：`./models/sft_v1_adapter/checkpoint-100`
- 偏好数据：`data/preferences/dpo_v1_data.jsonl`
- 输出路径：`./models/dpo_v1_adapter`

**训练参数：**
- 学习率：5e-6（比SFT更小）
- 训练步数：50步
- Beta值：0.1
- 批次大小：1（梯度累积步数：2）

### 步骤2: 模型评估

```bash
python scripts/04_evaluate_agent.py
```

**评估配置：**
- 评估任务：`click-button-v1`, `enter-text-v1`
- 每个任务评估次数：10次
- 对比模型：SFT基线 vs DPO强化

## 📊 评估结果解读

评估脚本会输出类似以下的报告：

```
==================================================
                最终评估报告
==================================================
SFT 基线模型平均成功率: 45.00%
DPO 强化模型平均成功率: 67.00%
--------------------------------------------------
绝对成功率提升: +22.00%
==================================================

🎉 结论：成功！DPO显著提升了Agent性能，已达成项目核心目标！
```

**成功标准：**
- 绝对成功率提升 ≥ 20% → 项目目标达成
- 绝对成功率提升 < 20% → 需要进一步优化

## 🔧 自定义配置

### 修改训练参数

编辑 `scripts/03_dpo_training.py` 中的配置：

```python
# DPO训练参数
learning_rate=5e-6,    # 学习率
max_steps=50,          # 训练步数
beta=0.1,              # DPO的beta参数
```

### 修改评估任务

编辑 `scripts/04_evaluate_agent.py` 中的任务列表：

```python
EVAL_TASKS = [
    'miniwob/click-button-v1',
    'miniwob/enter-text-v1',
    'miniwob/login-user-v1',  # 添加更多任务
]
```

### 增加偏好数据

向 `data/preferences/dpo_v1_data.jsonl` 添加更多偏好对：

```json
{"prompt": "任务描述", "chosen": "好的响应", "rejected": "差的响应"}
```

## 🐛 常见问题

### 1. 显存不足
- 减少 `per_device_train_batch_size`
- 增加 `gradient_accumulation_steps`
- 使用量化配置

### 2. 训练不收敛
- 降低学习率（如 1e-6）
- 增加训练步数
- 检查偏好数据质量

### 3. 评估环境错误
- 确保Chrome浏览器已安装
- 检查MiniWoB++环境注册
- 验证任务ID是否正确

## 📁 文件结构

```
Browser-DPO-Agent/
├── scripts/
│   ├── 03_dpo_training.py      # DPO训练脚本
│   ├── 04_evaluate_agent.py    # 最终评估脚本
│   └── test_dpo_setup.py       # 设置测试脚本
├── data/preferences/
│   └── dpo_v1_data.jsonl       # 偏好数据
├── models/
│   ├── sft_v1_adapter/         # SFT基线模型
│   └── dpo_v1_adapter/         # DPO训练结果
└── src/agent/
    └── model.py                # 包含train_dpo方法
```

## 🎯 下一步

完成DPO训练和评估后，可以考虑：

1. **超参数调优**：尝试不同的学习率、beta值
2. **数据增强**：收集更多高质量的偏好数据
3. **任务扩展**：在更多MiniWoB++任务上评估
4. **模型部署**：将最佳模型部署到生产环境

---

**注意**：DPO训练是一个迭代过程，可能需要多次实验才能达到最佳效果。建议保存每次训练的结果，以便对比分析。
