# 贡献指南 Contributing Guide

感谢您对 DPO-Driver 项目的关注！我们欢迎各种形式的贡献。

## 🤝 如何贡献

### 报告问题 (Issues)

如果您发现了bug或有功能建议，请：

1. 检查是否已有相关issue
2. 创建新issue，详细描述问题或建议
3. 提供复现步骤（如果是bug）
4. 包含环境信息（操作系统、Python版本等）

### 提交代码 (Pull Requests)

1. **Fork项目** 到您的GitHub账户
2. **创建分支** 用于您的修改
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **进行修改** 并确保代码质量
4. **运行测试** 确保所有测试通过
   ```bash
   poetry run pytest
   ```
5. **提交更改** 并推送到您的fork
   ```bash
   git commit -m "Add: your feature description"
   git push origin feature/your-feature-name
   ```
6. **创建Pull Request** 到主仓库

## 📝 代码规范

### Python代码风格

- 遵循 PEP 8 规范
- 使用 `black` 进行代码格式化
- 使用 `isort` 整理import语句
- 使用类型注解

```bash
# 格式化代码
poetry run black .
poetry run isort .
```

### 提交信息规范

使用清晰的提交信息：

- `Add: 新增功能`
- `Fix: 修复bug`
- `Update: 更新现有功能`
- `Docs: 文档更新`
- `Test: 测试相关`

## 🧪 测试

在提交代码前，请确保：

1. 所有现有测试通过
2. 为新功能添加测试
3. 测试覆盖率保持在合理水平

```bash
# 运行测试
poetry run pytest

# 查看测试覆盖率
poetry run pytest --cov=src
```

## 📚 文档

- 为新功能添加文档
- 更新README.md（如需要）
- 在代码中添加适当的注释和docstring

## 🔍 代码审查

所有Pull Request都会经过代码审查：

- 确保代码质量和一致性
- 验证功能正确性
- 检查测试覆盖率
- 确认文档完整性

## 🎯 开发环境设置

1. **克隆项目**
   ```bash
   git clone https://github.com/your-repo/dpo-driver.git
   cd dpo-driver
   ```

2. **安装依赖**
   ```bash
   poetry install
   ```

3. **激活虚拟环境**
   ```bash
   poetry shell
   ```

4. **运行测试**
   ```bash
   pytest
   ```

## 💡 贡献想法

我们特别欢迎以下方面的贡献：

- 🐛 Bug修复
- ✨ 新功能开发
- 📝 文档改进
- 🧪 测试用例增加
- 🔧 性能优化
- 🌐 多语言支持

## 📞 联系我们

如果您有任何问题或建议，可以通过以下方式联系：

- 创建GitHub Issue
- 发送邮件到项目维护者
- 参与项目讨论

## 🙏 致谢

感谢所有为 DPO-Driver 项目做出贡献的开发者！

---

**让我们一起构建更智能的AI Agent！** 🚀
