# 贡献指南

感谢你有兴趣为 PlateBlur 做出贡献！

## 环境准备

```bash
git clone https://github.com/zhouyp001/PlateBlur.git
cd PlateBlur
pip install -r requirements.txt
```

## 项目结构

```
.
├── main.py              # GUI 入口
├── gui/                 # Qt 界面
├── utils/               # 工具模块（处理引擎、日志、导出）
├── config/              # 配置文件
├── weights/             # YOLO 模型
├── tests/               # 测试
```

## 开发流程

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feat/my-feature`
3. 提交更改：`git commit -m "feat: 描述"`
4. 推送到你的 fork：`git push origin feat/my-feature`
5. 提交 Pull Request

## 提交规范

使用约定式提交格式：

- `feat:` 新功能
- `fix:` Bug 修复
- `refactor:` 重构
- `docs:` 文档更新
- `chore:` 构建 / 工具相关
- `style:` 格式调整

## 代码风格

- Python 代码使用 4 空格缩进
- 保持与现有代码风格一致
- 优先简洁，避免过度抽象

## 测试

```bash
python tests/test_video_player.py
python tests/test_video_player_qt.py
```

## Issue

- Bug 报告请使用 Bug 报告模板
- 功能建议请使用功能建议模板
