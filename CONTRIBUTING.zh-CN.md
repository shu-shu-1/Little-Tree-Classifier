# 贡献指南

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md)

感谢参与 Little Tree Classifier。贡献应保证可复现性、许可证合规以及面向独立第三方应用的接入稳定性。

## 开发环境

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python env_check.py
```

不要提交数据集、个人图片、checkpoint、生成报告、runtime 压缩包或凭据。

## 变更要求

- 保持训练、评估、导出和 runtime 构建可复现。
- 修改预处理时同步更新报告和包元数据。
- 修改 worker 协议时更新 `INTEGRATION.md` 并递增 `protocol_version`。
- 不提交没有再分发权利的图片、数据集或模型权重。
- 发布模型时保留 `MODEL_LICENSE.md` 要求的来源、版本、获取地址、修改说明和第三方声明。

## 验证

```powershell
python -m py_compile classifier_worker.py build_runtime.py
python build_runtime.py --checkpoint checkpoints\best.pt --version 0.0.0-dev --skip-pyinstaller
```

训练或预处理变化还应运行相关评估脚本并记录数据集、checkpoint、模型和指标变化。

## Pull Request

请说明目的、影响范围、协议或预处理变化、模型和数据集版本、验证命令和许可证检查结果。保持 PR 聚焦，不要混入生成文件或个人数据。
