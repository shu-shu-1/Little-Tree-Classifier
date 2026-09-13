# Little Tree Classifier

[English](README.md) | [简体中文](README.zh-CN.md)

Little Tree Classifier 是小树壁纸的附属图片分类项目，同时面向桌面应用、命令行工具和后端服务提供独立模型与运行时接入方案。使用者不需要使用小树壁纸，也不要求使用 Python。

## 项目范围

- 使用 PyTorch 训练和评估壁纸图片分类器。
- 将 checkpoint 导出为 ONNX。
- 构建版本化的 Windows x64 `classifier-runtime`。
- 通过独立 worker 和 JSON Lines 协议提供推理。
- 默认类别：`abstract`、`anime`、`cars`、`city`、`landscape`、`nature`、`pets`、`space`。

本项目不是通用图像识别模型，结果用于辅助整理壁纸。不要在未人工确认的情况下用结果执行不可逆操作。

## 接入与署名

请阅读[通用接入指南](INTEGRATION.md)以及[模型使用与署名条款](MODEL_LICENSE.md)。使用本项目发布的模型必须注明来源，并保留实际模型版本和项目或 Release 地址。微调、量化和格式转换也必须说明。

```text
图片分类模型：Little Tree Classifier
来源：Little Tree Studio，小树壁纸附属图片分类项目
模型版本：<实际版本>
模型地址：<实际项目或 Release URL>
修改说明：<无修改，或说明微调、量化、转换等变更>
```

代码采用 [MIT License](LICENSE)，模型权重和项目自有模型配置采用 [MODEL_LICENSE.md](MODEL_LICENSE.md)。第三方图片、预训练权重和运行时依赖仍适用各自许可证。

## 快速开始

```powershell
python -m pip install -r requirements.txt
python env_check.py
python download_dataset.py --per-class 200
python prepare_dataset.py
python train.py --epochs 10 --batch-size 16
python predict.py path\to\wallpaper.jpg --checkpoint checkpoints\best.pt
```

推荐训练流程：

```powershell
python prepare_dataset.py --output data\split_capped --max-per-class 300
python train.py --data data\split_capped --model resnet18 --pretrained --image-size 160 --balance sampler --epochs 10 --patience 3 --lr 0.0001 --output checkpoints\best.pt
```

请勿提交数据集、个人图片、checkpoint、生成报告、runtime 压缩包或凭据。重新发布前必须核对所有数据项的许可证和来源。

## 评估

```powershell
python audit_dataset.py --json checkpoints\dataset_audit.json
python external_eval.py "path\to\external-images" --checkpoint checkpoints\best.pt --output checkpoints\external_predictions.csv
```

外部图片仅用于独立评估，不会自动加入训练。当前模型存在领域限制，请在发布前查看评估报告。

## 构建独立运行时

```powershell
python -m pip install -r requirements.txt
python build_runtime.py --checkpoint checkpoints\best.pt --version 1.0.0
```

输出：

```text
release\classifier-runtime-v1.0.0-windows-x64.zip
```

协议测试版：

```powershell
python build_runtime.py --checkpoint checkpoints\best.pt --version 1.0.0 --skip-pyinstaller --debug-dir
```

更多生命周期、协议、安全、预处理和跨语言接入说明见[通用接入指南](INTEGRATION.md)。

## 项目文档

- [通用接入指南](INTEGRATION.md)
- [模型许可与署名](MODEL_LICENSE.md)
- [贡献指南](CONTRIBUTING.md)
- [安全政策](SECURITY.md)
- [行为准则](CODE_OF_CONDUCT.md)
