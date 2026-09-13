# Little Tree Classifier

[English](README.md) | [简体中文](README.zh-CN.md)

Little Tree Classifier is an image-classification companion project for Little Tree Wallpaper. It is also designed as a standalone model project for desktop applications, CLI tools, and backend services. Integrators do not need to use Little Tree Wallpaper or Python.

## Project Scope

- Train and evaluate wallpaper image classifiers with PyTorch.
- Export checkpoints to ONNX.
- Build a versioned Windows x64 `classifier-runtime` package.
- Expose inference through a standalone worker and JSON Lines protocol.
- Default classes: `abstract`, `anime`, `cars`, `city`, `landscape`, `nature`, `pets`, and `space`.

This project is not a general-purpose image-recognition model. Results are intended as assistance for wallpaper organization and should not drive irreversible actions without review.

## Integration and Attribution

Read the [general integration guide](INTEGRATION.md) to use the worker or the exported ONNX model. Any use, integration, hosted inference service, or redistribution of a released model must follow [MODEL_LICENSE.md](MODEL_LICENSE.md).

**Users must attribute the model source.** Put an accessible attribution in an About page, third-party notices, CLI help or documentation. Include the model version and the actual project or release URL. Fine-tuning, quantization, and format conversion must be disclosed.

```text
Image classification model: Little Tree Classifier
Source: Little Tree Studio, a Little Tree Wallpaper companion project
Model version: <actual version>
Model URL: <actual project or release URL>
Changes: <none, or describe fine-tuning, quantization, conversion, etc.>
```

The code is under the [MIT License](LICENSE). Model weights and project-owned model configuration are covered by [MODEL_LICENSE.md](MODEL_LICENSE.md). Third-party images, pretrained weights, and runtime dependencies retain their own licenses.

## Quick Start

```powershell
python -m pip install -r requirements.txt
python env_check.py
python download_dataset.py --per-class 200
python prepare_dataset.py
python train.py --epochs 10 --batch-size 16
python predict.py path\to\wallpaper.jpg --checkpoint checkpoints\best.pt
```

Recommended ResNet18 training flow:

```powershell
python prepare_dataset.py --output data\split_capped --max-per-class 300
python train.py --data data\split_capped --model resnet18 --pretrained --image-size 160 --balance sampler --epochs 10 --patience 3 --lr 0.0001 --output checkpoints\best.pt
```

Keep downloaded data, personal images, checkpoints, generated reports, and runtime archives out of source control. Check the license and provenance of every dataset item before redistribution.

## Evaluation

```powershell
python audit_dataset.py --json checkpoints\dataset_audit.json
python external_eval.py "path\to\external-images" --checkpoint checkpoints\best.pt --output checkpoints\external_predictions.csv
```

External images are for independent evaluation and are not automatically added to training. The current model has known domain limitations; see the evaluation reports before making quality claims.

## Build a Standalone Runtime

Install release dependencies and build a Windows x64 package:

```powershell
python -m pip install -r requirements.txt
python build_runtime.py --checkpoint checkpoints\best.pt --version 1.0.0
```

Output:

```text
release\classifier-runtime-v1.0.0-windows-x64.zip
```

For protocol testing without PyInstaller:

```powershell
python build_runtime.py --checkpoint checkpoints\best.pt --version 1.0.0 --skip-pyinstaller --debug-dir
```

The runtime uses JSON Lines. Start the worker, read its `ready` line, then send one request per line:

```json
{"action":"classify","image_path":"path/to/image.jpg","top_k":3}
```

See [INTEGRATION.md](INTEGRATION.md) for lifecycle, security, preprocessing, and cross-language examples.

## Project Documents

- [General integration guide](INTEGRATION.md)
- [Model license and attribution](MODEL_LICENSE.md)
- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Code of conduct](CODE_OF_CONDUCT.md)
