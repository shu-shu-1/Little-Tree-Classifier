# Contributing

[English](CONTRIBUTING.md) | [简体中文](CONTRIBUTING.zh-CN.md)

Thank you for contributing to Little Tree Classifier. Contributions should preserve reproducibility, license compliance, and stable integration for independent third-party applications.

## Development

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python env_check.py
```

Do not commit datasets, personal images, checkpoints, generated reports, runtime archives, or credentials.

## Changes

- Keep training, evaluation, export, and runtime builds reproducible.
- Update reports and package metadata when preprocessing changes.
- Update `INTEGRATION.md` and increment `protocol_version` when the worker protocol changes.
- Do not submit images, datasets, or weights without verified redistribution rights.
- When publishing models, preserve attribution, version, acquisition URL, modification notes, and third-party notices from `MODEL_LICENSE.md`.

## Verification

```powershell
python -m py_compile classifier_worker.py build_runtime.py
python build_runtime.py --checkpoint checkpoints\best.pt --version 0.0.0-dev --skip-pyinstaller
```

For training or preprocessing changes, run the relevant evaluation scripts and record the dataset, checkpoint, model, and metric changes.

## Pull Requests

Describe the purpose, affected scope, protocol or preprocessing changes, model and dataset versions, verification commands, and license review. Keep pull requests focused and do not mix generated files or personal data with code changes.
