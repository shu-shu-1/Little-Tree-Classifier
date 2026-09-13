# General Integration Guide

[English](INTEGRATION.md) | [简体中文](INTEGRATION.zh-CN.md)

Little Tree Classifier is a Little Tree Wallpaper companion model project, but its models and runtime can be used by other desktop applications, CLI tools, and backend services. This guide does not require Little Tree Wallpaper. Any language that can manage a child process and standard streams can integrate it.

## Attribution Is Required

Use, integration, hosted inference, and redistribution must follow [MODEL_LICENSE.md](MODEL_LICENSE.md), including fine-tuned, quantized, or converted models.

- Desktop or web app: show attribution in About, model settings, or third-party notices.
- CLI: include it in help output or shipped documentation.
- Hosted API: include it in public service documentation.
- Redistribution: ship the model terms, source, exact version, and acquisition URL.

```text
Image classification model: Little Tree Classifier
Source: Little Tree Studio, a Little Tree Wallpaper companion project
Model version: <actual version>
Model URL: <actual project or release URL>
Changes: <none, or describe modifications>
```

Attribution does not imply endorsement. It does not replace third-party license obligations.

## Integration Options

| Option | Requirements | Use case |
| --- | --- | --- |
| Standalone EXE worker | Child-process and pipe support | Desktop apps without Python or ONNX dependencies |
| Source worker | Python, ONNX Runtime, NumPy, and Pillow | Development, scripts, or internal services |
| Direct ONNX | ONNX Runtime for your language and matching preprocessing | Existing inference infrastructure |

The current build targets Windows x64. Other platforms require their own build and validation.

## Package and Installation

Build a source package for protocol testing:

```powershell
python build_runtime.py --checkpoint checkpoints\best.pt --version 1.0.0 --skip-pyinstaller --debug-dir
```

The source package contains `classifier-worker.py`, `model.onnx`, `labels.json`, `preprocessing.json`, and `manifest.json`. The normal EXE build embeds the model and configuration into `classifier-worker.exe`; install and replace the whole verified package when updating.

Download to a temporary file, verify the trusted manifest and archive hash, safely extract into a new version directory, validate platform and protocol version, run a health or known-input test, then atomically activate the version. Never overwrite a running version. Keep the prior version available for rollback.

## JSON Lines Protocol

The current manifest declares `protocol_version: 1`. Start the worker and read the first `ready` line:

```json
{"ok":true,"action":"ready","labels":["abstract","anime","cars","city","landscape","nature","pets","space"]}
```

Requests and responses are one UTF-8 JSON object per line:

```json
{"action":"health"}
{"action":"classify","image_path":"path/to/image.jpg","top_k":3}
{"action":"shutdown"}
```

Classification response:

```json
{"ok":true,"action":"classify","results":[{"label":"landscape","score":0.982,"index":4},{"label":"nature","score":0.012,"index":5},{"label":"city","score":0.004,"index":3}]}
```

The values above are structural examples. Results are sorted by score, and labels must be read from the actual model package. Errors may contain only `ok` and `error`:

```json
{"ok":false,"error":"..."}
```

The current protocol has no request IDs, cancellation, or asynchronous notifications. A worker processes requests sequentially. Queue requests or use one worker per concurrent caller. `image_path` is a local path on the worker machine, not a URL or a remote client path. Services must validate uploaded files and use controlled paths.

## Python Example

```python
import json
import subprocess
import sys
from pathlib import Path

runtime = Path("runtime/classifier-runtime-v1.0.0-windows-x64").resolve()
image = Path("example.jpg").resolve()
requests = [
    {"action": "health"},
    {"action": "classify", "image_path": str(image), "top_k": 3},
    {"action": "shutdown"},
]
result = subprocess.run(
    [sys.executable, "-X", "utf8", "-u", str(runtime / "classifier-worker.py")],
    input="".join(json.dumps(item, ensure_ascii=True) + "\n" for item in requests),
    capture_output=True,
    encoding="utf-8",
    timeout=60,
    check=True,
)
messages = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
if len(messages) != 4 or messages[0].get("action") != "ready":
    raise RuntimeError("Unexpected worker response sequence")
if any(not message.get("ok") for message in messages):
    raise RuntimeError("Worker request failed")
print(messages[2]["results"])
```

For an EXE, replace the Python command with `[str(runtime / "classifier-worker.exe")]`. Node.js can use `spawn`, C# can use `Process`, and Rust can use `Command` with the same protocol.

## Production Requirements

- Start a fixed, verified entry point with argument arrays; do not build shell commands.
- Keep stdout for protocol and drain stderr separately.
- Buffer stream bytes until a newline; one read is not necessarily one message.
- Set startup and inference timeouts and recover crashed processes deliberately.
- Handle initialization errors, EOF, non-zero exits, invalid JSON, and stale responses after forced termination.
- Limit image size, pixel count, queue length, and worker count. A worker is not a security sandbox.
- Hosted services must add authentication, upload limits, path isolation, rate limits, and retention policies.

## Direct ONNX

Use the package configuration and `classifier_worker.py` as the reference implementation. Convert to RGB, follow `image_size` and `mode`, use bilinear resize, apply the configured mean and standard deviation, convert HWC to NCHW float32, run inference, apply stable softmax, and map indices using `labels.json`. Reimplementations in another language must be compared against the reference worker because matching dimensions alone does not guarantee matching preprocessing.

## Quality Boundary

The current model is a closed-set wallpaper classifier. The worker does not produce `unknown`; integrators should add an application-level review state. Softmax scores are not calibrated accuracy or reliable out-of-distribution detection. Validate thresholds on domain-relevant data and do not use predictions for irreversible operations without review.
