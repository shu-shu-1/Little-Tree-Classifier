# 通用接入指南

[English](INTEGRATION.md) | [简体中文](INTEGRATION.zh-CN.md)

Little Tree Classifier 是小树壁纸的附属模型项目，但模型和 runtime 也可以被其他桌面应用、CLI 工具和后端服务独立使用。不要求安装小树壁纸。任何可以管理子进程和标准流的语言都可以接入。

## 必须注明模型来源

使用、集成、提供托管推理或再分发模型时必须遵守[模型使用与署名条款](MODEL_LICENSE.md)，微调、量化和转换格式后的模型同样如此。

- 桌面或 Web 应用：在关于页、模型设置或第三方声明中展示。
- CLI：在帮助信息或随附文档中展示。
- 托管 API：在用户可访问的服务文档中展示。
- 再分发：随模型保留条款、来源、准确版本和获取地址。

```text
图片分类模型：Little Tree Classifier
来源：Little Tree Studio，小树壁纸附属图片分类项目
模型版本：<实际版本>
模型地址：<实际项目或 Release URL>
修改说明：<无修改，或说明变更>
```

署名不代表项目背书，也不替代第三方许可证义务。

## 接入方式

| 方式 | 要求 | 适用场景 |
| --- | --- | --- |
| 独立 EXE worker | 支持子进程和管道 | 不想引入 Python 或 ONNX 依赖的桌面应用 |
| 源码 worker | Python、ONNX Runtime、NumPy、Pillow | 开发、脚本和内部服务 |
| 直接 ONNX | 对应语言的 ONNX Runtime 和一致预处理 | 已有推理基础设施的项目 |

当前构建目标为 Windows x64，其他平台需要自行构建和验证。

## 包与安装

源码测试包：

```powershell
python build_runtime.py --checkpoint checkpoints\best.pt --version 1.0.0 --skip-pyinstaller --debug-dir
```

源码包包含 `classifier-worker.py`、`model.onnx`、`labels.json`、`preprocessing.json` 和 `manifest.json`。常规 EXE 构建会将模型和配置嵌入 `classifier-worker.exe`，更新时替换整个已验证的版本包。

下载到临时文件，验证可信清单和压缩包哈希，安全解压到新版本目录，检查平台和协议版本，执行健康检查或已知输入测试后再原子切换。不要覆盖正在运行的版本，失败时保留旧版本。

## JSON Lines 协议

当前 manifest 的 `protocol_version` 为 1。启动 worker 后先读取 `ready`：

```json
{"ok":true,"action":"ready","labels":["abstract","anime","cars","city","landscape","nature","pets","space"]}
```

每个请求和响应都是一行 UTF-8 JSON：

```json
{"action":"health"}
{"action":"classify","image_path":"path/to/image.jpg","top_k":3}
{"action":"shutdown"}
```

分类响应：

```json
{"ok":true,"action":"classify","results":[{"label":"landscape","score":0.982,"index":4},{"label":"nature","score":0.012,"index":5},{"label":"city","score":0.004,"index":3}]}
```

数值仅为结构示例。结果按分数降序排列，标签必须读取实际模型包。错误可能只包含 `ok` 和 `error`：

```json
{"ok":false,"error":"..."}
```

当前协议没有请求 ID、取消或异步通知。worker 顺序处理请求，多调用者应排队或使用多个 worker。`image_path` 是 worker 所在机器的本地路径，不是 URL 或远端客户端路径。服务端必须验证上传文件并使用受控路径。

## Python 示例

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
    raise RuntimeError("Worker 响应顺序异常")
if any(not message.get("ok") for message in messages):
    raise RuntimeError("Worker 请求失败")
print(messages[2]["results"])
```

EXE 接入时将启动命令替换为 `[str(runtime / "classifier-worker.exe")]`。Node.js 使用 `spawn`，C# 使用 `Process`，Rust 使用 `Command` 即可实现同一协议。

## 生产要求

- 使用参数数组启动固定且已校验的入口，不拼接 shell 命令。
- stdout 只解析协议，stderr 单独排空或记录日志。
- 按字节流缓存到换行，不能假设一次读取就是一条消息。
- 分别设置启动和推理超时，处理崩溃、EOF、非零退出和无效 JSON。
- 限制图片大小、像素数量、队列长度和 worker 数量。worker 不是安全沙箱。
- 托管服务自行实现认证、上传限制、路径隔离、限流和数据保留。

## 直接使用 ONNX

请以 `classifier_worker.py` 和包内配置为参考：转 RGB，读取 `image_size` 和 `mode`，按配置执行缩放或中心裁剪、float32 归一化、HWC 到 NCHW 转换，然后执行 softmax 并根据 `labels.json` 映射。其他语言重新实现时应与参考 worker 对比，尺寸一致不代表预处理一致。

## 质量边界

当前模型是闭集壁纸分类器，worker 不会主动输出 `unknown`。接入方应增加未知或待确认状态。softmax 分数不是校准后的准确率，也不是可靠的分布外检测器。不要在未经审核时使用预测结果执行不可逆操作。
