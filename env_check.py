#!/usr/bin/env python3
"""检查图片分类训练所需的 Python 依赖和硬件。

直接运行 ``python env_check.py``，退出码为 0 表示核心依赖可用，1 表示
存在缺失依赖。脚本不修改环境，便于在新机器上快速诊断。
"""

from __future__ import annotations

import importlib
import platform
import sys
from dataclasses import dataclass


@dataclass
class ModuleStatus:
    name: str
    installed: bool
    version: str | None = None
    error: str | None = None


def module_status(name: str) -> ModuleStatus:
    try:
        module = importlib.import_module(name)
        return ModuleStatus(name, True, str(getattr(module, "__version__", "unknown")))
    except Exception as exc:  # import errors can originate in compiled extensions
        return ModuleStatus(name, False, error=f"{type(exc).__name__}: {exc}")


def main() -> int:
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"OS: {platform.platform()}")

    # torch, numpy, PIL and requests are required by the lightweight training
    # and download scripts.  torchvision is optional: the trainer can fall
    # back to a small CNN when pretrained torchvision models are unavailable.
    core = [module_status(name) for name in ("torch", "numpy", "PIL", "requests")]
    optional = [module_status(name) for name in ("torchvision", "bs4", "sklearn")]
    for title, statuses in (("Core modules", core), ("Optional modules", optional)):
        print(f"\n{title}:")
        for status in statuses:
            if status.installed:
                print(f"  [OK]   {status.name} {status.version}")
            else:
                print(f"  [MISS] {status.name}: {status.error}")

    try:
        torch = importlib.import_module("torch")
        cuda = bool(torch.cuda.is_available())
        print(f"\nPyTorch device: {'cuda' if cuda else 'cpu'}")
        if cuda:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("GPU unavailable; CPU training will be used.")
    except Exception:
        print("\nPyTorch device: unavailable (install torch first)")

    missing = [status.name for status in core if not status.installed]
    if missing:
        print("\nInstall missing core modules (choose a Python version supported by PyTorch):")
        print("  python -m pip install torch numpy pillow requests")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
