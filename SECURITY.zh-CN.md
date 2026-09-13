# 安全政策

[English](SECURITY.md) | [简体中文](SECURITY.zh-CN.md)

仅支持最新 runtime 和当前默认分支的安全修复。

不要在公开 Issue 中发布可利用的 runtime 问题、恶意包、被篡改的模型、凭据或个人图片。请使用 GitHub Security Advisories 或私下联系维护者，并提供受影响版本、复现步骤、影响范围和最小证据。

接入方应使用 HTTPS，校验压缩包哈希并优先校验签名，不执行下载包中的动态 Python 代码，不使用 pickle 作为模型发布格式，限制 worker 操作并验证图片路径。worker 只提供故障隔离，不是安全沙箱。
