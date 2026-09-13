# Security Policy

[English](SECURITY.md) | [简体中文](SECURITY.zh-CN.md)

Only the latest runtime and current default branch are supported for security fixes.

Do not publish exploitable runtime issues, malicious packages, tampered model archives, credentials, or private images in public issues. Use GitHub Security Advisories or contact the maintainers privately with the affected version, reproduction steps, impact, and minimal evidence.

Runtime integrators should use HTTPS, verify archive hashes and preferably signatures, never dynamically execute downloaded Python code, avoid pickle as a model distribution format, restrict worker operations, and validate image paths. A worker process provides fault isolation, not a security sandbox.
