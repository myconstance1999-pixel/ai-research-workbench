"""记录安全的环境元数据；不打印环境变量/凭据，不下载模型。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib import metadata, util
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys


def environment_report():
    packages = {}
    for package in ("numpy", "torch", "transformers", "peft", "accelerate", "huggingface-hub", "tokenizers", "safetensors"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    report = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "machine": platform.machine(), "working_disk_free_bytes": shutil.disk_usage(Path.cwd()).free,
        "mps_built": False, "mps_available": False,
        "packages": packages, "cuda_available": False, "gpu": None,
    }
    if util.find_spec("torch"):
        try:
            import torch
            report["torch_cuda_build"] = torch.version.cuda
            mps = getattr(torch.backends, "mps", None)
            report["mps_built"] = bool(mps and mps.is_built())
            report["mps_available"] = bool(mps and mps.is_available())
            report["cuda_available"] = torch.cuda.is_available()
            if report["cuda_available"]:
                props = torch.cuda.get_device_properties(0)
                report["gpu"] = {"name": props.name, "total_memory_bytes": props.total_memory,
                                 "bf16_supported": torch.cuda.is_bf16_supported()}
        except Exception as exc:
            report["torch_probe_error"] = type(exc).__name__
    if shutil.which("nvidia-smi"):
        try:
            p = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=10, check=False)
            report["nvidia_smi"] = p.stdout.strip() if p.returncode == 0 else "query_failed"
        except subprocess.TimeoutExpired:
            report["nvidia_smi"] = "query_timed_out"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = environment_report()
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
