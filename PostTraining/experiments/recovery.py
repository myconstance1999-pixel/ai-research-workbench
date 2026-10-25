"""短实验的日志与归档；不保存基础模型，不自动登录或续训。"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import uuid
import zipfile


def unique_run_dir(root=Path("runs")):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(root) / f"smoke_{stamp}_{uuid.uuid4().hex[:6]}"


def snapshot_sources(root, run_dir):
    """保留可重跑的实验源码、依赖、测试与notebook，不扫描个人目录。"""
    root, run_dir = Path(root).resolve(), Path(run_dir)
    destination = run_dir / "source_snapshot"
    for folder, suffixes in (("experiments", {".py", ".txt"}),
                             ("tests", {".py"}), ("notebooks", {".ipynb"})):
        for source in sorted((root / folder).rglob("*")):
            if source.is_file() and not source.is_symlink() and source.suffix in suffixes:
                target = destination / source.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    for name in ("README.md", "AGENTS.md"):
        if (root / name).is_file():
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / name, destination / name)


def archive_run(run_dir, recovery_dir=None):
    """同一个run的恢复包原子替换；已挂载目录由调用者显式提供。"""
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        return None
    archive = run_dir.parent / f"{run_dir.name}.zip"
    handle, temporary = tempfile.mkstemp(prefix=f".{run_dir.name}.", suffix=".zip", dir=run_dir.parent)
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for source in sorted(run_dir.rglob("*")):
                if source.is_file() and not source.is_symlink():
                    zf.write(source, Path(run_dir.name) / source.relative_to(run_dir))
            # 训练进程运行时，父进程先把console写到同级文件，避免预建run目录。
            live_console = run_dir.parent / f"{run_dir.name}.console.log"
            if live_console.is_file() and not (run_dir / "console.log").exists():
                zf.write(live_console, Path(run_dir.name) / "console.log")
        os.replace(temporary, archive)
    finally:
        Path(temporary).unlink(missing_ok=True)
    if recovery_dir is not None:
        recovery_dir = Path(recovery_dir)
        if not recovery_dir.is_dir():
            raise FileNotFoundError("RECOVERY_DIR必须是已存在、已挂载的目录；本脚本不会挂载账号。")
        handle, temporary = tempfile.mkstemp(prefix=f".{archive.stem}.", suffix=".zip", dir=recovery_dir)
        os.close(handle)
        try:
            shutil.copy2(archive, temporary)
            os.replace(temporary, recovery_dir / archive.name)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return archive


def _stop_process(process):
    try:
        if os.name == "posix":
            # 即使父进程已退出，也清理同组可能仍持有pipe/GPU的子进程。
            os.killpg(process.pid, signal.SIGTERM)
        elif process.poll() is None:
            process.terminate()
        if process.poll() is None:
            process.wait(timeout=5)
    except ProcessLookupError:
        if process.poll() is None:
            process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if process.poll() is None:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait(timeout=5)


def run_logged_process(command, *, cwd, run_dir, timeout_seconds=1200, recovery_dir=None):
    """执行一个完整pair；超时/中断时停止进程组，并归档已有文件。"""
    run_dir = Path(run_dir).resolve()
    if run_dir.exists():
        raise FileExistsError("已有run目录不可覆盖；重新生成RUN_DIR。")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds必须为正数")
    if recovery_dir is not None and not Path(recovery_dir).is_dir():
        raise FileNotFoundError("先指定已存在、已挂载的RECOVERY_DIR，或保留None。")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    console_path = run_dir.parent / f"{run_dir.name}.console.log"
    started = time.perf_counter()
    record = {"started_utc": datetime.now(timezone.utc).isoformat(),
              "timeout_seconds": timeout_seconds, "status": "starting", "returncode": None}
    process = reader = None
    try:
        with console_path.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1,
                                       start_new_session=(os.name == "posix"))

            def relay():
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)

            reader = threading.Thread(target=relay, daemon=True)
            reader.start()
            try:
                record["status"] = "running"
                record["returncode"] = process.wait(timeout=timeout_seconds)
                if record["returncode"] != 0:
                    record["status"] = "failed_partial_outputs"
                    raise subprocess.CalledProcessError(record["returncode"], command)
                record["status"] = "process_completed"
            except subprocess.TimeoutExpired:
                record["status"] = "timed_out_partial_outputs"
                raise
            except KeyboardInterrupt:
                record["status"] = "interrupted_partial_outputs"
                raise
            finally:
                _stop_process(process)
                reader.join(timeout=5)
                process.stdout.close()
                record["returncode"] = process.returncode
    except BaseException as exc:
        record["error_type"] = type(exc).__name__
        if record["status"] in ("starting", "running"):
            record["status"] = "failed_partial_outputs"
        raise
    finally:
        record["total_wall_seconds"] = time.perf_counter() - started
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        run_dir.mkdir(parents=True, exist_ok=True)
        if console_path.exists():
            shutil.move(console_path, run_dir / "console.log")
        (run_dir / "launcher.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if not (run_dir / "source_snapshot").exists():
            snapshot_sources(cwd, run_dir)
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.exists():
            status = record["status"]
            if status == "failed_partial_outputs":
                status = "failed_before_training_initialization"
            elif status == "process_completed":
                status = "process_completed_without_training_manifest"
            fallback = {
                "created_utc": record["started_utc"], "finished_utc": record["finished_utc"],
                "status": status, "kind": "launcher_only_diagnostics",
                "purpose": "No training manifest was created; GPU/model/results are unknown",
                "config": {"command": [str(part) for part in command], "timeout_seconds": timeout_seconds},
                "environment": {"gpu": None, "cuda_available": None}, "model_commit": None,
                "error_type": record.get("error_type"), "returncode": record["returncode"],
                "total_wall_seconds": record["total_wall_seconds"],
            }
            manifest_path.write_text(json.dumps(fallback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif record["status"] != "process_completed":
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["status"] = record["status"]
            manifest["finished_utc"] = record["finished_utc"]
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            archive = archive_run(run_dir, recovery_dir)
            print(f"已归档（{record['status']}）：{archive}", flush=True)
            if recovery_dir is None:
                print("归档仍在当前运行时，请下载到Mac；运行时消失后无法恢复。", flush=True)
            else:
                print(f"恢复包已复制到指定目录：{Path(recovery_dir) / archive.name}", flush=True)
        except Exception as exc:
            print(f"归档或复制失败：{type(exc).__name__}；保留的目录为 {run_dir}", flush=True)
    return record
