#!/usr/bin/env python3
"""离线导入 Colab 单次运行证据；私有轻量备份可保存到显式指定的外部位置。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import unicodedata
import zipfile


WORKSPACE = Path(__file__).resolve().parent.parent
MAX_FILES = 4000
MAX_TOTAL = 512 * 2**20
MAX_FILE = 256 * 2**20
MAX_ARCHIVE = 512 * 2**20
MAX_JSON = 4 * 2**20
RESERVED = {"result.md", "import_receipt.json"}
EXCLUDED_DIRS = {
    "__pycache__", "node_modules", "venv", "env", "models", "model", "checkpoints",
    "checkpoint", "adapter", "adapters", "tokenizer", "tokenizers", "cache", "caches",
    "backups", "archives", "dist", "build",
}
BACKUP_SUFFIXES = {
    ".md", ".txt", ".json", ".jsonl", ".csv", ".tsv", ".ris", ".bib", ".py",
    ".ipynb", ".html", ".css", ".js", ".sh", ".toml", ".yaml", ".yml", ".log",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_output(path: Path, workspace: Path) -> Path:
    """输出必须在工具所在工作台，且路径不得经过已有符号链接。"""
    workspace = workspace.resolve()
    candidate = path if path.is_absolute() else workspace / path
    # resolve 之前检查 links，避免用户目录中的 link 被当作普通输出目录。
    absolute = Path(os.path.abspath(candidate))
    try:
        rel = absolute.relative_to(workspace)
    except ValueError as exc:
        raise ValueError("输出路径必须位于当前 workbench-v2 工作台内") from exc
    if not rel.parts:
        raise ValueError("请使用工作台内的子目录或文件，不可覆盖工作台根目录")
    current = workspace
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("输出路径不能经过符号链接")
    return absolute


def backup_output(path: Path, workspace: Path) -> Path:
    """相对路径限定工作台；绝对路径可指向外部盘，仅逐级检查输出祖先。"""
    if not path.is_absolute():
        return local_output(path, workspace)
    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError("备份输出路径不能经过符号链接")
    return absolute


def safe_member(info: zipfile.ZipInfo) -> tuple[str, ...]:
    name = info.filename
    if "\x00" in info.orig_filename or "\\" in name or ":" in name or name.startswith("/"):
        raise ValueError(f"ZIP 含不安全路径：{name!r}")
    raw = name[:-1] if name.endswith("/") else name
    parts = tuple(raw.split("/"))
    if not raw or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"ZIP 含不安全路径：{name!r}")
    # macOS 常见文件系统不区分大小写，且会归一化 Unicode。
    if any(part.endswith((" ", ".")) for part in parts):
        raise ValueError(f"ZIP 路径以空格或点结尾：{name!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError(f"ZIP 不接受链接或特殊文件：{name!r}")
    if info.flag_bits & 1:
        raise ValueError("不接受加密 ZIP")
    return parts


def json_object(data: bytes) -> dict:
    def reject_constant(value):
        raise ValueError(f"JSON 非有限常数：{value}")

    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON 重复键：{key}")
            result[key] = value
        return result

    if len(data) > MAX_JSON:
        raise ValueError("JSON 元数据超过 4 MiB")
    value = json.loads(data, parse_constant=reject_constant, object_pairs_hook=unique_pairs)
    if not isinstance(value, dict):
        raise ValueError("JSON 元数据必须是对象")
    return value


def inspect_archive(archive: zipfile.ZipFile) -> tuple[dict[str, zipfile.ZipInfo], dict]:
    infos = archive.infolist()
    if len(infos) > MAX_FILES:
        raise ValueError("ZIP 条目过多")
    normalized = set()
    entries = []
    total = 0
    for info in infos:
        parts = safe_member(info)
        normalized_name = "/".join(unicodedata.normalize("NFC", p).casefold() for p in parts)
        if normalized_name in normalized:
            raise ValueError("ZIP 有重复或在 macOS 上冲突的归一化路径")
        normalized.add(normalized_name)
        if info.file_size > MAX_FILE or info.file_size < 0:
            raise ValueError("ZIP 单文件超过 256 MiB")
        total += info.file_size
        if total > MAX_TOTAL:
            raise ValueError("ZIP 解压总量超过 512 MiB")
        if info.file_size > 1024 * 1024 and info.file_size > max(info.compress_size, 1) * 1000:
            raise ValueError("ZIP 压缩比异常")
        entries.append((info, parts))
    manifests = [(i, p) for i, p in entries if not i.is_dir() and p[-1] == "manifest.json"]
    if len(manifests) != 1:
        raise ValueError("只接受包含一个 manifest.json 的单次运行 ZIP")
    manifest_info, manifest_path = manifests[0]
    if manifest_info.file_size > MAX_JSON:
        raise ValueError("manifest.json 超过 4 MiB")
    root = manifest_path[:-1]
    files = {}
    for info, parts in entries:
        if info.is_dir() and parts == root[:len(parts)]:
            continue  # ZIP 可包含 manifest 根目录的父目录条目。
        if parts[:len(root)] != root:
            raise ValueError("ZIP 含运行根目录之外的文件")
        relative = parts[len(root):]
        if not relative:
            raise ValueError("ZIP 的运行根目录与文件冲突")
        if not info.is_dir():
            name = "/".join(relative)
            if name.casefold() in RESERVED:
                raise ValueError("ZIP 含导入工具保留文件名")
            files[name] = info
    # a 文件与 a/b 文件无法同时安全落地；不留到写出一半才发现。
    normalized_files = {unicodedata.normalize("NFC", p).casefold() for p in files}
    for name in normalized_files:
        if any(str(parent) in normalized_files for parent in PurePosixPath(name).parents if str(parent) != "."):
            raise ValueError("ZIP 含文件与目录路径冲突")
    manifest = json_object(archive.read(manifest_info))
    return files, manifest


def shown(value) -> str:
    if value is None:
        return "未知"
    text = str(value).replace("|", "\\|").replace("`", "'")
    return " ".join(text.split())[:300] or "未知"


def nested(value: dict, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def result_markdown(manifest: dict, comparison: dict | None, comparison_error: str | None) -> str:
    status = manifest.get("status")
    if status == "smoke_complete_not_research_evidence":
        interpretation = "导出清单标记 smoke 完成；这是链路联调记录，不是研究结论。本地工具未重跑训练或验证声明。"
    elif isinstance(status, str) and "fail" in status.lower():
        interpretation = "运行失败；已保留部分输出，它们不能当作完整对照结果。"
    else:
        interpretation = "完成状态未确认；不能把已有文件或部分指标解释成训练成功。"
    lines = ["# 导入运行记录", "", interpretation, "", "| 项目 | 导出记录 |", "|---|---|",
             f"| manifest 状态 | {shown(status)} |",
             f"| comparison 状态 | {shown(nested(comparison, 'status'))} |",
             f"| 失败类型 | {shown(manifest.get('failure_type', manifest.get('error_type')))} |",
             f"| 模型 | {shown(nested(manifest, 'config', 'model'))} |",
             f"| 模型 commit | {shown(manifest.get('model_commit'))} |",
             f"| 数据类型 | {shown(nested(manifest, 'dataset', 'kind'))} |",
             f"| 随机种子 | {shown(nested(manifest, 'config', 'seed'))} |",
             f"| 每分支步数 | {shown(nested(manifest, 'config', 'max_steps'))} |",
             f"| GPU | {shown(nested(manifest, 'environment', 'gpu', 'name'))} |",
             f"| 总墙钟秒数 | {shown(manifest.get('total_wall_seconds'))} |", ""]
    if comparison_error:
        lines.extend([f"comparison.json：{comparison_error}；原文件已保留。", ""])
    rows = comparison.get("results", []) if isinstance(comparison, dict) else []
    if not isinstance(rows, list):
        rows = []
    if rows:
        lines.extend(["| 分支 | 状态 | 前 NLL | 后 NLL | 训练秒数 | 分支总秒数 | 峰值 CUDA 已分配字节 |", "|---|---|---|---|---|---|---|"])
        for row in rows:
            if not isinstance(row, dict):
                continue
            values = [row.get("objective"), row.get("status"), nested(row, "before_validation", "unweighted_nll"),
                      nested(row, "after_validation", "unweighted_nll"), row.get("training_seconds"),
                      row.get("branch_wall_seconds"), row.get("peak_cuda_allocated_bytes")]
            lines.append("| " + " | ".join(shown(v) for v in values) + " |")
        lines.append("")
    lines.extend(["数字仅转录自导出文件；缺失值保持未知。不要比较 SFT 与 DFT 的原始优化 loss。", "",
                  "完整配置、数据、输出及训练产物保留在本目录。原文件清单及内容哈希见 `import_receipt.json`。", ""])
    return "\n".join(lines)


def import_run(source: Path, destination: Path = Path("runs"), workspace: Path = WORKSPACE) -> dict:
    source = source.resolve(strict=True)
    if not source.is_file() or source.stat().st_size > MAX_ARCHIVE:
        raise ValueError("输入必须是至多 512 MiB 的 ZIP 文件")
    destination = local_output(destination, workspace)
    with zipfile.ZipFile(source) as archive:
        files, manifest = inspect_archive(archive)
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".import-", dir=destination) as staging_name:
            staging = Path(staging_name)
            hashes = {}
            for name, info in sorted(files.items()):
                target = staging.joinpath(*PurePosixPath(name).parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                actual_size = 0
                digest = hashlib.sha256()
                with archive.open(info) as original, target.open("xb") as output:
                    while block := original.read(1024 * 1024):
                        actual_size += len(block)
                        if actual_size > info.file_size or actual_size > MAX_FILE:
                            raise ValueError("ZIP 实际数据量超过声明")
                        digest.update(block)
                        output.write(block)
                if actual_size != info.file_size:
                    raise ValueError("ZIP 文件大小不匹配")
                hashes[name] = digest.hexdigest()
            fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
            final = destination / f"run-{fingerprint[:16]}"
            if final.exists() or final.is_symlink():
                if final.is_symlink() or not final.is_dir():
                    raise FileExistsError("导入目标已存在，拒绝覆盖")
                existing_entries = list(final.rglob("*"))
                if any(p.is_symlink() for p in existing_entries):
                    raise FileExistsError("同名运行目录含符号链接，拒绝读取或覆盖")
                actual = {p.relative_to(final).as_posix(): digest_file(p) for p in existing_entries
                          if p.is_file() and p.relative_to(final).as_posix() not in {"RESULT.md", "import_receipt.json"}}
                if actual != hashes:
                    raise FileExistsError("同名运行目录已有改动，拒绝覆盖")
                return {"status": "already_imported_identical", "path": str(final), "files": len(hashes)}
            comparison, comparison_error = None, None
            if (staging / "comparison.json").is_file():
                try:
                    if (staging / "comparison.json").stat().st_size > MAX_JSON:
                        raise ValueError("comparison.json 超过 4 MiB")
                    comparison = json_object((staging / "comparison.json").read_bytes())
                except (ValueError, UnicodeError) as exc:
                    comparison_error = f"无法解析（{type(exc).__name__}）"
            else:
                comparison_error = "未提供"
            (staging / "RESULT.md").write_text(result_markdown(manifest, comparison, comparison_error), encoding="utf-8")
            receipt = {"imported_utc": utc_now(), "source_archive_sha256": digest_file(source),
                       "content_sha256": fingerprint, "files_sha256": hashes,
                       "verification": "archive_integrity_only_not_training_reproduction"}
            (staging / "import_receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            # mkdir 原子占用新名称；即使目标已是空目录也不会覆盖。
            final.mkdir()
            try:
                for child in staging.iterdir():
                    shutil.move(str(child), str(final / child.name))
            except BaseException:
                shutil.rmtree(final)
                raise
    return {"status": "imported", "path": str(final), "files": len(hashes), "manifest_status": manifest.get("status")}


def backup(output: Path, workspace: Path = WORKSPACE) -> dict:
    workspace = workspace.resolve()
    output = backup_output(output, workspace)
    if output.suffix.lower() != ".zip":
        raise ValueError("备份输出应为 .zip")
    if output.exists():
        raise FileExistsError("备份文件已存在，拒绝覆盖")
    selected, skipped = [], []
    for directory, dirs, names in os.walk(workspace, followlinks=False):
        parent = Path(directory)
        kept = []
        for name in sorted(dirs):
            path = parent / name
            if name.startswith(".") or name.lower() in EXCLUDED_DIRS or path.is_symlink():
                skipped.append({"path": path.relative_to(workspace).as_posix() + "/", "reason": "hidden_cache_environment_or_model_directory"})
            else:
                kept.append(name)
        dirs[:] = kept
        for name in sorted(names):
            path = parent / name
            rel = path.relative_to(workspace).as_posix()
            if path == output or path.is_symlink() or name.startswith(".") or path.suffix.lower() not in BACKUP_SUFFIXES:
                skipped.append({"path": rel, "reason": "archive_binary_hidden_link_or_unsupported_type"})
                continue
            if path.stat().st_size > 32 * 2**20:
                skipped.append({"path": rel, "reason": "larger_than_32_MiB"})
                continue
            selected.append((path, rel))
    if sum(p.stat().st_size for p, _ in selected) > 128 * 2**20:
        raise ValueError("轻量备份超过 128 MiB；请缩小工作台中保留的日志或 notebook 输出")
    output.parent.mkdir(parents=True, exist_ok=True)
    description = {
        "created_utc": utc_now(), "kind": "private_lightweight_backup",
        "scope": "当前工作台的笔记、状态、文献索引、源码、notebook 与导入证据；不扫描工作台外部",
        "not_included": ["原有 Zotero 库与 .venv-zotero", "模型、适配器、tokenizer、缓存、虚拟环境", "PDF 与其他二进制文件、已有压缩包、隐藏文件", "单个超过 32 MiB 的文件"],
        "restore_limits": "此备份可恢复文本记录，不能单独恢复训练权重或 Zotero 数据。",
        "files_sha256": {rel: digest_file(path) for path, rel in selected}, "excluded": skipped,
    }
    try:
        # x 模式保证重复调用不能覆盖；不联网、不上传。
        with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            output.chmod(0o600)
            for path, rel in selected:
                archive.write(path, f"workbench-v2/{rel}")
            archive.writestr("BACKUP_MANIFEST.json", json.dumps(description, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError:
        raise
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    return {"status": "backup_created", "path": str(output), "files": len(selected),
            "excluded_items": len(skipped), "notice": "私有轻量备份；不含模型、适配器、Zotero 数据及缓存；未上传。详见 BACKUP_MANIFEST.json。"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    importer = sub.add_parser("import-run", help="导入带唯一 manifest.json 的单次 Colab 运行 ZIP")
    importer.add_argument("path", type=Path)
    importer.add_argument("--dest", type=Path, default=Path("runs"), help="工作台内路径，默认 runs；与启动目录无关")
    archiver = sub.add_parser("backup", help="制作不含权重/缓存/环境的私有 ZIP")
    archiver.add_argument("--output", type=Path, required=True,
                          help="新 .zip：相对路径位于工作台，显式绝对路径可保存至外部盘；不上传")
    args = parser.parse_args()
    try:
        result = import_run(args.path, args.dest) if args.command == "import-run" else backup(args.output)
    except (OSError, ValueError, zipfile.BadZipFile, RuntimeError) as exc:
        parser.exit(1, f"未完成：{exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
