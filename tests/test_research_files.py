"""文件工具的边界测试；不联网、不加载 ML 包。"""
import importlib.util
import json
from pathlib import Path
import stat
import tempfile
import unittest
import zipfile


SPEC = importlib.util.spec_from_file_location("research_files", Path(__file__).resolve().parents[1] / "tools/research_files.py")
rf = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rf)


class ResearchFilesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="research files ")
        self.root = Path(self.temp.name) / "workbench-v2 with spaces"
        self.root.mkdir()
        self.source = Path(self.temp.name) / "colab export.zip"

    def tearDown(self):
        self.temp.cleanup()

    def make_zip(self, extras=None, manifest=None):
        manifest = manifest if manifest is not None else {"status": "smoke_complete_not_research_evidence", "config": {"seed": 42}}
        with zipfile.ZipFile(self.source, "w") as archive:
            archive.writestr("smoke test/manifest.json", json.dumps(manifest))
            for name, value in (extras or {}).items():
                archive.writestr(name, value)

    def test_roundtrip_and_identical_import_is_noop(self):
        self.make_zip({"smoke test/sft/adapter/adapter_model.safetensors": b"fake test weights", "smoke test/output.json": "{}"})
        first = rf.import_run(self.source, workspace=self.root)
        folder = Path(first["path"])
        self.assertEqual((folder / "sft/adapter/adapter_model.safetensors").read_bytes(), b"fake test weights")
        self.assertIn("不是研究结论", (folder / "RESULT.md").read_text())
        before = (folder / "import_receipt.json").read_bytes()
        second = rf.import_run(self.source, workspace=self.root)
        self.assertEqual(second["status"], "already_imported_identical")
        self.assertEqual(before, (folder / "import_receipt.json").read_bytes())

    def test_changed_import_never_overwrites(self):
        self.make_zip()
        folder = Path(rf.import_run(self.source, workspace=self.root)["path"])
        (folder / "manifest.json").write_text("private local edit")
        with self.assertRaises(FileExistsError):
            rf.import_run(self.source, workspace=self.root)
        self.assertEqual((folder / "manifest.json").read_text(), "private local edit")

    def test_traversal_and_duplicate_normalized_names_rejected(self):
        for extras in [{"smoke test/../../outside.txt": "bad"}, {"smoke test/A.json": "{}", "smoke test/a.json": "{}"}, {"smoke test/x\\evil": "bad"}]:
            with self.subTest(extras=extras):
                self.make_zip(extras)
                with self.assertRaises(ValueError):
                    rf.import_run(self.source, workspace=self.root)
        self.assertFalse((self.root / "runs").exists())

    def test_symlink_rejected(self):
        self.make_zip()
        with zipfile.ZipFile(self.source, "a") as archive:
            link = zipfile.ZipInfo("smoke test/link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(link, "/etc/passwd")
        with self.assertRaises(ValueError):
            rf.import_run(self.source, workspace=self.root)

    def test_failed_and_unknown_status_preserved(self):
        self.make_zip(manifest={"status": "failed_partial_outputs_not_results", "failure_type": "RuntimeError"})
        folder = Path(rf.import_run(self.source, workspace=self.root)["path"])
        report = (folder / "RESULT.md").read_text()
        self.assertIn("运行失败", report)
        self.assertIn("RuntimeError", report)
        self.assertIn("未知", report)
        self.assertIn("完成状态未确认", rf.result_markdown({}, None, None))

    def test_recovery_error_type_and_explicit_failure_type_precedence(self):
        recovery_manifest = {"status": "failed_before_training_initialization",
                             "kind": "launcher_only_diagnostics",
                             "error_type": "CalledProcessError", "returncode": 7}
        for explicit, expected in (({}, "CalledProcessError"),
                                   ({"failure_type": "RuntimeError"}, "RuntimeError"),
                                   ({"failure_type": None}, "未知")):
            with self.subTest(explicit=explicit):
                manifest = {**recovery_manifest, **explicit}
                self.make_zip(manifest=manifest)
                folder = Path(rf.import_run(self.source, workspace=self.root)["path"])
                report = (folder / "RESULT.md").read_text()
                self.assertIn(f"| 失败类型 | {expected} |", report)
                self.assertIn("运行失败", report)
                self.assertEqual(json.loads((folder / "manifest.json").read_text()), manifest)

    def test_comparison_numeric_values_and_invalid_json_preserved(self):
        comparison = {"status": "smoke_complete_not_research_evidence", "results": [{"objective": "sft", "before_validation": {"unweighted_nll": 2.4}, "after_validation": {"unweighted_nll": 2.1}}]}
        self.make_zip({"smoke test/comparison.json": json.dumps(comparison)})
        folder = Path(rf.import_run(self.source, workspace=self.root)["path"])
        self.assertIn("2.4 | 2.1", (folder / "RESULT.md").read_text())
        self.make_zip({"smoke test/comparison.json": "not json"})
        folder = Path(rf.import_run(self.source, workspace=self.root)["path"])
        self.assertIn("无法解析", (folder / "RESULT.md").read_text())
        self.assertEqual((folder / "comparison.json").read_text(), "not json")

    def test_budget_and_multiple_runs_rejected(self):
        self.make_zip({"other/manifest.json": "{}"})
        with self.assertRaises(ValueError):
            rf.import_run(self.source, workspace=self.root)
        self.make_zip({"smoke test/data.txt": "1234567890"})
        original_limit = rf.MAX_TOTAL
        try:
            rf.MAX_TOTAL = 1
            with self.assertRaises(ValueError):
                rf.import_run(self.source, workspace=self.root)
        finally:
            rf.MAX_TOTAL = original_limit

    def test_import_and_relative_backup_outputs_stay_in_workspace(self):
        self.make_zip()
        with self.assertRaises(ValueError):
            rf.import_run(self.source, Path("../escaped"), self.root)
        with self.assertRaises(ValueError):
            rf.backup(Path("../escaped.zip"), self.root)

    def test_external_backup_destination_and_symlink_rejection(self):
        (self.root / "notes.md").write_text("private note")
        external = Path(self.temp.name).resolve() / "external drive"
        external.mkdir()
        result = rf.backup(external / "backup.zip", self.root)
        self.assertEqual(result["status"], "backup_created")
        self.assertIn("未上传", result["notice"])
        with self.assertRaises(FileExistsError):
            rf.backup(external / "backup.zip", self.root)
        link = Path(self.temp.name).resolve() / "linked drive"
        link.symlink_to(external, target_is_directory=True)
        with self.assertRaises(ValueError):
            rf.backup(link / "another.zip", self.root)
        self.assertFalse((external / "another.zip").exists())

    def test_result_uses_real_wall_time_gpu_and_memory_keys(self):
        manifest = {"status": "smoke_complete_not_research_evidence", "total_wall_seconds": 100.75,
                    "environment": {"gpu": {"name": "test GPU"}}}
        comparison = {"results": [{"objective": "sft", "training_seconds": 14.25,
                                   "branch_wall_seconds": 45.5, "peak_cuda_allocated_bytes": 12345678}]}
        self.make_zip({"smoke test/comparison.json": json.dumps(comparison)}, manifest=manifest)
        folder = Path(rf.import_run(self.source, workspace=self.root)["path"])
        report = (folder / "RESULT.md").read_text()
        self.assertIn("| GPU | test GPU |", report)
        self.assertIn("| 总墙钟秒数 | 100.75 |", report)
        self.assertIn("| 14.25 | 45.5 | 12345678 |", report)

    def test_backup_exclusions_roundtrip_and_no_overwrite(self):
        (self.root / "notes").mkdir()
        (self.root / "notes/a.md").write_text("my private note")
        for directory in (".venv-zotero", "runs/test/adapter", "cache"):
            (self.root / directory).mkdir(parents=True)
            (self.root / directory / "secret.json").write_text("not included")
        (self.root / "old.zip").write_bytes(b"not included")
        result = rf.backup(Path("backups/local.zip"), self.root)
        with zipfile.ZipFile(result["path"]) as archive:
            self.assertEqual(archive.read("workbench-v2/notes/a.md"), b"my private note")
            self.assertEqual(set(archive.namelist()), {"workbench-v2/notes/a.md", "BACKUP_MANIFEST.json"})
            self.assertTrue(json.loads(archive.read("BACKUP_MANIFEST.json"))["excluded"])
        with self.assertRaises(FileExistsError):
            rf.backup(Path("backups/local.zip"), self.root)


if __name__ == "__main__":
    unittest.main()
