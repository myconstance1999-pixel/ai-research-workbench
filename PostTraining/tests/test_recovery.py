"""检查真正的失败/超时归档路径；不下载模型、不调用GPU。"""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile

from experiments.recovery import archive_run, run_logged_process, snapshot_sources, unique_run_dir


class RecoveryTests(unittest.TestCase):
    def test_failure_preserves_console_and_status(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            run = unique_run_dir(root / "runs")
            with self.assertRaises(subprocess.CalledProcessError):
                run_logged_process([sys.executable, "-u", "-c", "print('failure evidence'); raise RuntimeError('expected')"],
                                   cwd=root, run_dir=run, timeout_seconds=5)
            record = json.loads((run / "launcher.json").read_text())
            self.assertEqual(record["status"], "failed_partial_outputs")
            self.assertIn("failure evidence", (run / "console.log").read_text())
            manifest = json.loads((run / "manifest.json").read_text())
            self.assertEqual(manifest["kind"], "launcher_only_diagnostics")
            self.assertEqual(manifest["status"], "failed_before_training_initialization")
            self.assertIsNone(manifest["model_commit"])
            with zipfile.ZipFile(run.parent / f"{run.name}.zip") as archive:
                self.assertIn(f"{run.name}/manifest.json", archive.namelist())
                self.assertIn(f"{run.name}/launcher.json", archive.namelist())
                self.assertIn(f"{run.name}/console.log", archive.namelist())

    def test_timeout_stops_and_exports_partial_output(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            run = unique_run_dir(root / "runs")
            started = time.monotonic()
            with self.assertRaises(subprocess.TimeoutExpired):
                run_logged_process([sys.executable, "-u", "-c", "import time; print('before timeout'); time.sleep(30)"],
                                   cwd=root, run_dir=run, timeout_seconds=0.2)
            self.assertLess(time.monotonic() - started, 8)
            record = json.loads((run / "launcher.json").read_text())
            self.assertEqual(record["status"], "timed_out_partial_outputs")
            self.assertIsNotNone(record["returncode"])
            self.assertIn("before timeout", (run / "console.log").read_text())
            self.assertTrue((run.parent / f"{run.name}.zip").is_file())

    def test_source_and_recovery_copy_exclude_model_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "experiments").mkdir()
            (root / "experiments" / "sample.py").write_text("print('source')\n")
            (root / "experiments" / "weights.safetensors").write_bytes(b"not a real model")
            (root / "notebooks").mkdir()
            (root / "notebooks" / "sample.ipynb").write_text("{}")
            run = unique_run_dir(root / "runs")
            run.mkdir(parents=True)
            snapshot_sources(root, run)
            recovery = root / "mounted_recovery"
            recovery.mkdir()
            archive = archive_run(run, recovery)
            self.assertEqual(archive.read_bytes(), (recovery / archive.name).read_bytes())
            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
                self.assertIn(f"{run.name}/source_snapshot/experiments/sample.py", names)
                self.assertFalse(any("weights.safetensors" in name for name in names))


if __name__ == "__main__":
    unittest.main()
