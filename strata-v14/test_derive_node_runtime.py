#!/usr/bin/env python3
"""Unit tests for derive_node_runtime FP8 GEMM prefill config installation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import derive_node_runtime as dnr  # noqa: E402


class TestDeriveNodeRuntimeFP8Configs(unittest.TestCase):
    def test_install_real_dot13_archive_and_preserve_symlink_source(self) -> None:
        tar_path = HERE / "dot13-fp8-configs.tar.gz"
        self.assertTrue(tar_path.is_file(), f"Missing {tar_path}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            base_cfg = tmp_path / "base_source_configs"
            base_cfg.mkdir(parents=True)
            (base_cfg / "existing.json").write_text('{"1": {}}\n', encoding="utf-8")

            rt = tmp_path / "runtime-ours"
            gemm_dir = rt / "sglang/kernels/ops/gemm"
            gemm_dir.mkdir(parents=True)
            (gemm_dir / "configs").symlink_to(base_cfg)

            receipt = dnr.install_fp8_gemm_configs(rt, tar_path)
            self.assertEqual(receipt["count"], 10)
            self.assertFalse((gemm_dir / "configs").is_symlink())
            self.assertTrue((gemm_dir / "configs/existing.json").is_file())
            # Base source must remain untouched!
            self.assertEqual(sorted(p.name for p in base_cfg.iterdir()), ["existing.json"])

            installed_pro5000 = sorted(
                p.name for p in (gemm_dir / "configs").glob("*NVIDIA_RTX_PRO_5000_72GB_Blackwell*.json")
            )
            self.assertEqual(len(installed_pro5000), 10)
            for name in installed_pro5000:
                payload = json.loads((gemm_dir / "configs" / name).read_text(encoding="utf-8"))
                self.assertEqual(sorted(payload.keys()), ["4095", "4096", "4097"])

    def test_install_fp8_configs_only_cli_updates_metadata(self) -> None:
        tar_path = HERE / "dot13-fp8-configs.tar.gz"
        with tempfile.TemporaryDirectory() as tmp:
            rt = Path(tmp) / "runtime-ours"
            (rt / "sglang/kernels/ops/gemm/configs").mkdir(parents=True)
            meta_path = rt / "orvix-trinode-runtime.json"
            meta_path.write_text(
                json.dumps({"schema": "orvix-trinode-t1-ours-runtime/1", "files_verified": {"pcie_ipc_ar": True}}),
                encoding="utf-8",
            )
            proc = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(HERE / "derive_node_runtime.py"),
                    "--target-runtime",
                    str(rt),
                    "--fp8-configs-tar",
                    str(tar_path),
                    "--install-fp8-configs-only",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            receipt = json.loads(proc.stdout)
            self.assertEqual(receipt["count"], 10)
            updated = json.loads(meta_path.read_text(encoding="utf-8"))
            self.assertTrue(updated["files_verified"]["fp8_gemm_configs"])
            self.assertEqual(updated["fp8_gemm_configs"]["count"], 10)
            self.assertEqual(len(updated["fp8_gemm_configs"]["files"]), 10)

    def test_rejects_path_traversal_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            rt = tmp_path / "runtime-ours"
            (rt / "sglang/kernels/ops/gemm/configs").mkdir(parents=True)
            bad_tar = tmp_path / "bad.tar.gz"
            dummy = tmp_path / "x.json"
            dummy.write_text("{}", encoding="utf-8")
            with tarfile.open(bad_tar, "w:gz") as tf:
                tf.add(dummy, arcname="../escape.json")
            with self.assertRaises(ValueError):
                dnr.install_fp8_gemm_configs(rt, bad_tar)


if __name__ == "__main__":
    unittest.main()
