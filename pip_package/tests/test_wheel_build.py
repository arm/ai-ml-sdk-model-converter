#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import importlib.util
import os
import pathlib
import tempfile
import unittest
from unittest import mock

MODULE_PATH = (
    pathlib.Path(__file__).resolve().parents[2] / "pip_package" / "wheel_build.py"
)
SPEC = importlib.util.spec_from_file_location("wheel_build", MODULE_PATH)
wheel_build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wheel_build)


class WheelBuildTest(unittest.TestCase):
    def test_source_mode_is_default(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(wheel_build.get_wheel_build_mode(), "source")

    def test_staged_mode_requires_directory(self):
        with mock.patch.dict(
            os.environ, {"MLSDK_WHEEL_BUILD_MODE": "staged"}, clear=True
        ):
            with self.assertRaisesRegex(RuntimeError, "MLSDK_WHEEL_STAGING_DIR"):
                wheel_build.get_staging_dir()

    def test_copy_staged_payload_validates_and_copies_linux_payload(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = pathlib.Path(temporary_dir)
            staging_dir = root / "staging"
            destination = root / "destination"
            required = wheel_build.required_payload_paths(staging_dir, "Linux")
            for path in required:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("payload", encoding="utf-8")

            wheel_build.copy_staged_payload(staging_dir, destination, "Linux")

            self.assertTrue((destination / "bin" / "model-converter").is_file())
            self.assertTrue(
                (destination / "lib" / "ai-ml-sdk-model-converter.spdx.json").is_file()
            )

    def test_copy_staged_payload_rejects_incomplete_payload(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            staging_dir = pathlib.Path(temporary_dir) / "staging"
            staging_dir.mkdir()

            with self.assertRaisesRegex(RuntimeError, "Incomplete Model Converter"):
                wheel_build.copy_staged_payload(
                    staging_dir, pathlib.Path(temporary_dir) / "destination", "Linux"
                )


if __name__ == "__main__":
    unittest.main()
