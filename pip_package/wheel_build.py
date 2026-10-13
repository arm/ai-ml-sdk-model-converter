#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
"""Helpers for building the Model Converter wheel payload."""

import os
import pathlib
import platform
import shutil

WHEEL_BUILD_MODE_ENV = "MLSDK_WHEEL_BUILD_MODE"
WHEEL_STAGING_DIR_ENV = "MLSDK_WHEEL_STAGING_DIR"
SOURCE_MODE = "source"
STAGED_MODE = "staged"


def get_wheel_build_mode():
    mode = os.environ.get(WHEEL_BUILD_MODE_ENV, SOURCE_MODE)
    if mode not in {SOURCE_MODE, STAGED_MODE}:
        raise RuntimeError(
            f"{WHEEL_BUILD_MODE_ENV} must be source or staged, got: {mode}"
        )
    return mode


def get_staging_dir():
    value = os.environ.get(WHEEL_STAGING_DIR_ENV)
    if not value:
        raise RuntimeError(
            f"{WHEEL_STAGING_DIR_ENV} is required when "
            f"{WHEEL_BUILD_MODE_ENV}={STAGED_MODE}"
        )
    return pathlib.Path(value).resolve()


def required_payload_paths(staging_dir, system=None):
    staging_dir = pathlib.Path(staging_dir)
    system = system or platform.system()
    binary_name = "model-converter.exe" if system == "Windows" else "model-converter"
    return [
        staging_dir / "bin" / binary_name,
        staging_dir / "lib" / "ai-ml-sdk-model-converter.spdx.json",
    ]


def copy_staged_payload(staging_dir, destination, system=None):
    staging_dir = pathlib.Path(staging_dir).resolve()
    missing = [
        path
        for path in required_payload_paths(staging_dir, system)
        if not path.is_file()
    ]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise RuntimeError(f"Incomplete Model Converter wheel payload: {missing_paths}")

    destination = pathlib.Path(destination)
    shutil.rmtree(destination, ignore_errors=True)
    shutil.copytree(staging_dir, destination)
