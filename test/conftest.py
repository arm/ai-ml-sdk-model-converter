#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import pathlib
import platform

import pytest


def valid_dir(value):
    path = pathlib.Path(value).resolve()
    if not path.is_dir():
        raise pytest.UsageError(f"{value} is not a directory")
    return path


# Add command line options
def pytest_addoption(parser):
    parser.addoption(
        "--build-dir",
        type=valid_dir,
        required=True,
        help="Path to ML SDK Model Converter build",
    )
    parser.addoption(
        "--sanitizers",
        action="store_true",
        default=False,
        required=False,
        help="Specifies if sanitizers are enabled",
    )


def exe_path(build_path, exe_name):
    if platform.system() == "Windows":
        return os.path.join(build_path, f"{exe_name}.exe")
    return os.path.join(build_path, exe_name)


@pytest.fixture
def vgf_dump_exe_path(request):
    model_converter_build_path = request.config.getoption("--build-dir")
    vgf_build_path = os.path.join(model_converter_build_path, "vgf-lib", "vgf_dump")
    return exe_path(vgf_build_path, "vgf_dump")


@pytest.fixture
def model_converter_exe_path(request):
    model_converter_build_path = request.config.getoption("--build-dir")
    return exe_path(model_converter_build_path, "model-converter")


@pytest.fixture
def opt_exe_path(request):
    model_converter_build_path = request.config.getoption("--build-dir")
    return exe_path(model_converter_build_path, "model-converter-opt")


def pytest_configure(config):
    if config.getoption("--sanitizers") and platform.system() == "Windows":
        asan_dll_path = os.getenv("ASAN_DLL_PATH")
        if asan_dll_path is None or asan_dll_path == "":
            raise pytest.UsageError("ASAN_DLL_PATH environment variable not set")
        os.add_dll_directory(asan_dll_path)
        os.environ["PATH"] += os.pathsep + asan_dll_path
