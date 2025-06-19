#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
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
        "--build-type",
        required=True,
        help="Build type",
    )


def exe_path(build_path, build_type, exe_name):
    if platform.system() == "Windows":
        return build_path / build_type / f"{exe_name}.exe"
    return build_path / exe_name


@pytest.fixture
def vgf_dump_exe_path(request):
    model_converter_build_path = request.config.getoption("--build-dir")
    vgf_build_path = model_converter_build_path / "vgf-lib" / "vgf_dump"
    build_type = request.config.getoption("--build-type")
    return exe_path(vgf_build_path, build_type, "vgf_dump")


@pytest.fixture
def model_converter_exe_path(request):
    model_converter_build_path = request.config.getoption("--build-dir")
    build_type = request.config.getoption("--build-type")
    return exe_path(model_converter_build_path, build_type, "model-converter")


@pytest.fixture
def opt_exe_path(request):
    model_converter_build_path = request.config.getoption("--build-dir")
    build_type = request.config.getoption("--build-type")
    return exe_path(model_converter_build_path, build_type, "model-converter-opt")
