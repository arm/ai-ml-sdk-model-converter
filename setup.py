#
# SPDX-FileCopyrightText: Copyright 2025-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#
import os
import pathlib
import platform
import shutil
import sys

from setuptools import setup
from setuptools.command.build import build as setuptools_build
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:
    from wheel.bdist_wheel import bdist_wheel


MODEL_CONVERTER_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(MODEL_CONVERTER_DIR))

SKIP_NATIVE_BUILD_ENV = "MODEL_CONVERTER_SKIP_NATIVE_BUILD"


class Build(setuptools_build):
    def initialize_options(self):
        super().initialize_options()
        self.build_base = str(pathlib.Path("build") / "python")


class BuildPy(build_py):
    def run(self):
        super().run()

        binary_name = (
            "model-converter.exe"
            if platform.system() == "Windows"
            else "model-converter"
        )
        staged_binary = (
            MODEL_CONVERTER_DIR
            / "pip_package"
            / "model_converter"
            / "binaries"
            / "bin"
            / binary_name
        )
        if os.environ.get(SKIP_NATIVE_BUILD_ENV) == "1" or staged_binary.is_file():
            return

        dependency_dir = MODEL_CONVERTER_DIR.parent.parent / "dependencies"
        vgf_lib_dir = MODEL_CONVERTER_DIR.parent / "vgf-lib"
        missing_paths = [
            path for path in (dependency_dir, vgf_lib_dir) if not path.is_dir()
        ]
        if missing_paths:
            missing = ", ".join(str(path) for path in missing_paths)
            raise RuntimeError(
                "The Model Converter native build requires an ML SDK checkout. "
                f"Missing: {missing}"
            )

        missing_tools = [tool for tool in ("cmake", "ninja") if not shutil.which(tool)]
        if missing_tools:
            raise RuntimeError(
                "The Model Converter native build requires: " + ", ".join(missing_tools)
            )

        from scripts.build import build as build_model_converter

        build_command = self.get_finalized_command("build")
        native_build_dir = pathlib.Path(build_command.build_temp) / "model_converter"
        native_install_dir = (
            pathlib.Path(self.build_lib) / "model_converter" / "binaries"
        )

        extra_args = []
        if os.environ.get("MODEL_CONVERTER_SKIP_LLVM_PATCH") == "1":
            extra_args.append("--skip-llvm-patch")
        result = build_model_converter(
            [
                "--build-dir",
                str(native_build_dir),
                "--install",
                str(native_install_dir),
                "--package-version",
                self.distribution.get_version(),
                "--threads",
                os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 1)),
            ]
            + extra_args
        )
        if result:
            raise RuntimeError(
                f"Model Converter native build failed with code {result}"
            )


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


class BDistWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform_tag = super().get_tag()
        return ("py3", "none", platform_tag)


setup(
    cmdclass={"build": Build, "build_py": BuildPy, "bdist_wheel": BDistWheel},
    distclass=BinaryDistribution,
)
