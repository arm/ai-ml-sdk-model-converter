#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import argparse
import os
import pathlib
import platform
import shutil
import subprocess
import sys
from datetime import datetime

try:
    import argcomplete
except:
    argcomplete = None

MODEL_CONVERTER_DIR = pathlib.Path(__file__).parent / ".."
MODEL_CONVERTER_DIR = MODEL_CONVERTER_DIR.resolve()
DEPENDENCY_DIR = MODEL_CONVERTER_DIR / ".." / ".." / "dependencies"
DEPENDENCY_DIR = DEPENDENCY_DIR.resolve()
CMAKE_TOOLCHAIN_PATH = MODEL_CONVERTER_DIR / "cmake" / "toolchain"


class Builder:
    """
    A  class that builds the ML SDK Model Converter.

    Parameters
    ----------
    args : 'dict'
        Dictionary with arguments to build the ML SDK Model Converter.
    """

    def __init__(self, args):
        self.build_dir = str(pathlib.Path(args.build_dir).resolve())
        self.threads = args.threads
        self.prefix_path = args.prefix_path
        self.external_llvm = args.external_llvm
        self.skip_llvm_patch = args.skip_llvm_patch
        self.run_tests = args.test
        self.build_type = args.build_type
        self.vgf_lib_path = args.vgf_lib_path
        self.json_path = args.json_path
        self.flatbuffers_path = args.flatbuffers_path
        self.tosa_tools_path = args.tosa_tools_path
        self.argparse_path = args.argparse_path
        self.doc = args.doc
        self.lint = args.lint
        self.enable_sanitizers = args.enable_sanitizers
        self.install = args.install
        self.target_platform = args.target_platform
        self.clang_tidy_fix = args.clang_tidy_fix

        self.package_dir = args.package_dir or self.build_dir
        self.package_version = args.package_version
        self.package_tgz = "tgz" in args.package_type
        self.package_zip = "zip" in args.package_type
        self.package_pip = "pip" in args.package_type

        if not self.install and self.package_pip:
            self.install = "pip_install"

    def setup_platform_build(self, cmake_cmd):
        system = platform.system()
        if self.target_platform == "host":
            if system == "Linux":
                cmake_cmd.append(
                    f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'gcc.cmake'}"
                )
                return True

            if system == "Darwin":
                cmake_cmd.append(
                    f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'clang.cmake'}"
                )
                return True

            if system == "Windows":
                cmake_cmd.append(
                    f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'windows-msvc.cmake'}"
                )
                cmake_cmd.append("-DMSVC=ON")
                return True

            print(f"Unsupported host platform {system}", file=sys.stderr)
            return False

        if self.target_platform == "linux-clang":
            if system != "Linux":
                print(
                    f"ERROR: target {self.target_platform} only supported on Linux. Host platform {system}",
                    file=sys.stderr,
                )
                return False
            cmake_cmd.append(
                f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'clang.cmake'}"
            )
            return True

        if self.target_platform == "aarch64":
            cmake_cmd.append(
                f"-DCMAKE_TOOLCHAIN_FILE={CMAKE_TOOLCHAIN_PATH / 'linux-aarch64-gcc.cmake'}"
            )
            cmake_cmd.append("-DHAVE_CLONEFILE=0")
            cmake_cmd.append("-DBUILD_TOOLS=OFF")
            cmake_cmd.append("-DBUILD_REGRESS=OFF")
            cmake_cmd.append("-DBUILD_EXAMPLES=OFF")
            cmake_cmd.append("-DBUILD_DOC=OFF")

            cmake_cmd.append("-DBUILD_WSI_WAYLAND_SUPPORT=OFF")
            cmake_cmd.append("-DBUILD_WSI_XLIB_SUPPORT=OFF")
            cmake_cmd.append("-DBUILD_WSI_XCB_SUPPORT=OFF")
            return True

        print(
            f"ERROR: Incorrect target platform option: {self.target_platform}",
            file=sys.stderr,
        )
        return False

    def generate_cmake_package(self, generator):
        cmake_package_cmd = [
            "cpack",
            "--config",
            f"{self.build_dir}/CPackConfig.cmake",
            "-C",
            self.build_type,
            "-G",
            generator,
            "-B",
            self.package_dir,
            "-D",
            "CPACK_INCLUDE_TOPLEVEL_DIRECTORY=OFF",
        ]
        subprocess.run(cmake_package_cmd, check=True)

    def run(self):
        cmake_setup_cmd = [
            "cmake",
            "-S",
            str(MODEL_CONVERTER_DIR),
            "-B",
            self.build_dir,
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
            "-G",
            "Ninja",
        ]

        if self.prefix_path:
            cmake_setup_cmd.append(f"-DCMAKE_PREFIX_PATH={self.prefix_path}")

        if self.external_llvm:
            cmake_setup_cmd.append(f"-DLLVM_PATH={self.external_llvm}")

        if not self.setup_platform_build(cmake_setup_cmd):
            return 1

        if self.vgf_lib_path:
            cmake_setup_cmd.append(f"-DML_SDK_VGF_LIB_PATH={self.vgf_lib_path}")

        if self.json_path:
            cmake_setup_cmd.append(f"-DJSON_PATH={self.json_path}")

        if self.flatbuffers_path:
            cmake_setup_cmd.append(f"-DFLATBUFFERS_PATH={self.flatbuffers_path}")

        if self.tosa_tools_path:
            cmake_setup_cmd.append(f"-DTOSA_TOOLS_PATH={self.tosa_tools_path}")

        if self.argparse_path:
            cmake_setup_cmd.append(f"-DARGPARSE_PATH={self.argparse_path}")

        if self.lint:
            cmake_setup_cmd.append("-DCMAKE_EXPORT_COMPILE_COMMANDS=ON")
        if self.doc:
            cmake_setup_cmd.append("-DMODEL_CONVERTER_BUILD_DOCS=ON")

        if self.enable_sanitizers:
            if self.target_platform != "host":
                print(
                    f"ERROR: sanitizer not supported for target platform: {self.target_platform}"
                )
                return 1

            system = platform.system()
            if system == "Linux":
                gcc_sanitizer_flags = [
                    "-g",
                    "-fsanitize=undefined,address",
                    "-fno-sanitize=vptr",
                    "-fno-sanitize=alignment",
                    "-fno-sanitize-recover=all",
                ]
                cmake_setup_cmd.append(
                    f"-DCMAKE_CXX_FLAGS={' '.join(gcc_sanitizer_flags)}"
                )
                cmake_setup_cmd.append(
                    "-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=undefined,address"
                )
            elif system == "Windows":
                cmake_setup_cmd.append("-DCMAKE_CXX_FLAGS=/fsanitize=address /Zi /MDd")
                cmake_setup_cmd.append(
                    "-DCMAKE_EXE_LINKER_FLAGS=/INFERASANLIBS /DEBUG /INCREMENTAL:NO"
                )
            else:
                print(f"ERROR: sanitizer is not supported on system: {system}")

        if self.skip_llvm_patch:
            cmake_setup_cmd.append("-DMODEL_CONVERTER_APPLY_LLVM_PATCH=OFF")

        cmake_build_cmd = [
            "cmake",
            "--build",
            self.build_dir,
            "-j",
            str(self.threads),
            "--config",
            self.build_type,
        ]

        try:
            subprocess.run(cmake_setup_cmd, check=True)
            subprocess.run(cmake_build_cmd, check=True)

            if self.clang_tidy_fix and not self.lint:
                print(
                    "WARNING: --clang-tidy-fix requires --lint to be enabled, argument ignored."
                )

            if self.lint:
                src_dirs = [
                    f"{MODEL_CONVERTER_DIR / 'src'}",
                ]

                lint_cmd = [
                    "cppcheck",
                    f"-j{str(self.threads)}",
                    "--std=c++17",
                    "--error-exitcode=1",
                    "--inline-suppr",
                    f"--cppcheck-build-dir={self.build_dir}/cppcheck",
                    "--enable=information,performance,portability,style",
                    f"-i={DEPENDENCY_DIR}",
                    f"--suppress=noValidConfiguration",
                    f"--suppress=unassignedVariable",
                    f"--suppress=unmatchedSuppression",
                    f"--suppress=variableScope",
                    f"--suppress=*:MachineIndependent*",
                    f"--suppress=*:{DEPENDENCY_DIR}*",
                ] + src_dirs
                subprocess.run(lint_cmd, check=True)

                clang_tidy_cmd = [
                    "run-clang-tidy",
                    f"-j{self.threads}",
                    f"-p{self.build_dir}",
                ] + src_dirs

                if self.clang_tidy_fix:
                    clang_tidy_cmd.append("-fix")

                subprocess.run(clang_tidy_cmd, check=True)

            if self.install:
                cmake_install_cmd = [
                    "cmake",
                    "--install",
                    self.build_dir,
                    "--prefix",
                    self.install,
                    "--config",
                    self.build_type,
                ]
                subprocess.run(cmake_install_cmd, check=True)

            if self.run_tests:
                pytest_cmd = [
                    sys.executable,
                    "-m",
                    "pytest",
                    "test",
                    "-n",
                    str(self.threads),
                    "--build-dir",
                    self.build_dir,
                ]
                if self.enable_sanitizers:
                    pytest_cmd.append("--sanitizers")
                subprocess.run(pytest_cmd, cwd=MODEL_CONVERTER_DIR, check=True)

            if self.package_tgz:
                self.generate_cmake_package("TGZ")

            if self.package_zip:
                self.generate_cmake_package("ZIP")

            if self.package_pip:
                os.makedirs("pip_package/model_converter/binaries/", exist_ok=True)
                shutil.copytree(
                    self.install,
                    "pip_package/model_converter/binaries/",
                    dirs_exist_ok=True,
                )
                shutil.copyfile("README.md", "pip_package/README.md")

                os.environ[
                    "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_AI_ML_SDK_MODEL_CONVERTER"
                ] = self.package_version
                result = subprocess.Popen(
                    [sys.executable, "-m", "build"],
                    env=os.environ,
                    cwd="pip_package",
                )
                result.communicate()
                if result.returncode != 0:
                    print("ERROR: Failed to generate pip package")
                    return 1

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"ModelConverterBuilder failed with {e}", file=sys.stderr)
            return 1

        return 0


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir",
        help="Name of folder where to build the ML SDK Model Converter. Default: build",
        default=f"{MODEL_CONVERTER_DIR / 'build'}",
    )
    parser.add_argument(
        "--threads",
        "-j",
        type=int,
        help="Number of threads to use for building. Default: %(default)s",
        default=16,
    )
    parser.add_argument(
        "--prefix-path",
        help="Path to prefix directory.",
    )
    parser.add_argument(
        "--external-llvm",
        help="Path to the LLVM repo and build.",
        default=f"{DEPENDENCY_DIR / 'llvm-project'}",
    )
    parser.add_argument(
        "-t",
        "--test",
        help="Run unit tests after build. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--build-type",
        help="Type of build to perform. Default: %(default)s",
        default="Release",
    )
    parser.add_argument(
        "--vgf-lib-path",
        help="Path to the ai-ml-sdk-vgf-library repo",
        default=f"{MODEL_CONVERTER_DIR / '..' / 'vgf-lib'}",
    )
    parser.add_argument(
        "--argparse-path",
        help="Path to argparse repo",
        default=f"{DEPENDENCY_DIR / 'argparse'}",
    )
    parser.add_argument(
        "--flatbuffers-path",
        help="Path to flatbuffers repo",
        default=f"{DEPENDENCY_DIR / 'flatbuffers'}",
    )
    parser.add_argument(
        "--json-path",
        help="Path to json repo",
        default=f"{DEPENDENCY_DIR / 'json'}",
    )
    parser.add_argument(
        "--tosa-tools-path",
        help="Path to the TOSA Tools repo",
        default=f"{DEPENDENCY_DIR / 'tosa-tools'}",
    )
    parser.add_argument(
        "--doc",
        help="Build documentation. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--enable-sanitizers",
        help="Enable sanitizers. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--skip-llvm-patch",
        help="Skip applying LLVM patch. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-l",
        "--lint",
        help="Run linter. Default: %(default)s",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--install",
        help="Install build artifacts into a provided location",
    )
    parser.add_argument(
        "--package-dir",
        help="Specify location for packages to be created. Default path is the build directory",
        default="",
    )
    parser.add_argument(
        "--package-type",
        action="append",
        choices=["zip", "tgz", "pip"],
        help="Create a package of a certain type",
        default=[],
    )
    parser.add_argument(
        "--package-version",
        help="Manually specify pip package version number",
        default=datetime.today().strftime("%m.%d"),
    )
    parser.add_argument(
        "--target-platform",
        help="Specify the target build platform",
        choices=["host", "aarch64", "linux-clang"],
        default="host",
    )
    parser.add_argument(
        "--clang-tidy-fix",
        help="Enable clang-tidy fix (requires --lint). Default: %(default)s",
        action="store_true",
        default=False,
    )

    if argcomplete:
        argcomplete.autocomplete(parser)

    args = parser.parse_args()
    return args


def main():
    builder = Builder(parse_arguments())
    sys.exit(builder.run())


if __name__ == "__main__":
    main()
