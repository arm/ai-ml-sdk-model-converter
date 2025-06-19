#
# SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import binascii
import filecmp
import json
import os
import pathlib
import tempfile

import pytest

test_folder = pathlib.Path(__file__).resolve().parents[0] / "type_narrowing_testfiles"


class NarrowingType:
    FULL = "full"
    PARTIAL = "partial"
    FULL_PRESERVE_IO = "full_preserve_io"


FULL_PRESERVE_IO_EXTENSION = "_preserve_io.tosa-ir"
PARTIAL_EXTENSION = "_partial.tosa-ir"
FULL_EXTENSION = "_full.tosa-ir"

LICENSE_HEADER = 4


def opt_cmd_compile(opt_exe_path, mode, input, output):
    cmd = []
    cmd.extend(
        [f"{opt_exe_path}", f'--type-narrowing="mode={mode}"', input, "-o", output]
    )

    return " ".join(cmd)


def cmd_compile(model_converter_exe_path, input, output, /, *, config):
    cmd = []
    cmd.extend(
        [f"{model_converter_exe_path}", "--input", input, "--output", output, config]
    )

    return " ".join(cmd)


def calculate_const_fp16_size(shape):
    res = 2  # fp16 = 2 bytes
    for s in shape:
        res *= s
    return res


def type_narrowing_common(
    narrowing_mode,
    vgf_dump_exe_path,
    model_converter_exe_path,
    test_path,
    nbr_constants,
):

    with tempfile.TemporaryDirectory() as tmp_path:

        # Save input IR
        input_ir = f"{test_path}"

        # Generate VGF
        output_vgf = f"{tmp_path}/output.vgf"
        cmd = cmd_compile(
            model_converter_exe_path,
            input_ir,
            output_vgf,
            config=("--type-narrowing=" + narrowing_mode),
        )
        error = os.system(cmd)
        if error:
            pytest.fail(f"Failed to run ML SDK Model Converter: {cmd}")

        # Dump model JSON and verify
        output_json = f"{tmp_path}/output.json"
        cmd = cmd_compile(vgf_dump_exe_path, output_vgf, output_json, config="")
        error = os.system(cmd)
        if error:
            pytest.fail(f"Failed to dump model JSON: {cmd}")

        actual_json = None
        with open(output_json, mode="r") as actual_json_file:
            actual_json = json.load(actual_json_file)

        # Dump model constants and verify

        assert len(actual_json["constants"]) == nbr_constants
        for i in range(len(actual_json["constants"])):
            constant_binary = f"{tmp_path}/constant{i}.bin"
            cmd = f"{vgf_dump_exe_path} --input {output_vgf} --dump-constant {i} --output {constant_binary}"

            error = os.system(cmd)
            if error:
                pytest.fail(f"Failed to dump model constants: {cmd}")

            index = actual_json["constants"][i]["mrt_index"]
            vk_format = actual_json["resources"][index]["vk_format"]
            shape_array = actual_json["resources"][index]["shape"]

            assert vk_format == "VK_FORMAT_R16_SFLOAT"
            constant_binary = f"{tmp_path}/constant{i}.bin"
            cmd = f"{vgf_dump_exe_path} --input {output_vgf} --dump-constant {i} --output {constant_binary}"
            error = os.system(cmd)
            if error:
                pytest.fail(f"Failed to dump model constants: {cmd}")

            constant_value = None
            with open(constant_binary, mode="rb") as constant_binary_file:
                constant_value = constant_binary_file.read()

            # FIXME: remove this line when constant dumping on Windows works again
            if test_path.endswith("conv3d.tosa-ir"):
                pytest.skip(
                    "The constant in this model contains 0x0a and fails on Windows"
                )

            assert len(constant_value) == calculate_const_fp16_size(shape_array)

            expected_constant_file = (
                f"{test_folder}/{pathlib.Path(test_path).stem}_const_{i}"
            )
            if pathlib.Path(expected_constant_file).is_file():
                expected_constant = None
                with open(expected_constant_file, mode="rb") as const:
                    expected_constant = const.read()
                assert constant_value == expected_constant


def flatbuffer_loop_external_files(
    narrowing_mode, vgf_dump_exe_path, model_converter_exe_path, test_path
):
    schema = 0  # temp solution, unsure if I need schema
    with tempfile.TemporaryDirectory() as tmp_path:
        tosa_1 = f"{tmp_path}/out_1.tosa"
        tosa_2 = f"{tmp_path}/out_2.tosa"
        config = (
            "--tosa-flatbuffer --type-narrowing "
            + narrowing_mode
            + ((" --tosa-flatbuffer-schema=" + schema) if schema else "")
        )

        cmd_line_1 = cmd_compile(
            model_converter_exe_path, test_path, tosa_1, config=config
        )
        error = os.system(cmd_line_1)
        if error:
            pytest.fail(
                f"Step [MLIR to TOSA]: run_cmd '{cmd_line_1}' returned non-zero value ({error}). Exiting."
            )
        if not os.path.exists(tosa_1):
            pytest.fail(f"No tosa file output after [MLIR to TOSA]: {tosa_1}. Exiting.")

        # Get the TOSA_Flatbuffer *from* the TOSA_Flatbuffer
        cmd_line_2 = cmd_compile(
            model_converter_exe_path, tosa_1, tosa_2, config=config
        )
        error = os.system(cmd_line_2)
        if error:
            pytest.fail(
                f"Step [TOSA to TOSA]: run_cmd '{cmd_line_2}' returned non-zero value ({error}). Exiting."
            )
        if not os.path.exists(tosa_2):
            pytest.fail(f"No tosa file output after [TOSA to TOSA]: {tosa_2}. Exiting.")

        # Compare output files
        if filecmp.cmp(tosa_1, tosa_2, shallow=False) == False:
            pytest.fail(
                f"Comparing TOSA files failed: The tosa outputs were not the same."
            )


def check_tosa_mlir(narrowing_mode, vgf_dump_exe_path, opt_exe_path, test_path):

    with tempfile.TemporaryDirectory() as tmp_path:

        tosa_ir = f"{tmp_path}/out.tosa-ir"
        cmd_line_1 = opt_cmd_compile(opt_exe_path, narrowing_mode, test_path, tosa_ir)

        error = os.system(cmd_line_1)
        if error:
            pytest.fail(
                f"Step [OPT]: run_cmd '{cmd_line_1}' returned non-zero value ({error}). Exiting."
            )
        if not os.path.exists(tosa_ir):
            pytest.fail(f"No tosa file output after [OPT]: {tosa_ir}. Exiting.")

        expected_file = test_folder
        match (narrowing_mode):
            case NarrowingType.PARTIAL:
                expected_file = (
                    f"{expected_file}/{pathlib.Path(test_path).stem}{PARTIAL_EXTENSION}"
                )
            case NarrowingType.FULL_PRESERVE_IO:
                expected_file = f"{expected_file}/{pathlib.Path(test_path).stem}{FULL_PRESERVE_IO_EXTENSION}"
            case NarrowingType.FULL:
                expected_file = (
                    f"{expected_file}/{pathlib.Path(test_path).stem}{FULL_EXTENSION}"
                )

        actual_tosa_ir = ""
        with open(tosa_ir, "r", errors="replace") as tosa_ir_file:
            actual_tosa_ir = "".join(tosa_ir_file.readlines()[:-1])

        if (
            narrowing_mode == NarrowingType.PARTIAL
            and not pathlib.Path(expected_file).is_file()
        ):
            pytest.skip("No expected file for PARTIAL")

        if (
            narrowing_mode == NarrowingType.FULL
            and not pathlib.Path(expected_file).is_file()
        ):
            pytest.skip("No expected file for FULL")

        expected_tosa_ir = ""
        with open(expected_file, "r") as expected_tosa_ir_file:
            expected_tosa_ir = "".join(
                expected_tosa_ir_file.readlines()[LICENSE_HEADER:]
            )

        assert actual_tosa_ir == expected_tosa_ir


test_files = [
    (
        "abs.tosa-ir",
        1,
    ),
    (
        "avg_pool2d.tosa-ir",
        0,
    ),
    (
        "conv2d.tosa-ir",
        1,
    ),
    (
        "depthwise_conv2d.tosa-ir",
        1,
    ),
]

test_acc_files = [
    "avg_pool2d.tosa-ir",
    "conv2d.tosa-ir",
    "depthwise_conv2d.tosa-ir",
]


@pytest.mark.parametrize(
    "test_file, nbr_consts",
    test_files,
)
def test_type_narrowing_common(
    vgf_dump_exe_path, model_converter_exe_path, test_file, nbr_consts
):
    test_path = f"{test_folder}/{test_file}"
    type_narrowing_common(
        NarrowingType.FULL_PRESERVE_IO,
        vgf_dump_exe_path,
        model_converter_exe_path,
        test_path,
        nbr_consts,
    )

    type_narrowing_common(
        NarrowingType.FULL,
        vgf_dump_exe_path,
        model_converter_exe_path,
        test_path,
        nbr_consts,
    )


@pytest.mark.parametrize(
    "test_file, nbr_consts",
    test_files,
)
def test_flatbuffer_loop_fp32_narrowing(
    vgf_dump_exe_path, model_converter_exe_path, test_file, nbr_consts
):
    test_path = f"{test_folder}/{test_file}"
    flatbuffer_loop_external_files(
        NarrowingType.FULL_PRESERVE_IO,
        vgf_dump_exe_path,
        model_converter_exe_path,
        test_path,
    )

    flatbuffer_loop_external_files(
        NarrowingType.FULL, vgf_dump_exe_path, model_converter_exe_path, test_path
    )


@pytest.mark.parametrize(
    "test_file, nbr_consts",
    test_files,
)
def test_tosa_mlir_type_narrowing_preserve_io(
    vgf_dump_exe_path, opt_exe_path, test_file, nbr_consts
):
    test_path = f"{test_folder}/{test_file}"
    check_tosa_mlir(
        NarrowingType.FULL_PRESERVE_IO, vgf_dump_exe_path, opt_exe_path, test_path
    )


@pytest.mark.parametrize(
    "test_file, nbr_consts",
    test_files,
)
def test_tosa_mlir_type_narrowing_full(
    vgf_dump_exe_path, opt_exe_path, test_file, nbr_consts
):
    test_path = f"{test_folder}/{test_file}"
    check_tosa_mlir(NarrowingType.FULL, vgf_dump_exe_path, opt_exe_path, test_path)


@pytest.mark.parametrize(
    "test_file",
    test_acc_files,
)
def test_tosa_mlir_fp32_partial_narrowing(vgf_dump_exe_path, opt_exe_path, test_file):
    test_path = f"{test_folder}/{test_file}"
    check_tosa_mlir(NarrowingType.PARTIAL, vgf_dump_exe_path, opt_exe_path, test_path)
