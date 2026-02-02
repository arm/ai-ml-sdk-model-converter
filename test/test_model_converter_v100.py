#
# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import binascii
import json
import os
import pathlib
import tempfile

import pytest

mlir_dir = pathlib.Path(__file__).resolve().parent / "mlir"
json_dir = pathlib.Path(__file__).resolve().parent / "json"

tests = [
    #
    # Test 1: Verify all sections and constant values encoded in the VGF.
    (
        "rescale.mlir",
        "rescale.json",
        [b"05000000", b"0800000009000000", b"1718"],
    ),
    #
    # Test 2: Verify Sparse Constants are correctly marked as sparse in the VGF.
    (
        "double_conv2d.mlir",
        "double_conv2d.json",
        [
            b"00000058005b0000000000e200006400760000000000000000006c5600000000000055750000005e000000000000000000380000000971000000000000af000000b70000000000001c0000007167000000000067000000780000000000000040000000d3000000227d0000000000007e000000000000be0000009c000000a300007300005200000000000078000000007a00000000850000007a000000fb00000061000000000000115d0000006a000000006d000000000000000057000000e5000000aa5400000000000034000000",
            b"00003758005b00580000250000cc00007600e2004a4f000000006c560000002500de0075470000007a004c0000",
        ],
    ),
    #
    # # Test 3: Verify parsing a VGF having both graph and compute segments.
    # (
    #     "double_custom.mlir",
    #     "double_custom.json",
    #     [],
    # ),
    # #
    # # Test 4: Single custom op
    # (
    #     "custom.mlir",
    #     "custom.json",
    #     [],
    # ),
    #
    # Test 5: Verify Rescale op attribute input_signed = True processed correctly when input is a tosa.const value
    (
        "rescale.mlir",
        "rescale.json",
        [b"05000000", b"0800000009000000", b"1718"],
    ),
    #
    # Test 5: Single op with rank 0
    (
        "add.mlir",
        "add.json",
        [],
    ),
    # Test 6: Single op with higher rank inlined constant
    (
        "inlined_higher_rank_constant.mlir",
        "inlined_higher_rank_constant.json",
        [],
    ),
    # Test 7: BF16 inputs
    (
        "bf16.mlir",
        "bf16.json",
        [],
    ),
]


def model_converter(
    vgf_dump_exe_path,
    model_converter_exe_path,
    input,
    expected_json_file,
    expected_values,
):

    with tempfile.TemporaryDirectory() as tmp_path:
        # Generate VGF
        output_vgf = f"{tmp_path}/output.vgf"
        cmd = f"{model_converter_exe_path} --input {mlir_dir / input} --output {output_vgf}"
        error = os.system(cmd)
        if error:
            pytest.fail(f"Failed to run ML SDK Model Converter: {cmd}")

        # Dump model JSON and verify
        output_json = f"{tmp_path}/output.json"
        cmd = f"{vgf_dump_exe_path} --input {output_vgf} --output {output_json}"
        error = os.system(cmd)
        if error:
            pytest.fail(f"Failed to dump model JSON: {cmd}")

        actual_json_str = ""
        with open(output_json, mode="r") as actual_json_file:
            actual_json_str = json.dumps(json.load(actual_json_file), sort_keys=False)

        expected_json = json.load(open(json_dir / expected_json_file))
        expected_json_str = json.dumps(expected_json, sort_keys=True)

        assert expected_json_str == actual_json_str

        # Dump model constants and verify
        for i in range(len(expected_json["constants"])):
            constant_binary = f"{tmp_path}/constant{i}.bin"
            cmd = f"{vgf_dump_exe_path} --input {output_vgf} --dump-constant {i} > {constant_binary}"
            error = os.system(cmd)
            if error:
                pytest.fail(f"Failed to dump model constants: {cmd}")

            constant_value = None
            with open(constant_binary, mode="rb") as constant_binary_file:
                constant_value = binascii.hexlify(constant_binary_file.read())

            assert constant_value == expected_values[i]


@pytest.mark.parametrize(
    "input, expected_json_file, expected_values",
    tests,
)
def test_model_converter(
    vgf_dump_exe_path,
    model_converter_exe_path,
    input,
    expected_json_file,
    expected_values,
):
    model_converter(
        vgf_dump_exe_path,
        model_converter_exe_path,
        input,
        expected_json_file,
        expected_values,
    )
