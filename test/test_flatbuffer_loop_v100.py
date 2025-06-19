#
# SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import filecmp
import os
import pathlib
import tempfile
from copy import deepcopy

import pytest

# Executables for tests
samples = pathlib.Path(__file__).resolve().parent / "mlir"


def cmd_compile(model_converter_exe_path, input, output, /, *, config):
    cmd = []
    cmd.extend(
        [f"{model_converter_exe_path}", "--input", input, "--output", output, config]
    )

    return " ".join(cmd)


def flatbuffer_loop(model_converter_exe_path, input, schema):
    with tempfile.TemporaryDirectory() as tmp_path:
        tosa_1 = f"{tmp_path}/out_1.tosa"
        tosa_2 = f"{tmp_path}/out_2.tosa"
        config = "--tosa-flatbuffer" + (
            (" --tosa-flatbuffer-schema=" + schema) if schema else ""
        )

        cmd_line_1 = cmd_compile(model_converter_exe_path, input, tosa_1, config=config)
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


def parametrize_samples(dir, inject, at):
    params = []
    for file in pathlib.Path(dir).iterdir():
        if file.name in ["custom.mlir", "double_custom.mlir"]:
            continue

        cpy = deepcopy(inject)
        cpy[at].append(f"{dir}/{file.name}")
        params.append(cpy)

    return params


"""
 log_label:     Appended to all log output.
 input:     Function to generate input.
 schema:        if used.
"""


@pytest.mark.parametrize(
    "log_label, input, schema",
    parametrize_samples(samples, ("file", [], None), 1),
)
def test_flatbuffer_loop(model_converter_exe_path, log_label, input, schema):
    flatbuffer_loop(model_converter_exe_path, input[0], schema)
