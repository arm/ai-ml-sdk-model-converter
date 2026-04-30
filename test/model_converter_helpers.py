#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import contextlib
import os
import pathlib
import tempfile

import pytest
from vgf_decoder import create_vgf_decoders


@contextlib.contextmanager
def converted_mlir(model_converter_exe_path, mlir):
    with tempfile.TemporaryDirectory() as tmp_path:
        input_mlir = f"{tmp_path}/input.mlir"
        output_vgf = f"{tmp_path}/output.vgf"
        pathlib.Path(input_mlir).write_text(mlir)

        cmd = f"{model_converter_exe_path} --input {input_mlir} --output {output_vgf}"
        error = os.system(cmd)
        if error:
            pytest.fail(f"Failed to run ML SDK Model Converter: {cmd}")

        yield create_vgf_decoders(output_vgf)
