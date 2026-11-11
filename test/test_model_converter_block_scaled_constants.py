#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import math
import subprocess

import pytest
import vgfpy
from model_converter_helpers import converted_mlir
from test_model_converter_core import main_mlir
from test_model_converter_graph_constants import graph_segment_constant_bindings
from vgf_decoder import VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT4E2M1_ARM
from vgf_decoder import VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT6E2M3_ARM
from vgf_decoder import VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT6E3M2_ARM
from vgf_decoder import VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM
from vgf_decoder import VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM
from vgf_decoder import VK_FORMAT_R8_SFLOAT_FPENCODING_MXINT8_ARM

pytestmark = pytest.mark.usefixtures("block_scaled_tensors_enabled")


def block_scaled_const_mlir(value_type, shape, values, scales):
    assert len(values) == math.prod(shape)
    tensor_shape = "x".join(str(dimension) for dimension in shape)
    element_type = f"!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:{value_type}>"
    scale_values = ", ".join(str(scale) for scale in scales)
    attribute_type = (
        f"!tosa.block_scaled<BLOCK_SHAPE_32:f8E8M0FNU:{value_type}, "
        f"{{{scale_values}}}>"
    )
    result_type = f"tensor<{tensor_shape}x{element_type}>"
    values_type = f"tensor<{tensor_shape}x{attribute_type}>"
    constant = (
        f"tosa.const values(dense_resource<weight_values> : {values_type}) "
        f": () -> {result_type}"
    )
    module = main_mlir(
        arguments="",
        result=f'({result_type} {{tf_saved_model.index_path = ["output_0"]}})',
        entry_inputs="",
        entry_outputs="output_0",
        body=f"    %c0 = {constant}",
        return_value=f"%c0 : {result_type}",
    )
    return module + f"""
{{-# dialect_resources: {{
  builtin: {{weight_values: "0x01000000{values.hex().upper()}"}}
}} #-}}
"""


FP6_PACKED = bytes.fromhex("40200c44611c48a22c4ce33c50244d54655d58a66d5ce77d")


@pytest.mark.parametrize(
    "value_type, shape, values, scales, packed_values, vk_format",
    [
        # Keep stored values 1.5 (0x3c) and -2.5 (0xc2) separate from their scales.
        pytest.param(
            "f8E4M3FN",
            (2, 32),
            bytes([0x3C] * 32 + [0xC2] * 32),
            (0.5, 4.0),
            bytes([0x3C] * 32 + [0xC2] * 32),
            VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM,
            id="fp8-e4m3-two-blocks",
        ),
        pytest.param(
            "f8E5M2",
            (32,),
            bytes(range(32)),
            (1.0,),
            bytes(range(32)),
            VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM,
            id="fp8-e5m2",
        ),
        pytest.param(
            "f4E2M1FN",
            (32,),
            bytes(range(16)) * 2,
            (1.0,),
            bytes.fromhex("1032547698badcfe1032547698badcfe"),
            VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT4E2M1_ARM,
            id="fp4",
        ),
        pytest.param(
            "f6E2M3FN",
            (32,),
            bytes(range(32)),
            (1.0,),
            FP6_PACKED,
            VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT6E2M3_ARM,
            id="fp6-e2m3",
        ),
        pytest.param(
            "f6E3M2FN",
            (32,),
            bytes(range(32)),
            (1.0,),
            FP6_PACKED,
            VK_FORMAT_OCPMXFP_32_SFLOAT_BLOCK_FPENCODING_FLOAT6E3M2_ARM,
            id="fp6-e3m2",
        ),
        pytest.param(
            "!tosa.mxint8",
            (32,),
            bytes(range(32)),
            (1.0,),
            bytes(range(32)),
            VK_FORMAT_R8_SFLOAT_FPENCODING_MXINT8_ARM,
            id="mxint8",
        ),
    ],
)
def test_block_scaled_const_payload(
    model_converter_exe_path,
    value_type,
    shape,
    values,
    scales,
    packed_values,
    vk_format,
):
    with converted_mlir(
        model_converter_exe_path,
        block_scaled_const_mlir(value_type, shape, values, scales),
    ) as vgf:
        assert vgf.constants.size() == 1
        resource_index = vgf.constants.getConstantMrtIndex(0)
        assert (
            vgf.resources.getCategory(resource_index) == vgfpy.ResourceCategory.Constant
        )
        assert int(vgf.resources.getVkFormat(resource_index)) == vk_format
        assert list(vgf.resources.getTensorShape(resource_index)) == list(shape)
        config = vgf.resources.getScaledTensorConfigHandle(resource_index)
        assert config is not None
        assert list(vgf.resources.getScaledTensorConfigBlockShape(config)) == [32]
        assert (
            vgf.resources.getScaledTensorConfigScaleFormat(config)
            == vgfpy.ScaleFormat.E8M0
        )
        assert (
            vgf.sequence.getBindingsSize(
                vgf.sequence.getModelSequenceInputBindingSlotsHandle()
            )
            == 0
        )
        expected_scales = bytes(127 + int(math.log2(scale)) for scale in scales)
        assert vgf.constants.getConstant(0).tobytes() == packed_values + expected_scales

        segment_bindings, module_ids = graph_segment_constant_bindings(vgf)
        assert segment_bindings == [[(0, 0)]]
        assert module_ids == [[0]]


def test_block_scaled_const_rejects_wrong_scale_count(
    model_converter_exe_path, tmp_path
):
    input_path = tmp_path / "input.mlir"
    output_path = tmp_path / "output.vgf"
    input_path.write_text(
        block_scaled_const_mlir("f8E4M3FN", (2, 32), bytes(range(64)), (1.0,))
    )
    result = subprocess.run(
        [
            model_converter_exe_path,
            "--input",
            input_path,
            "--output",
            output_path,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "expected 2, got 1" in result.stderr
    assert not output_path.exists()
