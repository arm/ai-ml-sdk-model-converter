#
# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import pytest
import vgfpy
from model_converter_helpers import converted_mlir
from vgf_decoder import VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM
from vgf_decoder import VK_FORMAT_R16_SINT
from vgf_decoder import VK_FORMAT_R16_UINT
from vgf_decoder import VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM
from vgf_decoder import VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM
from vgf_decoder import VK_FORMAT_R8_SINT
from vgf_decoder import VK_FORMAT_R8_UINT


def io_vk_format_mlir(element_type):
    return f"""
module {{
  func.func @main(%arg0: tensor<4x{element_type}> {{tf_saved_model.index_path = ["input_0"]}}) -> (tensor<4x{element_type}> {{tf_saved_model.index_path = ["output_0"]}}) attributes {{tf.entry_function = {{inputs = "input_0", outputs = "output_0"}}, tf_saved_model.exported_names = ["main"]}} {{
    %0 = tosa.reverse %arg0 {{axis = 0 : i32}} : (tensor<4x{element_type}>) -> tensor<4x{element_type}>
    return %0 : tensor<4x{element_type}>
  }}
}}
"""


def rescale_vk_format_mlir(input_unsigned, output_unsigned):
    return f"""
module {{
  func.func @main(%arg0: tensor<1x1xi8> {{tf_saved_model.index_path = ["input_0"]}}) -> (tensor<1x1xi16> {{tf_saved_model.index_path = ["output_0"]}}) attributes {{tf.entry_function = {{inputs = "input_0", outputs = "output_0"}}, tf_saved_model.exported_names = ["main"]}} {{
    %0 = "tosa.const"() <{{values = dense<[8]> : tensor<1xi32>}}> : () -> tensor<1xi32>
    %1 = "tosa.const"() <{{values = dense<[23]> : tensor<1xi8>}}> : () -> tensor<1xi8>
    %2 = "tosa.const"() <{{values = dense<[0]> : tensor<1xi8>}}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{{values = dense<[0]> : tensor<1xi16>}}> : () -> tensor<1xi16>
    %4 = tosa.rescale %arg0, %0, %1, %2, %3 {{rounding_mode = SINGLE_ROUND, per_channel = false, scale32 = true, input_unsigned = {str(input_unsigned).lower()}, output_unsigned = {str(output_unsigned).lower()}}} : (tensor<1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi16>) -> tensor<1x1xi16>
    return %4 : tensor<1x1xi16>
  }}
}}
"""


@pytest.mark.parametrize(
    "input_unsigned, output_unsigned, expected_input_format, expected_output_format",
    [
        (False, False, VK_FORMAT_R8_SINT, VK_FORMAT_R16_SINT),
        (True, False, VK_FORMAT_R8_UINT, VK_FORMAT_R16_SINT),
        (False, True, VK_FORMAT_R8_SINT, VK_FORMAT_R16_UINT),
        (True, True, VK_FORMAT_R8_UINT, VK_FORMAT_R16_UINT),
    ],
)
def test_rescale_signless_integer_resource_formats(
    model_converter_exe_path,
    input_unsigned,
    output_unsigned,
    expected_input_format,
    expected_output_format,
):
    with converted_mlir(
        model_converter_exe_path,
        rescale_vk_format_mlir(input_unsigned, output_unsigned),
    ) as vgf:
        assert vgf.resources.size() == 2
        assert vgf.resources.getCategory(0) == vgfpy.ResourceCategory.Input
        assert int(vgf.resources.getVkFormat(0)) == expected_input_format
        assert vgf.resources.getCategory(1) == vgfpy.ResourceCategory.Output
        assert int(vgf.resources.getVkFormat(1)) == expected_output_format


@pytest.mark.parametrize(
    "element_type, expected_format",
    [
        ("bf16", VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM),
        ("f8E4M3FN", VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM),
        ("f8E5M2", VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM),
    ],
)
def test_float_io_resource_formats(
    model_converter_exe_path,
    element_type,
    expected_format,
):
    with converted_mlir(
        model_converter_exe_path, io_vk_format_mlir(element_type)
    ) as vgf:
        assert vgf.resources.size() == 2
        assert vgf.resources.getCategory(0) == vgfpy.ResourceCategory.Input
        assert int(vgf.resources.getVkFormat(0)) == expected_format
        assert vgf.resources.getCategory(1) == vgfpy.ResourceCategory.Output
        assert int(vgf.resources.getVkFormat(1)) == expected_format
