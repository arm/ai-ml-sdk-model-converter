#
# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import json
import subprocess

import pytest
import vgfpy
from model_converter_helpers import converted_mlir
from vgf_decoder import VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM
from vgf_decoder import VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM
from vgf_decoder import VK_FORMAT_R8_SINT
from vgf_decoder import VK_FORMAT_R8_UINT
from vgf_decoder import VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM
from vgf_decoder import VK_FORMAT_R16_SINT
from vgf_decoder import VK_FORMAT_R16_UINT
from vgf_decoder import VK_FORMAT_R32_SINT

VK_DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000


def io_vk_format_mlir(element_type):
    return f"""
module {{
  func.func @main(%arg0: tensor<4x{element_type}> {{tf_saved_model.index_path = ["input_0"]}}) -> (tensor<4x{element_type}> {{tf_saved_model.index_path = ["output_0"]}}) attributes {{tf.entry_function = {{inputs = "input_0", outputs = "output_0"}}, tf_saved_model.exported_names = ["main"]}} {{
    %0 = tosa.reverse %arg0 axis(0) : (tensor<4x{element_type}>) -> tensor<4x{element_type}>
    return %0 : tensor<4x{element_type}>
  }}
}}
"""


def rescale_vk_format_mlir(input_unsigned, output_unsigned):
    return f"""
module {{
  func.func @main(%arg0: tensor<1x1xi8> {{tf_saved_model.index_path = ["input_0"]}}) -> (tensor<1x1xi16> {{tf_saved_model.index_path = ["output_0"]}}) attributes {{tf.entry_function = {{inputs = "input_0", outputs = "output_0"}}, tf_saved_model.exported_names = ["main"]}} {{
    %0 = tosa.const values(dense<[8]> : tensor<1xi32>) : () -> tensor<1xi32>
    %1 = tosa.const values(dense<[23]> : tensor<1xi8>) : () -> tensor<1xi8>
    %2 = tosa.const values(dense<[0]> : tensor<1xi8>) : () -> tensor<1xi8>
    %3 = tosa.const values(dense<[0]> : tensor<1xi16>) : () -> tensor<1xi16>
    %4 = tosa.rescale %arg0, %0, %1, %2, %3 scale32(true) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned({str(input_unsigned).lower()}) output_unsigned({str(output_unsigned).lower()}) : (tensor<1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi16>) -> tensor<1x1xi16>
    return %4 : tensor<1x1xi16>
  }}
}}
    """


def custom_shader_attrs(vk_format):
    return json.dumps(
        json.dumps(
            {
                "entry_point": "main",
                "is_vkshader": True,
                "workgroup_sizes": [1, 1, 1],
                "input_0_binding": 0,
                "input_0_descriptorset": 0,
                "input_0_type": "TENSOR",
                "input_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
                "input_0_vkformat": vk_format,
                "output_0_binding": 1,
                "output_0_descriptorset": 0,
                "output_0_type": "TENSOR",
                "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
                "output_0_vkformat": vk_format,
            },
            separators=(",", ":"),
        )
    )


def boundary_rescale_rematerialized_mlir():
    attrs = custom_shader_attrs("VK_FORMAT_R8_SINT")
    return f"""
module {{
  func.func @main(%arg0: tensor<4xi8> {{tf_saved_model.index_path = ["input_0"]}}) -> (tensor<4xi32> {{tf_saved_model.index_path = ["output_0"]}}) attributes {{tf.entry_function = {{inputs = "input_0", outputs = "output_0"}}, tf_saved_model.exported_names = ["main"]}} {{
    %multiplier = tosa.const values(dense<[1]> : tensor<1xi32>) : () -> tensor<1xi32>
    %shift = tosa.const values(dense<[0]> : tensor<1xi8>) : () -> tensor<1xi8>
    %input_zp = tosa.const values(dense<[0]> : tensor<1xi8>) : () -> tensor<1xi8>
    %flow_output_zp = tosa.const values(dense<[0]> : tensor<1xi8>) : () -> tensor<1xi8>
    %output_zp = tosa.const values(dense<[0]> : tensor<1xi32>) : () -> tensor<1xi32>
    %flow = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %flow_output_zp scale32(true) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<4xi8>
    %old_flow_i32 = tosa.rescale %flow, %multiplier, %shift, %input_zp, %output_zp scale32(true) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<4xi32>
    %shader = tosa.custom %flow operator_name("remat_shader") domain_name("com.arm.VulkanCustomShader") implementation_attrs({attrs}) : (tensor<4xi8>) -> tensor<4xi8>
    %shader_i32 = tosa.rescale %shader, %multiplier, %shift, %input_zp, %output_zp scale32(true) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<4xi32>
    %out = tosa.add %shader_i32, %old_flow_i32 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %out : tensor<4xi32>
  }}
}}
"""


def boundary_rescale_compute_use_preserved_mlir():
    attrs = custom_shader_attrs("VK_FORMAT_R32_SINT")
    return f"""
module {{
  func.func @main(%arg0: tensor<4xi8> {{tf_saved_model.index_path = ["input_0"]}}) -> (tensor<4xi32> {{tf_saved_model.index_path = ["output_0"]}}) attributes {{tf.entry_function = {{inputs = "input_0", outputs = "output_0"}}, tf_saved_model.exported_names = ["main"]}} {{
    %multiplier = tosa.const values(dense<[1]> : tensor<1xi32>) : () -> tensor<1xi32>
    %shift = tosa.const values(dense<[0]> : tensor<1xi8>) : () -> tensor<1xi8>
    %input_zp = tosa.const values(dense<[0]> : tensor<1xi8>) : () -> tensor<1xi8>
    %flow_output_zp = tosa.const values(dense<[0]> : tensor<1xi8>) : () -> tensor<1xi8>
    %output_zp = tosa.const values(dense<[0]> : tensor<1xi32>) : () -> tensor<1xi32>
    %flow = tosa.rescale %arg0, %multiplier, %shift, %input_zp, %flow_output_zp scale32(true) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<4xi8>
    %old_flow_i32 = tosa.rescale %flow, %multiplier, %shift, %input_zp, %output_zp scale32(true) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<4xi32>
    %shader = tosa.custom %old_flow_i32 operator_name("i32_shader") domain_name("com.arm.VulkanCustomShader") implementation_attrs({attrs}) : (tensor<4xi32>) -> tensor<4xi32>
    %out = tosa.add %shader, %old_flow_i32 : (tensor<4xi32>, tensor<4xi32>) -> tensor<4xi32>
    return %out : tensor<4xi32>
  }}
}}
"""


def binding_slots_by_binding(vgf, bindings_handle):
    return {
        vgf.sequence.getBindingSlotBinding(bindings_handle, binding_index): (
            vgf.sequence.getBindingSlotMrtIndex(bindings_handle, binding_index)
        )
        for binding_index in range(vgf.sequence.getBindingsSize(bindings_handle))
    }


def resource_formats(vgf, binding_slots):
    return [
        int(vgf.resources.getVkFormat(index))
        for _, index in sorted(binding_slots.items())
    ]


@pytest.fixture(scope="module")
def block_scaled_tensors_enabled(request):
    cache = request.config.getoption("--build-dir") / "CMakeCache.txt"
    enabled = any(
        line.startswith("BLOCK_SCALED_TENSOR:")
        and line.partition("=")[2].upper() in {"1", "ON", "TRUE", "YES"}
        for line in cache.read_text().splitlines()
    )
    if not enabled:
        pytest.skip("Model Converter was built without block-scaled tensor support")


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


def test_boundary_rescale_rematerialization_uses_i8_graph_resource(
    model_converter_exe_path,
):
    with converted_mlir(
        model_converter_exe_path,
        boundary_rescale_rematerialized_mlir(),
    ) as vgf:
        assert vgf.sequence.modelSequenceTableSize() == 3
        assert vgf.sequence.getSegmentType(0) == vgfpy.ModuleType.Graph
        assert vgf.sequence.getSegmentType(1) == vgfpy.ModuleType.Compute
        assert vgf.sequence.getSegmentType(2) == vgfpy.ModuleType.Graph

        graph0_outputs = binding_slots_by_binding(
            vgf, vgf.sequence.getSegmentOutputBindingSlotsHandle(0)
        )
        graph2_inputs = binding_slots_by_binding(
            vgf, vgf.sequence.getSegmentInputBindingSlotsHandle(2)
        )

        assert resource_formats(vgf, graph0_outputs) == [VK_FORMAT_R8_SINT]
        assert resource_formats(vgf, graph2_inputs) == [
            VK_FORMAT_R8_SINT,
            VK_FORMAT_R8_SINT,
        ]


def test_boundary_rescale_compute_i32_resource_is_preserved(
    model_converter_exe_path,
):
    with converted_mlir(
        model_converter_exe_path,
        boundary_rescale_compute_use_preserved_mlir(),
    ) as vgf:
        assert vgf.sequence.modelSequenceTableSize() == 3
        assert vgf.sequence.getSegmentType(0) == vgfpy.ModuleType.Graph
        assert vgf.sequence.getSegmentType(1) == vgfpy.ModuleType.Compute
        assert vgf.sequence.getSegmentType(2) == vgfpy.ModuleType.Graph

        graph0_outputs = binding_slots_by_binding(
            vgf, vgf.sequence.getSegmentOutputBindingSlotsHandle(0)
        )
        compute_inputs = binding_slots_by_binding(
            vgf, vgf.sequence.getSegmentInputBindingSlotsHandle(1)
        )
        graph2_inputs = binding_slots_by_binding(
            vgf, vgf.sequence.getSegmentInputBindingSlotsHandle(2)
        )

        assert resource_formats(vgf, graph0_outputs) == [
            VK_FORMAT_R8_SINT,
            VK_FORMAT_R32_SINT,
        ]
        assert resource_formats(vgf, compute_inputs) == [VK_FORMAT_R32_SINT]
        assert resource_formats(vgf, graph2_inputs) == [
            VK_FORMAT_R8_SINT,
            VK_FORMAT_R32_SINT,
        ]


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


def scaled_passthrough_mlir(
    value_type="f8E4M3FN",
    *,
    scale_type="f8E8M0FNU",
    block=32,
    size=64,
    alias_input_output=False,
):
    data_type = f"tensor<{size}x!spirv.arm.tensor_scaled_block<{value_type}, {scale_type}, [{block}]>>"
    if alias_input_output:
        function = f"""
  func.func @main(%input: {data_type}, %ordinary: tensor<4xf32>) -> ({data_type}, tensor<4xf32>) {{
    %result = tosa.reverse %ordinary axis(0) : (tensor<4xf32>) -> tensor<4xf32>
    return %input, %result : {data_type}, tensor<4xf32>
  }}
"""
    else:
        function = f"""
  func.func @main(%input: {data_type}) -> {data_type} {{
    return %input : {data_type}
  }}
"""
    return f"""
module attributes {{
  spirv.target_env = #spirv.target_env<#spirv.vce<v1.5,
    [VulkanMemoryModel, Shader, Int64, Float8EXT, Float8UnsignedE8M0EXT, TensorsARM, GraphARM, TensorsBlockScaledARM, ReplicatedCompositesEXT],
    [SPV_KHR_vulkan_memory_model, SPV_EXT_float8, SPV_EXT_ocp_microscaling_types, SPV_ARM_tensors, SPV_ARM_graph, SPV_ARM_tensors_block_scaled, SPV_EXT_replicated_composites]>, #spirv.resource_limits<>>
}} {{
{function}
}}
"""


@pytest.mark.parametrize(
    "value_type, value_format",
    [
        ("f8E4M3FN", VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM),
        ("f8E5M2", VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM),
    ],
)
@pytest.mark.usefixtures("block_scaled_tensors_enabled")
def test_scaled_graph_resources(model_converter_exe_path, value_type, value_format):
    with converted_mlir(
        model_converter_exe_path, scaled_passthrough_mlir(value_type)
    ) as vgf:
        assert vgf.resources.size() == 2
        assert vgf.modules.size() == 1
        assert vgf.sequence.modelSequenceTableSize() == 1
        assert vgf.modules.getModuleType(0) == vgfpy.ModuleType.Graph
        assert vgf.sequence.getSegmentType(0) == vgfpy.ModuleType.Graph
        assert vgf.modules.hasSPIRVCode(0)

        bindings = {}
        for handle, binding, category in [
            (
                vgf.sequence.getModelSequenceInputBindingSlotsHandle(),
                0,
                vgfpy.ResourceCategory.Input,
            ),
            (
                vgf.sequence.getModelSequenceOutputBindingSlotsHandle(),
                1,
                vgfpy.ResourceCategory.Output,
            ),
        ]:
            assert vgf.sequence.getBindingsSize(handle) == 1
            slots = binding_slots_by_binding(vgf, handle)
            assert list(slots) == [binding]
            resource_index = slots[binding]
            bindings[binding] = resource_index
            assert vgf.resources.getCategory(resource_index) == category
            assert (
                vgf.resources.getDescriptorType(resource_index)
                == VK_DESCRIPTOR_TYPE_TENSOR_ARM
            )
            assert int(vgf.resources.getVkFormat(resource_index)) == value_format
            assert list(vgf.resources.getTensorShape(resource_index)) == [64]
            assert vgf.resources.getAliasGroupId(resource_index) is None
            config = vgf.resources.getScaledTensorConfigHandle(resource_index)
            assert config is not None
            assert list(vgf.resources.getScaledTensorConfigBlockShape(config)) == [32]
            assert (
                vgf.resources.getScaledTensorConfigScaleFormat(config)
                == vgfpy.ScaleFormat.E8M0
            )

        assert bindings[0] != bindings[1]
        for handle, expected in [
            (vgf.sequence.getSegmentInputBindingSlotsHandle(0), {0: bindings[0]}),
            (vgf.sequence.getSegmentOutputBindingSlotsHandle(0), {1: bindings[1]}),
            (vgf.sequence.getDescriptorBindingSlotsHandle(0, 0), bindings),
        ]:
            assert vgf.sequence.getBindingsSize(handle) == len(expected)
            assert binding_slots_by_binding(vgf, handle) == expected


@pytest.mark.parametrize(
    "mlir, diagnostic",
    [
        pytest.param(
            scaled_passthrough_mlir(alias_input_output=True),
            "block scaled tensor resource alias groups are not supported",
            id="scaled-alias",
        ),
        pytest.param(
            scaled_passthrough_mlir(scale_type="f32"),
            "unsupported scale type",
            id="unsupported-scale",
        ),
        pytest.param(
            scaled_passthrough_mlir(block=4294967296, size=4294967296),
            "scaled tensor block dimension exceeds VGF uint32 range",
            id="block-dimension-overflow",
        ),
    ],
)
@pytest.mark.usefixtures("block_scaled_tensors_enabled")
def test_unsupported_scaled_graph_resources(
    model_converter_exe_path, tmp_path, mlir, diagnostic
):
    input_path = tmp_path / "input.mlir"
    output_path = tmp_path / "output.vgf"
    input_path.write_text(mlir)
    result = subprocess.run(
        [
            model_converter_exe_path,
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr
    assert not output_path.exists()
