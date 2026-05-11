#
# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import itertools
import json

import pytest
import vgfpy
from model_converter_helpers import converted_mlir
from vgf_decoder import VK_FORMAT_R32_SFLOAT
from vgf_decoder import VK_FORMAT_R32G32B32A32_SFLOAT

VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER = 1
VK_DESCRIPTOR_TYPE_STORAGE_IMAGE = 3
VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
VK_DESCRIPTOR_TYPE_TENSOR_ARM = 1000460000

SHADER_CASES = [
    {
        "id": "placeholder",
        "shader_language": "spirv",
        "shader_code_available": False,
        "attrs": {},
    },
    {
        "id": "glsl",
        "shader_language": "glsl",
        "shader_code_available": True,
        "attrs": {
            "shader_language": "GLSL",
            "shader_code": "#version 450\nvoid main() { }",
        },
    },
    {
        "id": "hlsl",
        "shader_language": "hlsl",
        "shader_code_available": True,
        "attrs": {
            "shader_language": "HLSL",
            "shader_code": "[numthreads(1,1,1)] void main() {}",
        },
    },
    {
        "id": "spirv",
        "shader_language": "spirv",
        "shader_code_available": True,
        "attrs": {
            "shader_language": "SPIR-V",
            "shader_code": "AwIjBwAAAQAAAAAAAAAAAA==",
        },
    },
]


def assert_module_shader(vgf, module_index, shader_case):
    if shader_case["shader_language"] == "hlsl":
        assert vgf.modules.isHLSL(module_index) is True
        assert (
            vgf.modules.hasHLSLCode(module_index)
            is shader_case["shader_code_available"]
        )
    elif shader_case["shader_language"] == "glsl":
        assert vgf.modules.isGLSL(module_index) is True
        assert (
            vgf.modules.hasGLSLCode(module_index)
            is shader_case["shader_code_available"]
        )
    else:
        assert vgf.modules.isSPIRV(module_index) is True
        assert (
            vgf.modules.hasSPIRVCode(module_index)
            is shader_case["shader_code_available"]
        )


def custom_shader_attrs(
    *,
    vk_format,
    workgroup_sizes,
    input_bindings,
    output_bindings,
    input_descriptor_sets=None,
    output_descriptor_sets=None,
    input_vk_descriptor_types=None,
    output_vk_descriptor_types=None,
    tensor_type="TENSOR",
    extra_attrs=None,
):
    input_descriptor_sets = input_descriptor_sets or [0] * len(input_bindings)
    output_descriptor_sets = output_descriptor_sets or [0] * len(output_bindings)
    input_vk_descriptor_types = input_vk_descriptor_types or [
        "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
    ] * len(input_bindings)
    output_vk_descriptor_types = output_vk_descriptor_types or [
        "VK_DESCRIPTOR_TYPE_TENSOR_ARM"
    ] * len(output_bindings)
    implementation_attrs = {
        "entry_point": "main",
        "is_vkshader": True,
        "workgroup_sizes": workgroup_sizes,
    }

    for index, binding in enumerate(input_bindings):
        implementation_attrs[f"input_{index}_binding"] = binding
        implementation_attrs[f"input_{index}_descriptorset"] = input_descriptor_sets[
            index
        ]
        implementation_attrs[f"input_{index}_type"] = tensor_type
        implementation_attrs[f"input_{index}_vkdescriptortype"] = (
            input_vk_descriptor_types[index]
        )
        implementation_attrs[f"input_{index}_vkformat"] = vk_format

    for index, binding in enumerate(output_bindings):
        implementation_attrs[f"output_{index}_binding"] = binding
        implementation_attrs[f"output_{index}_descriptorset"] = output_descriptor_sets[
            index
        ]
        implementation_attrs[f"output_{index}_type"] = tensor_type
        implementation_attrs[f"output_{index}_vkdescriptortype"] = (
            output_vk_descriptor_types[index]
        )
        implementation_attrs[f"output_{index}_vkformat"] = vk_format

    if extra_attrs is not None:
        implementation_attrs.update(extra_attrs)
    return json.dumps(json.dumps(implementation_attrs, separators=(",", ":")))


def custom_image_sampler_mlir():
    implementation_attrs = json.dumps(
        json.dumps(
            {
                "entry_point": "main",
                "input_0_binding": 0,
                "input_0_descriptorset": 0,
                "input_0_type": "Image",
                "input_0_vkdescriptortype": (
                    "VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER"
                ),
                "input_0_vkformat": "VK_FORMAT_R32G32B32A32_SFLOAT",
                "input_0_sampler": {
                    "min_filter": "VK_FILTER_LINEAR",
                    "mag_filter": "VK_FILTER_LINEAR",
                    "address_mode_u": "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER",
                    "address_mode_v": "VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER",
                    "border_color": "VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK",
                },
                "is_vkshader": True,
                "output_0_binding": 1,
                "output_0_descriptorset": 0,
                "output_0_type": "Image",
                "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_IMAGE",
                "output_0_vkformat": "VK_FORMAT_R32G32B32A32_SFLOAT",
                "workgroup_sizes": [8, 8, 1],
            },
            separators=(",", ":"),
        )
    )

    return f"""
module {{
  func.func @main(%arg0: tensor<1x8x8x4xf32> {{tf_saved_model.index_path = ["input_0"]}}) -> (tensor<1x8x8x4xf32> {{tf_saved_model.index_path = ["output_0"]}}) attributes {{tf.entry_function = {{inputs = "tosa_deserialized_input_0:0", outputs = "tosa_deserialized_output_0:0"}}, tf_saved_model.exported_names = ["tosa_deserialized"]}} {{
    %0 = tosa.custom %arg0 {{domain_name = "com.arm.VulkanCustomShader", implementation_attrs = {implementation_attrs}, operator_name = "test_image_sampler_shader"}} : (tensor<1x8x8x4xf32>) -> tensor<1x8x8x4xf32>
    return %0 : tensor<1x8x8x4xf32>
  }}
}}
"""


def custom_op(
    result,
    operands,
    operand_types,
    result_type,
    operator_name,
    shader_case,
    vk_format,
    workgroup_sizes,
    input_vk_descriptor_types=None,
    output_vk_descriptor_types=None,
):
    implementation_attrs = custom_shader_attrs(
        vk_format=vk_format,
        workgroup_sizes=workgroup_sizes,
        input_bindings=[0, 1],
        output_bindings=[2],
        input_vk_descriptor_types=input_vk_descriptor_types,
        output_vk_descriptor_types=output_vk_descriptor_types,
        extra_attrs=shader_case["attrs"],
    )
    operands = ", ".join(operands)
    operand_types = ", ".join(operand_types)
    return f"""
    {result} = tosa.custom {operands} {{domain_name = "com.arm.VulkanCustomShader", implementation_attrs = {implementation_attrs}, operator_name = "{operator_name}"}} : ({operand_types}) -> {result_type}
"""


def custom_binding_mlir(input_descriptor_set, output_descriptor_set):
    implementation_attrs = custom_shader_attrs(
        vk_format="VK_FORMAT_R32_SFLOAT",
        workgroup_sizes=[64, 1, 1],
        input_bindings=[0],
        output_bindings=[1],
        input_descriptor_sets=[input_descriptor_set],
        output_descriptor_sets=[output_descriptor_set],
        tensor_type="Tensor",
        extra_attrs={"push_constants": ""},
    )

    return f"""
module attributes {{tf_saved_model.semantics, tosa.description = "TOSA FBS Converted", tosa.fbs_version = "1.1.0d"}} {{
  func.func @main(%arg0: tensor<10xf32> {{tf_saved_model.index_path = ["input_0"]}}, %arg1: tensor<10xf32> {{tf_saved_model.index_path = ["input_1"]}}) -> (tensor<10xf32> {{tf_saved_model.index_path = ["output_0"]}}) attributes {{tf.entry_function = {{inputs = "tosa_deserialized_input_0:0,tosa_deserialized_input_1:0", outputs = "tosa_deserialized_output_0:0"}}, tf_saved_model.exported_names = ["tosa_deserialized"]}} {{
    %0 = tosa.add %arg0, %arg1 : (tensor<10xf32>, tensor<10xf32>) -> tensor<10xf32>
    %1 = tosa.custom %0 {{domain_name = "com.arm.VulkanCustomShader", implementation_attrs = {implementation_attrs}, operator_name = "thribrary.threee_pleee"}} : (tensor<10xf32>) -> tensor<10xf32>
    return %1 : tensor<10xf32>
  }}
}}
"""


def single_custom_mlir(shader_case):
    tensor_type = "tensor<1x16x16x16xf32>"
    custom = custom_op(
        "%0",
        ["%arg0", "%arg1"],
        [tensor_type, tensor_type],
        tensor_type,
        f"test_{shader_case['id']}_shader",
        shader_case,
        "VK_FORMAT_R32_SFLOAT",
        [16, 16, 16],
    )

    return f"""
module {{
  func.func @main(%arg0: tensor<1x16x16x16xf32> {{tf_saved_model.index_path = ["input_1"]}}, %arg1: tensor<1x16x16x16xf32> {{tf_saved_model.index_path = ["input_2"]}}) -> (tensor<1x16x16x16xf32> {{tf_saved_model.index_path = ["single_custom_op_layer"]}}) attributes {{tf.entry_function = {{inputs = "serving_default_input_1:0,serving_default_input_2:0", outputs = "PartitionedCall:0"}}, tf_saved_model.exported_names = ["serving_default"]}} {{
{custom}
    return %0 : tensor<1x16x16x16xf32>
  }}
}}
"""


def consecutive_custom_mlir(first_shader_case, second_shader_case):
    tensor_type = "tensor<1x16x16x16xf32>"
    first_custom = custom_op(
        "%0",
        ["%arg0", "%arg1"],
        [tensor_type, tensor_type],
        tensor_type,
        f"test_{first_shader_case['id']}_shader_0",
        first_shader_case,
        "VK_FORMAT_R32_SFLOAT",
        [16, 16, 16],
    )
    second_custom = custom_op(
        "%1",
        ["%0", "%arg2"],
        [tensor_type, tensor_type],
        tensor_type,
        f"test_{second_shader_case['id']}_shader_1",
        second_shader_case,
        "VK_FORMAT_R32_SFLOAT",
        [8, 8, 16],
    )

    return f"""
module {{
  func.func @main(%arg0: tensor<1x16x16x16xf32> {{tf_saved_model.index_path = ["input_0"]}}, %arg1: tensor<1x16x16x16xf32> {{tf_saved_model.index_path = ["input_1"]}}, %arg2: tensor<1x16x16x16xf32> {{tf_saved_model.index_path = ["input_2"]}}) -> (tensor<1x16x16x16xf32> {{tf_saved_model.index_path = ["output_0"]}}) attributes {{tf.entry_function = {{inputs = "input_0,input_1,input_2", outputs = "output_0"}}, tf_saved_model.exported_names = ["serving_default"]}} {{
{first_custom}
{second_custom}
    return %1 : tensor<1x16x16x16xf32>
  }}
}}
"""


def descriptor_type_aliasing_mlir(
    first_output_descriptor_type, second_input_descriptor_type
):
    tensor_type = "tensor<1x16x16x16xf32>"
    shader_case = SHADER_CASES[0]
    first_custom = custom_op(
        "%0",
        ["%arg0", "%arg1"],
        [tensor_type, tensor_type],
        tensor_type,
        "test_descriptor_type_aliasing_shader_0",
        shader_case,
        "VK_FORMAT_R32_SFLOAT",
        [16, 16, 16],
        output_vk_descriptor_types=[first_output_descriptor_type],
    )
    second_custom = custom_op(
        "%1",
        ["%0", "%arg2"],
        [tensor_type, tensor_type],
        tensor_type,
        "test_descriptor_type_aliasing_shader_1",
        shader_case,
        "VK_FORMAT_R32_SFLOAT",
        [8, 8, 16],
        input_vk_descriptor_types=[
            second_input_descriptor_type,
            "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
        ],
    )

    return f"""
module {{
  func.func @main(%arg0: tensor<1x16x16x16xf32> {{tf_saved_model.index_path = ["input_0"]}}, %arg1: tensor<1x16x16x16xf32> {{tf_saved_model.index_path = ["input_1"]}}, %arg2: tensor<1x16x16x16xf32> {{tf_saved_model.index_path = ["input_2"]}}) -> (tensor<1x16x16x16xf32> {{tf_saved_model.index_path = ["output_0"]}}) attributes {{tf.entry_function = {{inputs = "input_0,input_1,input_2", outputs = "output_0"}}, tf_saved_model.exported_names = ["serving_default"]}} {{
{first_custom}
{second_custom}
    return %1 : tensor<1x16x16x16xf32>
  }}
}}
"""


def custom_graph_custom_mlir():
    input_type = "tensor<1x16x16x16xi8>"
    pooled_type = "tensor<1x8x8x16xi8>"
    shader_case = SHADER_CASES[0]
    first_custom = custom_op(
        "%0",
        ["%arg0", "%arg2"],
        [input_type, input_type],
        input_type,
        "test_placeholder_shader_0",
        shader_case,
        "VK_FORMAT_R8_SINT",
        [16, 16, 16],
    )
    second_custom = custom_op(
        "%2",
        ["%arg1", "%1"],
        [pooled_type, pooled_type],
        pooled_type,
        "test_placeholder_shader_1",
        shader_case,
        "VK_FORMAT_R8_SINT",
        [8, 8, 16],
    )

    return f"""
module {{
  func.func @main(%arg0: tensor<1x16x16x16xi8> {{tf_saved_model.index_path = ["input_1"]}}, %arg1: tensor<1x8x8x16xi8> {{tf_saved_model.index_path = ["input_3"]}}, %arg2: tensor<1x16x16x16xi8> {{tf_saved_model.index_path = ["input_2"]}}) -> (tensor<1x8x8x16xi8> {{tf_saved_model.index_path = ["tf.quantization.fake_quant_with_min_max_vars_4"]}}) attributes {{tf.entry_function = {{inputs = "input_0,input_1,input_2", outputs = "output_0"}}, tf_saved_model.exported_names = ["serving_default"]}} {{
{first_custom}
    %1 = tosa.max_pool2d %0 {{kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>}} : (tensor<1x16x16x16xi8>) -> tensor<1x8x8x16xi8>
{second_custom}
    return %2 : tensor<1x8x8x16xi8>
  }}
}}
"""


@pytest.mark.parametrize(
    "shader_case",
    [pytest.param(shader_case, id=shader_case["id"]) for shader_case in SHADER_CASES],
)
def test_custom_shader_modules(model_converter_exe_path, shader_case):
    mlir = single_custom_mlir(shader_case)
    with converted_mlir(model_converter_exe_path, mlir) as vgf:
        assert vgf.modules.size() == 1
        assert vgf.modules.getModuleType(0) == vgfpy.ModuleType.Compute
        assert_module_shader(vgf, 0, shader_case)

        assert vgf.sequence.modelSequenceTableSize() == 1
        assert vgf.sequence.getSegmentType(0) == vgfpy.ModuleType.Compute
        assert vgf.sequence.getSegmentName(0) == "compute_segment_0"
        assert list(vgf.sequence.getSegmentDispatchShape(0)) == [16, 16, 16]

        assert vgf.resources.size() == 3
        assert vgf.resources.getCategory(0) == vgfpy.ResourceCategory.Input
        assert int(vgf.resources.getVkFormat(0)) == VK_FORMAT_R32_SFLOAT
        assert list(vgf.resources.getTensorShape(0)) == [1, 16, 16, 16]
        assert vgf.resources.getCategory(1) == vgfpy.ResourceCategory.Input
        assert int(vgf.resources.getVkFormat(1)) == VK_FORMAT_R32_SFLOAT
        assert list(vgf.resources.getTensorShape(1)) == [1, 16, 16, 16]
        assert vgf.resources.getCategory(2) == vgfpy.ResourceCategory.Output
        assert int(vgf.resources.getVkFormat(2)) == VK_FORMAT_R32_SFLOAT
        assert list(vgf.resources.getTensorShape(2)) == [1, 16, 16, 16]

        assert vgf.constants.size() == 0


def test_custom_image_sampler(model_converter_exe_path):
    with converted_mlir(model_converter_exe_path, custom_image_sampler_mlir()) as vgf:
        assert vgf.modules.size() == 1
        assert vgf.modules.getModuleType(0) == vgfpy.ModuleType.Compute
        assert vgf.modules.isSPIRV(0) is True
        assert vgf.modules.hasSPIRVCode(0) is False

        assert vgf.resources.size() == 2
        assert vgf.resources.getCategory(0) == vgfpy.ResourceCategory.Input
        assert (
            vgf.resources.getDescriptorType(0)
            == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
        )
        assert int(vgf.resources.getVkFormat(0)) == VK_FORMAT_R32G32B32A32_SFLOAT
        assert list(vgf.resources.getTensorShape(0)) == [1, 8, 8, 4]
        assert vgf.resources.getCategory(1) == vgfpy.ResourceCategory.Output
        assert vgf.resources.getDescriptorType(1) == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
        assert int(vgf.resources.getVkFormat(1)) == VK_FORMAT_R32G32B32A32_SFLOAT
        assert list(vgf.resources.getTensorShape(1)) == [1, 8, 8, 4]

        assert vgf.sequence.modelSequenceTableSize() == 1
        assert vgf.sequence.getSegmentType(0) == vgfpy.ModuleType.Compute
        assert vgf.sequence.getSegmentName(0) == "compute_segment_0"
        assert list(vgf.sequence.getSegmentDispatchShape(0)) == [8, 8, 1]

        inputs = vgf.sequence.getSegmentInputBindingSlotsHandle(0)
        assert vgf.sequence.getBindingsSize(inputs) == 1
        assert vgf.sequence.getBindingSlotBinding(inputs, 0) == 0
        assert vgf.sequence.getBindingSlotMrtIndex(inputs, 0) == 0
        outputs = vgf.sequence.getSegmentOutputBindingSlotsHandle(0)
        assert vgf.sequence.getBindingsSize(outputs) == 1
        assert vgf.sequence.getBindingSlotBinding(outputs, 0) == 1
        assert vgf.sequence.getBindingSlotMrtIndex(outputs, 0) == 1

        assert vgf.sequence.getSegmentDescriptorSetInfosSize(0) == 1
        assert vgf.sequence.getSegmentDescriptorSetIndex(0, 0) == 0
        descriptor_bindings = vgf.sequence.getDescriptorBindingSlotsHandle(0, 0)
        assert vgf.sequence.getBindingsSize(descriptor_bindings) == 2
        assert vgf.sequence.getBindingSlotBinding(descriptor_bindings, 0) == 0
        assert vgf.sequence.getBindingSlotMrtIndex(descriptor_bindings, 0) == 0
        assert vgf.sequence.getBindingSlotBinding(descriptor_bindings, 1) == 1
        assert vgf.sequence.getBindingSlotMrtIndex(descriptor_bindings, 1) == 1

        assert vgf.constants.size() == 0


@pytest.mark.parametrize(
    "first_shader_case, second_shader_case",
    [
        pytest.param(
            first_shader_case,
            second_shader_case,
            id=f"{first_shader_case['id']}-{second_shader_case['id']}",
        )
        for first_shader_case, second_shader_case in itertools.product(
            SHADER_CASES, repeat=2
        )
    ],
)
def test_consecutive_custom_shader_modules(
    model_converter_exe_path,
    first_shader_case,
    second_shader_case,
):
    mlir = consecutive_custom_mlir(first_shader_case, second_shader_case)
    with converted_mlir(model_converter_exe_path, mlir) as vgf:
        assert vgf.modules.size() == 2
        assert vgf.modules.getModuleType(0) == vgfpy.ModuleType.Compute
        assert_module_shader(vgf, 0, first_shader_case)
        assert vgf.modules.getModuleType(1) == vgfpy.ModuleType.Compute
        assert_module_shader(vgf, 1, second_shader_case)

        assert vgf.sequence.modelSequenceTableSize() == 2
        assert vgf.sequence.getSegmentType(0) == vgfpy.ModuleType.Compute
        assert vgf.sequence.getSegmentName(0) == "compute_segment_0"
        assert list(vgf.sequence.getSegmentDispatchShape(0)) == [16, 16, 16]
        assert vgf.sequence.getSegmentType(1) == vgfpy.ModuleType.Compute
        assert vgf.sequence.getSegmentName(1) == "compute_segment_1"
        assert list(vgf.sequence.getSegmentDispatchShape(1)) == [8, 8, 16]


def test_descriptor_type_change_creates_alias_group_for_intermediate(
    model_converter_exe_path,
):
    with converted_mlir(
        model_converter_exe_path,
        descriptor_type_aliasing_mlir(
            "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
            "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
        ),
    ) as vgf:
        assert vgf.resources.size() == 6
        assert vgf.resources.getCategory(3) == vgfpy.ResourceCategory.Intermediate
        assert vgf.resources.getDescriptorType(3) == VK_DESCRIPTOR_TYPE_TENSOR_ARM
        assert vgf.resources.getCategory(4) == vgfpy.ResourceCategory.Intermediate
        assert vgf.resources.getDescriptorType(4) == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER

        shared_alias_group = vgf.resources.getAliasGroupId(3)
        assert shared_alias_group is not None
        assert vgf.resources.getAliasGroupId(4) == shared_alias_group

        assert vgf.resources.getAliasGroupId(0) is None
        assert vgf.resources.getAliasGroupId(1) is None
        assert vgf.resources.getAliasGroupId(2) is None
        assert vgf.resources.getAliasGroupId(5) is None

        first_outputs = vgf.sequence.getSegmentOutputBindingSlotsHandle(0)
        assert vgf.sequence.getBindingsSize(first_outputs) == 1
        assert vgf.sequence.getBindingSlotBinding(first_outputs, 0) == 3
        assert vgf.sequence.getBindingSlotMrtIndex(first_outputs, 0) == 3

        second_inputs = vgf.sequence.getSegmentInputBindingSlotsHandle(1)
        assert vgf.sequence.getBindingsSize(second_inputs) == 2
        second_input_bindings = sorted(
            (
                vgf.sequence.getBindingSlotBinding(second_inputs, binding_index),
                vgf.sequence.getBindingSlotMrtIndex(second_inputs, binding_index),
            )
            for binding_index in range(vgf.sequence.getBindingsSize(second_inputs))
        )
        assert second_input_bindings == [(2, 2), (3, 4)]


def test_descriptor_type_match_does_not_create_alias_group(
    model_converter_exe_path,
):
    with converted_mlir(
        model_converter_exe_path,
        descriptor_type_aliasing_mlir(
            "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
            "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
        ),
    ) as vgf:
        assert vgf.resources.size() == 5
        assert vgf.resources.getCategory(3) == vgfpy.ResourceCategory.Intermediate
        assert vgf.resources.getDescriptorType(3) == VK_DESCRIPTOR_TYPE_TENSOR_ARM

        for resource_index in range(vgf.resources.size()):
            assert vgf.resources.getAliasGroupId(resource_index) is None

        first_outputs = vgf.sequence.getSegmentOutputBindingSlotsHandle(0)
        assert vgf.sequence.getBindingsSize(first_outputs) == 1
        assert vgf.sequence.getBindingSlotBinding(first_outputs, 0) == 3
        assert vgf.sequence.getBindingSlotMrtIndex(first_outputs, 0) == 3

        second_inputs = vgf.sequence.getSegmentInputBindingSlotsHandle(1)
        assert vgf.sequence.getBindingsSize(second_inputs) == 2
        second_input_bindings = sorted(
            (
                vgf.sequence.getBindingSlotBinding(second_inputs, binding_index),
                vgf.sequence.getBindingSlotMrtIndex(second_inputs, binding_index),
            )
            for binding_index in range(vgf.sequence.getBindingsSize(second_inputs))
        )
        assert second_input_bindings == [(2, 2), (3, 3)]


def test_custom_graph_custom_segments(model_converter_exe_path):
    with converted_mlir(model_converter_exe_path, custom_graph_custom_mlir()) as vgf:
        assert vgf.modules.size() == 3
        assert vgf.modules.getModuleType(0) == vgfpy.ModuleType.Compute
        assert vgf.modules.isSPIRV(0) is True
        assert vgf.modules.hasSPIRVCode(0) is False
        assert vgf.modules.getModuleType(1) == vgfpy.ModuleType.Graph
        assert vgf.modules.isSPIRV(1) is True
        assert vgf.modules.hasSPIRVCode(1) is True
        assert vgf.modules.getModuleType(2) == vgfpy.ModuleType.Compute
        assert vgf.modules.isSPIRV(2) is True
        assert vgf.modules.hasSPIRVCode(2) is False

        assert vgf.sequence.modelSequenceTableSize() == 3
        assert vgf.sequence.getSegmentType(0) == vgfpy.ModuleType.Compute
        assert vgf.sequence.getSegmentName(0) == "compute_segment_0"
        assert list(vgf.sequence.getSegmentDispatchShape(0)) == [16, 16, 16]
        assert vgf.sequence.getSegmentType(1) == vgfpy.ModuleType.Graph
        assert vgf.sequence.getSegmentName(1) == "graph_segment_0"
        assert list(vgf.sequence.getSegmentDispatchShape(1)) == [0, 0, 0]
        assert vgf.sequence.getSegmentType(2) == vgfpy.ModuleType.Compute
        assert vgf.sequence.getSegmentName(2) == "compute_segment_1"
        assert list(vgf.sequence.getSegmentDispatchShape(2)) == [8, 8, 16]


def test_custom_binding_on_compute_segment(model_converter_exe_path):
    with converted_mlir(model_converter_exe_path, custom_binding_mlir(1, 1)) as vgf:
        inputs = vgf.sequence.getSegmentInputBindingSlotsHandle(1)
        assert vgf.sequence.getBindingsSize(inputs) == 1
        assert vgf.sequence.getBindingSlotBinding(inputs, 0) == 2
        assert vgf.sequence.getBindingSlotMrtIndex(inputs, 0) == 2

        outputs = vgf.sequence.getSegmentOutputBindingSlotsHandle(1)
        assert vgf.sequence.getBindingsSize(outputs) == 1
        assert vgf.sequence.getBindingSlotBinding(outputs, 0) == 3
        assert vgf.sequence.getBindingSlotMrtIndex(outputs, 0) == 3

        assert vgf.sequence.getSegmentDescriptorSetInfosSize(1) == 1
        assert vgf.sequence.getSegmentDescriptorSetIndex(1, 0) == 1
        descriptor_bindings = vgf.sequence.getDescriptorBindingSlotsHandle(1, 0)
        assert vgf.sequence.getBindingsSize(descriptor_bindings) == 2
        assert vgf.sequence.getBindingSlotBinding(descriptor_bindings, 0) == 0
        assert vgf.sequence.getBindingSlotMrtIndex(descriptor_bindings, 0) == 2
        assert vgf.sequence.getBindingSlotBinding(descriptor_bindings, 1) == 1
        assert vgf.sequence.getBindingSlotMrtIndex(descriptor_bindings, 1) == 3


def test_custom_descriptor_set_indices(model_converter_exe_path):
    with converted_mlir(model_converter_exe_path, custom_binding_mlir(7, 9)) as vgf:
        assert vgf.sequence.getSegmentDescriptorSetInfosSize(1) == 2
        assert vgf.sequence.getSegmentDescriptorSetIndex(1, 0) == 7
        descriptor_bindings = vgf.sequence.getDescriptorBindingSlotsHandle(1, 0)
        assert vgf.sequence.getBindingsSize(descriptor_bindings) == 1
        assert vgf.sequence.getBindingSlotBinding(descriptor_bindings, 0) == 0
        assert vgf.sequence.getBindingSlotMrtIndex(descriptor_bindings, 0) == 2
        assert vgf.sequence.getSegmentDescriptorSetIndex(1, 1) == 9
        descriptor_bindings = vgf.sequence.getDescriptorBindingSlotsHandle(1, 1)
        assert vgf.sequence.getBindingsSize(descriptor_bindings) == 1
        assert vgf.sequence.getBindingSlotBinding(descriptor_bindings, 0) == 1
        assert vgf.sequence.getBindingSlotMrtIndex(descriptor_bindings, 0) == 3
