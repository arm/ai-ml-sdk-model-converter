#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import json

import vgfpy
from model_converter_helpers import converted_mlir
from test_model_converter_core_v100 import main_mlir
from vgf_decoder import VK_FORMAT_R8_SINT

OP_GRAPH_CONSTANT_ARM = 4181


def vulkan_custom_shader_attrs():
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
                "input_0_vkformat": "VK_FORMAT_R8_SINT",
                "output_0_binding": 1,
                "output_0_descriptorset": 0,
                "output_0_type": "TENSOR",
                "output_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_TENSOR_ARM",
                "output_0_vkformat": "VK_FORMAT_R8_SINT",
            },
            separators=(",", ":"),
        )
    )


def graph_constant_ids(spirv_words):
    ids = []
    words = list(spirv_words)
    offset = 5
    while offset < len(words):
        word = words[offset]
        word_count = word >> 16
        opcode = word & 0xFFFF
        if opcode == OP_GRAPH_CONSTANT_ARM:
            ids.append(words[offset + 3])
        offset += word_count
    return ids


def partitioned_table_mlir(shared_constant=False):
    shared_table = ""
    shared_return = "%4"
    if shared_constant:
        shared_table = (
            "    %5 = tosa.table %4, %c0 : "
            "(tensor<4xi8>, tensor<256xi8>) -> tensor<4xi8>\n"
        )
        shared_return = "%5"

    body = f"""    %c0 = "tosa.const"() {{values = dense<0> : tensor<256xi8>}} : () -> tensor<256xi8>
    %c1 = "tosa.const"() {{values = dense<1> : tensor<256xi8>}} : () -> tensor<256xi8>
    %c2 = "tosa.const"() {{values = dense<2> : tensor<256xi8>}} : () -> tensor<256xi8>
    %0 = tosa.table %arg0, %c0 : (tensor<4xi8>, tensor<256xi8>) -> tensor<4xi8>
    %1 = tosa.table %0, %c2 : (tensor<4xi8>, tensor<256xi8>) -> tensor<4xi8>
    %2 = tosa.custom %1 {{domain_name = "com.arm.VulkanCustomShader", implementation_attrs = {vulkan_custom_shader_attrs()}, operator_name = "partition_barrier"}} : (tensor<4xi8>) -> tensor<4xi8>
    %4 = tosa.table %2, %c1 : (tensor<4xi8>, tensor<256xi8>) -> tensor<4xi8>
{shared_table}"""

    return main_mlir(
        arguments='%arg0: tensor<4xi8> {tf_saved_model.index_path = ["input_0"]}',
        result='(tensor<4xi8> {tf_saved_model.index_path = ["output_0"]})',
        entry_inputs="input_0",
        entry_outputs="output_0",
        body=body,
        return_value=f"{shared_return} : tensor<4xi8>",
    )


def graph_segment_constant_indexes(vgf):
    segment_indexes = []
    module_ids = []
    for segment_index in range(vgf.sequence.modelSequenceTableSize()):
        if vgf.sequence.getSegmentType(segment_index) != vgfpy.ModuleType.Graph:
            continue
        module_index = vgf.sequence.getSegmentModuleIndex(segment_index)
        segment_indexes.append(
            list(vgf.sequence.getSegmentConstantIndexes(segment_index))
        )
        module_ids.append(
            sorted(graph_constant_ids(vgf.modules.getSPIRVModuleCode(module_index)))
        )
    return segment_indexes, module_ids


def segment_constant_indexes(vgf):
    indexes = []
    for segment_index in range(vgf.sequence.modelSequenceTableSize()):
        segment_constants = vgf.sequence.getSegmentConstantIndexes(segment_index)
        indexes.append([] if segment_constants is None else list(segment_constants))
    return indexes


def assert_constant_table(vgf, expected_values):
    assert vgf.constants.size() == len(expected_values)
    for index, expected_value in enumerate(expected_values):
        assert vgf.constants.getConstantMrtIndex(index) == index
        assert vgf.resources.getCategory(index) == vgfpy.ResourceCategory.Constant
        assert int(vgf.resources.getVkFormat(index)) == VK_FORMAT_R8_SINT
        assert list(vgf.resources.getTensorShape(index)) == [256]
        assert (
            vgf.constants.getConstant(index).tobytes() == bytes([expected_value]) * 256
        )


def test_partitioned_graph_constants_use_global_sparse_ids(model_converter_exe_path):
    with converted_mlir(model_converter_exe_path, partitioned_table_mlir()) as vgf:
        segment_constants, module_ids = graph_segment_constant_indexes(vgf)

        assert_constant_table(vgf, [0, 1, 2])
        assert segment_constant_indexes(vgf) == [[0, 2], [], [1]]
        assert segment_constants == [[0, 2], [1]]
        assert module_ids == [[0, 2], [1]]


def test_rematerialized_shared_graph_constant_serializes_once(model_converter_exe_path):
    with converted_mlir(
        model_converter_exe_path, partitioned_table_mlir(shared_constant=True)
    ) as vgf:
        segment_constants, module_ids = graph_segment_constant_indexes(vgf)

        assert_constant_table(vgf, [0, 1, 2])
        assert segment_constant_indexes(vgf) == [[0, 2], [], [0, 1]]
        assert segment_constants == [[0, 2], [0, 1]]
        assert module_ids == [[0, 2], [0, 1]]
