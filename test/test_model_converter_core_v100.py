#
# SPDX-FileCopyrightText: Copyright 2023-2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import pytest
import vgfpy
from model_converter_helpers import converted_mlir
from vgf_decoder import VK_FORMAT_R16_SINT
from vgf_decoder import VK_FORMAT_R32_SFLOAT
from vgf_decoder import VK_FORMAT_R32_SINT
from vgf_decoder import VK_FORMAT_R8_SINT

CONV2D_CONSTANT_VALUES = [
    bytes.fromhex(
        "51266ef1c48a006465b17e2a633507f73841a3812cbf384d83c5cd09263fe5baa9eb25f5d757332ef4d2b64e2f9d0868affb5055023e127d7f70f4016857b4c6bd01a70cf2df535414454067821f1a2d0658e9b27c026d4714ed4962ff1b3c8c0d372139b749e339a41c033e530d686bcde610ea33c2a7cd81f52c658bec1e00abee94d130d4abb9e8527f8a1e7d0b7cd49593961d7eb688470903c14af52d1777e06042d12c2df009eba15e9c4b7b998d649ec406ecfa0a83702e1d06913b433af158fef959bdeda1651435eff87fbb7dc2c09e3b4a923a373c1ec19840c8f8fa4e1ae0ee20c30204a81b58e023d0337d7fc0280a1423cb6a5721818c8bb73aa344a68e7f09c9efa0b2e67ec9f460245457f4fa41066cefc3d9e4e5916a85450ebbad2cc62fbb3699414f09240cf22fb72277db60df3db2a466ba3c49c38527d85ac3dd512f263cd760e4d8b29fd13f707d25b613d481db5db8046dae7ead8dc0bacaeb6d7663936b91a65da22cac9c3732ff5325f1e28b0d4be7785dc9303af747a7fe182bfb7ee32fb0e0f329df088d33deea295e597fa2e6eb4a2a944b9bb9ed6c42f6b85c00b09e756a44f9cb3026025b554d338751f317db17ca96545899b92ee597fa06bdd7b38a0cc1f6b567f4856f3df9814130f495f931cf14e4c3f11ef425bedc2dd3496afc97633496c65ba2f9ee0e51cb87ea10336bd84d17eee32601fc7c3946da41f6626b47852ac25b756c5acf869ddae68e11001b741d7fae3976de9b64a240dc02ace72c8ccac49382b6570f0bf6750fae9efee38ae899dcef77863556d7a0e8a7921860372a476c34bf213914146f87548a6f1f8a8c8ab373bf92d50c2121fac912d0aa16955a968344df728ca791c2737f5e3c67e3d2e99a79e6870ab62c8a7e164dfee7a120d3f32a1224b51f812712e2915031670acb1a4bf32539970b556689bd23dd5dd7f162beeb2d097f68cb872ebcdc2d464a8ed718ec66ac52dcd5f7d2cd82c634a00082f7bdb35bddf5eb3d61c525b8dfb954b2601d81da71ea1ca088dfe1f40ae309be51fac36304c7e4654690f1954973f98fe77f3f7612ab07bd4296bc00b00919d737c4a4e2a51f9d0982cbfce8af5d5eb8a23b8f2e317094878dfc9d41c0c8d4fbb6abf1e4505384963a26c3a647df670f4b878fd4dd85b22035bc4cc3c06acf60ee9483b0c84f1f7a71eca8bf355afdabe4bd6d936aa768e45277ed99f68879358364b6a8c7235c747fc0066e8342a642373c3b0660e4110850c978e113751ea7ba47ec6ab18c6487c4c633f8b87269f0975bec548c98de7af37f9bbeed12055e39fb933500d9cd5063681717fbb950242f8269861b15d3e2592c7f539a076c50b673dc0a91b07ebde6efe0c5d9e4ab8a1c03173c75a699992030bbae302d39b027f540b2a3d906cacf40c1727b66"
    ),
]

DOUBLE_CONV2D_CONSTANT_VALUES = [
    bytes.fromhex(
        "00000058005b0000000000e200006400760000000000000000006c5600000000000055750000005e000000000000000000380000000971000000000000af000000b70000000000001c0000007167000000000067000000780000000000000040000000d3000000227d0000000000007e000000000000be0000009c000000a300007300005200000000000078000000007a00000000850000007a000000fb00000061000000000000115d0000006a000000006d000000000000000057000000e5000000aa5400000000000034000000"
    ),
    bytes.fromhex(
        "00003758005b00580000250000cc00007600e2004a4f000000006c560000002500de0075470000007a004c0000"
    ),
]


def main_mlir(
    *,
    arguments,
    result,
    body,
    return_value,
    entry_inputs,
    entry_outputs,
    module_attrs=None,
    exported_name=None,
):
    module_header = "module"
    if module_attrs is not None:
        module_header += f" attributes {{{module_attrs}}}"

    function_attrs = (
        f'tf.entry_function = {{inputs = "{entry_inputs}", '
        f'outputs = "{entry_outputs}"}}'
    )
    if exported_name is not None:
        function_attrs += f', tf_saved_model.exported_names = ["{exported_name}"]'

    return f"""
{module_header} {{
  func.func @main({arguments}) -> {result} attributes {{{function_attrs}}} {{
{body}
    return {return_value}
  }}
}}
"""


def rescale_mlir():
    return main_mlir(
        module_attrs='tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32',
        arguments='%arg0: tensor<1x1xi32> {tf_saved_model.index_path = ["x"]}',
        result='(tensor<1x1xi16> {tf_saved_model.index_path = ["model_input"]})',
        entry_inputs="serving_default_x:0",
        entry_outputs="StatefulPartitionedCall:0",
        exported_name="serving_default",
        body="""    %1 = "tosa.const"() {values = dense<[8]> : tensor<1xi32>} : () -> tensor<1xi32>
    %2 = "tosa.const"() {values = dense<[23]> : tensor<1xi8>} : () -> tensor<1xi8>
    %3 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %4 = "tosa.const"() {values = dense<0> : tensor<1xi16>} : () -> tensor<1xi16>
    %5 = tosa.rescale %arg0, %1, %2, %3, %4 {rounding_mode = DOUBLE_ROUND, per_channel = false, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<1x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x1xi16>""",
        return_value="%5 : tensor<1x1xi16>",
    )


def add_mlir():
    return main_mlir(
        arguments='%arg0: tensor<f32> {tf_saved_model.index_path = ["input_1"]}, %arg1: tensor<f32> {tf_saved_model.index_path = ["input_2"]}',
        result='(tensor<f32> {tf_saved_model.index_path = ["single_custom_op_layer"]})',
        entry_inputs="serving_default_input_1:0,serving_default_input_2:0",
        entry_outputs="PartitionedCall:0",
        exported_name="serving_default",
        body="    %0 = tosa.add %arg0, %arg1 : (tensor<f32>, tensor<f32>) -> tensor<f32>",
        return_value="%0 : tensor<f32>",
    )


def direct_passthrough_mlir():
    return main_mlir(
        arguments='%arg0: tensor<49x38x55xi16> {tf_saved_model.index_path = ["input_0"]}',
        result='(tensor<49x38x55xi16> {tf_saved_model.index_path = ["output_0"]})',
        entry_inputs="tosa_deserialized_input_0:0",
        entry_outputs="tosa_deserialized_output_0:0",
        exported_name="tosa_deserialized",
        body="",
        return_value="%arg0 : tensor<49x38x55xi16>",
    )


def conv2d_mlir():
    body = """
    %0 = "tosa.const"() {values = dense<0> : tensor<16xi32>} : () -> tensor<16xi32>
    %1 = "tosa.const"() {values = dense<"0x__CONV2D_CONSTANT__"> : tensor<16x2x2x16xi8>} : () -> tensor<16x2x2x16xi8>
    %2 = "tosa.const"() {values = dense<[1931928506, 1951328493, 1755276611, 1934572170, 1932872072, 1944302913, 1932623941, 1949223780, 1952764318, 1948369062, 1947668025, 1899102491, 1913241564, 1948406415, 1912688338, 1935896203]> : tensor<16xi32>} : () -> tensor<16xi32>
    %3 = "tosa.const"() {values = dense<42> : tensor<16xi8>} : () -> tensor<16xi8>
    %4 = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
    %5 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %6 = tosa.conv2d %arg0, %1, %0, %4, %5 {dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>, acc_type = i32} : (tensor<1x16x16x16xi8>, tensor<16x2x2x16xi8>, tensor<16xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x16xi32>
    %7 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %8 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %9 = tosa.rescale %6, %2, %3, %7, %8 {rounding_mode = DOUBLE_ROUND, per_channel = true, scale32 = true,  input_unsigned = false, output_unsigned = false} : (tensor<1x8x8x16xi32>, tensor<16xi32>, tensor<16xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
""".replace("__CONV2D_CONSTANT__", CONV2D_CONSTANT_VALUES[0].hex().upper())

    return main_mlir(
        module_attrs='tf_saved_model.semantics, tfl.description = "MLIR Converted.", tfl.schema_version = 3 : i32',
        arguments='%arg0: tensor<1x16x16x16xi8> {tf_saved_model.index_path = ["input_3"]}',
        result='(tensor<1x8x8x16xi8> {tf_saved_model.index_path = ["conv2d"]})',
        entry_inputs="serving_default_input_3:0",
        entry_outputs="StatefulPartitionedCall:0",
        exported_name="serving_default",
        body=body,
        return_value="%9 : tensor<1x8x8x16xi8>",
    )


def maxpool_mlir():
    return main_mlir(
        module_attrs='tfl.description = "TOCO Converted.", tfl.schema_version = 3 : i32',
        arguments="%arg0: tensor<1x16x16x16xi8>",
        result="tensor<1x8x8x16xi8>",
        entry_inputs="data/Placeholder",
        entry_outputs="pool0/max_pooling2d/MaxPool",
        body="    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>, nan_mode = IGNORE} : (tensor<1x16x16x16xi8>) -> tensor<1x8x8x16xi8>",
        return_value="%0 : tensor<1x8x8x16xi8>",
    )


def double_conv2d_mlir():
    body = """
    %0 = "tosa.const"() {values = dense<"0x__DOUBLE_CONV2D_CONSTANT_0__"> : tensor<1x3x3x23xi8>} : () -> tensor<1x3x3x23xi8>
    %1 = "tosa.const"() {values = dense<12> : tensor<1xi32>} : () -> tensor<1xi32>
    %2 = "tosa.const"() {values = dense<"0x__DOUBLE_CONV2D_CONSTANT_1__"> : tensor<5x3x3x1xi8>} : () -> tensor<5x3x3x1xi8>
    %3 = "tosa.const"() {values = dense<12> : tensor<5xi32>} : () -> tensor<5xi32>
    %4 = "tosa.const"() {values = dense<2026291432> : tensor<1xi32>} : () -> tensor<1xi32>
    %5 = "tosa.const"() {values = dense<40> : tensor<1xi8>} : () -> tensor<1xi8>
    %6 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %7 = tosa.conv2d %arg0, %0, %1, %6, %6 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, acc_type = i32} : (tensor<1x32x32x23xi8>, tensor<1x3x3x23xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x32x32x1xi32>
    %8 = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
    %9 = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
    %10 = tosa.rescale %7, %4, %5, %9, %8 {rounding_mode = DOUBLE_ROUND, per_channel = true, scale32 = true, input_unsigned = false, output_unsigned = false} : (tensor<1x32x32x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x32x32x1xi8>
    %11 = tosa.conv2d %10, %2, %3, %8, %8 {dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, acc_type = i32} : (tensor<1x32x32x1xi8>, tensor<5x3x3x1xi8>, tensor<5xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x32x32x5xi32>
""".replace(
        "__DOUBLE_CONV2D_CONSTANT_0__",
        DOUBLE_CONV2D_CONSTANT_VALUES[0].hex().upper(),
    ).replace(
        "__DOUBLE_CONV2D_CONSTANT_1__",
        DOUBLE_CONV2D_CONSTANT_VALUES[1].hex().upper(),
    )

    return main_mlir(
        module_attrs='tf_saved_model.semantics, tosa.description = "Tosa FBS Converted", tosa.fbs_version = "0.60.0"',
        arguments='%arg0: tensor<1x32x32x23xi8> {tf_saved_model.index_path = ["input_0"]}',
        result='(tensor<1x32x32x5xi32> {tf_saved_model.index_path = ["output_0"]})',
        entry_inputs="tosa_deserialized_input_0:0",
        entry_outputs="tosa_deserialized_output_0:0",
        exported_name="tosa_deserialized",
        body=body,
        return_value="%11 : tensor<1x32x32x5xi32>",
    )


def inlined_higher_rank_constant_mlir():
    return main_mlir(
        arguments='%arg0: tensor<1x3x3x1xi8> {tf_saved_model.index_path = ["input_0"]}',
        result='(tensor<1x3x3x1xi8> {tf_saved_model.index_path = ["output_0"]})',
        entry_inputs="input_0",
        entry_outputs="output_0",
        body="""    %weight = "tosa.const"() {values = dense<[[[[0], [1], [2]], [[3], [4], [5]], [[6], [7], [8]]]]> : tensor<1x3x3x1xi8>} : ()-> tensor<1x3x3x1xi8>
    %bias = "tosa.const"() {values = dense<0> : tensor<1xi32>} : ()-> tensor<1xi32>
    %input_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : ()-> tensor<1xi8>
    %weight_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : ()-> tensor<1xi8>
    %0 = tosa.conv2d %arg0, %weight, %bias, %input_zp, %weight_zp {acc_type = i32, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>, dilation = array<i64: 1, 1>} : (tensor<1x3x3x1xi8>, tensor<1x3x3x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x3x3x1xi32>
    %1 = tosa.cast %0 : (tensor<1x3x3x1xi32>) -> tensor<1x3x3x1xi8>""",
        return_value="%1 : tensor<1x3x3x1xi8>",
    )


@pytest.mark.parametrize(
    "mlir, expected",
    [
        pytest.param(
            rescale_mlir(),
            {
                "resources": [
                    (vgfpy.ResourceCategory.Input, VK_FORMAT_R32_SINT, [1, 1]),
                    (vgfpy.ResourceCategory.Output, VK_FORMAT_R16_SINT, [1, 1]),
                ],
                "segment_inputs": [(0, 0)],
                "segment_outputs": [(1, 1)],
                "model_inputs": [(0, 0, "serving_default_x:0")],
                "model_outputs": [(1, 1, "StatefulPartitionedCall:0")],
                "constants": [],
            },
            id="rescale",
        ),
        pytest.param(
            add_mlir(),
            {
                "resources": [
                    (vgfpy.ResourceCategory.Input, VK_FORMAT_R32_SFLOAT, []),
                    (vgfpy.ResourceCategory.Input, VK_FORMAT_R32_SFLOAT, []),
                    (vgfpy.ResourceCategory.Output, VK_FORMAT_R32_SFLOAT, []),
                ],
                "segment_inputs": [(0, 0), (1, 1)],
                "segment_outputs": [(2, 2)],
                "model_inputs": [
                    (0, 0, "serving_default_input_1:0"),
                    (1, 1, "serving_default_input_2:0"),
                ],
                "model_outputs": [(2, 2, "PartitionedCall:0")],
                "constants": [],
            },
            id="add",
        ),
        pytest.param(
            direct_passthrough_mlir(),
            {
                "resources": [
                    (
                        vgfpy.ResourceCategory.Input,
                        VK_FORMAT_R16_SINT,
                        [49, 38, 55],
                    ),
                    (
                        vgfpy.ResourceCategory.Output,
                        VK_FORMAT_R16_SINT,
                        [49, 38, 55],
                    ),
                ],
                "segment_inputs": [(0, 0)],
                "segment_outputs": [(1, 1)],
                "model_inputs": [(0, 0, "tosa_deserialized_input_0:0")],
                "model_outputs": [(1, 1, "tosa_deserialized_output_0:0")],
                "constants": [],
            },
            id="direct-passthrough",
        ),
        pytest.param(
            conv2d_mlir(),
            {
                "resources": [
                    (
                        vgfpy.ResourceCategory.Constant,
                        VK_FORMAT_R8_SINT,
                        [16, 2, 2, 16],
                    ),
                    (
                        vgfpy.ResourceCategory.Input,
                        VK_FORMAT_R8_SINT,
                        [1, 16, 16, 16],
                    ),
                    (
                        vgfpy.ResourceCategory.Output,
                        VK_FORMAT_R8_SINT,
                        [1, 8, 8, 16],
                    ),
                ],
                "segment_inputs": [(0, 1)],
                "segment_outputs": [(1, 2)],
                "model_inputs": [(0, 1, "serving_default_input_3:0")],
                "model_outputs": [(1, 2, "StatefulPartitionedCall:0")],
                "constants": [
                    {
                        "mrt_index": 0,
                        "sparsity_dimension": -1,
                        "value": CONV2D_CONSTANT_VALUES[0],
                    },
                ],
            },
            id="conv2d",
        ),
        pytest.param(
            maxpool_mlir(),
            {
                "resources": [
                    (
                        vgfpy.ResourceCategory.Input,
                        VK_FORMAT_R8_SINT,
                        [1, 16, 16, 16],
                    ),
                    (
                        vgfpy.ResourceCategory.Output,
                        VK_FORMAT_R8_SINT,
                        [1, 8, 8, 16],
                    ),
                ],
                "segment_inputs": [(0, 0)],
                "segment_outputs": [(1, 1)],
                "model_inputs": [(0, 0, "data/Placeholder")],
                "model_outputs": [(1, 1, "pool0/max_pooling2d/MaxPool")],
                "constants": [],
            },
            id="maxpool",
        ),
        pytest.param(
            double_conv2d_mlir(),
            {
                "resources": [
                    (
                        vgfpy.ResourceCategory.Constant,
                        VK_FORMAT_R8_SINT,
                        [1, 3, 3, 23],
                    ),
                    (
                        vgfpy.ResourceCategory.Constant,
                        VK_FORMAT_R8_SINT,
                        [5, 3, 3, 1],
                    ),
                    (
                        vgfpy.ResourceCategory.Input,
                        VK_FORMAT_R8_SINT,
                        [1, 32, 32, 23],
                    ),
                    (
                        vgfpy.ResourceCategory.Output,
                        VK_FORMAT_R32_SINT,
                        [1, 32, 32, 5],
                    ),
                ],
                "segment_inputs": [(0, 2)],
                "segment_outputs": [(1, 3)],
                "model_inputs": [(0, 2, "tosa_deserialized_input_0:0")],
                "model_outputs": [(1, 3, "tosa_deserialized_output_0:0")],
                "constants": [
                    {
                        "mrt_index": 0,
                        "sparsity_dimension": 3,
                        "value": DOUBLE_CONV2D_CONSTANT_VALUES[0],
                    },
                    {
                        "mrt_index": 1,
                        "sparsity_dimension": -1,
                        "value": DOUBLE_CONV2D_CONSTANT_VALUES[1],
                    },
                ],
            },
            id="double-conv2d",
        ),
        pytest.param(
            inlined_higher_rank_constant_mlir(),
            {
                "resources": [
                    (
                        vgfpy.ResourceCategory.Input,
                        VK_FORMAT_R8_SINT,
                        [1, 3, 3, 1],
                    ),
                    (
                        vgfpy.ResourceCategory.Output,
                        VK_FORMAT_R8_SINT,
                        [1, 3, 3, 1],
                    ),
                ],
                "segment_inputs": [(0, 0)],
                "segment_outputs": [(1, 1)],
                "model_inputs": [(0, 0, "input_0")],
                "model_outputs": [(1, 1, "output_0")],
                "constants": [],
            },
            id="inlined-higher-rank-constant",
        ),
    ],
)
def test_core_vgf_sections(model_converter_exe_path, mlir, expected):
    with converted_mlir(model_converter_exe_path, mlir) as vgf:
        assert vgf.modules.size() == 1
        assert vgf.modules.getModuleType(0) == vgfpy.ModuleType.Graph
        assert vgf.modules.isSPIRV(0) is True
        assert vgf.modules.hasSPIRVCode(0) is True

        assert_resources(vgf.resources, expected["resources"])
        assert_sequence(vgf.sequence, expected)
        assert_constants(vgf.constants, expected["constants"])


def assert_resources(resources, expected_resources):
    assert resources.size() == len(expected_resources)
    for index, (category, vk_format, shape) in enumerate(expected_resources):
        tensor_shape = resources.getTensorShape(index)
        if tensor_shape is None:
            tensor_shape = []

        assert resources.getCategory(index) == category
        assert int(resources.getVkFormat(index)) == vk_format
        assert list(tensor_shape) == shape


def assert_sequence(sequence, expected):
    assert sequence.modelSequenceTableSize() == 1
    assert sequence.getSegmentType(0) == vgfpy.ModuleType.Graph
    assert sequence.getSegmentName(0) == "graph_segment_0"
    assert list(sequence.getSegmentDispatchShape(0)) == [0, 0, 0]

    assert_bindings(
        sequence,
        sequence.getSegmentInputBindingSlotsHandle(0),
        expected["segment_inputs"],
    )
    assert_bindings(
        sequence,
        sequence.getSegmentOutputBindingSlotsHandle(0),
        expected["segment_outputs"],
    )
    assert_named_bindings(
        sequence,
        sequence.getModelSequenceInputBindingSlotsHandle(),
        sequence.getModelSequenceInputNamesHandle(),
        expected["model_inputs"],
    )
    assert_named_bindings(
        sequence,
        sequence.getModelSequenceOutputBindingSlotsHandle(),
        sequence.getModelSequenceOutputNamesHandle(),
        expected["model_outputs"],
    )


def assert_bindings(sequence, bindings, expected_bindings):
    assert sequence.getBindingsSize(bindings) == len(expected_bindings)
    for index, (binding, mrt_index) in enumerate(expected_bindings):
        assert sequence.getBindingSlotBinding(bindings, index) == binding
        assert sequence.getBindingSlotMrtIndex(bindings, index) == mrt_index


def assert_named_bindings(sequence, bindings, names, expected_bindings):
    assert sequence.getBindingsSize(bindings) == len(expected_bindings)
    for index, (binding, mrt_index, name) in enumerate(expected_bindings):
        assert sequence.getBindingSlotBinding(bindings, index) == binding
        assert sequence.getBindingSlotMrtIndex(bindings, index) == mrt_index
        assert sequence.getName(names, index) == name


def assert_constants(constants, expected_constants):
    assert constants.size() == len(expected_constants)
    for index, expected in enumerate(expected_constants):
        assert constants.getConstantMrtIndex(index) == expected["mrt_index"]
        assert (
            constants.getConstantSparsityDimension(index)
            == expected["sparsity_dimension"]
        )

        value = constants.getConstant(index).tobytes()
        if "value" in expected:
            assert value == expected["value"]
        else:
            assert len(value) == expected["size"]
