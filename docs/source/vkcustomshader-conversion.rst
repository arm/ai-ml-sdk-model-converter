Vulkan® Custom Shader Conversion to VGF
=======================================

Overview
--------

The exact ``domain_name`` value ``com.arm.VulkanCustomShader`` selects a TOSA
custom operation for Vulkan® compute shader lowering. ``operator_name`` supplies
the shader's logical name, and ``implementation_attrs`` must be a JSON object
conforming to :doc:`vkshader-custom-op-encoding`. That reference defines the
entry point, workgroup sizes, shader code, indexed resource and descriptor
metadata, optional sampler configuration, and push-constant layout.

A selected operation with malformed or invalid required attributes fails
conversion; it does not fall back to ordinary custom-operation handling.
Operations in other domains are not selected by this shader-lowering path.
Attribute validation does not establish compatibility with every target device.

This page explains how selected compute operations partition a mixed ML and
shader model and how their dependencies are represented in VGF. See
:ref:`model-converter-architecture` for the surrounding conversion pipeline.


Model Partitioning
------------------

ML models are represented as directed acyclic graphs (DAGs),
where nodes are operations and edges represent data dependencies.

Models may contain a mix of:

- Regular ML operations
- Shader custom operations (as defined above)

Vulkan® execution constraints require that:

- ML workloads cannot be dispatched in the same pipeline call
  as shader workloads.
- Neural Engine accelerators and classical GPU engines must be
  scheduled through separate Vulkan® pipelines.

Therefore, before VGF generation, the model is partitioned into
sub-models that can be executed independently.


Partitioning Strategy
---------------------

The partitioning follows these principles:

- Each shader custom operation forms its own partition.
- Regular ML operations occurring:
  - before the first shader custom operation,
  - between two shader custom operations,
  - or after the last shader custom operation,
  are grouped into ML partitions.
- Connections crossing partition boundaries become explicit
  inputs and outputs of the generated sub-models.

This transforms a single model DAG into a sequence (or DAG)
of sub-graphs, each mapped to an independent Vulkan® pipeline.


Execution Correctness
---------------------

Partitioning preserves model data dependencies in the ordered graph and compute
segments and their resource bindings. Conversion does not record Vulkan® commands
or insert runtime memory barriers or cache operations.

The runtime consumer must honor those dependencies when recording and executing
the segments, including the required synchronization. The encoded sequence alone
does not guarantee correct synchronization for every execution environment.


Partitioning Algorithm (Conceptual)
-----------------------------------

The model is traversed in breadth-first order.
Each operation is assigned a ``partition_id``.

- Shader custom operations always receive a new partition.
- Regular ML operations inherit or create partitions based on
  the partitions of their parent operations.

All operations sharing the same ``partition_id`` belong to the same
sub-model.

The algorithm ensures:

- Parent operations are processed before their children.
- Shader and ML operations never coexist in the same partition.
- Partition identifiers increase monotonically.


Conversion Semantics
--------------------

When a TOSA custom operation satisfies the required constraints:

1. The operation is isolated into its own partition.
2. A VGF compute node is generated for that partition.
3. The shader is stored as SPIR-V™ binary (decoded from base64 if necessary)
   or embedded GLSL or HLSL source. The serializer does not compile textual
   shaders; a runtime consumer must compile supported source during pipeline
   setup or supply an implementation for a module without embedded code.
4. Descriptor-set and resource-binding metadata is encoded according to the indexed
   ``input_<index>_*`` and ``output_<index>_*`` attributes, including
   optional per-resource sampler configuration.
5. Workgroup sizes are mapped to the compute dispatch configuration.
6. The ordered segments and their bindings describe the dependencies that the
   runtime consumer must honor when dispatching work and recording barriers.

The conversion is deterministic and fully driven by the schema-defined
attributes.


Minimal Example
---------------

.. code-block:: json

   {
     "domain_name": "com.arm.VulkanCustomShader",
     "operator_name": "MyComputeKernel",
     "implementation_attrs": {
       "entry_point": "main",
       "workgroup_sizes": [8, 8, 1],
       "shader_language": "GLSL",
       "shader_code": "void main() { }",

       "input_0_vkformat": "VK_FORMAT_R32_SFLOAT",
       "input_0_vkdescriptortype": "VK_DESCRIPTOR_TYPE_STORAGE_BUFFER",
       "input_0_binding": 0,
       "input_0_descriptorset": 0
     }
   }


Worked mixed-segment mapping
----------------------------

Consider a tensor flowing through ``tosa.abs``, a shader custom operation,
and another ``tosa.abs``. The following is a synthetic mapping to explain the
records, not a promise of exact Model Converter output or numeric identifiers.
Assume ordinary float32 tensors with matching shapes and
``VK_FORMAT_R32_SFLOAT`` views throughout. Both shader resources explicitly use
``VK_DESCRIPTOR_TYPE_TENSOR_ARM``, matching the graph tensor descriptors; the
shader implementation must support those tensor resources.

.. uml:: puml/mixed-segment-example.puml
   :align: center
   :caption: Illustrative graph, compute, graph sequence and shared resources.
   :alt: Graph segment 0 uses module 0 to read MRT 0 and write MRT 1. Compute segment 1 uses module 1 to read MRT 1 and write MRT 2. Graph segment 2 uses module 2 to read MRT 2 and write MRT 3.

The ordered segments reference module-table entries 0, 1 and 2 respectively;
module indices identify implementations, while segment positions identify their
uses. MRT means model resource table. Model input binding 0 refers to MRT 0,
and model output binding 3 refers to MRT 3. The graph segments use the logical
bindings associated with their corresponding input and output values.

For the compute segment, logical input binding 1 refers to MRT 1 and logical
output binding 2 refers to MRT 2. Its descriptor metadata separately maps set 2,
binding 7 to MRT 1 and set 2, binding 9 to MRT 2. The custom-operation attributes
``input_0_descriptorset=2``, ``input_0_binding=7``,
``output_0_descriptorset=2`` and ``output_0_binding=9`` select the descriptor
locations; those locations are not the logical input/output binding numbers.
The same MRT reference connects each producer output to the next consumer input.

A resource's category and view requirements can require multiple MRT entries;
the one-entry-per-edge picture is deliberately simplified. Module and resource
numbering depends on the actual conversion. The runtime consumer must preserve
these dependencies and supply synchronization between segment executions.
