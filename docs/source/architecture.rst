.. _model-converter-architecture:

Model Converter architecture
============================

Model Converter converts a TOSA model into a VGF file. It combines
LLVM/MLIR transformations with SDK passes that describe an execution sequence,
connect its resources, and package its modules and constants through VGF Library.
An application uses a runtime to read the file, create Vulkan® resources and
pipelines, and record and submit work.

Ordinary TOSA operations are lowered to the SPIR-V™ graph/TOSA representation.
They remain graph operations in the VGF file; compute execution is
provided by the runtime's Vulkan® implementation, which may include the
Emulation Layer. Models can also contain explicitly supplied custom compute
shaders, represented as separate compute segments.

.. contents:: On this page
   :local:
   :depth: 2

Inputs and deployment boundary
------------------------------

Frameworks export models to TOSA before conversion begins. Training and
framework-to-TOSA conversion happen outside Model Converter. Model Converter
accepts TOSA FlatBuffers and TOSA MLIR in textual or bytecode form.
TOSA Tools supplies FlatBuffer deserialization into MLIR;
MLIR's parser reads the other input forms. Both paths produce a module with
TOSA operations in functions for the same pass pipeline.

VGF output combines graph modules, custom compute modules where present,
ordered execution segments, resource and binding metadata, and constant data.
Compute modules can carry SPIR-V™ binaries, GLSL/HLSL source, or placeholders
whose implementation the runtime supplies. Source shader support is compute-only.
The converter records their code and execution metadata in VGF; source shader
compilation is handled by the runtime.

The application connects model inputs and outputs to its data and uses the
runtime to manage Vulkan® execution. Model Converter does not allocate device
resources, create Vulkan® pipelines, or submit commands.

The ``--tosa-flatbuffer`` option selects an alternative output path. It serializes
the prepared TOSA representation through TOSA Tools before graph-constant
marking, partitioning, graph lowering, and VGF packaging. The common normalization
and narrowing stages still run, so this path can change the input representation
and types. See :doc:`usage` for invocation and options.

Conversion pipeline
-------------------

``Compiler`` owns the MLIR context, dialect registry, and pass manager. It
registers the Func, TOSA, SPIR-V™, and converter-specific VGF dialects, constructs
the pipeline from command-line options, parses the input, and runs the passes.
The command-line driver enables pass verification; parsing or pass failure
causes conversion to fail.

.. uml:: puml/conversion-pipeline.puml
   :align: center
   :alt: TOSA input enters common preparation, then branches to TOSA Tools FlatBuffer output or VGF preparation, partitioning, LLVM/MLIR graph lowering and finalization, and VGF Library encoding.
   :caption: Principal conversion stages. Model Converter assembles the pipeline; labels identify the implementation responsible for each stage.

Input preparation
^^^^^^^^^^^^^^^^^

The common pipeline normalizes integer types to signless MLIR integers, inlines
dense-resource attributes, and optionally rejects unranked or dynamically shaped
tensors. Canonicalization simplifies functions before type narrowing and cleans
up casts and operations afterwards. LLVM/MLIR provides integer normalization,
canonicalization, and the TOSA narrowing passes; the converter supplies resource
inlining and the optional shape check.

I64-to-I32 and F64-to-F32 narrowing run on the common path, including function
boundaries. Optional F32-to-F16 narrowing controls whether function boundaries
and accumulator types also change. Narrowing can reduce numeric range and
precision; the F32-to-F16 modes differ in how they change input/output and
accumulator types. Their detailed options are described in :doc:`usage`.

Sequence construction
^^^^^^^^^^^^^^^^^^^^^

For VGF output, the converter first records integer signedness information needed
by the eventual resource interfaces. LLVM/MLIR's TOSA layerwise constant-folding
pass then simplifies constant expressions, and its graph-constant marking pass
assigns graph-constant IDs to constants whose data is stored separately from
the graph module. This happens before partitioning so cloned constants retain
their IDs across graph segments.

The converter's partitioning passes introduce its VGF dialect: a ``vgf.sequence``
contains graph or compute ``vgf.segment`` operations and their invocations.
These operations represent the structure of the VGF file: the sequence becomes a
Model Sequence Table, and segment invocations become its ordered ``SegmentInfo``
records, each referencing a graph or compute module in the Module Table.
Graph segments retain TOSA functions for lowering; compute segments contain
shader placeholders with interface metadata and optional shader code.
Values between segment invocations describe the dependencies that become VGF
resource bindings. See :doc:`vkcustomshader-conversion` for the custom shader
requirements and :doc:`vkshader-custom-op-encoding` for its attributes.

After partitioning, SDK passes detect supported constant sparsity, encode graph
constant data, and assign attributes describing the graph's application binary
interface (ABI). Constant encoding and final serialization share a
``VGFBuilder``, which retains the VGF Library
encoder and the mapping from graph-constant IDs to encoded constants.

LLVM/MLIR graph lowering and finalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The upstream ``TosaToSPIRVTosa`` pass wraps graph functions in ``spirv.module``
and ``spirv.ARM.Graph``, converts tensor and shape types to SPIR-V™ ARM tensor
types, and rewrites supported TOSA operations to ``spirv.Tosa.*`` operations.
Marked constants
become ``spirv.ARM.GraphConstant`` references; inline constants remain in the
module. Explicit graph interface bindings are preserved through conversion.
Target-environment legality and conversion patterns determine which input
operations and types can be lowered.

TOSA custom operations use the conversion's ``spirv.ExperimentalML.Call`` path
only when their domain has an explicit opcode mapping, supplied through
``--custom-op-domain-to-opcode`` or the predefined mapping enabled by
``--enable-bespoke``. Unmapped custom operations that reach graph lowering are
unsupported and cause conversion to fail. Custom Vulkan® shader operations
follow a separate path: the converter partitions them into compute segments
before graph lowering. See :doc:`usage` for domain-mapping options.

Within each resulting SPIR-V™ module, LLVM/MLIR optionally converts splat
composite constants to replicated composites, lowers interface ABI attributes to
graph entry-point interfaces, and infers the required version, capabilities, and
extensions. Its SPIR-V™ serializer then produces graph module binaries for VGF.

The converter chooses and orders these upstream passes alongside its own
passes. The SDK also carries a small LLVM patch for the conversion's experimental
analysis diagnostics, adding to the upstream lowering implementation.
The checked-out LLVM/MLIR dependency and the SDK's patch together determine the
behavior used by a build.

Resource planning and emission
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The converter's final serialization pass plans sequence inputs, intermediates,
and outputs, including the resource views and alias relationships needed across
graph and compute segments. It maps these values to resource and binding records,
collects graph-constant references, and adds module and segment records through
VGF Library's encoder. Graph module bindings and VGF resource metadata describe
the same interfaces.

VGF Library owns binary container encoding and writing. Model Converter owns
the translation from its MLIR sequence to those records and the choice of
modules, bindings, and constants to encode. After the encoder is finalized, the
serialization pass writes the VGF file. The runtime allocates device memory
and executes the model.
