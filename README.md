# ML SDK Model Converter

The ML SDK Model Converter is a command line application that translate TOSA ML
Models to VGF files. A VGF file is a container for SPIR-V™ modules and
constants that are required to execute the converter model through the Arm®
Vulkan® ML extensions. The ML SDK Model Converter supports several different
TOSA encodings as inputs:

- TOSA FlatBuffers
- TOSA MLIR bytecode
- TOSA MLIR textual format

The ML SDK Model Converter can also produce TOSA FlatBuffers from its input,
without performing any conversion.

You can also use the ML SDK Model Converter to check that all tensors specified
in the input model are ranked and have fixed, non-dynamic shapes. If a dynamic
tensor is detected, the program will exit with an error.

The suggested workflow for this tool as part of the ML SDK for Vulkan® is:

1.  A TOSA MLIR file is converted to a VGF file using the ML SDK Model Converter
    (this project).
2.  The generated VGF file and VGF library VGF Dump Tool is used to create a
    JSON scenario template file. The template file is edited with the correct
    filenames and paths.
3.  Using the generated VGF file and scenario file, the ML SDK Scenario Runner
    then dispatches the contained SPIR-V™ modules to the Arm® Vulkan® ML
    extensions.

## How to build

[Build](BUILD.md)

## License

[Apache-2.0](LICENSES/Apache-2.0.txt)

## Security

If you believe you have discovered a security issue please refer to the
[Security Section](SECURITY.md)

## Trademark notice

Arm® is a registered trademarks of Arm Limited (or its subsidiaries) in the US
and/or elsewhere.

Khronos®, Vulkan® and SPIR-V™ are registered trademarks of the
[Khronos® Group](https://www.khronos.org/legal/trademarks).
