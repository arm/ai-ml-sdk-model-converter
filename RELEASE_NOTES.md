# Model Converter — Release Notes

---

## Version 0.8.0 – *Toolchain Refresh & Packaging Expansion*

### Converter & Toolchain

- Switched the converter to use the new `tosa_tools` repository as dependency and
  refreshed the bundled LLVM/FlatBuffers revisions. This will need a repo tool sync.
- Updated unit tests and core conversion logic to match the new 'tosa_tools' dependency
  and behavior, keeping partitioning, type narrowing, and VGF serialization aligned
  with the refreshed MLIR updates.

### Build, Packaging & Developer Experience

- Modernized the pip package: switched to `pyproject.toml`, added the missing
  metadata, and fixed package naming/installation ordering issues that affected
  `--install`.
- Defaulted the build system to Ninja, refined the CMake packaging flow.
- Introduced `clang-tidy` configuration and streamlined cppcheck
  invocation/CLI integration (including build-script driven execution).

### Platform & Compliance

- Added Darwin targets for AArch64 to the pip packaging matrix.
- Refreshed SBOM data and adopted usage of `REUSE.toml`.

### Supported Platforms

The following platform combinations are supported:

- Linux - AArch64 and x86-64
- Windows® - x86-64
- Darwin - AArch64 (experimental)

---

## Version 0.7.0 – *Initial Public Release*

## Purpose

Converts **TOSA models** into **VGF files** with embedded SPIR-V™ modules, constants and metadata.

## Features

### Input Format Support

- **TOSA FlatBuffers**: Direct conversion from binary TOSA FlatBuffer files
- **TOSA MLIR bytecode**: Conversion from compiled MLIR bytecode format
- **TOSA MLIR textual format**: Support for human-readable MLIR text files

### Output Capabilities

- **VGF file generation**: Primary output format containing SPIR-V™ modules and
  constants for ML extensions in Vulkan®
- **TOSA FlatBuffer passthrough**: Generate TOSA FlatBuffers from input without
  conversion for validation, optimization and debugging

### Model Validation & Analysis

- **Tensor shape validation**: Ensures all tensors are ranked with fixed,
  non-dynamic shapes
- **Dynamic tensor detection**: Program exits with error if dynamic tensors are
  detected
- **Model integrity checking**: Validates input model structure and
  compatibility
- **Type narrowing**: Support type narrowing from fp32 to fp16

### Integration & Workflow

- **ML SDK for Vulkan® integration**: Seamless integration as part of the
  complete ML SDK workflow
- **VGF Dump Tool compatibility**: Generated VGF files work with VGF library
  tools for JSON scenario template creation
- **Scenario Runner support**: Output files compatible with ML SDK Scenario
  Runner for SPIR-V™ module dispatch to Vulkan®

### Command Line Interface

- **Flexible input/output options**: Simple command-line interface with
  customizable input and output paths
- **Format detection**: Automatic detection of input format type
- **Help and documentation**: Built-in help system with usage examples

## Platform Support

The following platform combinations are supported:

- Linux - X86-64
- Windows® - X86-64

---
