# Model Converter — Release Notes

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

---
