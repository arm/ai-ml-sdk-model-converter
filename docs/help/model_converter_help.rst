OVERVIEW: ML SDK Model Converter

USAGE: model-converter [options]

OPTIONS:

Generic Options:

  --help                                - Display available options (--help-hidden for more)
  --help-list                           - Display list of available options (--help-list-hidden for more)
  -v                                    - Alias for --version
  --version                             - Display the version of this program

Model Converter options:

  --custom-op-domain-to-opcode=<string> - Map TOSA custom op domains to Arm.ExperimentalMLOperations CALL Opcode literal integers. Entries use <domain>:<opcode> and can be specified multiple times.
  --disable-replicated-composites       - Do not emit SPIR-V features that require VK_EXT_shader_replicated_composites
  --dump-mlir                           - Dump MLIR between each pass to std error
  --emit-debug-info                     - Produce debug info instructions in SPIR-V assembly
  --enable-bespoke                      - Enable com.arm.bespoke custom op lowering as an Arm.ExperimentalMLOperations CALL with Opcode 0.
  --experimental-analysis               - Print analysis output (what operator lower and which errors out) for the input. [EXPERIMENTAL]
  -i                                    - Alias for --input
  --input=<string>                      - the input file to read TOSA FlatBuffer or TOSA MLIR data from
  -o                                    - Alias for --output
  --output=<string>                     - the output file to write VGF data to
  --require-static-shape                - Require all tensors to be ranked and have a specified shape. Terminate on deviation.
  --target-tosa-version=<string>        - Set the target TOSA import ceiling for VGF output (ignored with --tosa-flatbuffer)
  --tosa-flatbuffer                     - write tosa FlatBuffer instead of VGF
  --tosa-flatbuffer-schema=<string>     - path to the tosa FlatBuffer schema
  --type-narrowing=<value>              - Perform type-narrowing to all operator operands/results from fp32 -> fp16
    =none                               -   Disable type narrowing
    =full                               -   Narrow all supported operations
    =partial                            -   Narrow supported operations selectively
    =full_preserve_io                   -   Preserve input/output types
