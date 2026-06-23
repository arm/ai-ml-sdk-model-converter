Usage: ./model-converter [--help] [--version] [--input VAR] [--output VAR] [--tosa-flatbuffer] [--tosa-flatbuffer-schema VAR] [--dump-mlir] [--emit-debug-info] [--disable-replicated-composites] [--require-static-shape] [--experimental-analysis] [--custom-op-domain-to-opcode VAR]... [--enable-bespoke] [--type-narrowing VAR]

Optional arguments:
  -h, --help                       shows help message and exits
  -v, --version                    prints version information and exits
  -i, --input                      the input file to read TOSA FlatBuffer or TOSA MLIR data from [nargs=0..1] [default: "-"]
  -o, --output                     the output file to write VGF data to [nargs=0..1] [default: "-"]
  --tosa-flatbuffer                write tosa FlatBuffer instead of VGF
  --tosa-flatbuffer-schema         path to the tosa FlatBuffer schema [nargs=0..1] [default: ""]
  --dump-mlir                      Dump MLIR between each pass to std error
  --emit-debug-info                Produce debug info instructions in SPIR-V assembly
  --disable-replicated-composites  Do not emit SPIR-V features that require VK_EXT_shader_replicated_composites
  --require-static-shape           Require all tensors to be ranked and have a specified shape. Terminate on deviation.
  --experimental-analysis          Print analysis output (what operator lower and which errors out) for the input. [EXPERIMENTAL]
  --custom-op-domain-to-opcode     Map TOSA custom op domains to Arm.ExperimentalMLOperations CALL Opcode literal integers. Entries use <domain>:<opcode> and can be specified multiple times. [nargs=0..1] [default: {}] [may be repeated]
  --enable-bespoke                 Enable com.arm.bespoke custom op lowering as an Arm.ExperimentalMLOperations CALL with Opcode 0.
  --type-narrowing                 Perform type-narrowing to all operator operands/results from fp32 -> fp16 [nargs=0..1] [default: "none"]
