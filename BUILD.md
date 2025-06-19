# Building the ML SDK Model Converter from source

The build system must have:

- CMake 3.25 or later.
- C/C++ 17 compiler: GCC, or optionally Clang on Linux and MSVC on Windows®.
- Python 3.10 or later. Required python libraries for building are listed in
  `tooling-requirements.txt`.
- FlatBuffers flatc compiler.

The following dependencies are also needed:

- [Argument Parser for Modern C++](https://github.com/p-ranav/argparse).
- [LLVM](https://github.com/llvm/llvm-project).
- [TOSA Serialization Library](https://review.mlplatform.org/plugins/gitiles/tosa/serialization_lib).
- [TOSA MLIR Translator](https://review.mlplatform.org/plugins/gitiles/tosa/tosa_mlir_translator).

For the preferred dependency versions see the manifest file.

## Providing Flatc

There are 3 options for providing the flatc binary.

1.  Installing flatc to the system. To install flatc to the system and make it
    available on the searchable `PATH`, see the
    [FlatBuffers documentation](https://flatbuffers.dev/).

2.  Building flatc from repo source. When the repository is initialized using
    the repo manifest, you can find a FlatBuffers source checkout in
    `<repo-root>/dependencies/flatbuffers/`. Navigate to this location and run
    the following commands:

```bash
cmake -B build && cmake --build build --parallel $(nproc)
```

```{note}
If you build into the `build` folder, the VGF-lib cmake scripts
automatically find the flatc binary.
```

3.  Providing a custom flatc path. If flatc cannot be searched using `PATH` or
    found in the default `<repo-root>/dependencies/flatbuffers/build` path, you
    can provide a custom binary file path to the build script using the
    `--flatc-path <path_to_flatc>` option, see
    [Building with the script](#building-with-the-script).

<a name="building-with-the-script"></a>

## Building with the script

Arm® provides a python build script to make build configuration options easily
discoverable. When the script is run from a git-repo manifest checkout, the
script uses default paths and does not require any additional arguments.
Otherwise the paths to the dependencies must be specified.

To build on the current platform, for example on Linux or Windows®, run:

```bash
python3 $SDK_PATH/sw/model-converter/scripts/build.py -j $(nproc) \
    --vgf-lib-path ${PATH_TO_VGF_LIB_CHECKOUT} \
    --flatbuffers-path ${PATH_TO_FLATBUFFERS_CHECKOUT} \
    --argparse-path ${PATH_TO_ARGPARSE_CHECKOUT} \
    --tosa-mlir-translator-path ${PATH_TO_TOSA_MLIR_TRANSLATOR_CHECKOUT} \
    --external-llvm ${PATH_TO_LLVM_CHECKOUT}
```

If the components are in their default locations, it is not necessary to specify
the `--vgf-lib-path`, `--flatbuffers-path`, `--argparse-path`,
`--tosa-mlir-translator-path`, and `--external-llvm` options.

Tests can be enabled and run with `--test` and linting by `--lint`. The
documentation can be built with `--doc`. To build the documentation, sphinx and
doxygen must be installed on the machine.

You can install the project build artifacts into a specified location by passing
the option `--install` with the required path.

To create an archive with the build artifacts option, add `--package`. The
archive will be stored in the provided location."

For more information, see the help output:

```bash
python3 scripts/build.py --help
```

## Usage

To generate a VGF file, run:

```bash
./build/model-converter --input ${INPUT_TOSA} --output ${OUTPUT_VGF}
```

To generate a TOSA flatbuffer file, run:

```bash
./build/model-converter --tosa-flatbuffer --input ${INPUT_TOSA} --output ${OUTPUT_TOSA_FB}
```

For more information, see the help output:

```bash
./build/model-converter --help
```
