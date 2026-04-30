#
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import pathlib

import vgfpy as vgf

VK_FORMAT_R8_UINT = 13
VK_FORMAT_R8_SINT = 14
VK_FORMAT_R16_UINT = 74
VK_FORMAT_R16_SINT = 75
VK_FORMAT_R32_SINT = 99
VK_FORMAT_R32_SFLOAT = 100
VK_FORMAT_R32G32B32A32_SFLOAT = 109
VK_FORMAT_R16_SFLOAT_FPENCODING_BFLOAT16_ARM = 1000460001
VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E4M3_ARM = 1000460002
VK_FORMAT_R8_SFLOAT_FPENCODING_FLOAT8E5M2_ARM = 1000460003


def create_vgf_decoders(vgf_path):
    return VgfDecoders(vgf_path)


class VgfDecoders:
    def __init__(self, vgf_path):
        self._data = memoryview(pathlib.Path(vgf_path).read_bytes())
        self.header = vgf.CreateHeaderDecoder(
            self._data, vgf.HeaderSize(), len(self._data)
        )
        assert self.header is not None

        self.modules = vgf.CreateModuleTableDecoder(
            self._section(
                self.header.GetModuleTableOffset(),
                self.header.GetModuleTableSize(),
            ),
            self.header.GetModuleTableSize(),
        )
        self.resources = vgf.CreateModelResourceTableDecoder(
            self._section(
                self.header.GetModelResourceTableOffset(),
                self.header.GetModelResourceTableSize(),
            ),
            self.header.GetModelResourceTableSize(),
        )
        self.sequence = vgf.CreateModelSequenceTableDecoder(
            self._section(
                self.header.GetModelSequenceTableOffset(),
                self.header.GetModelSequenceTableSize(),
            ),
            self.header.GetModelSequenceTableSize(),
        )
        self.constants = vgf.CreateConstantDecoder(
            self._section(
                self.header.GetConstantsOffset(),
                self.header.GetConstantsSize(),
            ),
            self.header.GetConstantsSize(),
        )

        assert self.modules is not None
        assert self.resources is not None
        assert self.sequence is not None
        assert self.constants is not None

    def _section(self, offset, size):
        return self._data[offset : offset + size]
