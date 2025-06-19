#
# SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
import os
import sys

sys.path.insert(0, os.path.abspath("."))


# ML SDK Model Converter project config
BE_project = "ML SDK Model Converter"
copyright = "2022-2025, Arm Limited and/or its affiliates <open-source-office@arm.com>"
author = "Arm Limited"
git_repo_tool_url = "https://gerrit.googlesource.com/git-repo"

# Set home project name
project = BE_project

rst_epilog = """
.. |BE_project| replace:: %s
.. |git_repo_tool_url| replace:: %s
""" % (
    BE_project,
    git_repo_tool_url,
)

# Enabled extensions
extensions = [
    "breathe",
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
]

# Disable superfluous warnings
suppress_warnings = ["sphinx.ext.autosectionlabel.*"]
# Breathe Configuration
breathe_projects = {"ModelConverter": "../generated/xml"}
breathe_default_project = "ModelConverter"
breathe_domain_by_extension = {"h": "c"}

# Enable RTD theme
html_theme = "sphinx_rtd_theme"

# Stand-alone builds need to include some base docs (Security and Contributor guide)
tags.add("WITH_BASE_MD")
