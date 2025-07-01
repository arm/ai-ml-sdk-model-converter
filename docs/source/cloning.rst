Cloning the repository
======================

To clone the |MC_project| as a stand-alone repository, you can use regular git clone commands. However we recommend
using the :code:`git-repo` tool to clone the repository as part of the ML SDK for Vulkan® suite. The tool is available here:
(|git_repo_tool_url|). This ensures all dependencies are fetched and in a suitable default location on the file
system.

For a minimal build and to initialize only the |MC_project| and its dependencies, run:

.. code-block:: bash

    repo init -u <server>/ml-sdk-for-vulkan-manifest -g model-converter

Alternatively, to initialize the repo structure for the entire ML SDK for Vulkan®, including the ML SDK Model Converter, run:

.. code-block:: bash

    repo init -u <server>/ml-sdk-for-vulkan-manifest -g all

After the repo is initialized, you can fetch the contents with:

.. code-block:: bash

    repo sync

.. note::
    You must enable long paths on Windows®. To ensure nested submodules do not exceed the maximum long path length,
    you must clone close to the root directory or use a symlink.


Due to a bug in :code:`git-repo`, nested submodules do not always update as part of :code:`repo sync` and need to
be manually updated, for example:

.. code-block:: bash

    cd dependencies/SPIRV-Tools
    git submodule update --init --recursive

After the sync command completes successfully, you can find the ML SDK Model Converter in :code:`<repo_root>/sw/model-converter/`.
You can also find all the dependencies required by the ML SDK Model Converter in :code:`<repo_root>/dependencies/`.
