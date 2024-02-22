# motile_toolbox

[![tests](https://github.com/funkelab/motile_toolbox/actions/workflows/tests.yaml/badge.svg)](https://github.com/funkelab/motile_toolbox/actions/workflows/tests.yaml)
[![black](https://github.com/funkelab/motile_toolbox/actions/workflows/black.yaml/badge.svg)](https://github.com/funkelab/motile_toolbox/actions/workflows/black.yaml)
[![mypy](https://github.com/funkelab/motile_toolbox/actions/workflows/mypy.yaml/badge.svg)](https://github.com/funkelab/motile_toolbox/actions/workflows/mypy.yaml)
[![codecov](https://codecov.io/gh/funkelab/motile_toolbox/branch/main/graph/badge.svg)](https://codecov.io/gh/funkelab/motile_toolbox)

# Install Motile Toolbox
Motile Toolbox depends on (motile)[https://github.com/funkelab/motile], which in turn depends on gurobi and ilpy. These dependencies must be installed with conda before installing motile toolbox with pip.
"""
conda install -c conda-forge -c funkelab -c gurobi ilpy
pip install motile_toolbox
"""
