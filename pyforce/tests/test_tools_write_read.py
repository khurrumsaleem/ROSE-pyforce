import pytest
import numpy as np
from pathlib import Path
from pyvista import examples

from pyforce.tools.write_read import ReadFromOF

## add OpenFOAM import test using pyvista examples
@pytest.fixture
def setup_of_reader():
    filename = examples.download_cavity(load=False)[:-9]
    return ReadFromOF(filename)