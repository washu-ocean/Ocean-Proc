from ..nuisance import GenerateNuisanceMatrix
from pathlib import Path
import pandas as pd
import pytest


@pytest.fixture
def in_filepath():
    return Path(__file__).parent / "data" / "confounds.tsv"


def test_GenerateNuisanceMatrix(in_filepath, tmp_path_factory):
    confounds_columns_group = [
        (["framewise_displacement"]),
        (["framewise_displacement", "csf"])
    ]
    out_filepath = tmp_path_factory.mktemp("data") / "out.tsv"
    for confounds_columns in confounds_columns_group:
        res = GenerateNuisanceMatrix(in_file=in_filepath,
                                     confounds_columns=confounds_columns,
                                     out_file=out_filepath).run()
