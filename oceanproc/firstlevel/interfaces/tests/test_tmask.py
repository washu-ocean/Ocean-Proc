from ..tmask import MakeTmask
import pandas as pd
import numpy as np
import pytest


@pytest.mark.parametrize("fd,start_censor,neighbors,expected",[
    (
        [0,0,0,0,0,0,0,0,0,0],
        3,
        0,
        [0,0,0,1,1,1,1,1,1,1]
    ),
    (
        [0,0,0,0,0,0,0,0,0,0],
        0,
        0,
        [1,1,1,1,1,1,1,1,1,1]
    ),
    (
        [1,1,1,1,1,1,1,1,1,1],
        0,
        0,
        [0,0,0,0,0,0,0,0,0,0]
    ),
    (
        [0,0,0,0,0,1,0,0],
        0,
        2,
        [1,1,1,0,0,0,0,0]
    ),
    (
        [0,0,0,0,0,1,0,0],
        3,
        2,
        [0,0,0,0,0,0,0,0]
    )
])
def test_MakeTmask(fd, start_censor, neighbors, expected, tmp_path):
    pd.DataFrame({"framewise_displacement": fd}).to_csv(
        in_file := tmp_path / "confounds.csv"
    )
    out_file = tmp_path / "tmask.txt"
    MakeTmask(in_file=in_file,
              out_file=out_file,
              fd_threshold=0.9,
              minimum_unmasked_neighbors=neighbors,
              start_censoring=start_censor).run()
    # assert out_file.exists()
    # assert np.loadtxt(out_file).astype(bool) == np.array(expected).astype(bool)
