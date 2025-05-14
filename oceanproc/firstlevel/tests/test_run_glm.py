import random
import re
import itertools
from ..run_glm import *
import pytest
import pandas as pd
import numpy as np
from scipy import signal

TR = 1.2

# define some fixtures


@pytest.fixture
def confounds_df():
    df = pd.DataFrame(data={"framewise_displacement": np.random.rand(50),
                            "global_signal": np.random.rand(50) * 5000,
                            "volterra": np.ones(50)})
    return df


@pytest.fixture
def confounds_columns(confounds_df):
    cols = [col for col in confounds_df.columns if "volterra" not in col]
    return cols


@pytest.fixture
def random_fdata():
    return np.random.rand(50, 91282)


@pytest.fixture
def random_eventdf():
    d = {"trial_type": random.choices(('a','b','c'), k=50),  # <-- 50 random a, b, or c
         "onset": np.linspace(0, 60, 50),  # <-- 50 intervals of 1.2 seconds}
         "duration": np.ones(50)}  # <-- 50 random correct, incorrect
    return pd.DataFrame(data=d)


@pytest.fixture
def random_signaldf():
    d = {"flat_signal": np.ones(50) * 50,
         "random_signal": np.random.rand(50) * 50,
         "sine_signal": np.sin(np.linspace(-np.pi, np.pi, 50))}
    return pd.DataFrame(data=d)


def test_make_option():
    assert make_option(True) == " "
    assert make_option(True, key="myopt") == "--myopt "
    assert make_option(['1','2','3']) == " 1 2 3"
    assert make_option("hello") == " hello"
    assert make_option("hello", key="myopt") == "--myopt hello"


@pytest.mark.parametrize("demean,linear_trend,spike_threshold,volterra_expansion,volterra_columns",
                         [(False, False, None, None, None),
                          (True, False, None, None, None),
                          (False, True, None, None, None),
                          (True, True, None, None, None),
                          (True, True, 0.5, None, None),
                          (True, True, 0.5, 1, ["volterra"]),
                          (True, True, 0.5, 3, ["volterra"])])
def test_make_noise_ts(tmp_path,
                       confounds_df,
                       confounds_columns,
                       demean,
                       linear_trend,
                       spike_threshold,
                       volterra_expansion,
                       volterra_columns):
    confounds_file = tmp_path / "tmp.tsv"
    confounds_df.to_csv(confounds_file, sep='\t')
    noise_ts = make_noise_ts(confounds_file,
                             confounds_columns,
                             demean,
                             linear_trend,
                             spike_threshold,
                             volterra_expansion,
                             volterra_columns)
    if spike_threshold:
        assert spike_threshold > 0
        spike_column_num = (noise_ts.loc[:, "framewise_displacement"] > spike_threshold).sum()
        assert len([col for col in noise_ts.columns if "spike" in col]) == spike_column_num

    assert "volterra_expansion" in locals() and "volterra_columns" in locals()
    if volterra_expansion:
        for i in range(volterra_expansion):
            # Check that each volterra column has been shifted appropriately
            assert (noise_ts.loc[0:i, f"volterra_{i+1}"] == 0).all()


@pytest.mark.parametrize("fir,hrf,fir_list,hrf_list",
                         [(None, None, None, None),
                          (3, None, ['a', 'b'], None),
                          (3, None, None, None),
                          (3, (5, 10), None, None),
                          (3, (5, 10), ['a','b'], None),
                          (3, (5, 10), None, ['a','b']),
                          (3, (5, 10), ['a','b'], ['a','b'])])
def test_events_to_design(tmp_path,
                          random_fdata,
                          random_eventdf,
                          fir,
                          hrf,
                          fir_list,
                          hrf_list):
    def run():
        return events_to_design(func_data,
                                TR,
                                event_file,
                                fir=fir,
                                hrf=hrf,
                                fir_list=fir_list,
                                hrf_list=hrf_list)
    func_data = random_fdata
    event_file = tmp_path / "events.tsv"
    random_eventdf.to_csv(event_file, sep='\t')
    # both fir and hrf specified, but no list of columns for either
    if (fir and hrf) and not (fir_list or hrf_list):
        with pytest.raises(RuntimeError):
            events_df, run_conditions = run()
        return
    # fir_list, hrf_list have overlapping column names
    elif (fir_list and hrf_list) and not set(fir_list).isdisjoint(hrf_list):
        with pytest.raises(RuntimeError):
            events_df, run_conditions = run()
        return
    events_df, run_conditions = run()
    if fir_list:  # Check if a_00, a_01, ... a_nn exist, and if the number of a columns matches the fir int
        pattern = re.compile(f"{fir_list[0]}_\\d\\d")
        assert len([col for col in events_df.columns if re.match(pattern, col)]) == fir


@pytest.mark.parametrize("demean,linear_trend,spike_threshold,volterra_expansion,volterra_columns",
                         [(False, False, None, None, None),
                          (True, False, None, None, None),
                          (False, True, None, None, None),
                          (True, True, None, None, None),
                          (True, True, 0.5, None, None),
                          (True, True, 0.5, 1, ["volterra"]),
                          (True, True, 0.5, 3, ["volterra"])])
def test_nuisance_regression(tmp_path,
                             confounds_df,
                             confounds_columns,
                             random_fdata,
                             demean,
                             linear_trend,
                             spike_threshold,
                             volterra_expansion,
                             volterra_columns):
    confounds_file = tmp_path / "tmp.tsv"
    confounds_df.to_csv(confounds_file, sep='\t')
    noise_ts = make_noise_ts(confounds_file,
                             confounds_columns,
                             demean,
                             linear_trend,
                             spike_threshold,
                             volterra_expansion,
                             volterra_columns)
    data_minus_residuals = nuisance_regression(random_fdata, noise_ts)


# def test_hrf_convolve_features(tmp_path,
    # random_signaldf):
    # cfeats = hrf_convolve_features(random_signaldf)
    # for col in random_signaldf.columns:
    # recovered, remainder = signal.deconvolve(cfeats[col], )
