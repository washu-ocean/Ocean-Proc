from ..nuisance import GenerateNuisanceMatrix


def test_GenerateNuisanceMatrix(confounds_filepath, tmp_path_factory):
    confounds_columns_group = [
        (["framewise_displacement"]),
        (["framewise_displacement", "csf"])
    ]
    out_filepath = tmp_path_factory.mktemp("data") / "out.tsv"
    for confounds_columns in confounds_columns_group:
        GenerateNuisanceMatrix(in_file=confounds_filepath,
                               confounds_columns=confounds_columns,
                               out_file=out_filepath).run()
