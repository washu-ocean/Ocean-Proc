from ..filter import FilterData


def test_FilterData_with_nifti(nifti_timeseries_path, tmask_path):
    out_file = nifti_timeseries_path.parent / nifti_timeseries_path.name.replace(".nii.gz", "_filtered.nii.gz")
    print(out_file)
    FilterData(in_file=nifti_timeseries_path,
               in_mask=tmask_path,
               out_file=out_file)


def test_FilterData_with_cifti(cifti_timeseries_path, tmask_path):
    out_file = cifti_timeseries_path.parent / cifti_timeseries_path.name.replace(".dtseries.nii", "_filtered.dtseries.nii")
    print(out_file)
    FilterData(in_file=cifti_timeseries_path,
               in_mask=tmask_path,
               out_file=out_file)
