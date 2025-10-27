import pytest


@pytest.fixture(scope="session")
def nifti_timeseries_path(tmp_path_factory):
    """
    Fixture to a dummy NIFTI timeseries.

    MAKE SURE THIS IS THE FIRST FIXTURE
    """
    from numpy.random import RandomState
    import numpy as np
    import nibabel as nib
    prng = RandomState(42)
    fdata = prng.rand(48, 64, 48, 30)
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(fdata, affine)
    nifti_path = tmp_path_factory.mktemp("data") / "mock_nifti.nii.gz"
    nib.save(nifti_img, nifti_path)
    return nifti_path


@pytest.fixture(scope="session")
def tmask_path(nifti_timeseries_path):
    """
    Fixture to a tmask for a dummy NIFTI timeseries.
    """
    import numpy as np
    np.savetxt(
        (tmask_path := nifti_timeseries_path.parent / "mock_nifti_tmask.txt"),
        np.array([
            0, 0, 0, 0, 0,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            0, 0, 0, 0, 0
        ])
    )
    return tmask_path


@pytest.fixture(scope="session")
def nifti_mask_path(nifti_timeseries_path):
    """
    Fixture to a mask for the dummy NIFTI timeseries (really simple, just a cube of 1s)
    """
    import numpy as np
    import nibabel as nib
    img = nib.load(nifti_timeseries_path)
    fdata = img.get_fdata()
    mask = np.zeros(fdata.shape[0:3])  # just the spatial component
    mask[3:-3, 3:-3, 3:-3] = 1
    mask_img = img.__class__(mask, affine=img.affine)
    mask_path = nifti_timeseries_path.parent / "mock_nifti_mask.nii.gz"
    nib.save(mask_img, mask_path)
    return mask_path


@pytest.fixture(scope="session")
def cifti_timeseries_path(nifti_timeseries_path, tmask_path):
    from numpy.random import RandomState
    import numpy as np
    import nibabel as nib
    prng = RandomState(42)
    fdata = prng.rand(30, 1000)
    brain_model_axis = nib.cifti2.BrainModelAxis.from_mask(np.ones(1000, dtype=bool), "CIFTI_STRUCTURE_CORTEX_LEFT")
    series_axis = nib.cifti2.ScalarAxis(np.arange(30))
    cifti_img = nib.cifti2.cifti2.Cifti2Image(fdata, (series_axis, brain_model_axis))
    cifti_path = nifti_timeseries_path.parent / "mock_cifti.dtseries.nii"
    nib.save(cifti_img, cifti_path)
    return cifti_path
