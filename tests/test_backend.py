import radiofisher


def test_backend_capability_contract_is_immutable():
    capabilities = radiofisher.get_backend_capabilities()

    assert radiofisher.BACKEND_ID == "radiofisher"
    assert radiofisher.BACKEND_VERSION == "1.0.0"
    assert radiofisher.__version__ == "1.0.0"
    assert radiofisher.BACKEND_API_VERSION == 1
    assert capabilities is radiofisher.BACKEND_CAPABILITIES
    assert isinstance(capabilities, frozenset)
    assert {
        "P_res",
        "astrophysical_model_profiles",
        "explicit_physical_densities",
        "kpar_min_fn",
        "noise_freq_mode",
        "noise_freq_weight",
        "vol_frac",
    } <= capabilities


def test_package_root_exposes_only_the_supported_surface():
    namespace = {}
    exec("from radiofisher import *", namespace)

    assert "fisher" in namespace
    assert "get_astrophysical_profile" in namespace
    assert "np" not in namespace
    assert "os" not in namespace
    assert not hasattr(radiofisher, "np")
    assert not hasattr(radiofisher, "os")
    assert not hasattr(radiofisher, "experiments_galaxy")
    assert len(radiofisher.__all__) == len(set(radiofisher.__all__))
    assert all(hasattr(radiofisher, name) for name in radiofisher.__all__)
