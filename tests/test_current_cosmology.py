"""Pin a single published column and the exact physical-density convention."""
import pytest
import radiofisher as rf
from radiofisher import cosmologies


def test_default_is_the_august_2026_joint_fiducial():
    c = rf.get_cosmology()
    assert rf.DEFAULT_COSMOLOGY == 'cmbspa2026'
    # Table 8, kappa_GPA + CMB_SPA + BAO_DESI; 100*omega rows converted once.
    assert (c['h'], c['ombh2'], c['omch2'], c['ns'], c['sigma_8'], c['mnu']) == pytest.approx((
        0.6832, 2.2506/100, 11.755/100, 0.9742, 0.8117, 0.06), rel=0, abs=1e-16)
    densities = rf.physical_density_parameters(c)
    assert sum(densities.values()) == pytest.approx(c['omega_M_0']*c['h']**2)
    assert c['omega_M_0'] == pytest.approx(0.3014, abs=5e-5)
    assert c['omega_M_0'] + c['omega_lambda_0'] == 1
    assert c['w0'] == -1 and c['wa'] == 0
    assert c['fiducial_source']['arxiv'] == '2608.31136v1'


def test_factories_do_not_share_mutable_state_or_rewrite_legacy():
    original = rf.experiments.cosmo.copy()
    c = rf.get_cosmology()
    c['foregrounds']['A'][0] = -1
    c['fiducial_source']['date'] = 'changed'
    assert rf.get_cosmology()['foregrounds']['A'][0] == original['foregrounds']['A'][0]
    assert rf.get_cosmology()['fiducial_source']['date'] == '2026-08-31'
    assert rf.get_cosmology('planck2013') == original
    with pytest.raises(ValueError, match='unknown fiducial'):
        rf.get_cosmology('latest')
    with pytest.raises(TypeError):
        cosmologies.CMBSPA2026['h'] = 0.7
