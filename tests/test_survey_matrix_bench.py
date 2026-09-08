"""Binning, survey responses, and Fisher bookkeeping against algebraic oracles."""
import copy
import numpy as np
import pytest
from radiofisher import baofisher as rf, experiments


@pytest.mark.parametrize('kind,kwargs', [
    ('fixed', {'dz': .1}), ('equal_spaced', {'bins': 12}), ('equal_spaced', {'dz': .1}),
    ('const_dr', {'bins': 12}), ('const_dr', {'initial_dz': .1}),
    ('const_dnu', {'bins': 12}), ('const_dnu', {'dnu': 20.}), ('const_dnu', {'initial_dz': .1}),
    ('split_width', {'zsplit': .1}), ('split_width', {'zsplit': 5.}),
])
def test_binning_covers_band_and_preserves_its_coordinate(kind, kwargs):
    e = copy.deepcopy(experiments.CHIME)
    fn = getattr(rf, 'zbins_'+kind)
    args = (e, experiments.cosmo) if kind in ('const_dr', 'const_dnu') else (e,)
    edges, centers = fn(*args, **kwargs)
    low, high = e['nu_line']/e['survey_numax']-1, e['nu_line']/(e['survey_numax']-e['survey_dnutot'])-1
    np.testing.assert_allclose(edges[[0, -1]], [low, high], atol=.01)
    assert np.all(np.diff(edges) > 0)
    np.testing.assert_allclose(centers, (edges[:-1]+edges[1:])/2)
    if kind == 'const_dnu' and 'bins' in kwargs:
        np.testing.assert_allclose(np.diff(e['nu_line']/(1+edges)), -e['survey_dnutot']/12)
    if kind == 'const_dr' and 'bins' in kwargs:
        distance = rf.background_evolution_splines(experiments.cosmo, nsamples=4000)[1]
        spacings = np.diff(distance(edges))
        np.testing.assert_allclose(spacings, spacings.mean(), rtol=1e-4)


def test_overlapping_instruments_use_dish_beam_weighted_parameters():
    first, second = copy.deepcopy(experiments.exptS), copy.deepcopy(experiments.exptS)
    first.update(survey_numax=1000., survey_dnutot=600., Ndish=10, Nbeam=1, Tinst=20., Ddish=10.)
    second.update(survey_numax=900., survey_dnutot=300., Ndish=20, Nbeam=2, Tinst=40., Ddish=20.)
    combined = rf.overlapping_expts({'overlap': (first, second)}, zlow=1., zhigh=1.1, Sarea=2.)
    assert combined['Ndish'] == 50 and combined['Nbeam'] == 1
    assert combined['Tinst'] == 36 and combined['Ddish'] == 18 and combined['Sarea'] == 2
    only_first = rf.overlapping_expts({'overlap': (first, second)}, zlow=1.8, zhigh=1.9)
    assert only_first['Ndish'] == 10 and only_first['Tinst'] == 20


@pytest.mark.parametrize('mode', ['i', 'icyl'])
@pytest.mark.parametrize('custom', [False, True])
def test_interferometer_beam_density_and_spectral_suppression(mode, custom):
    e = dict(mode=mode, Ddish=5., Dmin=10., Dmax=200., Ndish=50, nu_line=1420., dnu=.4)
    c = dict(aperp=1., apar=1., r=2000., rnu=3000., z=1.)
    q = np.array([0., 300., 10000.])
    if custom:
        e['n(x)'] = lambda x: np.full_like(x, 1e5)
    response = rf.interferometer_response(q, np.zeros(3), c, e)
    assert np.isfinite(response).all() and np.all(response > 0)
    broadened = rf.interferometer_response(q, np.full(3, 500.), c, e)
    assert np.all(broadened > response)
    if not custom or mode == 'i':
        assert response[0] == rf.INF_NOISE
    if not custom:
        assert response[-1] == rf.INF_NOISE


def test_voxel_noise_radiometer_time_scaling():
    e = copy.deepcopy(experiments.exptS)
    e.update(Dmin=10., Dmax=200., dnutot=10.)
    for function in [rf.noise_rms_per_voxel, rf.noise_rms_per_voxel_interferom]:
        initial = function(1., e)
        deeper = dict(e, ttot=e['ttot']*4)
        assert function(1., deeper) == pytest.approx(initial/2)


def test_sampled_parameter_combination_equals_design_matrix_projection():
    matrices = [np.array([[2., .5], [.5, 3.]]), np.array([[4., .7], [.7, 5.]])]
    result, names = rf.combined_fisher_matrix(matrices, ['shared', 'bias'], expand=['bias'])
    J0 = np.array([[1, 0, 0], [0, 1, 0]])
    J1 = np.array([[1, 0, 0], [0, 0, 1]])
    np.testing.assert_allclose(result, J0.T@matrices[0]@J0+J1.T@matrices[1]@J1)
    assert names == ['shared', 'bias0', 'bias1']
    np.testing.assert_array_equal(rf.indexes_for_sampled_fns(1, 2, [1]), [1, 2])
    reduced, names = rf.combined_fisher_matrix(matrices, ['shared', 'bias'], exclude=['bias'])
    np.testing.assert_allclose(reduced, [[6.]])


@pytest.mark.parametrize('fs8', [False, True])
@pytest.mark.parametrize('exclude', [[], [1]])
def test_eos_matrix_extension_is_a_quadratic_form(fs8, exclude):
    derivs = rf.eos_fisher_matrix_derivs(experiments.cosmo, None, fsigma8=fs8)
    names = ['bias', 'fs8' if fs8 else 'f', 'aperp', 'apar', 'pk0']
    F = np.diag([1., 2., 3., 4., 5.])
    projected, labels = rf.expand_fisher_matrix(1., derivs, F, names, fsigma8=fs8, exclude=exclude)
    rng = np.random.default_rng(52)
    shift = rng.normal(size=len(labels))
    oldshift = np.array([shift[labels.index(name)] for name in names])
    for i in range(3):
        if i not in exclude:
            oldshift[i+1] += sum(float(fn(.5))*shift[4+j] for j, fn in enumerate(derivs[i]))
    assert shift@projected@shift == pytest.approx(oldshift@F@oldshift)


def test_load_baseline_and_parameter_name_files(tmp_path):
    p = tmp_path/'baseline.dat'
    np.savetxt(p, [[.1, 3], [.2, 7], [.3, 9]])
    fn = rf.load_interferom_file(p)
    assert fn(.15) == pytest.approx(5) and fn(.01) == 1/rf.INF_NOISE
    p.write_text('# A b_HI f\n1 2 3\n')
    assert rf.load_param_names(p) == ['A', 'b_HI', 'f']
