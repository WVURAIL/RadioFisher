"""Configuration syntax, comments and deliberate malformed-input refusal."""
import numpy as np
import pytest
from radiofisher import read_config as cfg


@pytest.mark.parametrize('text,value', [('True', True), ('FALSE # x', False), ('None', None),
                                      (' 2.5 # comment', 2.5), ('word', 'word'), ('[1,2]', [1,2]),
                                      ('linspace(1,3,3)', [1,2,3]), ('logspace(0,2,3)', [1,10,100])])
def test_supported_values(text, value):
    np.testing.assert_equal(cfg.parse(text), value)


@pytest.mark.parametrize('value', ['[1,', 'linspace(1,2)', 'linspace(1,2,-3)'])
def test_malformed_structures_are_rejected(value):
    with pytest.raises(ValueError):
        cfg.parse(value)


def test_file_parsing_and_debug_output(tmp_path, capsys):
    path = tmp_path/'settings.ini'
    path.write_text('[survey]\narea=3\nname=CHIME # test\n[grid]\nk=logspace(-2,0,3)\n')
    values = cfg.load_config(path, debug=True)
    assert values['area'] == 3 and values['name'] == 'CHIME'
    np.testing.assert_allclose(values['k'], [0.01, 0.1, 1])
    assert 'area' in capsys.readouterr().out
