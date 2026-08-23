import os

import pytest

from radiofisher import camb_wrapper


def test_run_camb_does_not_change_process_working_directory(tmp_path, monkeypatch):
    parameter_directory = tmp_path / "paramfiles"
    parameter_directory.mkdir()
    (parameter_directory / "params.ini").write_text("output_root=test\n")
    executable_directory = tmp_path / "camb-bin"
    executable_directory.mkdir()
    monkeypatch.chdir(tmp_path)
    original_cwd = os.getcwd()

    def fake_check_output(command, cwd, text):
        assert command == [
            str(executable_directory / "camb"),
            str(parameter_directory / "params.ini"),
        ]
        assert cwd == str(executable_directory)
        assert text is True
        return "sigma8 = 0.81\nz_EQ = 3400.0\n"

    monkeypatch.setattr(camb_wrapper.subprocess, "check_output", fake_check_output)

    values = camb_wrapper.run_camb("params.ini", executable_directory)

    assert os.getcwd() == original_cwd
    assert values["sigma8"] == pytest.approx(0.81)
    assert values["z_EQ"] == pytest.approx(3400.0)
