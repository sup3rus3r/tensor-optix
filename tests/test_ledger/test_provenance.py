import subprocess

from tensor_optix.ledger.provenance import capture_environment


def test_capture_environment_never_raises_without_git(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    env = capture_environment()
    assert env["source_revision"] is None


def test_capture_environment_includes_installed_libraries():
    env = capture_environment()
    assert "numpy" in env["library_versions"]
    assert "torch" in env["library_versions"]
    assert env["library_versions"]["numpy"]
    assert env["library_versions"]["torch"]
    assert "hostname" in env
