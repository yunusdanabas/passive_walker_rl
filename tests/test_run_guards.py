# tests/test_run_guards.py
from pathlib import Path
from passive_walker.core.runlog import make_run_dir, atomic_write_text, atomic_write_json

def test_make_run_dir_suffix(tmp_path):
    a = make_run_dir("bc", root=tmp_path, run_name="x")
    b = make_run_dir("bc", root=tmp_path, run_name="x")
    assert a != b

def test_make_run_dir_force(tmp_path):
    a = make_run_dir("bc", root=tmp_path, run_name="y", force=False)
    b = make_run_dir("bc", root=tmp_path, run_name="y", force=True)
    assert a == b  # Should reuse the same directory

def test_atomic_write_text(tmp_path):
    p = tmp_path / "x.txt"
    atomic_write_text(p, "hello world")
    assert p.exists()
    assert p.read_text() == "hello world"

def test_atomic_write_json(tmp_path):
    p = tmp_path / "x.json"
    data = {"a": 1, "b": [2, 3]}
    atomic_write_json(p, data)
    assert p.exists()
    import json
    assert json.loads(p.read_text()) == data

def test_make_run_dir_no_run_name(tmp_path):
    run_dir = make_run_dir("test", root=tmp_path, run_name=None)
    assert run_dir.exists()
    assert "test" in run_dir.name

def test_make_run_dir_default_root(tmp_path):
    # Temporarily change cwd to tmp_path to test default root
    import os
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        run_dir = make_run_dir("default_test", run_name="myrun")
        assert run_dir.exists()
        assert "results/default_test" in str(run_dir)
    finally:
        os.chdir(old_cwd)