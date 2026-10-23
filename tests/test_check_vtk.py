import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from tools import check_vtk


def saved_fields():
    return {
        "x": np.arange(18, dtype=float).reshape(3, 2, 3) / 20,
        "speed": np.array([[0., 0.], [1., 2.], [3., 4.]]),
        "damage": np.array([[0., 0.], [.2, .3], [.6, 1.]]),
        "temperature": np.array([[300., 300.], [301., 302.], [303., 304.]]),
        "failed": np.array([[0, 0], [0, 0], [0, 1]], dtype=np.uint8),
        "body": np.array([0, 1], dtype=np.int32),
    }


def write_mapped_case(folder):
    fields = saved_fields()
    np.savez(folder / "trajectory.npz", time=[0., .5, 1.],
             body=fields["body"], failed=fields["failed"])
    for name, values in fields.items():
        if name != "body":
            np.save(folder / f"trajectory_{name}.npy", values)
    (folder / "summary.json").write_text(json.dumps({"completed": True}))
    return fields


def test_completed_index_loads_mmap_only_scalars_and_releases_files(tmp_path):
    expected = write_mapped_case(tmp_path)
    indices, positions, fields = check_vtk.trajectory_samples(tmp_path, {"completed": True})
    assert indices == [0, 2]
    np.testing.assert_array_equal(positions, expected["x"][[0, 2]])
    assert set(fields) == set(expected) - {"x"}
    for name, values in fields.items():
        np.testing.assert_array_equal(values, expected[name] if name == "body" else expected[name][[0, 2]])
    for path in tmp_path.glob("trajectory_*.npy"):
        path.unlink()
    np.testing.assert_array_equal(fields["damage"], expected["damage"][[0, 2]])


@pytest.mark.parametrize("summary", [{}, {"completed": False}, {"completed": True}])
def test_archive_only_legacy_fields_remain_readable(tmp_path, summary):
    expected = saved_fields()
    np.savez(tmp_path / "trajectory.npz", time=[0., .5, 1.], **expected)
    indices, positions, fields = check_vtk.trajectory_samples(tmp_path, summary)
    np.testing.assert_array_equal(positions, expected["x"][indices])
    for name, values in fields.items():
        np.testing.assert_array_equal(values, expected[name] if name == "body" else expected[name][indices])


def test_partial_mmaps_do_not_replace_completed_archive_frames(tmp_path):
    expected = saved_fields()
    np.savez(tmp_path / "trajectory.npz", time=[0., .5, 1.],
             x=expected["x"], damage=expected["damage"])
    np.save(tmp_path / "trajectory_damage.npy", np.full((5, 2), .9))
    np.save(tmp_path / "trajectory_speed.npy", np.full((5, 2), 99.))
    _, _, fields = check_vtk.trajectory_samples(tmp_path, {"completed": False})
    assert set(fields) == {"damage"}
    np.testing.assert_array_equal(fields["damage"], expected["damage"][[0, 2]])


@pytest.mark.parametrize("corrupt", [None, "damage", "speed", "temperature"])
def test_readback_checks_scalars_missing_from_npz_index(tmp_path, monkeypatch, corrupt):
    expected = write_mapped_case(tmp_path)
    requested = []

    class Reader:
        def SetFileName(self, filename):
            self.frame = int(Path(filename).stem.rsplit("_", 1)[1])

        def Update(self):
            pass

        def GetOutput(self):
            def array(name):
                requested.append(name)
                value = expected[name] if name == "body" else expected[name][self.frame]
                return value + 1 if name == corrupt else value

            return SimpleNamespace(
                GetNumberOfPoints=lambda: 2,
                GetPoints=lambda: SimpleNamespace(GetData=lambda: expected["x"][self.frame]),
                GetPointData=lambda: SimpleNamespace(GetArray=array),
            )

    reader_module = ModuleType("vtkmodules.vtkIOXML")
    reader_module.vtkXMLPolyDataReader = Reader
    numpy_module = ModuleType("vtkmodules.util.numpy_support")
    numpy_module.vtk_to_numpy = np.asarray
    monkeypatch.setitem(sys.modules, "vtkmodules.vtkIOXML", reader_module)
    monkeypatch.setitem(sys.modules, "vtkmodules.util.numpy_support", numpy_module)
    monkeypatch.setattr(sys, "argv", ["check_vtk.py", str(tmp_path)])
    if corrupt is None:
        check_vtk.main()
        assert set(requested) == set(expected) - {"x"}
    else:
        with pytest.raises(AssertionError):
            check_vtk.main()
        assert corrupt in requested
