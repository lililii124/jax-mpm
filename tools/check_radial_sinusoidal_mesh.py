"""Run with pvpython to verify the actual rendered warped mesh; no solver run."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from paraview.simple import Sphere, Delete, CreateView, Show, Render, SaveScreenshot
from paraview import servermanager
from vtkmodules.util.numpy_support import vtk_to_numpy

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("radial_scene", root / "mpm_jax/paraview_scene.py")
scene = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scene)
sphere = Sphere()
sphere.Radius = .16
sphere.ThetaResolution = 256
sphere.PhiResolution = 192
profile = {"amplitude_m": .008, "wavelength_m": .08, "axis": "y"}
warped = scene.radial_sinusoidal_mesh(sphere, profile)
checks = []
for center in ([.5,.48,.5], [.5,.8,.5]):
    sphere.Center = center
    sphere.UpdatePipeline()
    warped.UpdatePipeline()
    original = vtk_to_numpy(servermanager.Fetch(sphere).GetPoints().GetData())
    actual = vtk_to_numpy(servermanager.Fetch(warped).GetPoints().GetData())
    rho = np.linalg.norm(original[:,(0,2)]-np.asarray(center)[(0,2),], axis=1)
    height = .008*(np.cos(2*np.pi*rho/.08)-1)
    expected = original.astype(np.float64).copy()
    expected[:,1] += height
    error = float(np.max(np.abs(actual-expected)))
    unwarped = actual-np.asarray(center)
    unwarped[:,1] -= height
    gap_error = float(np.max(np.abs(np.linalg.norm(unwarped,axis=1)-.16)))
    assert error < 1e-7 and gap_error < 1e-7, (error,gap_error)
    assert np.ptp(actual[:,1]-original[:,1]) > .015
    checks.append({"center":center,"points":len(actual),"maximum_mesh_map_error_m":error,
                   "maximum_force_gap_zero_set_error_m":gap_error})
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--screenshot", type=Path)
args = parser.parse_args()
view = CreateView("RenderView")
view.OrientationAxesVisibility = 0
view.CenterAxesVisibility = 0
view.ViewSize = [640,640]
surface = Show(warped,view)
surface.DiffuseColor = [.59,.65,.71]
view.CameraPosition = [1.0, .45, 1.0]
view.CameraFocalPoint = [.5,.77,.5]
view.CameraViewUp = [0,1,0]
view.ResetCamera()
Render(view)
if args.screenshot:
    args.screenshot.parent.mkdir(parents=True,exist_ok=True)
    SaveScreenshot(str(args.screenshot.resolve()),view,ImageResolution=[640,640])
Delete(warped)
Delete(sphere)
print(json.dumps({"status":"pass", "actual_paraview_mesh_checks":checks}, indent=2))
