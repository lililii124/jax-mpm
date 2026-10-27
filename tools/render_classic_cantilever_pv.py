"""ParaView 6.2 OSMesa render of the actual SIMP density field history."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import vtk
from vtk.util.numpy_support import numpy_to_vtk


LENGTH, HEIGHT, THICKNESS = 0.75, 0.25, 0.1875
X0, Y0, Z0 = 0.125, 0.375, 0.40625
ISO = 0.20


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""): h.update(chunk)
    return h.hexdigest()


def image_frame_count(path):
    with Image.open(path) as im:
        return int(im.n_frames)


def make_pipeline(field, threshold=ISO):
    field=np.asarray(field,dtype=np.float32)
    shape=field.shape
    spacing=(LENGTH/shape[0],HEIGHT/shape[1],THICKNESS/shape[2])
    # Add one zero-density layer around the voxel design, so the same contour
    # generates an exterior and any internal void surfaces for every frame.
    padded=np.pad(field,1,mode="constant",constant_values=0)
    data=vtk.vtkImageData()
    data.SetDimensions(padded.shape[0]+1,padded.shape[1]+1,padded.shape[2]+1)
    data.SetSpacing(*spacing)
    data.SetOrigin(X0-spacing[0],Y0-spacing[1],Z0-spacing[2])
    scalars=numpy_to_vtk(np.ascontiguousarray(padded.ravel(order="F")),deep=True,array_type=vtk.VTK_FLOAT)
    scalars.SetName("solid_fraction")
    data.GetCellData().SetScalars(scalars)
    c2p=vtk.vtkCellDataToPointData();c2p.SetInputData(data);c2p.PassCellDataOn()
    contour=vtk.vtkContourFilter();contour.SetInputConnection(c2p.GetOutputPort());contour.SetValue(0,threshold)
    normals=vtk.vtkPolyDataNormals();normals.SetInputConnection(contour.GetOutputPort());normals.SplittingOff();normals.ConsistencyOn()
    mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(normals.GetOutputPort())
    mapper.ScalarVisibilityOn();mapper.SetScalarModeToUsePointFieldData();mapper.SelectColorArray("solid_fraction")
    mapper.SetScalarRange(0.0,1.0)
    lut=vtk.vtkColorTransferFunction()
    for x,rgb in ((0.0,(0.10,0.19,0.30)),(0.2,(0.08,0.43,0.56)),(0.4,(0.30,0.72,0.67)),(0.65,(0.89,0.77,0.43)),(1.0,(0.81,0.31,0.21))):
        lut.AddRGBPoint(x,*rgb)
    mapper.SetLookupTable(lut)
    actor=vtk.vtkActor();actor.SetMapper(mapper)
    prop=actor.GetProperty();prop.SetAmbient(0.20);prop.SetDiffuse(0.78);prop.SetSpecular(0.12);prop.SetSpecularPower(24)
    return data, c2p, contour, normals, mapper, lut, actor


def make_binary_surface(field):
    """Build exact exterior voxel faces without cell-to-point interpolation."""
    solid=np.asarray(field).reshape((48,16,12))>=0.5
    spacing=(LENGTH/48,HEIGHT/16,THICKNESS/12)
    point_ids={};points=[];quads=[]
    faces=(
        ((-1,0,0),((0,0,0),(0,0,1),(0,1,1),(0,1,0))),
        ((1,0,0),((1,0,0),(1,1,0),(1,1,1),(1,0,1))),
        ((0,-1,0),((0,0,0),(1,0,0),(1,0,1),(0,0,1))),
        ((0,1,0),((0,1,0),(0,1,1),(1,1,1),(1,1,0))),
        ((0,0,-1),((0,0,0),(0,1,0),(1,1,0),(1,0,0))),
        ((0,0,1),((0,0,1),(1,0,1),(1,1,1),(0,1,1))),
    )
    def point_id(key):
        if key not in point_ids:
            point_ids[key]=len(points)
            points.append((X0+key[0]*spacing[0],Y0+key[1]*spacing[1],Z0+key[2]*spacing[2]))
        return point_ids[key]
    exposed_faces=0
    for i,j,k in np.argwhere(solid):
        for (di,dj,dk),corners in faces:
            ni,nj,nk=i+di,j+dj,k+dk
            if 0<=ni<48 and 0<=nj<16 and 0<=nk<12 and solid[ni,nj,nk]:
                continue
            quads.append([point_id((i+a,j+b,k+c)) for a,b,c in corners])
            exposed_faces+=1
    vtk_points=vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(np.asarray(points,dtype=np.float64),deep=True,array_type=vtk.VTK_DOUBLE))
    polygons=vtk.vtkCellArray()
    for face in quads:
        quad=vtk.vtkQuad()
        for n,pid in enumerate(face):quad.GetPointIds().SetId(n,pid)
        polygons.InsertNextCell(quad)
    poly=vtk.vtkPolyData();poly.SetPoints(vtk_points);poly.SetPolys(polygons)
    normals=vtk.vtkPolyDataNormals();normals.SetInputData(poly);normals.ConsistencyOn();normals.SplittingOn();normals.SetFeatureAngle(30.0)
    mapper=vtk.vtkPolyDataMapper();mapper.SetInputConnection(normals.GetOutputPort());mapper.ScalarVisibilityOff()
    actor=vtk.vtkActor();actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.16,0.49,0.53);actor.GetProperty().SetAmbient(0.20);actor.GetProperty().SetDiffuse(0.78);actor.GetProperty().SetSpecular(0.12);actor.GetProperty().SetSpecularPower(24)
    remaining=set(map(tuple,np.argwhere(solid)));components=0
    while remaining:
        components+=1;stack=[remaining.pop()]
        while stack:
            i,j,k=stack.pop()
            for di,dj,dk in ((-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)):
                neighbor=(i+di,j+dj,k+dk)
                if neighbor in remaining:remaining.remove(neighbor);stack.append(neighbor)
    return actor, int(solid.sum()), int(exposed_faces), int(components)


def render(field, output, camera=None, size=800, threshold=ISO):
    _,_,_,_,mapper,lut,actor=make_pipeline(field,threshold=threshold)
    return render_actor(actor,output,camera,size)


def render_actor(actor,output,camera=None,size=800):
    renderer=vtk.vtkRenderer();renderer.SetBackground(0.965,0.974,0.986);renderer.AddActor(actor)
    renderer.SetUseDepthPeeling(1);renderer.SetMaximumNumberOfPeels(4);renderer.SetOcclusionRatio(0.1)
    rw=vtk.vtkRenderWindow();rw.SetOffScreenRendering(1);rw.SetMultiSamples(0);rw.SetSize(size,size);rw.AddRenderer(renderer)
    cam=renderer.GetActiveCamera();cam.ParallelProjectionOn()
    if camera is None:
        center=(X0+LENGTH/2,Y0+HEIGHT/2,Z0+THICKNESS/2)
        cam.SetFocalPoint(*center);cam.SetPosition(center[0]+0.70,center[1]-0.78,center[2]+1.05);cam.SetViewUp(0,1,0);cam.SetParallelScale(0.43)
        renderer.ResetCameraClippingRange()
        camera={"position":list(cam.GetPosition()),"focal_point":list(cam.GetFocalPoint()),"view_up":list(cam.GetViewUp()),"parallel_scale":cam.GetParallelScale()}
    else:
        cam.SetPosition(*camera["position"]);cam.SetFocalPoint(*camera["focal_point"]);cam.SetViewUp(*camera["view_up"]);cam.SetParallelScale(camera["parallel_scale"])
        renderer.ResetCameraClippingRange()
    rw.Render()
    capture=vtk.vtkWindowToImageFilter();capture.SetInput(rw);capture.SetInputBufferTypeToRGB();capture.ReadFrontBufferOff();capture.Update()
    writer=vtk.vtkPNGWriter();writer.SetFileName(str(output));writer.SetInputConnection(capture.GetOutputPort());writer.Write()
    rw.Finalize()
    return camera


def downsample(source,dest):
    with Image.open(source) as im: im.convert("RGB").resize((400,400),Image.Resampling.LANCZOS).save(dest)


def gif(paths,out,duration=120):
    rgb=[Image.open(p).convert("RGB") for p in paths]
    sample_ids=np.unique(np.linspace(0,len(rgb)-1,min(len(rgb),12),dtype=int))
    palette_sheet=Image.new("RGB",(400*len(sample_ids),400))
    for col,idx in enumerate(sample_ids):palette_sheet.paste(rgb[int(idx)],(400*col,0))
    palette=palette_sheet.quantize(colors=192,method=Image.Quantize.MEDIANCUT)
    frames=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in rgb]
    frames[0].save(out,save_all=True,append_images=frames[1:],duration=duration,loop=0,optimize=False,disposal=2)
    for frame in frames: frame.close()
    for frame in rgb: frame.close()
    palette.close();palette_sheet.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--iso",type=float,default=0.35,help="display isosurface density; separate from the phi=0.50 audit")
    p.add_argument("--camera-manifest",type=Path,default=None,help="reuse a prior render manifest camera")
    p.add_argument("--final-only",action="store_true",help="render only the optimized field, for the phi=0.50 audit view")
    p.add_argument("--final-binary-input",type=Path,default=None,help="render rho_binary from the independently thresholded solve")
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    initial=np.load(args.input/"refined-initial.npz")["rho"]
    optimized=np.load(args.input/"refined-optimized.npz")["rho"]
    final_source="continuous optimized SIMP field"
    if args.final_binary_input:
        optimized=np.load(args.final_binary_input)["rho_binary"]
        final_source="rho_binary from the independent phi=0.50 thresholded solve; ersatz void retained in the saved field"
    evolution=np.load(args.input/"design-evolution.npz")
    frames=np.asarray(evolution["rho_history"])
    betas=np.asarray(evolution["beta"]);iterations=np.asarray(evolution["iteration"])
    if frames.ndim!=4 or frames.shape[1:]!=(36,12,9):raise ValueError(f"unexpected coarse design history shape {frames.shape}")
    if initial.shape!=(48,16,12) or optimized.shape!=initial.shape:raise ValueError("refined field shapes differ from expected 64-background-grid beam")
    camera=None
    if args.camera_manifest:
        camera=json.loads(args.camera_manifest.read_text(encoding="utf-8"))["camera"]
    render_dir=args.output/"rendered-800";small_dir=args.output/"frames-400";render_dir.mkdir(exist_ok=True);small_dir.mkdir(exist_ok=True)
    manifests={"renderer":"ParaView/VTK 6.2 OSMesa pvpython","visualization":"continuous p=3 SIMP density for the design history; final-only may use the separately thresholded rho_binary. Neither visualization certifies manufacturability.","simp_penalty":3,"display_contour_fraction":args.iso,"binary_audit_threshold":0.50,"field_source":final_source if args.final_only else "continuous p=3 SIMP density history","view":"optimized_only" if args.final_only else "full_design_history","deformation_scale":1.0,"camera":None,"color_range":[0.0,1.0],"source_background_grid":48,"render_resolution_px":[800,800],"output_resolution_px":[400,400],"optimization_frames":int(len(frames)),"frame_labels":[{"beta":float(b),"iteration":int(i)} for b,i in zip(betas,iterations)]}
    if args.final_only:
        b=render_dir/"optimized.png"
        if args.final_binary_input:
            actor,solid_voxels,exposed_faces,components=make_binary_surface(optimized)
            camera=render_actor(actor,b,camera)
            manifests["binary_surface"]={"construction":"direct exterior faces of occupied rho_binary cells; no cell-to-point interpolation or contour smoothing","solid_cells":solid_voxels,"exposed_quad_faces":exposed_faces,"six_connected_components":components,"cell_dimensions":[48,16,12]}
        else:
            camera=render(optimized,b,camera,threshold=args.iso)
        downsample(b,small_dir/"optimized.png")
    else:
        a=render_dir/"initial.png";camera=render(initial,a,camera,threshold=args.iso);downsample(a,small_dir/"initial.png")
        b=render_dir/"optimized.png";camera=render(optimized,b,camera,threshold=args.iso);downsample(b,small_dir/"optimized.png")
    frame_paths=[]
    if not args.final_only:
        for idx,field in enumerate(frames):
            big=render_dir/f"iteration-{idx:03d}.png";camera=render(field,big,camera,threshold=args.iso)
            small=small_dir/f"iteration-{idx:03d}.png";downsample(big,small);frame_paths.append(small)
    manifests["camera"]=camera
    initial_gif=args.output/"cantilever-initial.gif";optimized_gif=args.output/"cantilever-optimized.gif";evolution_gif=args.output/"cantilever-design-evolution.gif"
    if args.final_only:
        gif([small_dir/"optimized.png"],optimized_gif,duration=1000)
        outputs=(optimized_gif,)
    else:
        gif([small_dir/"initial.png"],initial_gif,duration=1000)
        gif([small_dir/"optimized.png"],optimized_gif,duration=1000)
        gif(frame_paths,evolution_gif,duration=120)
        outputs=(initial_gif,optimized_gif,evolution_gif)
    manifests["gifs"]={p.name:{"sha256":sha256(p),"frames":image_frame_count(p),"bytes":p.stat().st_size} for p in outputs}
    if not args.final_only:manifests["unique_history_frames"]=len({sha256(p) for p in frame_paths})
    manifests["camera_and_color_range_shared"]=True
    (args.output/"render-manifest.json").write_text(json.dumps(manifests,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(manifests,separators=(",",":")),flush=True)


if __name__=="__main__":main()
