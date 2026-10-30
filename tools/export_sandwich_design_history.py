"""Export fixed-mass control snapshots as core-volume fraction fields."""
from pathlib import Path
import argparse,base64,hashlib,json,os,struct
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_name,'4')
import numpy as np


def export(bundle,output):
    import jax
    jax.config.update('jax_enable_x64',True)
    import jax.numpy as jnp
    from examples.blast_sandwich_design import Settings,build_model
    bundle,output=Path(bundle),Path(output)
    provenance=json.loads(bundle.with_suffix('.json').read_text(encoding='utf-8'))
    if hashlib.sha256(bundle.read_bytes()).hexdigest()!=provenance['controls_artifact_sha256']:
        raise ValueError('Control bundle differs from its receipt')
    with np.load(bundle,allow_pickle=False) as saved:controls=saved['controls']
    if controls.ndim!=2 or controls.shape[1]!=1728 or len(controls)!=len(provenance['states']) or not np.isfinite(controls).all():
        raise ValueError('Expected finite 1728-variable snapshots with provenance')
    model=build_model(Settings(xy_step=.01,z_step=.001,controls=(24,24,3),stage='fixed-mass'))
    properties=jax.jit(model['properties'])
    position=np.asarray(model['support'].positions);volume=np.asarray(model['support'].volume)
    centres=position.reshape(-1,8,3).mean(axis=1)
    axes=[np.unique(centres[:,axis]) for axis in range(3)]
    shape=tuple(len(axis) for axis in axes)
    np.testing.assert_allclose(centres,np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3),rtol=0,atol=1e-14)
    skin=np.asarray(model['geometry']['skin']).reshape(-1,8)
    assert np.all(skin==skin[:,:1])
    core=(~skin[:,0]).reshape(shape)
    layers=np.flatnonzero(core[0,0])
    assert len(layers)>0 and np.all(np.diff(layers)==1) and np.array_equal(core,np.broadcast_to(core[0,0],shape))
    first,last=int(layers[0]),int(layers[-1])+1
    edges=[axis[0]-.5*np.median(np.diff(axis))+np.arange(len(axis)+1)*np.median(np.diff(axis)) for axis in axes]
    edges[2]=edges[2][first:last+1]
    extent=' '.join(str(v) for n in [shape[0],shape[1],last-first] for v in (0,n))
    cell_volume=volume.reshape(-1,8).sum(axis=1)
    output.mkdir(parents=True,exist_ok=False);records=[]
    def array(name,values):
        raw=np.asarray(values,dtype='<f8').tobytes()
        data=base64.b64encode(struct.pack('<I',len(raw))+raw).decode('ascii')
        return f'<DataArray type="Float64" Name="{name}" NumberOfComponents="1" format="binary">{data}</DataArray>'
    for index,(q,source) in enumerate(zip(controls,provenance['states'])):
        qsha=hashlib.sha256(q.tobytes()).hexdigest()
        if qsha!=source['controls_sha256']:raise ValueError('Snapshot control hash mismatch')
        phi,density,_,_=properties(jnp.asarray(q));phi,density=np.asarray(phi),np.asarray(density)
        mass=float(np.dot(volume,density))
        if not np.isfinite(phi).all() or abs(mass/1.8468-1)>=1e-10:raise ValueError('Invalid fixed-mass field')
        mean=np.sum(phi.reshape(-1,8)*volume.reshape(-1,8),axis=1)/cell_volume
        field=mean.reshape(shape)[:,:,first:last].ravel(order='F')
        np.testing.assert_allclose(np.sum(cell_volume*mean)*2700,mass,rtol=1e-12)
        xml='<?xml version="1.0"?><VTKFile type="RectilinearGrid" version="0.1" byte_order="LittleEndian">'
        xml+=f'<RectilinearGrid WholeExtent="{extent}"><Piece Extent="{extent}"><CellData Scalars="core_fraction">'
        xml+=array('core_fraction',field)+'</CellData><Coordinates>'
        xml+=''.join(array(name,values) for name,values in zip(('X','Y','Z'),edges))
        xml+='</Coordinates></Piece></RectilinearGrid></VTKFile>'
        path=output/f'design-{index:03d}.vtr';path.write_text(xml,encoding='utf-8')
        records.append(dict(index=index,source=source,file=path.name,mass_kg=mass,
            fraction_range=[float(field.min()),float(field.max())],
            field_sha256=hashlib.sha256(field.tobytes()).hexdigest(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report=dict(scope='Discrete retained material designs; not a dynamic trajectory or independently verified optimum.',
        source_bundle_sha256=provenance['controls_artifact_sha256'],frames=records,spacing_m=[.01,.01,.001],
        field='Reference-volume average of the eight original integration points per core cell',
        color_range=[.08,.35],physical_scale=1,exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (output/'export.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return dict(frames=len(records),output=str(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps(export(args.bundle,args.output)))
