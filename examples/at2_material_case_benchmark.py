"""Run an actual zero-initial-damage anisotropic or hard-inclusion load path.

Uses the baseline accepted-state integrator, KKT, rollback and energy checks.
The isolated process supplies an extended variational model to that integrator.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',choices=['anisotropic','hard-inclusion','hard-tough-inclusion'],required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--increments',type=int,default=24);parser.add_argument('--displacement',type=float,default=8e-5);parser.add_argument('--angle',type=float,default=25.);parser.add_argument('--anisotropy-ratio',type=float,default=9.)
    parser.add_argument('--inclusion-x',type=float,default=.65);parser.add_argument('--inclusion-y',type=float,default=.53);parser.add_argument('--inclusion-radius',type=float,default=.12);parser.add_argument('--modulus-ratio',type=float,default=8.)
    parser.add_argument('--gc-ratio',type=float,default=None)
    args=parser.parse_args()
    import mpm_jax.at2_quasistatic as base
    from mpm_jax.at2_material_cases import MaterialCaseModel
    from examples.at2_sharp_crack_benchmark import run
    models=[];verification=[];original=base.Model
    def factory(*aa,**kw):
        model=MaterialCaseModel(*aa,case=args.case,preferred_angle_degrees=args.angle,anisotropy_ratio=args.anisotropy_ratio,inclusion_center=(args.inclusion_x,args.inclusion_y),inclusion_radius=args.inclusion_radius,inclusion_modulus_ratio=args.modulus_ratio,inclusion_gc_ratio=args.gc_ratio if args.gc_ratio is not None else (10. if args.case=='hard-tough-inclusion' else 1.),**kw);models.append(model)
        u=np.c_[2e-5*model.xy[:,0]+1e-5*model.xy[:,1],4e-5*model.xy[:,1]].ravel();d=.2*np.exp(-abs(model.xy[:,1]-.5)/model.ell)
        direction=np.random.default_rng(274).normal(size=u.shape)*1e-6;h=1e-4
        energy,force,plus,eps,K=model.mechanical(u,d,True);ep,fp,*_=model.mechanical(u+h*direction,d);em,fm,*_=model.mechanical(u-h*direction,d)
        gradient_error=abs((ep-em)/(2*h)-force@direction)/max(abs(force@direction),1e-20)
        tangent_error=abs((fp-fm)/(2*h)-K@direction).max()/max(abs(K@direction).max(),1e-20)
        vals=np.linalg.eigvalsh(eps);tr=vals.sum(axis=1);lam=1e6*.2/(1.2*.6);mu=1e6/2.4;dp=np.sum(model.weight*d[model.nodes],axis=1)
        tensile=.5*lam*np.maximum(tr,0)**2+mu*np.sum(np.maximum(vals,0)**2,axis=1);compressive=.5*lam*np.minimum(tr,0)**2+mu*np.sum(np.minimum(vals,0)**2,axis=1)
        independent=float(np.sum(model.volume*model.particle_modulus_ratio*((1e-7+(1-1e-7)*(1-dp)**2)*tensile+compressive)))
        v={'energy_gradient_relative_error':float(gradient_error),'mechanical_tangent_relative_error':float(tangent_error),'independent_heterogeneous_energy_relative_error':float(abs(independent-energy)/max(abs(energy),1e-20)),'anisotropic_damage_weak_operator_relative_error':model.operator_equivalence(plus,d)}
        v['tensile_damage_driving_relative_error']=float(abs(plus-model.particle_modulus_ratio*tensile).max()/max(abs(plus).max(),1e-20))
        gd=np.einsum('pk,pki->pi',d[model.nodes],model.grad[:,:,:2]);gc=model.gc*model.particle_gc_ratio
        direct_fracture=.5*np.sum(model.volume*gc*(dp**2/model.ell+model.ell*np.einsum('pi,ij,pj->p',gd,model.crack_metric,gd)))
        v['heterogeneous_Gc_local_and_gradient_energy_error']=float(abs(direct_fracture-model.fracture(d))/max(abs(direct_fracture),1e-20))
        dd=.1*d*np.random.default_rng(731).normal(size=d.shape)
        fd=(model.fracture(d+h*dd)-model.fracture(d-h*dd))/(2*h);analytic=dd@(model.fracture_matrix@d)
        v['heterogeneous_Gc_damage_energy_gradient_error']=float(abs(fd-analytic)/max(abs(analytic),1e-20))
        A=model.fracture_matrix+model.assemble_d(2*(1-model.residual)*plus[:,None,None]*model.mass)
        rhs=np.bincount(model.nodes.ravel(),weights=((2*(1-model.residual)*plus*model.volume)[:,None]*model.weight).ravel(),minlength=len(d))
        total_p=model.mechanical(u,d+h*dd)[0]+model.fracture(d+h*dd);total_m=model.mechanical(u,d-h*dd)[0]+model.fracture(d-h*dd)
        exact=dd@(A@d-rhs)
        v['coupled_fixed_displacement_damage_gradient_error']=float(abs((total_p-total_m)/(2*h)-exact)/max(abs(exact),1e-20))
        v['passed']=all(value<1e-6 for value in v.values());assert v['passed'],v;verification.append(v)
        # Persist the constitutive fields before continuation so a bounded job
        # retains its exact material region even if interrupted at a later load.
        folder=Path(args.output)
        (folder/'material-kernel-verification.json').write_text(json.dumps(v,indent=2))
        (folder/'material-case.json').write_text(json.dumps(model.case_description,indent=2))
        np.savez_compressed(folder/'material-fields.npz',particle_modulus_ratio=model.particle_modulus_ratio,particle_gc_ratio=model.particle_gc_ratio,particle_inclusion=model.particle_inclusion,crack_metric=model.crack_metric)
        print(json.dumps({'kernel_verification':v}),flush=True);return model
    config=argparse.Namespace(n=48,ny=49,ell=.04,young=1e6,poisson=.2,gc=.001,residual=1e-7,crack_length=.375,initial_damage_mode='geometry-only',competitor_peak=.9,energy_competitor=True,adaptive_energy=True,max_alternations=400,damage_relaxation=1.6,displacement=args.displacement,increments=args.increments,output=args.output,restart=None,allow_source_migration=False,source_migration_reason=None,max_load_refinements=8,energy_tolerance=.05,stagger_tolerance=1e-7,coupled_kkt=2e-6,strain_limit=.02)
    base.Model=factory
    try:summary=run(config)
    finally:base.Model=original
    model=models[0];out=Path(args.output)
    description={**model.case_description,'new_module_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'mpm_jax/at2_material_cases.py').read_bytes()).hexdigest(),'example_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'all_initial_damage_zero':True,'same_nonlinear_acceptance_checks_as_isotropic_baseline':True,'summary':summary}
    (out/'material-kernel-verification.json').write_text(json.dumps(verification[0],indent=2))
    (out/'material-case.json').write_text(json.dumps(description,indent=2));np.savez_compressed(out/'material-fields.npz',particle_modulus_ratio=model.particle_modulus_ratio,particle_gc_ratio=model.particle_gc_ratio,particle_inclusion=model.particle_inclusion,crack_metric=model.crack_metric)
    print(json.dumps(description,indent=2));return 0 if summary['completed'] else 2

if __name__=='__main__':raise SystemExit(main())
