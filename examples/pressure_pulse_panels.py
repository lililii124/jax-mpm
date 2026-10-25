"""Four clamped3D material panels under an assumed follower pressure pulse."""
import argparse
from pathlib import Path
import hashlib
import json
import math
import jax
import jax.numpy as jnp
import numpy as np
from mpm_jax import Config,block,fixed_box
from mpm_jax.pressure_pulse import pressure,pressure_integral,surface_force
from mpm_jax.particle_fields import material_metadata
from .common import run
from .material_engineering import tensile_aluminum
from .scenarios import Case,material_model

MODELS=('jc','hjc','jh2','rht')
PEAK_PRESSURES={'jc':50e6,'hjc':15e6,'jh2':80e6,'rht':15e6}
DOMAIN=.24
LOWER=(.03,.114,.03)
UPPER=(.21,.126,.21)
SOURCE=('examples/pressure_pulse_panels.py','mpm_jax/pressure_pulse.py','examples/common.py',
        'examples/material_engineering.py','examples/scenarios.py','mpm_jax/core.py',
        'mpm_jax/initialization.py','mpm_jax/boundaries.py','mpm_jax/materials/history.py',
        'mpm_jax/materials/__init__.py','mpm_jax/materials/johnson_cook.py',
        'mpm_jax/materials/hjc.py','mpm_jax/materials/mat111_point.py',
        'mpm_jax/materials/jh2.py','mpm_jax/materials/rht.py')


def setup(model, *,grid=128,ppc=2,peak_pressure=None,pulse_duration=80e-6,duration=200e-6):
    if model not in MODELS:raise ValueError('Unknown material preset')
    peak_pressure=PEAK_PRESSURES[model] if peak_pressure is None else peak_pressure
    if type(grid) is not int or grid<64 or type(ppc) is not int or ppc<1:raise ValueError('Use grid>=64 and positive integer ppc; engineering defaults128/2')
    if not np.isfinite([peak_pressure,pulse_duration,duration]).all() or not 0<peak_pressure or not 0<pulse_duration<duration:raise ValueError('Require positive pressure and0<pulse-duration<duration')
    if model=='jc':
        material=tensile_aluminum();density=material.density;kind='johnson-cook'
        speed=math.sqrt((material.bulk+4*material.mu/3)/density)
        material_note='Existing tensile_aluminum factory; AA2024-T351 JC parameters and thermal assumptions as documented in MATERIAL_ENGINEERING.md, not a pressure-test calibration'
    else:
        material,density,kind,speed=material_model(model)
        material_note='Existing examples.scenarios.material_model repository demonstration preset; HJC legacy-v3, JH2 selected MAT110, RHT three-surface model as documented in MATERIALS.md'
    dx=DOMAIN/grid
    steps=20*math.ceil(duration/(20*.1*dx/speed))
    dt=duration/steps;frames=steps//20;grip=.01
    boundaries=(fixed_box((0,0,0),(LOWER[0]+grip,DOMAIN,DOMAIN)),
                fixed_box((UPPER[0]-grip,0,0),(DOMAIN,DOMAIN,DOMAIN)),
                fixed_box((0,0,0),(DOMAIN,DOMAIN,LOWER[2]+grip)),
                fixed_box((0,0,UPPER[2]-grip),(DOMAIN,DOMAIN,DOMAIN)))
    config=Config(dim=3,n_grid=grid,domain_size=DOMAIN,dt=dt,material=kind,
                  gravity=(0.,0.,0.),boundary_cells=0,boundaries=boundaries,failure_mode='retain')
    state=block(config,LOWER,UPPER,particles_per_cell=ppc,density=density,material=material,dtype=jnp.float64)
    reference=np.asarray(state.x);counts=np.ceil((np.asarray(UPPER)-LOWER)/dx*ppc).astype(int)
    spacing=(np.asarray(UPPER)-LOWER)/counts
    top=np.isclose(reference[:,1],reference[:,1].max(),rtol=0,atol=1e-12)
    area=jnp.where(jnp.asarray(top),state.volume/spacing[1],0.)
    reference_area=float(area.sum());expected_area=(UPPER[0]-LOWER[0])*(UPPER[2]-LOWER[2])
    if not np.isclose(reference_area,expected_area,rtol=1e-13,atol=0):raise ValueError('Discrete material-face quadrature area differs from plate area')
    centre=top&(np.abs(reference[:,0]-.12)<.005)&(np.abs(reference[:,2]-.12)<.005)
    if not centre.any():raise ValueError('Grid does not sample the central surface')
    clamped=(reference[:,0]<LOWER[0]+grip-1.5*dx)|(reference[:,0]>UPPER[0]-grip+1.5*dx)|(reference[:,2]<LOWER[2]+grip-1.5*dx)|(reference[:,2]>UPPER[2]-grip+1.5*dx)

    def forces(states,time):
        return (surface_force(states[0].F,area,time,peak_pressure,pulse_duration),)

    def observe(states,time):
        current=states[0];position=np.asarray(current.x);velocity=np.asarray(current.v)
        force=np.asarray(forces(states,time)[0]);history=current.history;vol=np.asarray(current.volume);mass=np.asarray(current.mass)
        normal=np.cross(np.asarray(current.F)[:,:,2],np.asarray(current.F)[:,:,0])
        return dict(pressure_Pa=float(pressure(time,peak_pressure,pulse_duration)),load_sample_time_s=float(time),
                    reference_pressure_impulse_Pa_s=float(pressure_integral(time,peak_pressure,pulse_duration)),
                    flat_reference_resultant_impulse_Ns=float(reference_area*pressure_integral(time,peak_pressure,pulse_duration)),
                    current_loaded_area_m2=float(np.sum(np.asarray(area)*np.linalg.norm(normal,axis=1))),
                    resultant_pressure_force_x_N=float(force[:,0].sum()),resultant_pressure_force_y_N=float(force[:,1].sum()),resultant_pressure_force_z_N=float(force[:,2].sum()),
                    centre_surface_deflection_y_m=float((position[centre,1]-reference[centre,1]).mean()),
                    centre_surface_velocity_y_m_per_s=float(velocity[centre,1].mean()),maximum_displacement_m=float(np.linalg.norm(position-reference,axis=1).max()),
                    fully_supported_clamp_max_motion_m=float(np.linalg.norm(position[clamped]-reference[clamped],axis=1).max(initial=0.)),
                    maximum_equivalent_plastic_strain=float(np.asarray(history.plastic_strain).max()),maximum_damage=float(np.asarray(history.damage).max()),
                    maximum_temperature_K=float(np.asarray(history.temperature).max()),maximum_pressure_history_Pa=float(np.asarray(history.pressure).max()),
                    mean_compaction_reference_volume=float(np.dot(vol,np.asarray(history.compaction))/vol.sum()),
                    material_history_energy_J=float(np.dot(mass,np.asarray(history.energy))),material_history_work_J=float(np.dot(mass,np.asarray(history.work))),
                    failed_reference_volume_fraction=float(vol[np.asarray(history.failed)].sum()/vol.sum()),
                    all_material_history_valid=bool(np.asarray(history.valid).all()))

    root=Path(__file__).resolve().parents[1]
    source={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in SOURCE}
    definition=dict(case='pressure-pulse-panel',model=model,material_note=material_note,material_parameters_SI=material_metadata(material),units='m,s,kg,Pa,N,J,K',
                    geometry=dict(lower_m=LOWER,upper_m=UPPER,plate_size_m=[.18,.012,.18],free_span_m=[.16,.16],clamp_width_m=grip,backing=False),
                    boundary_condition='Four grid-node edge strips fixed in all3 velocity components through full thickness; no backing or bottom support; remaining faces traction-free except loaded +Y face',
                    load=dict(law='p(t)=p0*sin(pi*t/T)^2 for0<t<T, zero otherwise; positive pressure acts inward',peak_pressure_Pa=peak_pressure,pulse_duration_s=pulse_duration,
                              pressure_impulse_Pa_s=peak_pressure*pulse_duration/2,flat_reference_resultant_impulse_Ns=reference_area*peak_pressure*pulse_duration/2,
                              free_areal_mass_velocity_scale_m_per_s=peak_pressure*pulse_duration/(2*density*(UPPER[1]-LOWER[1])),
                              velocity_scale_scope='I/(rho*thickness) for an unconstrained flat sheet; dimensional load scale, not clamped-panel response prediction',
                              preset_status='Assumed engineering pressure severity, not a fitted experimental or explosive-source pressure',
                              discrete_flat_reference_impulse_Ns=float(reference_area*dt*np.asarray(pressure(jnp.arange(steps)*dt,peak_pressure,pulse_duration)).sum()),
                              reference_face_area_m2=reference_area,face_particles=int(top.sum()),surface_quadrature='Topmost material particle layer; A0=particle reference volume / actual reference layer thickness',
                              follower_traction='f_p=-p(t)*A0_p*cross(F_p*e_z,F_p*e_x) in Newtons; cofactor(F)*e_y includes current area and normal',
                              temporal_sampling='Existing common.run left-end force sample; work uses trapezoidal particle velocities with that force',
                              damage_scope='Reference material face remains loaded as it deforms, including damaged points; no new crack-face topology or fragment pressure model'),
                    material_eos_scope='JC uses the existing p=K*(1/J-1) density-linear law, adiabatic plastic heating and clipped thermal softening; HJC/JH2/RHT retain their own original EOS and histories',
                    duration_s=duration,planned_steps=steps,frames=frames,substeps=20,time_step_s=dt,initial_acoustic_CFL=dt*speed/dx,initial_CFL_limit=.1,estimated_initial_wave_speed_m_per_s=speed,
                    grid=grid,particles_per_cell_per_axis=ppc,particle_count=len(reference),particle_spacing_m=spacing.tolist(),source_sha256=source,
                    scope='Assumed equivalent mechanical pressure input; no explosive chemistry, JWL or blast-flow coupling; not experimental/IJIE calibration',
                    energy_scope='Actual applied impulse/work and material histories recorded; standard APIC ledger is not a complete discrete energy certificate; numerical loss is not assigned to heat',
                    qualification=dict(full_physical_validation_completed=False,spatial_temporal_convergence_established=False,pressure_experiment_validated=False))
    return Case((state,),(material,),(config,),duration,definition,forces=forces,observer=observe)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--grid',type=int,default=128);p.add_argument('--ppc',type=int,default=2)
    p.add_argument('--pressure','--peak-pressure',dest='peak_pressure',type=float,help='Peak compressive Pa; defaults JC50/HJC15/JH2 80/RHT15MPa');p.add_argument('--pulse-duration',type=float,default=80e-6);p.add_argument('--duration',type=float,default=200e-6)
    p.add_argument('--output',type=Path,default=Path('outputs/pressure-pulse-panels'));p.add_argument('--resume',action='store_true');p.add_argument('--stop-after-steps',type=int)
    a=p.parse_args();jax.config.update('jax_enable_x64',True)
    case=setup(a.model,grid=a.grid,ppc=a.ppc,peak_pressure=a.peak_pressure,pulse_duration=a.pulse_duration,duration=a.duration)
    stop=None
    if a.stop_after_steps is not None:
        if not 0<a.stop_after_steps<=case.description['planned_steps'] or a.stop_after_steps%20:p.error('Stop must be a positive multiple of20 within the fixed physical plan')
        stop=a.stop_after_steps//20
    output=a.output/a.model;output.mkdir(parents=True,exist_ok=True)
    if (output/'checkpoint.json').exists() and not a.resume:p.error('Existing checkpoint requires --resume or a fresh output directory')
    if a.resume and not (output/'checkpoint.json').exists():p.error('--resume requires the original checkpoint directory')
    (output/'case.json').write_text(json.dumps(case.description,indent=2)+'\n')
    print(json.dumps(dict(model=a.model,particles=len(case.states[0].x),planned_steps=case.description['planned_steps'],stop_after_steps=a.stop_after_steps,grid=a.grid,ppc=a.ppc)),flush=True)
    run(case.states,case.materials,case.configs,frames=case.description['frames'],substeps=20,output=output,label='Pressure pulse panel / '+a.model,
        forces=case.forces,observer=case.observer,metadata=case.description,render_output=False,resume=a.resume,stop_after_frame=stop,
        point_fields={**{'reference_'+axis:np.asarray(case.states[0].x)[:,d] for d,axis in enumerate('xyz')},
                      'loaded_surface':np.asarray(np.isclose(np.asarray(case.states[0].x)[:,1],UPPER[1]-.5*case.description['particle_spacing_m'][1],rtol=0,atol=1e-12),dtype=np.int32)})


if __name__=='__main__':main()
