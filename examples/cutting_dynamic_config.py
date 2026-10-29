"""Tracing-only configuration copy; actual inputs retain original validation.

Do not pass an unvalidated dt to the executable. The host wrapper calls the
original Config constructor through dataclasses.replace for each actual dt.
No production Config or original material/contact implementation is modified.
"""
from dataclasses import fields,make_dataclass,replace
import jax.numpy as jnp

def runtime_config_factory(base):
    # Config's original isfinite(dt) check is a Python input-validation check;
    # it cannot evaluate a tracer. The unchanged original validation therefore
    # runs on each real host dt before entering this tracing-only value holder.
    runtime_type=make_dataclass('ValidatedRuntimeDtConfig',[(f.name,f.type) for f in fields(base)],
        frozen=True,namespace={'dx':type(base).dx})
    static_values={f.name:getattr(base,f.name) for f in fields(base)}
    def config(dt):
        return runtime_type(**{**static_values,'dt':dt})
    return config

def validated_scalar(base,dt,dtype):
    replace(base,dt=float(dt)) # All original Config and boundary checks.
    return jnp.asarray(dt,dtype=dtype)
