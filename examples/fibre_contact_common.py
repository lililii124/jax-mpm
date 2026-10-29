"""Shared fibre-contact serialization and complete checkpoint restoration."""

import hashlib
import json
import math

import numpy as np


def descriptor(array):
    value = np.ascontiguousarray(array)
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(memoryview(value).cast('B')).hexdigest()}


def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    return None if isinstance(value, float) and not math.isfinite(value) else value


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def restore_contact_history(template, checkpoint, array_module=np, *, parameters=None):
    """Restore every law carry leaf; never reset an adhesive branch on resume."""
    values = {}
    for key, reference in template._asdict().items():
        name = 'contact_active' if key == 'active' else 'contact_law_valid' if key == 'valid' else 'contact_local_'+key
        if name not in checkpoint:
            raise ValueError(f'Missing complete local-contact history: {name}')
        value = np.asarray(checkpoint[name])
        expected = np.asarray(reference)
        if value.shape != expected.shape or value.dtype != expected.dtype or not np.isfinite(value).all():
            raise ValueError(f'Invalid local-contact history shape, dtype or value: {name}')
        if key == 'gap' and np.any(value < 0):
            raise ValueError('Cohesive ring gaps cannot be negative')
        if key == 'valid' and not value.all():
            raise ValueError('Invalid local-contact state is not resumable')
        values[key] = array_module.asarray(value)
    if parameters is not None and hasattr(parameters, 'base'):
        profile = np.asarray(values['previous_profile'])
        base_profile = np.asarray(parameters.base.profile)
        ratio = profile/base_profile
        if (np.any(profile <= 0) or not np.allclose(ratio, ratio[..., :1], rtol=1e-12, atol=0)):
            raise ValueError('Saved contact curvature profile is not a positive circular paraboloid')
        gap = np.asarray(values['gap'])
        work = np.broadcast_to(np.asarray(parameters.work_of_adhesion), gap.shape[:-1])[..., None]
        interaction_range = np.broadcast_to(np.asarray(parameters.interaction_range), gap.shape[:-1])[..., None]
        active = np.any(np.where(work > 0, gap < interaction_range, gap == 0), axis=-1)
        if not np.array_equal(active, np.asarray(values['active'])):
            raise ValueError('Saved adhesive branch flag contradicts its ring gaps')
    return type(template)(**values)


def restore_contact_model(parameters, checkpoint):
    """Retain the original BEM table across CPUs, with independent table checks."""
    if not hasattr(parameters, 'base'):
        return parameters
    import jax.numpy as jnp
    base = parameters.base
    values = {}
    for key, reference in base._asdict().items():
        name = 'contact_bem_'+key
        if name not in checkpoint:
            raise ValueError('Resume requires the complete original BEM model table')
        value, expected = np.asarray(checkpoint[name]), np.asarray(reference)
        if value.shape != expected.shape or value.dtype != expected.dtype or not np.isfinite(value).all():
            raise ValueError('Invalid saved BEM model table')
        if key not in ('compliance', 'stiffness') and not np.array_equal(value, expected):
            raise ValueError('Saved BEM geometry or parameters differ from the physical plan')
        values[key] = value
    C, K = values['compliance'], values['stiffness']
    if (not np.allclose(C, np.asarray(base.compliance), rtol=1e-12, atol=0)
            or not np.allclose(C, C.T, rtol=1e-12, atol=0)
            or not np.allclose(K, K.T, rtol=1e-12, atol=0)
            or np.linalg.norm(C@K-np.eye(len(C)), ord=np.inf) > 1e-9):
        raise ValueError('Saved BEM compliance or inverse is inconsistent')
    try:
        np.linalg.cholesky(C)
    except np.linalg.LinAlgError as error:
        raise ValueError('Saved BEM compliance is not positive definite') from error
    return parameters._replace(base=type(base)(**{key: jnp.asarray(value) for key, value in values.items()}))
