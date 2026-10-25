import numpy as np
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('fibre_hash_layout',Path(__file__).resolve().parents[1]/'mpm_jax/fibre_hash_layout.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
build_tables,lookup_numpy=module.build_tables,module.lookup_numpy


def test_hash_sparse_global_addresses_and_missing_stencils():
    ids=np.unique(np.concatenate([np.arange(0,5000,3),np.arange(400_000_000,400_005_000,5)]))
    keys,values,rounds=build_tables(ids)
    assert np.array_equal(lookup_numpy(ids,keys,values),np.arange(len(ids)))
    assert (lookup_numpy(np.array([-1,2,400_000_002,2**31-1]),keys,values)==-1).all()
    assert 1<=rounds<=128 and keys.nbytes+values.nbytes<ids.max()
