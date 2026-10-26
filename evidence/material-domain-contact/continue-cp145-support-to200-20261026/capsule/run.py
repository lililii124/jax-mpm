import importlib, json, runpy, sys
sys.path.insert(0, "/work/source")
import mpm_jax
modules=("examples.metal_sphere_impact_fracture", "mpm_jax.total_lagrangian",
"mpm_jax.metal_tl_midpoint", "mpm_jax.material_domain_contact",
"mpm_jax.fragment_v6_discrete_work", "mpm_jax.fragment_v6_midpoint_strength",
"mpm_jax.fragment_multifield_v6_mpm", "mpm_jax.fragment_v6_matrixfree_projection",
"mpm_jax.materials.linear_metal", "mpm_jax.materials.shock_metal",
"mpm_jax.tl_fracture_support")
for name in modules: importlib.import_module(name)
print(json.dumps({"startup_imports":"passed", "source_commit":"de31cc2",
"source_root":"/work/source", "import_count":len(modules)},separators=(",",":")),flush=True)
runpy.run_path("/work/source/run_metal_tl_fresh.py",run_name="__main__")
