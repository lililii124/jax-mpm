"""Reproduce Euler-Bernoulli/Timoshenko checks for the published MPM beam.

This uses only the Python standard library and reads model constants from the
example source with a small, literal-only AST evaluator (no JAX import).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples" / "classic_cantilever_topopt.py"
DEFAULT_SUMMARY = ROOT / "docs" / "results" / "jax-design--classic-cantilever" / "summary.json"
DEFAULT_OUTPUT = ROOT / "docs" / "results" / "jax-design--classic-cantilever" / "beam-theory-reference.json"


def numeric_literal(node: ast.AST) -> float:
    """Evaluate only numeric constants and basic arithmetic from source AST."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = numeric_literal(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp):
        left, right = numeric_literal(node.left), numeric_literal(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    raise ValueError(f"not a supported numeric source literal: {ast.dump(node)}")


def model_constants(source_text: str) -> tuple[dict[str, float], dict[str, float]]:
    tree = ast.parse(source_text)
    names = {"E0", "NU", "FORCE_N", "LENGTH", "HEIGHT", "THICKNESS", "X0", "Y0", "Z0"}
    constants: dict[str, float] = {}
    pad: dict[str, float] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    constants[target.id] = numeric_literal(node.value)
    geometry = next(
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "geometry"
    )
    for node in ast.walk(geometry):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            targets = node.targets[0].elts
            if [x.id for x in targets if isinstance(x, ast.Name)] == ["pad_y", "pad_z"]:
                values = node.value.elts
                pad = {"width_y_m": numeric_literal(values[0]), "thickness_z_m": numeric_literal(values[1])}
                break
    missing = names - constants.keys()
    if missing or not pad:
        raise ValueError(f"model constants missing from source: {sorted(missing)}; pad={pad}")
    return constants, pad


def cross(a: list[float], b: list[float]) -> list[float]:
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def close_vector(a: list[float], b: list[float], tol: float = 1e-8) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol * max(1.0, abs(x), abs(y)) for x, y in zip(a, b))


def build_reference(summary: dict, zero_void_audit: dict, source_path: Path = SOURCE) -> dict:
    source_text = source_path.read_text(encoding="utf-8")
    c, pad = model_constants(source_text)
    L, b, t = c["LENGTH"], c["HEIGHT"], c["THICKNESS"]
    E, nu, F = c["E0"], c["NU"], c["FORCE_N"]
    area = b * t
    inertia_z = t * b**3 / 12.0  # F_y bends the beam about the z axis.
    shear_modulus = E / (2.0 * (1.0 + nu))
    kappa = 5.0 / 6.0  # rectangular section, Timoshenko shear correction
    delta_bend = F * L**3 / (3.0 * E * inertia_z)
    delta_shear = F * L / (kappa * shear_modulus * area)
    delta_timoshenko = delta_bend + delta_shear

    audit = zero_void_audit["force_and_reaction_balance"]
    applied_force = audit["applied_resultant_N"]
    applied_moment_origin = audit["applied_moment_about_origin_Nm"]
    load_centroid = [c["X0"] + L, c["Y0"] + b / 2.0, c["Z0"] + t / 2.0]
    root_centroid = [c["X0"], c["Y0"] + b / 2.0, c["Z0"] + t / 2.0]
    expected_origin_moment = cross(load_centroid, applied_force)
    moment_root = [m - r for m, r in zip(applied_moment_origin, cross(root_centroid, applied_force))]
    expected_root_moment = cross([L, 0.0, 0.0], applied_force)
    if not close_vector(applied_force, [0.0, -F, 0.0]):
        raise ValueError(f"published applied resultant does not match model force: {applied_force}")
    if not close_vector(applied_moment_origin, expected_origin_moment):
        raise ValueError(f"published applied moment does not match the centered end-pad load: {applied_moment_origin}")
    if not close_vector(moment_root, expected_root_moment):
        raise ValueError(f"moment shifted to clamp centroid is inconsistent: {moment_root}")
    if not summary["strict_zero_void_audit"]["all_loaded_nodes_retained_and_supported"] or not zero_void_audit["all_loaded_nodes_supported"]:
        raise ValueError("the published zero-void audit did not retain/support all loaded nodes")

    pad_area = pad["width_y_m"] * pad["thickness_z_m"]
    refined = summary["independent_refined_forward"]
    mpm = {}
    for label, key in (("initial_38_percent_SIMP_field", "initial_compliance_Nm"),
                       ("optimized_38_percent_SIMP_field", "optimized_compliance_Nm")):
        compliance = refined[key]
        mpm[label] = {
            "compliance_Nm": compliance,
            "equivalent_load_weighted_pad_displacement_m": compliance / F,
            "ratio_to_full_solid_Timoshenko_compliance": compliance / (F * delta_timoshenko),
            "mass_kg": refined["same_mass_kg"],
            "volume_fraction": refined["initial_and_optimized_volume_fraction"],
        }
    return {
        "case": "analytical end-load cantilever reference and comparison to the published variable-density MPM solves",
        "source": {
            "model_file": str(source_path.relative_to(ROOT)).replace("\\", "/"),
            "model_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "summary_file": str(DEFAULT_SUMMARY.relative_to(ROOT)).replace("\\", "/"),
        },
        "model_and_load_audit": {
            "span_from_fixed_beam_face_to_load_face_m": L,
            "global_clamp_face_x_m": c["X0"],
            "global_load_face_x_m": c["X0"] + L,
            "rectangular_cross_section_y_width_m": b,
            "rectangular_cross_section_z_thickness_m": t,
            "load_pad_y_width_m": pad["width_y_m"],
            "load_pad_z_thickness_m": pad["thickness_z_m"],
            "load_pad_area_m2": pad_area,
            "uniform_equivalent_pad_traction_Pa": F / pad_area,
            "applied_resultant_N": applied_force,
            "applied_moment_about_global_origin_Nm": applied_moment_origin,
            "load_moment_about_clamp_section_centroid_Nm": moment_root,
            "load_line_passes_through_cross_section_centroid": True,
            "audit_force_balance_relative_error": audit["relative_force_balance_error"],
            "audit_moment_balance_relative_error": audit["relative_moment_balance_error"],
            "interpretation": "The code applies a 2x2 Gauss quadrature traction over a finite centered end-face pad. Its resultant and root bending moment match a point force at the free-end centroid for the global beam response; local pad stresses are three-dimensional and are not represented by beam theory.",
        },
        "full_solid_linear_elastic_beam_reference": {
            "material_E_Pa": E,
            "poisson_ratio": nu,
            "shear_modulus_Pa": shear_modulus,
            "cross_section_area_m2": area,
            "second_moment_about_z_m4": inertia_z,
            "Timoshenko_rectangular_shear_factor": kappa,
            "Euler_Bernoulli_bending_deflection_m": delta_bend,
            "Timoshenko_shear_deflection_m": delta_shear,
            "Timoshenko_total_deflection_m": delta_timoshenko,
            "Euler_Bernoulli_compliance_Nm": F * delta_bend,
            "Timoshenko_total_compliance_Nm": F * delta_timoshenko,
            "shear_share_of_Timoshenko_deflection": delta_shear / delta_timoshenko,
            "assumptions": [
                "uniform, fully solid, isotropic linear-elastic rectangular beam",
                "rigid full-section clamp and small deflection",
                "point resultant at the centroid of the free-end cross-section for global response",
                "Timoshenko shear correction 5/6 for a rectangular cross-section",
            ],
        },
        "published_refined_MPM_comparison": {
            "background_grid": refined["background_grid"],
            "material_quadrature_points": refined["material_quadrature_points"],
            "initial_and_optimized_mass_kg": refined["same_mass_kg"],
            "initial_and_optimized_volume_fraction": refined["initial_and_optimized_volume_fraction"],
            "initial_and_optimized_MPM_equivalent_pad_responses": mpm,
            "warning": "These MPM fields contain only about 38% mean material volume fraction with fully solid non-design clamp/load pads and SIMP ersatz stiffness. The analytical values are for a full solid beam. The ratios quantify the difference between these stated models; they are not an MPM discretization error or a same-material validation. The optimized field is nonuniform and must not be interpreted as a uniform beam.",
        },
        "checks": {
            "reported_resultant_matches_source_force": True,
            "reported_origin_moment_matches_centered_end_load": True,
            "shifted_root_moment_matches_F_times_span": True,
            "all_loaded_nodes_retained_and_supported": True,
            "clamp_load_span_aspect_ratio_L_over_height": L / b,
            "beam_is_slender_under_EB_assumption": False,
            "reason_for_reporting_Timoshenko": "L/H=3, so transverse shear is material (about 8.0% of the Timoshenko deflection).",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    audit_path = args.summary.parent / "true-void-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    reference = build_reference(summary, audit)
    reference["source"]["summary_file"] = str(args.summary.resolve().relative_to(ROOT)).replace("\\", "/")
    reference["source"]["zero_void_audit_file"] = str(audit_path.resolve().relative_to(ROOT)).replace("\\", "/")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "reference": reference["full_solid_linear_elastic_beam_reference"],
                      "checks": reference["checks"]}, indent=2))


if __name__ == "__main__":
    main()
