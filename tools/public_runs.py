"""Select completed runs for the public documentation."""

from fnmatch import fnmatchcase
import hashlib
import json
from pathlib import Path


HIDDEN_HOME_IDS = {
    "completed--phase-field-at2-tension",
    "material-engineering--concrete-bending-large--hjc",
    "refinement--grid32--wave--neo-hookean",
    "refinement--grid64--wave--linear",
    "refinement--half_dt--wave--neo-hookean",
    "suite--wave--neo-hookean",
    "visible-motion--wave-grid32--neo-hookean",
    "visible-motion--wave-grid48--neo-hookean",
    "visible-motion--wave-grid64--neo-hookean",
    "visible-motion--wave-half-dt--neo-hookean",
}
HIDDEN_HOME_IDS.update({
    f"woven--{architecture}--{loading}"
    for architecture in ("plain", "triaxial", "twill-2x2")
    for loading in ("transverse-compression", "transverse-bending", "hemispherical-compression")
})


HOME_REPLACEMENTS = {
    "material-engineering--ceramic-indentation-wide--jh2": "completed--ceramic-impact-debris",
    "material-engineering--jc-tension--johnson-cook": "completed--jc-necking-refined",
    "material-engineering--concrete-bending--hjc": "material-engineering--concrete-bending-large--hjc",
    "material-engineering--concrete-bending-large--hjc": "material-engineering--concrete-bending-symmetric--hjc",
    "material-engineering--ceramic-indentation--jh2": "material-engineering--ceramic-indentation-wide--jh2",
    "adhesion--nonadhesive-pull-off": "adhesion-visible--nonadhesive-pull-off",
    "adhesion--adhesive-pull-off": "adhesion-visible--adhesive-pull-off",
    "suite--wave--neo-hookean": "visible-motion--wave-grid48--neo-hookean",
    "refinement--grid32--wave--neo-hookean": "visible-motion--wave-grid32--neo-hookean",
    "refinement--grid64--wave--linear": "visible-motion--wave-grid64--neo-hookean",
    "refinement--half_dt--wave--neo-hookean": "visible-motion--wave-half-dt--neo-hookean",
    "suite--cantilever--neo-hookean": "visible-motion--cantilever--neo-hookean",
    "suite--shear--neo-hookean": "visible-motion--shear--neo-hookean",
    "suite--torsion--neo-hookean": "visible-motion--torsion--neo-hookean",
    "suite--surface-load--neo-hookean": "visible-motion--surface-load--neo-hookean",
}
HOME_REPLACEMENTS.update({
    f"woven-preview--{architecture}--{loading}": f"woven--{architecture}--{loading}"
    for architecture in ("plain", "triaxial", "twill-2x2")
    for loading in ("warp-tension", "weft-tension", "in-plane-shear", "transverse-compression", "transverse-bending")
})


def is_public_run(identifier):
    parts = str(identifier).replace("\\", "/").replace("/", "--").lower().split("--")
    excluded_sources = ("compression", "compression_*", "dense*", "elastic_*", "elatic_*")
    return not (
        any(fnmatchcase(parts[0], pattern) for pattern in excluded_sources)
        or parts[:2] == ["suite", "compression"]
        or parts == ["adhesion", "dugdale-array"]
    )


def is_ready_for_publication(folder, identifier, summary):
    if not is_public_run(identifier):
        return False
    canonical = str(identifier).replace("\\", "/").replace("/", "--").lower()
    if canonical == "fibres--dry-three-strand-forming":
        return is_dry_three_strand_forming_ready(Path(folder), summary)
    if canonical == "fibres--three-strand-forming":
        return is_three_strand_forming_ready(Path(folder), summary)
    if canonical == "fibres--three-strand-tension":
        return is_three_strand_prefix_ready(Path(folder), summary)
    if summary.get("completed") is False:
        return False
    if canonical == "completed--phase-field-at2-tension":
        try:
            folder = Path(folder)
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("publication_status") == "withdrawn":
                return False
            crack = json.loads((folder / "crack-checks.json").read_text(encoding="utf-8"))
            media = json.loads((folder / "media-review.json").read_text(encoding="utf-8"))
            home = json.loads((folder / "homepage-media-review.json").read_text(encoding="utf-8"))
            image_root = folder.parents[1] / "images"
            gif = image_root / "home" / (canonical + ".gif")
            cover = image_root / (canonical + ".png")
            return (metadata["id"] == canonical and metadata["status"] == "completed"
                    and metadata["global_steps"] == summary["steps"] == 2800
                    and metadata["frames"] == media["frames"] == 113
                    and crack["passed"] is True and all(crack["criteria"].values())
                    and crack["components"][0]["extension"] >= crack["minimum_extension"]
                    and crack["components"][0]["thickness_fraction"] >= crack["minimum_thickness_fraction"]
                    and media["passed"] is True and media["annotations"] is False
                    and media["axes"] is False and media["resolution"] == [400, 400]
                    and media["all_decoded_frames_equal_exact_png_quantization"] is True
                    and home["passed"] is True
                    and home["full_113_gif_sha256"] == media["gif_sha256"]
                    and home["full_113_frames"] == 113 and home["home_gif_frames"] == 57
                    and home["source_indices"] == list(range(0, 113, 2))
                    and home["physical_steps"] == list(range(0, 2801, 50))
                    and home["resolution"] == media["resolution"]
                    and home["all_selected_frames_rgb_exact"] is True
                    and home["maximum_channel_error"] == home["changed_pixels"] == 0
                    and home["no_interpolation_or_fabricated_frames"] is True
                    and home["white_background_no_text_no_axes_unchanged"] is True
                    and home["threshold_and_geometry_unchanged"] is True
                    and home["first_saved_geometric_admission"]["retained"] is True
                    and home["last_frame"]["retained"] is True
                    and home["total_duration_ms"] == 11300
                    and home["home_gif_bytes"] < 10_000_000
                    and gif.stat().st_size == home["home_gif_bytes"]
                    and hashlib.sha256(gif.read_bytes()).hexdigest() == home["home_gif_sha256"]
                    and hashlib.sha256(cover.read_bytes()).hexdigest() == media["cover_sha256"])
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            return False
    if canonical == "completed--jc-necking-refined":
        if not is_archived_completed_ready(Path(folder), summary):
            return False
        try:
            physical = json.loads((Path(folder)/"physical-review.json").read_text())
            media = json.loads((Path(folder)/"media-audit.json").read_text())
            return (physical["passed"] is True and physical["full_protocol_completed"] is True
                    and physical["resolved_saved_history_fracture"] is True
                    and physical["coordinate_neck_audit"]["passed"] is True
                    and all(physical["fracture_gates"].values())
                    and media["pipeline_coordinates_verified"] is True
                    and media["smoke_only"] is False and media["frames_rendered"] == 61
                    and media["unique_png_sha256_count"] > 2
                    and media["gif"]["unique_decoded_pixel_states"] > 2
                    and media["gif"]["frames"] == 61)
        except (OSError, ValueError, KeyError, TypeError):
            return False
    if canonical in {"completed--woven-plate-impact", "completed--crossed-fibre-hertz", "completed--ceramic-impact-debris", "completed--crossed-fibre-jkr", "completed--crossed-fibre-md", "completed--crossed-fibre-self-consistent"}:
        return is_archived_completed_ready(Path(folder), summary)
    if canonical == "completed--radial-sinusoidal-pull-off":
        if not is_archived_completed_ready(Path(folder), summary, expected_frames=51):
            return False
        try:
            contact = json.loads((Path(folder)/"full-contact-review.json").read_text())
            mesh = json.loads((Path(folder)/"vtk-full-mesh-audit.json").read_text())
            return (contact["passed"] is True and all(contact["checks"].values())
                    and contact["frames_checked"] == mesh["actual_particle_frames_checked"] == 51
                    and mesh["passed"] is True and len(mesh["actual_warped_rigid_mesh_checks"]) == 6)
        except (OSError, ValueError, KeyError, TypeError):
            return False
    source = canonical.split("--")[0]
    if source == "woven-preview":
        return is_woven_preview_ready(Path(folder), canonical, summary)
    revised_material = canonical in (
        "material-engineering--jc-necking--johnson-cook",
        "material-engineering--concrete-bending-large--hjc",
        "material-engineering--concrete-bending-symmetric--hjc",
        "material-engineering--ceramic-indentation-wide--jh2",
    )
    if source not in ("adhesion", "adhesion-visible", "visible-motion", "woven") and not revised_material:
        return True
    if summary.get("completed") is not True:
        return False
    folder = Path(folder)
    try:
        review = json.loads((folder / "interpretation.json").read_text(encoding="utf-8"))
        if not isinstance(review, dict) or review.get("passed") is not True:
            return False
        checks = review.get("checks")
        sources = review.get("sources")
        if not isinstance(checks, dict) or not checks or not isinstance(sources, dict):
            return False
        if not all(value is True for value in checks.values()):
            return False
        for name in ("summary.json", "diagnostics.csv", "observables.csv"):
            content = (folder / name).read_bytes().replace(b"\r\n", b"\n")
            if sources.get(name) != hashlib.sha256(content).hexdigest():
                return False
        if canonical == "material-engineering--concrete-bending-symmetric--hjc":
            audit_bytes = (folder / "runtime-symmetry.json").read_bytes()
            audit = json.loads(audit_bytes)
            reference = review.get("runtime_symmetry_report", {})
            if not (reference.get("file") == "runtime-symmetry.json"
                    and reference.get("sha256") == hashlib.sha256(audit_bytes).hexdigest()
                    and audit.get("passed") is True and audit.get("checks")
                    and all(value is True for value in audit["checks"].values())
                    and audit.get("thresholds") == {"maximum_saved_damage_mirror_rms": .01, "maximum_saved_left_right_damaged_fraction_difference": .01}):
                return False
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False
    return True


def is_dry_three_strand_forming_ready(folder, summary):
    """Admit the complete dry process while retaining the failed tight screen."""
    try:
        read = lambda name: json.loads((folder/name).read_text(encoding="utf-8"))
        physical, media, render, snapshots, ledger, review = map(read, (
            "physical-review.json", "media-review.json", "render.json",
            "source-snapshots.json", "full-trajectory-ledger.json", "publication-review.json"))
        state, coverage = physical["state"], physical["coverage"]
        checkpoint = "06ff311f05f4e06bee30c08fb75353babf68fbfce9d9e9b3439e2adbb11366c5"
        scope = summary["accuracy_scope"]
        required = {"complete_saved_process", "source_bound_states", "physical_scope_explicit",
                    "actual_paraview", "source_bound_media", "visual_review", "mechanical_screen_false_preserved"}
        if not (summary["completed"] is True and summary["production_eligible"] is False
                and summary["case"] == "Dry three-strand forming"
                and summary["steps"] == state["step"] == ledger["steps"] == 12394
                and abs(summary["simulation_time"]-.06) < 1e-12
                and summary["frames"] == render["frames_rendered"] == media["frames"] == 61
                and summary["source_checkpoint_sha256"] == state["checkpoint_sha256"] == media["source_checkpoint_sha256"] == checkpoint
                and summary["source_commit"] == state["source"] == "5d2e4149f588c5455052d18c39cc58e45782615f"
                and state["passed"] is True and state["full_turn"] is True and state["all_geometry_valid"] is True
                and state["minJ"] > 0 and summary["max_damage"] == 0
                and state["full_turn_geometry"]["full_support"] is True
                and state["full_turn_geometry"]["collar_error_m"] < 1e-9
                and coverage["all_deep_checks_pass"] is True
                and scope["three_pair_70percent_tight_contact_screen_passed"] is False
                and coverage["major_grid_contact_candidate"] is False
                and scope["free_span_contact_fractions"] == [q["free_span_contact_fraction"] for q in coverage["pairs"]]
                and scope["energy_temporal_spatial_convergence_certified"] is False
                and scope["manufacturing_accuracy_certified"] is False
                and scope["whole_trajectory_energy_remainder_over_absolute_work"] == ledger["remainder_over_absolute_fixture_work"]
                and scope["final_segment_energy_remainder_over_absolute_work"] == state["remainder_over_abswork"]
                and review["schema"] == "dry-forming-process-publication-v1" and review["passed"] is True
                and set(review["checks"]) == required and all(review["checks"].values())
                and render["renderer"] == "ParaView" and render["geometry_scale"] == 1
                and render["resolution"] == media["resolution"] == [400,400]
                and render["axes"] is False and render["annotations"] is False
                and render["camera_position"] == [-.020,.016,.012]
                and render["full_turn_completed"] is snapshots["full_turn_completed"] is True
                and render["actual_steps"] == [f["step"] for f in snapshots["frames"]]
                and render["actual_times_s"] == [f["time_s"] for f in snapshots["frames"]]
                and render["actual_steps"][0] == 16 and render["actual_steps"][-1] == 12394
                and media["passed"] is media["visual_review_passed"] is True
                and media["source_manifest_sha256"] == hashlib.sha256((folder/"source-snapshots.json").read_bytes().replace(b"\r\n",b"\n")).hexdigest()
                and media["position_interpolation"] is False and media["decoded_frames"] == 61
                and media["distinct_decoded_frames"] == 61 and media["all_frame_timings_match"] is True and media["loop"] == 0):
            return False
        required_sources = {"summary.json","physical-review.json","media-review.json","render.json",
                            "original-render.json","original-render-review.json","source-snapshots.json","full-trajectory-ledger.json","provenance.json"}
        if not required_sources <= set(review["sources"]):return False
        for name,digest in review["sources"].items():
            if Path(name).name != name or hashlib.sha256((folder/name).read_bytes().replace(b"\r\n",b"\n")).hexdigest() != digest:return False
        images=folder.parents[1]/"images";gif=images/"home/fibres--dry-three-strand-forming.gif";cover=images/"dry-three-strand-forming.png"
        return gif.stat().st_size == media["gif_bytes"] and hashlib.sha256(gif.read_bytes()).hexdigest() == media["gif_sha256"] and hashlib.sha256(cover.read_bytes()).hexdigest() == media["cover_sha256"]
    except (OSError,ValueError,TypeError,KeyError,IndexError):return False


def is_three_strand_forming_ready(folder, summary):
    """Admit a complete illustrative process with explicit accuracy limits."""
    try:
        read = lambda name: json.loads((folder / name).read_text())
        review, physical, media, render, snapshots = map(read, (
            "publication-review.json", "physical-review.json", "media-review.json",
            "render.json", "source-snapshots.json"))
        checkpoint = "1e807f9aae76e6df5970b2d1e7cfa73323d182a62d167b4183f0987e6af2d2e2"
        required = {"complete_saved_process", "source_bound_states", "physical_scope_explicit",
                    "actual_paraview", "source_bound_media", "visual_review"}
        if not (summary["completed"] is True and summary["production_eligible"] is False
                and summary["steps"] == snapshots["last_step"] == 5164
                and summary["frames"] == render["frames_rendered"] == media["frames"] == 63
                and summary["source_checkpoint_sha256"] == physical["checkpoint_sha256"] == snapshots["checkpoint_sha256"] == checkpoint
                and summary["source_commit"] == physical["source"] == "ab410cfdb4b864f4481c2d3254752df9b0e53049"
                and snapshots["full_turn_completed"] is True and physical["full_turn_completed"] is True
                and physical["all_geometry_valid"] is True and physical["all_capacity_valid"] is True
                and physical["latest_geometry"]["full_support"] is True
                and physical["latest_geometry"]["collar_error_m"] < 1e-9
                and physical["minJ"] > 0 and physical["contact_loss_negative_steps"] == 0
                and physical["production_eligible"] is False
                and summary["min_J"] == physical["minJ"]
                and summary["accuracy_scope"]["energy_temporal_spatial_convergence_certified"] is False
                and summary["accuracy_scope"]["final_deceleration_energy_remainder_ratio"] == physical["remainder_ratio"]
                and summary["accuracy_scope"]["manufacturing_accuracy_certified"] is False
                and review["schema"] == "illustrative-process-publication-v1" and review["passed"] is True
                and set(review["checks"]) == required and all(review["checks"].values())
                and render["renderer"] == "ParaView" and render["geometry_scale"] == 1
                and render["resolution"] == media["resolution"] == [400, 400]
                and render["axes"] is False and render["annotations"] is False
                and render["full_turn_completed"] is True
                and render["actual_steps"] == [f["step"] for f in snapshots["frames"]]
                and render["actual_times_s"] == [f["time_s"] for f in snapshots["frames"]]
                and render["actual_steps"][0] == 0 and render["actual_steps"][-1] == 5164
                and media["passed"] is True and media["visual_review_passed"] is True
                and media["source_manifest_sha256"] == hashlib.sha256((folder/"source-snapshots.json").read_bytes()).hexdigest()
                and media["position_interpolation"] is False and media["decoded_frames"] == 63
                and media["all_frame_timings_match"] is True and media["loop"] == 0
                and media["distinct_decoded_frames"] == 63):
            return False
        for name, digest in review["sources"].items():
            if Path(name).name != name or hashlib.sha256((folder/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest:
                return False
        if not {"summary.json", "physical-review.json", "media-review.json", "render.json", "provenance.json", "source-snapshots.json", "original-render-review.json"} <= set(review["sources"]):
            return False
        images = folder.parents[1] / "images"
        gif, cover = images/"home/fibres--three-strand-forming.gif", images/"three-strand-forming.png"
        return (gif.stat().st_size == media["gif_bytes"]
                and hashlib.sha256(gif.read_bytes()).hexdigest() == media["gif_sha256"]
                and hashlib.sha256(cover.read_bytes()).hexdigest() == media["cover_sha256"])
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def is_three_strand_prefix_ready(folder, summary):
    """Admit the reviewed loading segment without marking the protocol complete."""
    checkpoint = "4152797ee8f85ab2b551bf721efeaabd48402775ee74bd08d5c1fea3dc8d316e"
    try:
        review = json.loads((folder / "publication-review.json").read_text())
        physical = json.loads((folder / "physical-review.json").read_text())
        media = json.loads((folder / "media-review.json").read_text())
        render = json.loads((folder / "render.json").read_text())
        required = {"saved_prefix_verified", "independent_saved_state_checks", "actual_paraview",
                    "source_bound_media", "visual_review", "partial_protocol_explicit"}
        if not (
            summary["completed"] is False and summary["displayed_segment_complete"] is True
            and summary["steps"] == physical["steps"] == 3500
            and summary["last_saved_frame"] == physical["frame"] == 12
            and summary["frames"] == render["frames_rendered"] == media["frames"] == 13
            and summary["source_checkpoint_sha256"] == physical["checkpoint_sha256"] == checkpoint
            and media["source_checkpoint_sha256"] == render["source_checkpoint_sha256"] == checkpoint
            and summary["source_run"] == 40116768349 and summary["source_artifact"] == 12569841297
            and all(physical["saved_array_checks"].values())
            and physical["f12_material"]["all_fields_finite"] is True
            and physical["all_rows_online_contact_valid"] is True
            and physical["max_abs_energy_ledger_over_max_work"] < .15
            and review["schema"] == "saved-prefix-publication-v1" and review["passed"] is True
            and set(review["checks"]) == required and all(review["checks"].values())
            and render["renderer"] == "ParaView" and render["geometry_scale"] == 1
            and render["resolution"] == media["resolution"] == [400, 400]
            and render["axes"] is False and render["annotations"] is False
            and media["passed"] is True and media["visual_review_passed"] is True
            and media["no_interpolation"] is True and media["actual_source_frames"] == list(range(13))
            and media["unique_png_states"] > 2 and media["decoded_frames_match_quantized_pngs"] is True
        ):
            return False
        names = {"summary.json", "physical-review.json", "render.json", "media-review.json", "provenance.json"}
        if not names <= set(review["sources"]):
            return False
        for name, expected in review["sources"].items():
            if Path(name).name != name:
                return False
            if hashlib.sha256((folder / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected:
                return False
        images = folder.parents[2] / "docs/images"
        gif = images / "home/fibres--three-strand-tension.gif"
        cover = images / "three-strand-tension.png"
        return (gif.stat().st_size == media["gif_bytes"]
                and hashlib.sha256(gif.read_bytes()).hexdigest() == media["gif_sha256"]
                and hashlib.sha256(cover.read_bytes()).hexdigest() == media["cover_sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return False


def is_archived_completed_ready(folder, summary, *, expected_frames=61):
    """Keep imported full-protocol media bound to its reviewed publication files."""
    try:
        review = json.loads((folder / "publication-review.json").read_text(encoding="utf-8"))
        render = json.loads((folder / "render.json").read_text(encoding="utf-8"))
        required = {"complete_numerical_protocol", "independent_physical_checks", "actual_paraview",
                    "full_saved_history", "source_bound_media", "visual_review"}
        if (summary.get("completed") is not True or review.get("schema") != "completed-media-publication-v1"
                or review.get("passed") is not True or set(review.get("checks", {})) != required
                or not all(value is True for value in review["checks"].values())
                or render.get("renderer") != "ParaView" or render.get("frames_rendered") != expected_frames
                or render.get("axes") is not False or render.get("annotations") is not False
                or render.get("geometry_scale") != 1):
            return False
        if not {"summary.json", "render.json", "physical-review.json", "provenance.json"} <= set(review["sources"]):
            return False
        for name, expected in review["sources"].items():
            if Path(name).name != name:
                return False
            raw = (folder / name).read_bytes().replace(b"\r\n", b"\n")
            if hashlib.sha256(raw).hexdigest() != expected:
                return False
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False
    return True


def is_woven_preview_ready(folder, identifier, summary):
    if summary.get("completed") is not True or identifier not in HOME_REPLACEMENTS:
        return False
    try:
        _, architecture, loading = identifier.split("--")
        definition = summary["definition"]
        if (definition["architecture"], definition["loading"]) != (architecture, loading):
            return False
        eligibility = json.loads((folder / "preview-eligibility.json").read_text(encoding="utf-8"))
        review = json.loads((folder / "preview-review.json").read_text(encoding="utf-8"))
        original = json.loads((folder / "interpretation.json").read_text(encoding="utf-8"))
        allowed = {"at_least_one_hundred_thousand_particles", "grid_resolves_yarn_height"}
        if loading == "transverse-compression":
            allowed.add("required_moving_samples_exist")
        checks = review["checks"]
        if not checks or any(type(value) is not bool for value in checks.values()):
            return False
        failed = {key for key, value in checks.items() if not value}
        required = (
            "positive_final_stored_elastic_energy", "final_piola_section_forces_match",
            "final_reference_section_moment_matches",
        )
        if not (
            eligibility["kind"] == "coarse-woven-preview-v1"
            and eligibility["eligible"] is True
            and eligibility["review_file"] == "preview-review.json"
            and eligibility["source_review_passed"] is False and review["passed"] is False
            and original.get("passed") is False
            and failed and failed <= allowed
            and set(eligibility["allowed_failed_checks"]) == allowed
            and set(eligibility["observed_failed_checks"]) == failed
            and all(checks.get(name) is True for name in required)
            and isinstance(eligibility["limitations"], list) and eligibility["limitations"]
        ):
            return False
        names = ("summary.json", "diagnostics.csv", "observables.csv", "interpretation.json", "preview-review.json")
        if set(eligibility["sources"]) != set(names):
            return False
        for name in names:
            content = (folder / name).read_bytes().replace(b"\r\n", b"\n")
            digest = hashlib.sha256(content).hexdigest()
            if eligibility["sources"][name] != digest:
                return False
            if name in names[:3] and review["sources"].get(name) != digest:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def check_home_adhesion_pair(root, catalogue):
    indexed = {record["id"]: record for record in catalogue}
    roles = {"nonadhesive": "nonadhesive-pull-off", "adhesive": "adhesive-pull-off"}
    identifiers = {role: "adhesion-visible--" + name for role, name in roles.items()}
    if not any(identifier in indexed for identifier in identifiers.values()):
        return
    if not all(identifier in indexed for identifier in identifiers.values()):
        raise ValueError("Both members of the lift-hold pair must be published together")
    results = Path(root) / "docs/results"
    try:
        report = json.loads((results / "adhesion-visible-comparison.json").read_text(encoding="utf-8"))
        checks = report["checks"]
        if report.get("passed") is not True or not checks or not all(value is True for value in checks.values()) or checks.get("lift_hold_has_visible_deformation_contrast") is not True:
            raise ValueError("The lift-hold pair has no passed deformation comparison")
        if set(report["sources"]) != set(roles):
            raise ValueError("The lift-hold pair has different comparison roles")
        for role, identifier in identifiers.items():
            hashes = report["sources"][role]["files"]
            for name in ("summary.json", "observables.csv", "diagnostics.csv", "interpretation.json"):
                digest = hashlib.sha256((results / identifier / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
                if hashes.get(name) != digest or indexed[identifier]["sha256"].get(name) != digest:
                    raise ValueError(f"Lift-hold comparison source changed: {identifier}/{name}")
        table = report["csv"]
        if table["filename"] != "adhesion-visible-comparison.csv":
            raise ValueError("The lift-hold response table has a different name")
        digest = hashlib.sha256((results / table["filename"]).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if table.get("lf_sha256") != digest:
            raise ValueError("The lift-hold response table has changed")
    except (OSError, KeyError, TypeError) as error:
        raise ValueError("The lift-hold pair requires its complete archived comparison") from error


def homepage_records(root, catalogue):
    """Select reviewed replacements while retaining their numerical references."""
    identifiers = [record["id"] for record in catalogue]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Run catalogue contains duplicate identifiers")
    for identifier in identifiers:
        folder = Path(root) / "docs/results" / identifier
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if identifier == "completed--phase-field-at2-tension":
            metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
            if metadata.get("publication_status") == "withdrawn":
                continue
        if not is_ready_for_publication(folder, identifier, summary):
            raise ValueError(f"Run has no matching passed review: {identifier}")
    present = set(identifiers)
    check_home_adhesion_pair(root, catalogue)
    replaced = {old for old, new in HOME_REPLACEMENTS.items() if new in present}
    return [
        record for record in catalogue
        if record["id"] not in replaced and record["id"] not in HIDDEN_HOME_IDS
        and not record["id"].startswith("woven-preview--")
    ]
