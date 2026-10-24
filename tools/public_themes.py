"""Themes shared by the homepage and gallery."""

THEMES = (
    "Metal plasticity and impact",
    "Concrete",
    "Ceramics",
    "Woven plate impact",
    "Plain weave: in-plane loading",
    "Triaxial weave: in-plane loading",
    "Twill weave: in-plane loading",
    "Other woven composites",
    "Adhesive contact and pull-off",
    "Nonadhesive contact",
    "Fluid dynamics",
    "Fluid-solid interaction",
    "Structural elasticity",
    "Numerical stage previews",
)


def theme(value):
    value = value.lower()
    if "metal-stage" in value:
        return THEMES[13]
    if any(s in value for s in (
        "jc-tension", "johnson-cook", "metal-sphere", "metal-impact",
        "metal-linear", "5kms",
    )):
        return THEMES[0]
    if any(s in value for s in ("concrete", "hjc", "rht")):
        return THEMES[1]
    if any(s in value for s in ("ceramic", "jh2")):
        return THEMES[2]
    if "woven" in value:
        if any(s in value for s in ("plate-impact", "deep-impact", "woven-impact")):
            return THEMES[3]
        if "plain" in value:
            return THEMES[4]
        if "triaxial" in value:
            return THEMES[5]
        if "twill" in value:
            return THEMES[6]
        return THEMES[7]
    if any(s in value for s in (
        "adhesion", "adhesive", "pull-off", "sinusoid", "jkr", "dugdale",
        "maugis", "self-consistent", "crossed-fibre-md", "crossed-fibre-sc",
        "lateral-release", "three-sphere",
    )):
        return THEMES[8]
    if any(s in value for s in (
        "crossed-fibre", "sphere-contact", "head-on", "oblique", "sliding",
        "rebound", "rubber-ball", "ball-beam", "hertz",
    )):
        return THEMES[9]
    if "unbaffled-tank" in value:
        return THEMES[10]
    if any(s in value for s in (
        "fsi", "elastic-gate", "floating-block", "water-entry", "baffled-tank",
    )):
        return THEMES[11]
    if any(s in value for s in ("tank", "dam-break", "sloshing")):
        return THEMES[10]
    return THEMES[12]


def order(value):
    value = value.lower()
    rank = (
        0 if "nonadhesive-pull-off" in value or "nonadhesive pull-off" in value
        else 1 if "adhesive-pull-off" in value or "adhesive pull-off" in value
        else 2 if "sinusoid" in value
        else 3
    )
    return rank, value


def metal_stage_cell(prefix="docs/"):
    return (
        '<img src="' + prefix + 'images/stages/metal-impact-f16.gif" '
        'alt="Metal impact numerical stage preview" width="100%"><br>'
        '<a href="' + prefix + 'stages/metal-impact-f16/README.md">'
        'Metal impact: f16 numerical stage</a><br>'
        '17 saved states; phase/EOS extrapolation. Full protocol pending.'
    )


def ceramic_section_cell(prefix="docs/"):
    return (
        '<img src="' + prefix + 'images/views/ceramic-cross-section.gif" '
        'alt="JH2 ceramic central cross section" width="100%"><br>'
        '<a href="' + prefix + 'views/ceramic-cross-section/README.md">'
        'JH2 ceramic central cross section</a><br>'
        'Second view of the same 61-state result.'
    )
